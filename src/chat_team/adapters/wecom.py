"""WeCom (企业微信) AI Bot adapter — long-connection (WebSocket) mode.

Implements the wire protocol documented in ``docs/wechat_bot_api.md`` /
``docs/wechat_bot_接收消息.md`` directly on top of ``websockets``. Three
co-operating asyncio tasks run while a connection is alive:

* **reader**  — pulls frames from the socket, dispatches by ``cmd``.
* **heartbeat** — sends ``{"cmd": "ping"}`` every 30 seconds.
* **writer**  — single drain task on an ``asyncio.Queue`` that holds JSON
  payloads to send. All writes go through this so frames cannot interleave.

Each ``aibot_msg_callback`` is dispatched as a fire-and-forget task so a
slow user turn does not block the reader.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import websockets

from ..config import Settings
from . import wecom_media
from .base import (
    BotAdapter,
    ChatType,
    ContentBlock,
    IncomingMessage,
    MessageHandler,
    StreamHandle,
    blocks_to_text,
    coalesce_text_blocks,
)

WorkspaceResolver = Callable[[str], Path]

log = logging.getLogger(__name__)

WECOM_WS_URL = "wss://openws.work.weixin.qq.com"
HEARTBEAT_INTERVAL = 30
STREAM_PUSH_MIN_INTERVAL = 1.0          # seconds between intermediate stream frames
WRITE_QUEUE_MAXSIZE = 1024
INBOUND_PREFETCH_TIMEOUT_SECONDS = 240.0
STREAM_CONTENT_MAX_BYTES = 2048            # WeCom silently drops larger stream refresh/final frames
MARKDOWN_CONTENT_MAX_BYTES = 20 * 1024     # documented markdown.content limit

# Reconnect backoff: 1s, 2s, 4s, … capped at 5 min; +0.5s jitter per attempt.
RECONNECT_INITIAL_DELAY = 1.0
RECONNECT_MAX_DELAY = 300.0
RECONNECT_JITTER_CAP = 0.5

UPLOAD_CHUNK_SIZE = 256 * 1024          # raw bytes per chunk; ~341KB base64, under 512KB cap
UPLOAD_RESPONSE_TIMEOUT = 30.0          # per-step ack timeout
MEDIA_SIZE_LIMITS = {
    "image": 10 * 1024 * 1024,
    "file": 20 * 1024 * 1024,
}

_MENTION_RE = re.compile(r"^@\S+\s+")

# Slash command matching. Anchored at start, requires a word boundary after
# the command name so "/newton" does NOT match "/new". English names are
# case-insensitive; Chinese aliases are normalised to their canonical names.
# Group 1 captures the bare command name.
_SLASH_CMD_RE = re.compile(
    r"^/(new|stop|status|running|刷新|停止|状态)\b",
    re.IGNORECASE,
)
_SLASH_CMD_ALIASES = {
    "刷新": "new",
    "停止": "stop",
    "状态": "status",
}
# Commands that operate on the current session (group OR private).
_SESSION_SLASH_CMDS = {"new", "stop", "status"}
# /running is private-chat only (it inspects OTHER sessions' busy state).
_PRIVATE_ONLY_SLASH_CMDS = {"running"}


def _strip_mention_from_first_text(blocks: list[ContentBlock]) -> list[ContentBlock]:
    """Remove a leading ``@bot `` mention from the first text block.
    Image / non-text blocks before the first text block are preserved as-is.
    Quote markers (added later) live after this strip is applied to the
    current-message side, so quote interiors are never touched."""
    out = list(blocks)
    for i, block in enumerate(out):
        if block.get("type") == "text":
            text = (block.get("text") or "")
            stripped = _MENTION_RE.sub("", text, count=1).strip()
            out[i] = {"type": "text", "text": stripped}
            break
    return out


def _new_req_id() -> str:
    return uuid.uuid4().hex


class _LRU(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def add(self, key: str) -> bool:
        """Return True if key was newly inserted, False if already present."""
        if key in self:
            self.move_to_end(key)
            return False
        self[key] = True
        if len(self) > self.capacity:
            self.popitem(last=False)
        return True


@dataclass
class _InboundTurn:
    """One inbound message after the reader has assigned its FIFO slot.

    Media resolution runs concurrently across turns, but dispatcher entry is
    released by ``seq`` so user-visible model processing stays in receive
    order for each session.
    """

    session_id: str
    seq: int
    frame: dict[str, Any]
    inbound: IncomingMessage
    msgtype: str
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    drop: bool = False
    timed_out: bool = False


@dataclass
class _InboundQueue:
    next_seq: int = 0
    dispatch_seq: int = 0
    turns: dict[int, _InboundTurn] = field(default_factory=dict)
    skipped: set[int] = field(default_factory=set)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    worker: asyncio.Task | None = None


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to at most ``max_bytes`` of UTF-8 without splitting a codepoint."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class WeComStreamHandle:
    """Streaming reply backed by aibot_respond_msg / msgtype=stream.

    Throttles intermediate frames to ``STREAM_PUSH_MIN_INTERVAL`` seconds.
    Final ``finish()`` is always sent (un-throttled) with finish=true.
    """

    def __init__(self, adapter: "WeComBotAdapter", req_id: str):
        self._adapter = adapter
        self._req_id = req_id
        self._stream_id = uuid.uuid4().hex
        self._content = ""
        self._last_push = 0.0
        self._closed = False
        # Set by the adapter when WeCom rejects a stream frame with errcode
        # 846608 (stream update window expired, >10 minutes after the
        # inbound message). push()/status() become no-ops; finish() skips
        # the closing stream frame but still attempts the markdown reply.
        self.expired = False
        adapter._active_streams[req_id] = self

    async def push(self, chunk: str, *, append: bool = True) -> None:
        if self._closed or self.expired:
            return
        if not chunk.strip():
            return
        self._content = (self._content + chunk) if append else chunk
        if time.monotonic() - self._last_push < STREAM_PUSH_MIN_INTERVAL:
            return                                          # throttle silently
        await self._send_frame(self._content, finish=False)

    async def status(self, note: str) -> bool:
        """Push a transient status frame.

        Returns ``False`` once the stream is closed or WeCom-expired so
        long-running callers (the dispatcher progress heartbeat) can stop
        pushing instead of generating one rejected frame per interval.
        """
        if self._closed or self.expired:
            return False
        if time.monotonic() - self._last_push < STREAM_PUSH_MIN_INTERVAL:
            return True                      # throttled, but stream alive
        # status messages don't accumulate into the body; they're transient.
        await self._send_frame(f"{self._content}\n\n_{note}_" if self._content else f"_{note}_",
                               finish=False)
        return True

    async def finish(self, final_text: str) -> None:
        if self._closed:
            return
        self._closed = True
        cur = self._adapter._active_streams.get(self._req_id)
        if cur is self:
            self._adapter._active_streams.pop(self._req_id, None)
        text = final_text or "(空回复)"
        log.info(
            "wecom stream finish: role=%s req_id=%s bytes=%d expired=%s",
            self._adapter.role_name, self._req_id, len(text.encode("utf-8")),
            self.expired,
        )
        # Close the live "思考中…" stream with a plain-text frame first, then
        # send the actual response as a markdown message.  WeCom stream frames
        # are intended for transient plain-text status; sending markdown inside
        # ``stream.content`` (tables/heavy bold/headings) often fails silently
        # and leaves the prior spinner text on screen.  When the stream is
        # already WeCom-expired the closing frame would just be rejected with
        # 846608 — skip it, but still attempt the markdown reply.
        if not self.expired:
            await self._send_frame("处理完成。", finish=True)
        await self._adapter._send_markdown_reply(self._req_id, text)

    async def _send_frame(self, content: str, *, finish: bool) -> None:
        content = _truncate_utf8(content, STREAM_CONTENT_MAX_BYTES)
        payload = {
            "cmd": "aibot_respond_msg",
            "headers": {"req_id": self._req_id},
            "body": {
                "msgtype": "stream",
                "stream": {
                    "id": self._stream_id,
                    "finish": finish,
                    "content": content,
                },
            },
        }
        await self._adapter._enqueue_write(payload)
        self._last_push = time.monotonic()

    async def send_image(self, path: Path, *, filename: str | None = None) -> None:
        await self._send_media(path, kind="image", filename=filename)

    async def send_file(self, path: Path, *, filename: str | None = None) -> None:
        await self._send_media(path, kind="file", filename=filename)

    async def _send_media(self, path: Path, *, kind: str, filename: str | None) -> None:
        data = await asyncio.to_thread(path.read_bytes)
        name = filename or path.name
        media_id = await self._adapter.upload_media(data, kind=kind, filename=name)
        payload = {
            "cmd": "aibot_respond_msg",
            "headers": {"req_id": self._req_id},
            "body": {"msgtype": kind, kind: {"media_id": media_id}},
        }
        await self._adapter._enqueue_write(payload)


class WeComBotAdapter(BotAdapter):
    def __init__(
        self,
        settings: Settings,
        workspace_resolver: WorkspaceResolver | None = None,
        *,
        bot_id: str | None = None,
        secret: str | None = None,
        role_name: str | None = None,
    ):
        self.settings = settings
        self.bot_id = bot_id or ""
        self.secret = secret or ""
        self.role_name = role_name
        self._handler: MessageHandler | None = None
        # Per-adapter (lifetime) state ----------------------------------
        # ``_msgid_lru`` survives across reconnects so server-side replays
        # are still deduped.
        self._msgid_lru = _LRU(settings.session.msgid_lru_size)
        self._workspace_resolver = workspace_resolver
        # ``_shutdown`` is set only by close()/SIGINT; signals the outer
        # run_forever loop to exit.
        self._shutdown = asyncio.Event()
        # Strong refs to in-flight callback handlers. asyncio only weakly
        # references tasks (CPython implementation detail) — without this
        # set, a slow callback can be garbage-collected mid-execution and
        # silently disappear under load. Spans reconnects so an in-flight
        # turn survives a brief WS blip.
        self._bg_tasks: set[asyncio.Task] = set()
        # Per-session inbound ordering. Message media is prefetched by the
        # callback tasks concurrently, while model dispatch is released in
        # the exact order the reader assigned these sequence numbers.
        self._inbound_queues: dict[str, _InboundQueue] = {}
        # Per-connection state (rebuilt each connect iteration) ---------
        self._ws: Any = None
        self._write_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(WRITE_QUEUE_MAXSIZE)
        self._tasks: list[asyncio.Task] = []
        # ``_connection_dead`` flips when the current connection ends for
        # any reason (server close, disconnected_event, write error, …).
        # Recreated per connection so a fresh wait() resolves correctly.
        self._connection_dead = asyncio.Event()
        self._pending_acks: dict[str, asyncio.Future] = {}
        # Live stream handles keyed by inbound req_id. Populated by
        # WeComStreamHandle.__init__, removed by finish(). Used by
        # _dispatch_ack to mark a stream expired when WeCom answers a
        # stream frame with errcode 846608 (>10min update window), so the
        # progress heartbeat stops pushing into a dead stream.
        self._active_streams: dict[str, "WeComStreamHandle"] = {}

    # ---- BotAdapter interface ---------------------------------------------

    def set_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def connect(self) -> None:
        """Open one WS connection and subscribe.

        Kept for backward-compat with code/tests that drive ``connect()`` +
        ``run()`` directly. In production use ``run_forever()`` instead,
        which handles transient connection loss with exponential backoff.
        """
        if not self.bot_id or not self.secret:
            raise RuntimeError("WECOM_BOT_ID / WECOM_SECRET missing — set them under 'bots:' in ~/.chat_team/config.yaml")
        log.info("connecting to %s", WECOM_WS_URL)
        # WeCom uses application-level heartbeat ({"cmd":"ping"} every ~30s);
        # disable the websockets library's protocol-level auto-ping so it does
        # not close the socket when WeCom predictably ignores it.
        self._ws = await websockets.connect(
            WECOM_WS_URL,
            max_size=8 * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
        )
        await self._subscribe()

    async def run(self) -> None:
        """Run one connection until it dies. Use ``run_forever`` in prod."""
        if self._ws is None:
            raise RuntimeError("call connect() before run()")
        self._tasks = [
            asyncio.create_task(self._writer_loop(), name="wecom-writer"),
            asyncio.create_task(self._heartbeat_loop(), name="wecom-heartbeat"),
            asyncio.create_task(self._reader_loop(), name="wecom-reader"),
        ]
        try:
            await self._connection_dead.wait()
        finally:
            await self._tear_down_connection()

    async def run_forever(self) -> None:
        """Production entry point: keep reconnecting until ``close()``.

        On any non-shutdown disconnect, sleeps with exponential backoff
        (1s, 2s, 4s, …, capped at 5 min plus jitter) then reconnects.
        Backoff resets to 1s after a connection that survived long enough
        to subscribe successfully.
        """
        backoff = RECONNECT_INITIAL_DELAY
        while not self._shutdown.is_set():
            try:
                await self._open_connection()
                # Subscribe succeeded; future disconnects are "transient".
                backoff = RECONNECT_INITIAL_DELAY
                await self._serve_one_connection()
            except Exception as exc:                             # noqa: BLE001
                log.warning("ws connection ended: %r", exc)
            finally:
                await self._tear_down_connection()
            if self._shutdown.is_set():
                break
            sleep_for = backoff + random.uniform(0, RECONNECT_JITTER_CAP)
            log.info("reconnecting in %.2fs", sleep_for)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=sleep_for)
                break                                            # shutdown won the race
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, RECONNECT_MAX_DELAY)

    async def close(self) -> None:
        self._shutdown.set()
        self._connection_dead.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:                                # noqa: BLE001
                pass
            self._ws = None

    # ---- connection lifecycle --------------------------------------------

    async def _open_connection(self) -> None:
        """Reset per-connection state, open socket, subscribe."""
        if not self.bot_id or not self.secret:
            raise RuntimeError("WECOM_BOT_ID / WECOM_SECRET missing — set them under 'bots:' in ~/.chat_team/config.yaml")
        self._connection_dead = asyncio.Event()
        self._write_queue = asyncio.Queue(WRITE_QUEUE_MAXSIZE)
        self._pending_acks = {}
        log.info("connecting to %s", WECOM_WS_URL)
        self._ws = await websockets.connect(
            WECOM_WS_URL,
            max_size=8 * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
        )
        await self._subscribe()

    async def _serve_one_connection(self) -> None:
        """Start the 3 cooperating tasks and wait for the connection to die."""
        self._tasks = [
            asyncio.create_task(self._writer_loop(), name="wecom-writer"),
            asyncio.create_task(self._heartbeat_loop(), name="wecom-heartbeat"),
            asyncio.create_task(self._reader_loop(), name="wecom-reader"),
        ]
        await self._connection_dead.wait()

    async def _tear_down_connection(self) -> None:
        """Cancel per-connection tasks, fail any in-flight ack futures, drop
        the websocket. Safe to call multiple times (idempotent)."""
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):      # noqa: BLE001
                pass
        self._tasks = []
        # Any caller blocked on _send_and_await must wake up with an error
        # rather than hang forever once the socket is gone.
        for req_id, fut in list(self._pending_acks.items()):
            if not fut.done():
                fut.set_exception(
                    ConnectionResetError(f"connection lost; req_id={req_id}")
                )
        self._pending_acks = {}
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:                                # noqa: BLE001
                pass
            self._ws = None

    # ---- internals --------------------------------------------------------

    async def _subscribe(self) -> None:
        req_id = _new_req_id()
        payload = {
            "cmd": "aibot_subscribe",
            "headers": {"req_id": req_id},
            "body": {"bot_id": self.bot_id, "secret": self.secret},
        }
        await self._ws.send(json.dumps(payload, ensure_ascii=False))
        # Wait for the subscribe ack before declaring the connection live.
        raw = await asyncio.wait_for(self._ws.recv(), timeout=15)
        msg = json.loads(raw)
        if msg.get("errcode") not in (0, None):
            raise RuntimeError(f"subscribe failed: {msg!r}")
        log.info("subscribe ok: %s", msg)

    async def _writer_loop(self) -> None:
        while True:
            payload = await self._write_queue.get()
            if payload is None:
                return
            try:
                await self._ws.send(json.dumps(payload, ensure_ascii=False))
            except Exception:                                # noqa: BLE001
                log.warning("ws write failed; signalling reconnect", exc_info=True)
                self._connection_dead.set()
                return

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await self._enqueue_write({"cmd": "ping", "headers": {"req_id": _new_req_id()}})

    async def _reader_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("dropping non-json frame: %r", raw[:200])
                    continue
                cmd = msg.get("cmd") or ""
                if cmd in ("aibot_msg_callback", "aibot_event_callback"):
                    body = msg.get("body") or {}
                    log.info(
                        "wecom inbound callback: cmd=%s msgtype=%s session=%s msgid=%s",
                        cmd,
                        body.get("msgtype"),
                        self._session_id_from_body(body),
                        body.get("msgid"),
                    )
                if cmd == "aibot_msg_callback":
                    order_key = self._reserve_inbound_order(msg)
                    self._spawn_bg(
                        self._handle_msg_callback(msg, order_key), name="wecom-msg-cb",
                    )
                elif cmd == "aibot_event_callback":
                    self._spawn_bg(
                        self._handle_event_callback(msg), name="wecom-event-cb",
                    )
                elif cmd == "" and msg.get("errmsg") is not None:
                    self._dispatch_ack(msg)                 # upload/heartbeat/subscribe acks
                else:
                    log.debug("frame ignored: cmd=%s", cmd)
        except websockets.ConnectionClosed as err:
            log.warning("ws closed: %s", err)
        finally:
            # Always flag the connection dead so run_forever can reconnect
            # (or run() can return cleanly). _shutdown is unaffected.
            self._connection_dead.set()

    async def _enqueue_write(self, payload: dict[str, Any]) -> None:
        await self._write_queue.put(payload)

    async def _send_markdown_reply(self, req_id: str, content: str) -> None:
        """Send a full-size answer as a markdown reply after the short stream close."""
        content = _truncate_utf8(content, MARKDOWN_CONTENT_MAX_BYTES)
        payload = {
            "cmd": "aibot_respond_msg",
            "headers": {"req_id": req_id},
            "body": {
                "msgtype": "markdown",
                "markdown": {"content": content},
            },
        }
        await self._enqueue_write(payload)

    def _spawn_bg(self, coro, *, name: str | None = None) -> asyncio.Task:
        """Create a background task and hold a strong reference to it until
        completion. Without this, asyncio's weak-ref-only Task tracking can
        let an in-flight handler be garbage-collected mid-await — Python's
        docs explicitly warn about this. ``add_done_callback`` releases the
        ref when the task finishes."""
        task = asyncio.create_task(coro, name=name)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    def _queue_for(self, session_id: str) -> _InboundQueue:
        q = self._inbound_queues.get(session_id)
        if q is None:
            q = _InboundQueue()
            self._inbound_queues[session_id] = q
        return q

    def _reserve_inbound_order(self, frame: dict[str, Any]) -> tuple[str, int] | None:
        session_id = self._session_id_from_body(frame.get("body") or {})
        if session_id is None:
            return None
        q = self._queue_for(session_id)
        seq = q.next_seq
        q.next_seq += 1
        return session_id, seq

    async def _skip_inbound_order(self, order_key: tuple[str, int] | None) -> None:
        if order_key is None:
            return
        session_id, seq = order_key
        q = self._queue_for(session_id)
        async with q.cond:
            q.skipped.add(seq)
            if q.worker is None or q.worker.done():
                q.worker = self._spawn_bg(
                    self._inbound_worker(session_id),
                    name=f"wecom-inbound-worker-{session_id}",
                )
            q.cond.notify_all()

    async def _enqueue_inbound_turn(self, turn: _InboundTurn) -> None:
        q = self._queue_for(turn.session_id)
        async with q.cond:
            q.turns[turn.seq] = turn
            if q.worker is None or q.worker.done():
                q.worker = self._spawn_bg(
                    self._inbound_worker(turn.session_id),
                    name=f"wecom-inbound-worker-{turn.session_id}",
                )
            q.cond.notify_all()

    async def _mark_inbound_ready(self, turn: _InboundTurn) -> None:
        turn.ready.set()
        q = self._queue_for(turn.session_id)
        async with q.cond:
            q.cond.notify_all()

    async def _inbound_worker(self, session_id: str) -> None:
        q = self._queue_for(session_id)
        try:
            while not self._shutdown.is_set():
                async with q.cond:
                    await q.cond.wait_for(
                        lambda: (
                            self._shutdown.is_set()
                            or q.dispatch_seq in q.skipped
                            or q.dispatch_seq in q.turns
                        )
                    )
                    if self._shutdown.is_set():
                        return
                    if q.dispatch_seq in q.skipped:
                        q.skipped.remove(q.dispatch_seq)
                        q.dispatch_seq += 1
                        if not q.turns and not q.skipped and q.dispatch_seq == q.next_seq:
                            self._inbound_queues.pop(session_id, None)
                            return
                        q.cond.notify_all()
                        continue
                    turn = q.turns[q.dispatch_seq]

                try:
                    await asyncio.wait_for(
                        turn.ready.wait(), timeout=INBOUND_PREFETCH_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "inbound media prefetch timed out: session=%s seq=%d msgtype=%s",
                        session_id, turn.seq, turn.msgtype,
                    )
                    turn.timed_out = True
                    turn.inbound.content_blocks = [{
                        "type": "text",
                        "text": f"[用户发来 {turn.msgtype},但媒体下载超时]",
                    }]
                    turn.inbound.text = blocks_to_text(turn.inbound.content_blocks)
                    turn.ready.set()

                try:
                    if not turn.drop:
                        await self._dispatch_ready_turn(turn)
                finally:
                    turn.done.set()
                    async with q.cond:
                        q.turns.pop(q.dispatch_seq, None)
                        q.dispatch_seq += 1
                        if not q.turns and not q.skipped and q.dispatch_seq == q.next_seq:
                            self._inbound_queues.pop(session_id, None)
                            return
                        q.cond.notify_all()
        finally:
            if self._inbound_queues.get(session_id) is q:
                q.worker = None

    async def _dispatch_ready_turn(self, turn: _InboundTurn) -> None:
        handler = self._handler
        if handler is None:
            log.error("no handler registered; dropping message")
            return

        stream = WeComStreamHandle(self, req_id=turn.inbound.reply_token)
        # Initial 思考中 frame is sent only when this message reaches the
        # FIFO dispatch head. Later text messages no longer appear to start
        # processing before earlier media messages.
        await stream._send_frame("思考中…", finish=False)

        try:
            await handler(turn.inbound, stream)
        except Exception:                                    # noqa: BLE001
            log.exception("handler raised")
            try:
                await stream.finish("(系统错误,请稍后再试)")
            except Exception:                                # noqa: BLE001
                pass

    # ---- message dispatch -------------------------------------------------

    async def _handle_msg_callback(
        self,
        frame: dict[str, Any],
        order_key: tuple[str, int] | None = None,
    ) -> None:
        if order_key is None:
            order_key = self._reserve_inbound_order(frame)
        order_consumed = False
        try:
            body = frame.get("body") or {}
            msg_id = body.get("msgid") or ""
            if msg_id and not self._msgid_lru.add(msg_id):
                log.info("dedup msgid=%s", msg_id)
                await self._skip_inbound_order(order_key)
                order_consumed = True
                return
            try:
                inbound = self._parse_metadata(frame)
            except Exception:                                # noqa: BLE001
                log.exception("failed to parse callback")
                await self._skip_inbound_order(order_key)
                order_consumed = True
                return
            if inbound is None:
                await self._skip_inbound_order(order_key)
                order_consumed = True
                return                                       # unsupported msgtype, already logged

            # Private-chat policy gate. Group chats always pass through;
            # only single chats are filtered by ``private_chat`` in config.yaml.
            # Done BEFORE the 思考中 frame so a blocked user doesn't see the
            # "thinking..." spinner forever.
            if inbound.chat_type == ChatType.SINGLE:
                pc = self.settings.private_chat
                if not pc.allows(inbound.user_id):
                    log.info(
                        "private chat blocked: user=%s mode=%s",
                        inbound.user_id, pc.mode,
                    )
                    try:
                        await self._reply_blocked(
                            inbound.reply_token, pc.blocked_reply, inbound.user_id,
                        )
                    finally:
                        await self._skip_inbound_order(order_key)
                        order_consumed = True
                    return

            # ---- slash command interception -------------------------------
            # Done here (after the private_chat gate, before media resolution
            # and inbound queueing) so /stop is NOT serialised behind the very
            # turn it is supposed to cancel. Slash commands bypass the inbound
            # queue entirely: they reply with a single aibot_respond_msg stream
            # frame and never create a Session or enter the dispatcher.
            msgtype_early = body.get("msgtype") or "text"
            slash_cmd = self._match_slash_command(inbound, msgtype_early)
            if slash_cmd is not None:
                # /running is private-chat only; in a group it degrades to a
                # polite refusal rather than executing.
                if slash_cmd in _PRIVATE_ONLY_SLASH_CMDS and inbound.chat_type != ChatType.SINGLE:
                    await self._reply_slash(
                        inbound.reply_token,
                        "/running 仅在私聊中可用。请在 1:1 对话中发送。",
                    )
                else:
                    await self._handle_slash_command(slash_cmd, inbound)
                await self._skip_inbound_order(order_key)
                order_consumed = True
                return
            # ---- end slash interception -----------------------------------

            msgtype = body.get("msgtype") or "text"
            if order_key is None:
                log.error("unable to assign inbound order; dropping msgid=%s", msg_id)
                return
            _, seq = order_key
            turn = _InboundTurn(
                session_id=inbound.session_id,
                seq=seq,
                frame=frame,
                inbound=inbound,
                msgtype=msgtype,
            )
            await self._enqueue_inbound_turn(turn)
            order_consumed = True
            try:
                blocks = await self._resolve_inbound_blocks(
                    body, msgtype, inbound.session_id, inbound.chat_type,
                )
            except Exception:                                # noqa: BLE001
                log.exception("failed to resolve inbound blocks for msgtype=%s", msgtype)
                blocks = [{"type": "text",
                           "text": f"[用户发来 {msgtype},但下载/解密失败]"}]
            if turn.timed_out:
                await turn.done.wait()
                return
            if blocks is None:
                log.info("unsupported msgtype=%s; ignoring", msgtype)
                turn.drop = True
                await self._mark_inbound_ready(turn)
                await turn.done.wait()
                return
            blocks = coalesce_text_blocks(blocks)
            if not blocks:
                blocks = [{"type": "text", "text": "(空消息)"}]
            inbound.content_blocks = blocks
            inbound.text = blocks_to_text(blocks)
            await self._mark_inbound_ready(turn)
            await turn.done.wait()
        finally:
            if order_key is not None and not order_consumed:
                await self._skip_inbound_order(order_key)

    def _parse_metadata(self, frame: dict[str, Any]) -> IncomingMessage | None:
        body = frame.get("body") or {}
        headers = frame.get("headers") or {}
        chat_type_raw = body.get("chattype") or "single"
        chat_type = ChatType.GROUP if chat_type_raw == "group" else ChatType.SINGLE
        chat_id = body.get("chatid")
        aibot_id = body.get("aibotid") or self.bot_id
        from_user = (body.get("from") or {}).get("userid") or "anonymous"

        if chat_type == ChatType.GROUP and chat_id:
            session_id = f"wecom-group-{chat_id}"
        else:
            session_id = f"wecom-single-{aibot_id}-{from_user}"

        return IncomingMessage(
            session_id=session_id,
            chat_type=chat_type,
            user_id=from_user,
            text="",                                          # filled in async stage
            msg_id=body.get("msgid") or "",
            bot_id=aibot_id,
            chat_id=chat_id,
            reply_token=headers.get("req_id"),
            raw=body,
        )

    async def _resolve_inbound_blocks(
        self,
        body: dict[str, Any],
        msgtype: str,
        session_id: str,
        chat_type: ChatType,
    ) -> list[ContentBlock] | None:
        """Build the ordered content-block list for an inbound message body.

        Handles WeCom's sibling ``quote`` field by recursively flattening
        the quote payload and prefixing the current-message blocks with
        text-boundary markers so the LLM can tell what was quoted vs. what
        the user just sent. The group ``@bot `` mention strip is applied to
        the first text block of the *current* message side only — never to
        the quote interior or the marker blocks.

        Returns ``None`` for genuinely unsupported msgtypes so the caller
        can drop the message; otherwise always returns a non-empty list.
        """
        msg_id = body.get("msgid") or ""
        # voice msgtype is unique: WeCom delivers it pre-transcribed text,
        # not media to download. Empty transcription → drop entirely.
        if msgtype == "voice":
            transcribed = ((body.get("voice") or {}).get("content") or "").strip()
            current_blocks: list[ContentBlock] = (
                [{"type": "text", "text": transcribed}] if transcribed else []
            )
            if not current_blocks and not body.get("quote"):
                return None
        else:
            current_blocks = await self._flatten_payload(
                body, msgtype, session_id, msg_id, idx=0,
            )

        if chat_type == ChatType.GROUP:
            current_blocks = _strip_mention_from_first_text(current_blocks)

        quote = body.get("quote") or {}
        if quote and quote.get("msgtype"):
            quote_blocks = await self._flatten_payload(
                quote, quote.get("msgtype"), session_id, f"{msg_id}-q", idx=0,
            )
        else:
            quote_blocks = []

        if not quote_blocks:
            return current_blocks or [{"type": "text", "text": "(空消息)"}]

        # Wrap the quote in text-boundary markers so the model can
        # distinguish quoted context from the new message body.
        return (
            [{"type": "text", "text": "[引用开始]"}]
            + quote_blocks
            + [{"type": "text", "text": "[引用结束 — 以下为本条新消息]"}]
            + current_blocks
        )

    async def _flatten_payload(
        self,
        payload: dict[str, Any],
        msgtype: str,
        session_id: str,
        msg_id: str,
        *,
        idx: int = 0,
    ) -> list[ContentBlock]:
        """Recursively flatten a WeCom message body (or a quote sub-payload)
        into an ordered ContentBlock list. Handles ``text``, ``image``,
        ``mixed`` (recurses into ``msg_item``), and falls back to a text
        placeholder for ``voice`` / ``file`` / ``video`` (their bytes are
        saved to inbox but not sent to the LLM as vision)."""
        if msgtype == "text":
            content = ((payload.get("text") or {}).get("content") or "").strip()
            return [{"type": "text", "text": content}] if content else []

        if msgtype == "image":
            image_payload = payload.get("image") or {}
            rel = await self._save_media_bytes(
                image_payload, "image", session_id, f"{msg_id or 'msg'}-{idx}",
            )
            if rel is None:
                return [{"type": "text", "text": "[图片下载失败]"}]
            return [{"type": "image", "path": rel}]

        if msgtype == "mixed":
            items = (payload.get("mixed") or {}).get("msg_item") or []
            # Download all sub-trees CONCURRENTLY rather than serially.
            #
            # Why: WeCom image download URLs expire ~5 minutes after the
            # callback is delivered. The old serial loop downloaded one
            # image at a time; with N × ~1MB images (e.g. a 9-photo vehicle
            # inspection burst) the cumulative wall time exceeded the URL
            # lifetime, so the tail URLs 404'd and the user saw
            # ``[图片下载失败]`` placeholders for items they did send.
            # Concurrent download parallelises the wait so all URLs are
            # fetched inside their validity window.
            #
            # Order is preserved: ``asyncio.gather`` returns results in
            # input order, so the assembled block list matches the user's
            # photo order.
            async def _flatten_one(i: int, it: dict[str, Any]) -> list[ContentBlock]:
                it_type = it.get("msgtype") or ""
                sub = await self._flatten_payload(
                    it, it_type, session_id, msg_id, idx=i,
                )
                if not sub and it_type:
                    sub = [{"type": "text", "text": f"[未支持的 mixed 子项: {it_type}]"}]
                return sub

            sub_lists = await asyncio.gather(
                *(_flatten_one(i, it) for i, it in enumerate(items))
            )
            blocks: list[ContentBlock] = []
            for sub in sub_lists:
                blocks.extend(sub)
            return blocks

        if msgtype in ("file", "video"):
            sub_payload = payload.get(msgtype) or {}
            placeholder = await self._save_media_placeholder(
                sub_payload, msgtype, session_id, f"{msg_id or 'msg'}-{idx}",
            )
            return [{"type": "text", "text": placeholder}]

        # Anything else: emit a text placeholder so the message is still
        # routable rather than silently dropped.
        return [{"type": "text", "text": f"[未支持: {msgtype}]"}]

    # WeCom errcode for "stream message update expired (>10 minutes),
    # cannot update". Stream frames (and the progress heartbeat) for this
    # req_id are dead; mark the handle so pushers stop.
    _STREAM_EXPIRED_ERRCODE = 846608

    def _dispatch_ack(self, msg: dict[str, Any]) -> None:
        req_id = (msg.get("headers") or {}).get("req_id")
        if (
            req_id
            and msg.get("errcode") == self._STREAM_EXPIRED_ERRCODE
            and req_id in self._active_streams
        ):
            handle = self._active_streams[req_id]
            if not handle.expired:
                handle.expired = True
                log.info(
                    "wecom stream expired (>10min) for req_id=%s; "
                    "stopping further stream pushes",
                    req_id,
                )
        fut = self._pending_acks.pop(req_id, None) if req_id else None
        if fut is not None and not fut.done():
            fut.set_result(msg)
        else:
            log.debug("ack: %s", msg)

    async def _send_and_await(
        self,
        payload: dict[str, Any],
        *,
        timeout: float = UPLOAD_RESPONSE_TIMEOUT,
    ) -> dict[str, Any]:
        req_id = payload["headers"]["req_id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_acks[req_id] = fut
        try:
            await self._enqueue_write(payload)
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_acks.pop(req_id, None)

    async def send_proactive_markdown(self, chat_id: str, content: str) -> None:
        """Push one markdown message through this adapter's existing socket."""
        safe_chat_id = str(chat_id or "").strip()
        safe_content = _truncate_utf8(str(content or "").strip(), MARKDOWN_CONTENT_MAX_BYTES)
        if not safe_chat_id or not safe_content:
            raise RuntimeError("proactive chat_id and markdown content are required")
        response = await self._send_and_await(
            {
                "cmd": "aibot_send_msg",
                "headers": {"req_id": _new_req_id()},
                "body": {
                    "chatid": safe_chat_id,
                    "msgtype": "markdown",
                    "markdown": {"content": safe_content},
                },
            }
        )
        if int(response.get("errcode") or 0) != 0:
            raise RuntimeError(
                f"WeCom proactive markdown rejected: {response.get('errmsg') or response.get('errcode')}"
            )

    async def send_proactive_media(
        self,
        chat_id: str,
        path: Path,
        *,
        kind: str,
        filename: str | None = None,
    ) -> None:
        """Upload and push one image/file through the existing bot connection."""
        if kind not in {"image", "file"}:
            raise RuntimeError(f"unsupported proactive media kind: {kind}")
        data = await asyncio.to_thread(path.read_bytes)
        media_id = await self.upload_media(
            data,
            kind=kind,
            filename=filename or path.name,
        )
        response = await self._send_and_await(
            {
                "cmd": "aibot_send_msg",
                "headers": {"req_id": _new_req_id()},
                "body": {
                    "chatid": str(chat_id or "").strip(),
                    "msgtype": kind,
                    kind: {"media_id": media_id},
                },
            }
        )
        if int(response.get("errcode") or 0) != 0:
            raise RuntimeError(
                f"WeCom proactive {kind} rejected: {response.get('errmsg') or response.get('errcode')}"
            )

    async def upload_media(self, data: bytes, *, kind: str, filename: str) -> str:
        """Upload bytes via aibot_upload_media_init/chunk/finish; return media_id."""
        if kind not in ("image", "file"):
            raise RuntimeError(f"unsupported media kind: {kind}")
        size = len(data)
        if size < 5:
            raise RuntimeError("file too small (WeCom requires ≥5 bytes)")
        cap = MEDIA_SIZE_LIMITS[kind]
        if size > cap:
            raise RuntimeError(f"{kind} exceeds {cap} bytes (got {size})")
        total_chunks = max(1, math.ceil(size / UPLOAD_CHUNK_SIZE))
        md5 = hashlib.md5(data).hexdigest()

        init_resp = await self._send_and_await({
            "cmd": "aibot_upload_media_init",
            "headers": {"req_id": _new_req_id()},
            "body": {
                "type": kind,
                "filename": filename,
                "total_size": size,
                "total_chunks": total_chunks,
                "md5": md5,
            },
        })
        if init_resp.get("errcode") not in (0, None):
            raise RuntimeError(f"upload_init failed: {init_resp!r}")
        upload_id = (init_resp.get("body") or {}).get("upload_id") or ""
        if not upload_id:
            raise RuntimeError(f"upload_init missing upload_id: {init_resp!r}")

        for idx in range(total_chunks):
            chunk = data[idx * UPLOAD_CHUNK_SIZE : (idx + 1) * UPLOAD_CHUNK_SIZE]
            chunk_resp = await self._send_and_await({
                "cmd": "aibot_upload_media_chunk",
                "headers": {"req_id": _new_req_id()},
                "body": {
                    "upload_id": upload_id,
                    "chunk_index": idx,
                    "base64_data": base64.b64encode(chunk).decode("ascii"),
                },
            })
            if chunk_resp.get("errcode") not in (0, None):
                raise RuntimeError(f"upload_chunk[{idx}] failed: {chunk_resp!r}")

        finish_resp = await self._send_and_await({
            "cmd": "aibot_upload_media_finish",
            "headers": {"req_id": _new_req_id()},
            "body": {"upload_id": upload_id},
        })
        if finish_resp.get("errcode") not in (0, None):
            raise RuntimeError(f"upload_finish failed: {finish_resp!r}")
        media_id = (finish_resp.get("body") or {}).get("media_id") or ""
        if not media_id:
            raise RuntimeError(f"upload_finish missing media_id: {finish_resp!r}")
        return media_id

    async def _save_media_bytes(
        self,
        payload: dict[str, Any],
        msgtype: str,
        session_id: str,
        media_tag: str,
    ) -> str | None:
        """Download + decrypt + persist media bytes to the session's inbox.

        Returns the workspace-relative path (``./inbox/<file>``) on success
        or ``None`` on any failure. Caller decides how to render failure —
        image flow turns it into a text block, file/video flow into a
        placeholder string. ``media_tag`` typically encodes msgid + index
        so concurrent downloads don't collide on the same-second filename.
        """
        url = payload.get("url") or ""
        aeskey = payload.get("aeskey") or ""
        if not url or not aeskey:
            log.warning("media payload missing url/aeskey for %s", msgtype)
            return None
        if self._workspace_resolver is None:
            log.warning("no workspace_resolver wired; cannot save media")
            return None
        try:
            plain = await wecom_media.download_and_decrypt(url, aeskey)
        except Exception:                                    # noqa: BLE001
            log.exception("download/decrypt failed for %s", msgtype)
            return None
        cwd = self._workspace_resolver(session_id)
        inbox = cwd / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        ext = wecom_media.sniff_extension(plain, msgtype)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", media_tag)[-32:] or "media"
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{ts}-{safe_tag}.{ext}"
        out = inbox / fname
        await asyncio.to_thread(out.write_bytes, plain)
        return f"./inbox/{fname}"

    async def _save_media_placeholder(
        self,
        payload: dict[str, Any],
        msgtype: str,
        session_id: str,
        media_tag: str,
    ) -> str:
        """Wrap _save_media_bytes for non-vision media (file/video). Always
        returns a printable placeholder, never None — failure becomes a
        ``下载失败`` placeholder so the LLM still sees something."""
        rel = await self._save_media_bytes(payload, msgtype, session_id, media_tag)
        if rel is None:
            return f"[用户发来 {msgtype},但下载失败]"
        try:
            cwd = self._workspace_resolver(session_id)
            size = (cwd / rel.lstrip("./")).stat().st_size
        except Exception:                                    # noqa: BLE001
            size = -1
        size_part = f" ({size} bytes)" if size >= 0 else ""
        return f"[用户发来 {msgtype}: {rel}{size_part}]"

    async def _handle_event_callback(self, frame: dict[str, Any]) -> None:
        body = frame.get("body") or {}
        headers = frame.get("headers") or {}
        event = (body.get("event") or {}).get("eventtype") or ""
        msg_id = body.get("msgid") or ""
        if msg_id and not self._msgid_lru.add(msg_id):
            return
        if event == "enter_chat":
            # Apply the same private-chat policy gate to enter_chat welcomes
            # so a blocked user doesn't see the welcome message even though
            # their subsequent messages would be dropped.
            chat_type_raw = body.get("chattype") or "single"
            from_user = (body.get("from") or {}).get("userid") or ""
            if chat_type_raw != "group":
                pc = self.settings.private_chat
                if from_user and not pc.allows(from_user):
                    log.info(
                        "enter_chat blocked (single): user=%s mode=%s",
                        from_user, pc.mode,
                    )
                    return
            session_id = self._session_id_from_body(body)
            await self._reply_welcome(headers.get("req_id"), session_id)
        elif event == "disconnected_event":
            log.warning("received disconnected_event; will reconnect")
            self._connection_dead.set()
        else:
            log.info("event ignored: %s", event)

    def _session_id_from_body(self, body: dict[str, Any]) -> str | None:
        """Compute the same session_id _parse_metadata would, using only the
        routing fields available on every callback (msg or event). Returns
        None if the body lacks enough info to identify the session."""
        chat_type_raw = body.get("chattype") or "single"
        chat_id = body.get("chatid")
        aibot_id = body.get("aibotid") or self.bot_id
        from_user = (body.get("from") or {}).get("userid") or ""
        if chat_type_raw == "group" and chat_id:
            return f"wecom-group-{chat_id}"
        if from_user:
            return f"wecom-single-{aibot_id}-{from_user}"
        return None

    # ---- slash command machinery -----------------------------------------

    def _dispatcher_ref(self) -> Any:
        """Return the Dispatcher instance backing this adapter's handler, or
        None if the handler isn\'t a bound method of a Dispatcher (e.g. a
        test fake). We resolve this lazily rather than wiring an explicit
        Dispatcher reference at construction time so the adapter stays
        decoupled from the dispatcher module (no circular import) and so
        existing test fakes that register a plain coroutine keep working.
        The slash commands simply degrade to "command unavailable" when
        this returns None."""
        h = self._handler
        if h is None:
            return None
        disp = getattr(h, "__self__", None)
        if disp is None:
            return None
        # Duck-type: the methods we need.
        if all(hasattr(disp, m) for m in ("is_busy", "busy_group_sessions", "reset_session_history", "current_role_for")):
            return disp
        return None

    def _match_slash_command(
        self,
        inbound: IncomingMessage,
        msgtype: str,
    ) -> str | None:
        """Return the canonical command name if this inbound text message is
        a recognised slash command, else None.

        Rules:
          * Only ``msgtype == "text"`` is considered. Images/mixed/voice/etc
            can\'t be commands.
          * The command must be the first token of the message body
            (anchored at start). A word boundary after the name is required
            so "/newton" does not match "/new".
          * In GROUP chats the message MUST mention the bot first
            (``@bot /new``); a bare "/new" in a group is treated as ordinary
            text and forwarded to the agent. This matches the existing
            group-@bot contract for normal conversations and prevents a
            member mentioning "/new" mid-sentence from resetting the session.
          * In SINGLE (private) chats no @bot is required — the
            private_chat gate already controls who can reach this point.
          * English command names are case-insensitive. Chinese aliases
            ``/刷新``, ``/停止``, and ``/状态`` map to ``new``, ``stop``,
            and ``status`` respectively.
        """
        if msgtype != "text":
            return None
        # Raw text content of the body, before block resolution. We don\'t
        # need full _resolve_inbound_blocks for a plain text message — the
        # content is right there in body.text.content. Quote handling is
        # irrelevant: a slash command would never be sent as a quote reply.
        body = inbound.raw or {}
        content = ((body.get("text") or {}).get("content") or "").strip()
        if not content:
            return None

        text = content
        if inbound.chat_type == ChatType.GROUP:
            # Strip a leading "@<botname> " mention. We reuse the same regex
            # as _strip_mention_from_first_text so the matching semantics stay
            # identical to normal group-message handling.
            stripped = _MENTION_RE.sub("", text, count=1).strip()
            if stripped == text:
                # No @bot prefix in a group → not a command invocation.
                return None
            text = stripped

        m = _SLASH_CMD_RE.match(text)
        if m is None:
            return None
        matched = m.group(1).lower()
        return _SLASH_CMD_ALIASES.get(matched, matched)

    async def _reply_slash(self, req_id: Any, text: str) -> None:
        """Reply to a slash command with a single finished stream frame.
        Reuses the same WeComStreamHandle path as _reply_blocked — the only
        frame format WeCom honours for ordinary message callbacks is
        msgtype=stream with finish=true."""
        stream = WeComStreamHandle(self, req_id=req_id)
        await stream.finish(text)

    def drain_pending_turns(self, session_id: str) -> int:
        """Mark all queued-but-not-yet-dispatched inbound turns for this
        session as dropped, so /stop doesn\'t leave the next queued message
        immediately re-triggering the agent. Returns the count of turns
        drained.

        The currently-running turn (if any) is NOT touched here — it\'s
        cancelled separately via _cancel_running_turn. This method only
        handles turns sitting in the queue behind the running one."""
        q = self._inbound_queues.get(session_id)
        if q is None:
            return 0
        drained = 0
        for turn in q.turns.values():
            if not turn.drop:
                turn.drop = True
                # Mark ready so the worker\'s wait_for(turn.ready.wait())
                # doesn\'t block on a drained turn that will never resolve
                # media. drop=True short-circuits dispatch anyway.
                if not turn.ready.is_set():
                    turn.ready.set()
                drained += 1
        return drained

    async def _cancel_running_turn(self, session_id: str) -> bool:
        """Cancel the inbound worker task currently running a turn for this
        session, if any. Returns True if a task was actually cancelled.

        The running turn lives inside ``_inbound_queues[session_id].worker``
        — that worker task is the one ``await``\'ing
        ``handler(...)`` (i.e. ``dispatcher.handle``). Cancelling it raises
        CancelledError inside dispatcher.handle, which propagates out of the
        ``async with session.lock`` block; the dispatcher\'s
        ``_handle_locked`` finally resets turn counters and the outer
        ``handle`` finally clears the busy state. The agent\'s own
        ``except BaseException`` then rolls back its half-appended history
        so the next turn starts clean."""
        q = self._inbound_queues.get(session_id)
        if q is None or q.worker is None:
            return False
        task = q.worker
        if task.done():
            return False
        # Drain queued turns FIRST so they don\'t immediately re-run when
        # the worker task ends and a new worker is spawned for the next
        # inbound message.
        self.drain_pending_turns(session_id)
        task.cancel()
        # Don\'t await the task here — we\'re inside a different callback
        # and we want to return the slash reply promptly. The cancelled
        # task will resolve on its own; its finally-clause cleans up
        # q.worker = None.
        return True

    async def _handle_slash_command(
        self,
        cmd: str,
        inbound: IncomingMessage,
    ) -> None:
        """Dispatch a recognised slash command. ``inbound.chat_type`` has
        already been validated (private-only commands filtered upstream)."""
        sid = inbound.session_id
        req_id = inbound.reply_token
        disp = self._dispatcher_ref()

        if cmd == "new":
            if disp is None:
                await self._reply_slash(req_id, "⚠️ 命令不可用（未连接调度器）。")
                return
            if disp.is_busy(sid):
                await self._reply_slash(
                    req_id,
                    "⚠️ 会话正在执行任务中，请先发送 /stop 中止后再 /new。",
                )
                return
            try:
                role = await disp.reset_session_history(sid)
            except Exception:                          # noqa: BLE001
                log.exception("/new failed for session=%s", sid)
                await self._reply_slash(req_id, "⚠️ 重置失败，请稍后重试。")
                return
            await self._reply_slash(
                req_id,
                f"✅ 会话已重置（保留工作区文件）。当前角色: {role}",
            )
            return

        if cmd == "stop":
            if disp is None:
                await self._reply_slash(req_id, "⚠️ 命令不可用（未连接调度器）。")
                return
            cancelled = await self._cancel_running_turn(sid)
            if cancelled:
                await self._reply_slash(req_id, "⏹ 已中止当前任务。")
            else:
                # Still drain any pending queued turns for tidiness.
                drained = self.drain_pending_turns(sid)
                if drained:
                    await self._reply_slash(
                        req_id,
                        f"⏹ 当前无运行中任务，已清空 {drained} 条排队消息。",
                    )
                else:
                    await self._reply_slash(req_id, "⚪ 当前没有正在执行的任务。")
            return

        if cmd == "status":
            if disp is None:
                await self._reply_slash(req_id, "⚠️ 命令不可用（未连接调度器）。")
                return
            if disp.is_busy(sid):
                role = disp.current_role_for(sid) or "?"
                await self._reply_slash(req_id, f"🟢 正在执行任务（角色: {role}）")
            else:
                await self._reply_slash(req_id, "⚪ 当前空闲。")
            return

        if cmd == "running":
            if disp is None:
                await self._reply_slash(req_id, "⚠️ 命令不可用（未连接调度器）。")
                return
            busy = disp.busy_group_sessions()
            if not busy:
                await self._reply_slash(req_id, "⚪ 当前没有群会话正在执行任务。")
                return
            lines = [f"🟢 当前有 {len(busy)} 个群会话正在执行任务："]
            for sid_i in busy:
                # Strip the "wecom-group-" prefix for readability.
                lines.append(f"  • {sid_i}")
            await self._reply_slash(req_id, "\n".join(lines))
            return

        # Should not reach here — _match_slash_command only returns the four
        # known names — but degrade gracefully.
        await self._reply_slash(req_id, f"⚠️ 未知命令: /{cmd}")

    async def _reply_welcome(
        self,
        req_id: str | None,
        session_id: str | None = None,
    ) -> None:
        from ..roles.registry import RoleRegistry
        from ..session.persistence import load_state
        roles = RoleRegistry.load(self.settings.paths.user_roles_dir)
        # Solo mode: always use the pinned role.
        if self.role_name and roles.has(self.role_name):
            role_name = self.role_name
        else:
            # Pick whichever role the user will *actually* talk to next: the
            # persisted current_role for this session if any, otherwise the
            # global default. Without this, a returning user sees admin's
            # "我是小管" while their messages still route to e.g. engineer.
            role_name = self.settings.default_role
            if session_id and self._workspace_resolver is not None:
                try:
                    cwd = self._workspace_resolver(session_id)
                    state = load_state(cwd)
                except Exception:                                # noqa: BLE001
                    state = None
                prior_role = (state or {}).get("current_role")
                if prior_role and roles.has(prior_role):
                    role_name = prior_role
        welcome = ""
        if roles.has(role_name):
            welcome = (roles.get(role_name).welcome_message or "").strip()
        if not welcome:
            welcome = "你好,我是这个团队的机器人助手。"
        payload = {
            "cmd": "aibot_respond_welcome_msg",
            "headers": {"req_id": req_id or _new_req_id()},
            "body": {"msgtype": "text", "text": {"content": welcome}},
        }
        await self._enqueue_write(payload)

    async def _reply_blocked(
        self,
        req_id: str | None,
        text: str,
        user_id: str = "",
    ) -> None:
        """Send a single text reply to a blocked private chat.

        ``text`` is treated as a template and supports one placeholder:

          ``{userid}`` — replaced with the blocked sender's WeCom ``userid``
                         (the same value used by ``private_chat.whitelist``
                         / ``blacklist``).

        The {userid} placeholder exists because end users have no way to
        learn their own WeCom userid (it isn't surfaced in the client) yet
        the maintainer needs that exact string to add them to whitelist.
        A common pattern is:

            blocked_reply: "私聊未开放。你的企业微信账号是 {userid},请联系管理员开通。"

        — the user sends one message and immediately learns the string to
        hand to the maintainer.

        When ``text`` is empty the message is silently dropped — appropriate
        for ``closed`` mode where you want no feedback.
        """
        if not text:
            return
        # str.format with an explicit mapping (not **locals) so a literal
        # stray ``{`` in the text can't raise and so unknown placeholders
        # survive verbatim rather than blowing up the reply path.
        try:
            content = text.format(userid=user_id)
        except (KeyError, IndexError):
            # Unknown placeholder like {foo} — render literally so a typo
            # in config.yaml can't take the bot's reply path down.
            content = text.replace("{userid}", user_id)
        # IMPORTANT: replies to ``aibot_msg_callback`` MUST use the stream
        # frame format (msgtype=stream, finish=true). A plain msgtype=text
        # frame is silently dropped by WeCom for ordinary message callbacks
        # (text frames only work for ``aibot_respond_welcome_msg`` welcome
        # events). Reuse ``WeComStreamHandle`` — the same path the normal
        # handler uses — to deliver a single finished stream frame; this is
        # the only format WeCom honours here.
        stream = WeComStreamHandle(self, req_id=req_id)
        await stream.finish(content)
        log.info("blocked_reply delivered: user=%s", user_id)
