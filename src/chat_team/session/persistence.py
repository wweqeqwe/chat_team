"""Per-session debounced + atomic persistence.

State target: ``<cwd>/.chat_team/session.json``
Schema::

    {
      "session_id": "<sanitized>",
      "session_uuid": "<stable UUID sent as x-session-id>",
      "current_role": "team_admin",
      "histories": {
        "team_admin": [{"role": "user", "content": "...", ...}, ...]
      }
    }

Notebook is NOT included — it lives in its own ``notebook.md`` already.
Per-turn counters reset to zero, so they're not part of the snapshot either.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..llm.base import ChatMessage, ToolCall

if TYPE_CHECKING:
    from ..config import Settings
    from .session import Session

log = logging.getLogger(__name__)

STATE_FILENAME = "session.json"


# ---- (de)serialisation -----------------------------------------------------


def _serialize_message(m: ChatMessage) -> dict[str, Any]:
    if isinstance(m.content, list):
        content: Any = m.content
    else:
        content = m.content or ""
    d: dict[str, Any] = {"role": m.role, "content": content}
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments or {}}
            for c in m.tool_calls
        ]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.name:
        d["name"] = m.name
    return d


def _deserialize_message(d: dict[str, Any]) -> ChatMessage:
    tool_calls = [
        ToolCall(
            id=tc.get("id", ""),
            name=tc.get("name", ""),
            arguments=tc.get("arguments") or {},
        )
        for tc in (d.get("tool_calls") or [])
    ]
    raw = d.get("content")
    content: Any = raw if isinstance(raw, list) else (raw or "")
    return ChatMessage(
        role=d.get("role", "user"),
        content=content,
        tool_calls=tool_calls,
        tool_call_id=d.get("tool_call_id"),
        name=d.get("name"),
    )


def _state_path(cwd: Path, filename: str = STATE_FILENAME) -> Path:
    return cwd / ".chat_team" / filename


# ---- snapshot / write / load ----------------------------------------------


def snapshot(session: "Session") -> dict[str, Any]:
    histories = {
        role: [_serialize_message(m) for m in agent.history]
        for role, agent in session.agents_by_role.items()
    }
    return {
        "session_id": session.session_id,
        "session_uuid": session.session_uuid,
        "current_role": session.current_role,
        "histories": histories,
    }


def write_atomic(cwd: Path, data: dict[str, Any], filename: str = STATE_FILENAME) -> None:
    target = _state_path(cwd, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{filename}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name
    os.replace(tmp_path, target)


def load_state(cwd: Path, filename: str = STATE_FILENAME) -> dict[str, Any] | None:
    target = _state_path(cwd, filename)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        log.exception("failed to load %s; ignoring", target)
        return None


def restored_histories(cwd: Path, filename: str = STATE_FILENAME) -> dict[str, list[ChatMessage]]:
    state = load_state(cwd, filename)
    if not state:
        return {}
    out: dict[str, list[ChatMessage]] = {}
    for role, msgs in (state.get("histories") or {}).items():
        if not isinstance(msgs, list):
            continue
        out[role] = [_deserialize_message(m) for m in msgs if isinstance(m, dict)]
    return out


# ---- debounced flush manager ----------------------------------------------


class PersistenceManager:
    """One pending debounced flush per session.

    ``schedule(session)`` cancels any in-flight pending flush and starts a
    new one ``debounce`` seconds out. ``flush_now`` writes synchronously.
    ``flush_all`` cancels pending tasks and force-flushes every passed-in
    session — call from shutdown hooks.
    """

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self._pending: dict[str, asyncio.Task] = {}

    def schedule(self, session: "Session") -> None:
        # Caller is responsible for holding session.lock so the snapshot below
        # is consistent (Dispatcher._post_turn does). We snapshot here, while
        # the lock is still held, then write asynchronously after a debounce —
        # the worker never touches `session` again, so it can't race the next
        # turn's mutations of agents_by_role / agent.history.
        sid = session.session_id
        old = self._pending.get(sid)
        if old and not old.done():
            old.cancel()
        delay = self.settings.session.persistence_debounce_seconds
        snap = snapshot(session)
        cwd = session.cwd
        filename = session.state_filename
        self._pending[sid] = asyncio.create_task(
            self._delayed_flush(sid, cwd, snap, delay, filename)
        )

    async def _delayed_flush(
        self,
        session_id: str,
        cwd: Path,
        snap: dict[str, Any],
        delay: float,
        filename: str = STATE_FILENAME,
    ) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        try:
            await asyncio.to_thread(write_atomic, cwd, snap, filename)
        except Exception:                                     # noqa: BLE001
            log.exception("debounced flush failed for %s", session_id)

    def flush_now(self, session: "Session") -> None:
        write_atomic(session.cwd, snapshot(session), session.state_filename)

    async def flush_all(self, sessions: list["Session"]) -> None:
        tasks = [t for t in self._pending.values() if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for s in sessions:
            try:
                self.flush_now(s)
            except Exception:                                 # noqa: BLE001
                log.exception("final flush failed for %s", s.session_id)
        self._pending.clear()
