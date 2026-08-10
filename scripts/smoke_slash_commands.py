"""Offline smoke test for slash commands and their Chinese aliases.

These commands live in ``WeComBotAdapter._handle_msg_callback`` (intercepted
before inbound queueing) and call into Dispatcher's command-support surface
(``is_busy`` / ``current_role_for`` / ``busy_group_sessions`` /
``reset_session_history``). This test wires a real Dispatcher + ScriptedLLM
through a real WeComBotAdapter (with ``set_handler(dispatcher.handle)`` so
the adapter's ``_dispatcher_ref()`` introspection finds it) and drives
``_handle_msg_callback`` directly with crafted frames, asserting on the
replied stream frames.

Run: ``python3 scripts/smoke_slash_commands.py`` — no network.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_slash_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.adapters.base import ChatType, IncomingMessage
from chat_team.adapters.wecom import WeComBotAdapter
from chat_team.agent.tools.base import ToolRegistry
from chat_team.agent.tools.notebook_tools import NotebookReadTool, NotebookWriteTool
from chat_team.agent.tools.transfer_tool import TransferToEmployeeTool
from chat_team.config import load_settings
from chat_team.dispatcher import Dispatcher
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager


# ---------- helpers ---------------------------------------------------------


class CapturingStream:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.final: str | None = None

    async def push(self, chunk: str, *, append: bool = True) -> None: ...
    async def status(self, note: str) -> None:
        self.statuses.append(note)

    async def finish(self, final_text: str) -> None:
        self.final = final_text

    async def send_image(self, path: Path, *, filename: str | None = None) -> None: ...
    async def send_file(self, path: Path, *, filename: str | None = None) -> None: ...


class ScriptedLLM(LLMProvider):
    """Returns a queued sequence of responses; blocks forever when empty so
    /stop has something concrete to cancel."""

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[CompletionRequest] = []
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        # Block forever — used by /stop tests so the turn is genuinely
        # in-flight when the /stop frame arrives.
        await asyncio.Event().wait()
        raise RuntimeError("unreachable")


def reply(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def build_dispatcher(settings, llm: LLMProvider) -> Dispatcher:
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    tools.register(NotebookReadTool())
    tools.register(NotebookWriteTool())
    tools.register(TransferToEmployeeTool(available_employees=roles.names()))
    return Dispatcher(settings, SessionManager(settings), roles, tools, llm)


def build_adapter(settings, dispatcher: Dispatcher):
    """Wire a real WeComBotAdapter with dispatcher.handle as the handler, plus
    a capturing _enqueue_write so we can inspect replied frames."""
    adapter = WeComBotAdapter(
        settings,
        workspace_resolver=lambda sid: settings.workspace_root / Path(sid).name,
        bot_id="BOT",
        secret="S",
    )
    adapter.set_handler(dispatcher.handle)

    sent: list[dict] = []

    async def fake_enqueue(payload):
        sent.append(payload)

    adapter._enqueue_write = fake_enqueue  # noqa: SLF001
    return adapter, sent


def text_frame(
    msgid: str,
    content: str,
    *,
    chattype: str = "single",
    chatid: str | None = None,
    userid: str = "U1",
) -> dict:
    body: dict = {
        "msgid": msgid,
        "aibotid": "BOT",
        "chattype": chattype,
        "from": {"userid": userid},
        "msgtype": "text",
        "text": {"content": content},
    }
    if chatid is not None:
        body["chatid"] = chatid
    return {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": f"rq-{msgid}"},
        "body": body,
    }


def reply_text(sent_frames: list[dict]) -> str | None:
    """Pull the content of the last aibot_respond_msg stream frame."""
    for fr in reversed(sent_frames):
        body = fr.get("body") or {}
        if fr.get("cmd") == "aibot_respond_msg" and body.get("msgtype") == "stream":
            return (body.get("stream") or {}).get("content")
    return None


def load_open_settings():
    """load_settings() but with private_chat opened so single-chat frames reach
    the slash interception layer (which lives behind the private_chat gate)."""
    s = load_settings()
    s.private_chat.mode = "open"
    return s


# ---------- tests -----------------------------------------------------------


async def test_match_slash_command():
    settings = load_open_settings()
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="S")

    def mk(chattype, content, userid="U1", chatid=None):
        body = {
            "msgid": "m", "aibotid": "BOT", "chattype": chattype,
            "from": {"userid": userid}, "msgtype": "text",
            "text": {"content": content},
        }
        if chatid:
            body["chatid"] = chatid
        return IncomingMessage(
            session_id="sid", chat_type=ChatType(chattype),
            user_id=userid, text=content, msg_id="m", bot_id="BOT",
            chat_id=chatid, raw=body,
        )

    # Private chat: bare command works.
    assert adapter._match_slash_command(mk("single", "/new"), "text") == "new"
    assert adapter._match_slash_command(mk("single", "/STOP"), "text") == "stop"
    assert adapter._match_slash_command(mk("single", "/status now"), "text") == "status"
    assert adapter._match_slash_command(mk("single", "/刷新"), "text") == "new"
    assert adapter._match_slash_command(mk("single", "/停止"), "text") == "stop"
    assert adapter._match_slash_command(mk("single", "/状态 现在"), "text") == "status"
    # Word boundary: /newton is NOT /new.
    assert adapter._match_slash_command(mk("single", "/newton ideas"), "text") is None
    assert adapter._match_slash_command(mk("single", "/刷新一下"), "text") is None
    # Non-text msgtype is never a command.
    assert adapter._match_slash_command(mk("single", "/new"), "image") is None

    # Group chat: bare command is rejected (must @bot).
    assert adapter._match_slash_command(mk("group", "/new", chatid="G1"), "text") is None
    # Group chat with @bot works.
    assert adapter._match_slash_command(mk("group", "@BOT /new", chatid="G1"), "text") == "new"
    assert adapter._match_slash_command(mk("group", "@BOT /刷新", chatid="G1"), "text") == "new"
    assert adapter._match_slash_command(mk("group", "@小管 /停止", chatid="G1"), "text") == "stop"
    assert adapter._match_slash_command(mk("group", "@小管 /状态", chatid="G1"), "text") == "status"
    assert adapter._match_slash_command(mk("group", "@小管 /running", chatid="G1"), "text") == "running"
    # @bot followed by ordinary text is not a command.
    assert adapter._match_slash_command(mk("group", "@BOT hello", chatid="G1"), "text") is None
    print("  ✓ _match_slash_command: English + Chinese, private bare, group @bot, boundaries")


async def test_status_idle_and_busy():
    settings = load_open_settings()
    # Busy scenario: a long-running turn (ScriptedLLM blocks forever) keeps
    # the session busy while we issue /status from a separate frame.
    llm = ScriptedLLM([])  # blocks on first call
    dispatcher = build_dispatcher(settings, llm)
    adapter, sent = build_adapter(settings, dispatcher)

    sid = "wecom-single-BOT-U1"

    # Start a turn in the background; it will block inside the LLM call.
    frame_chat = text_frame("chat1", "hello", userid="U1")
    bg_task = asyncio.create_task(adapter._handle_msg_callback(frame_chat))
    # Yield once so the turn reaches the LLM block.
    await asyncio.sleep(0.05)

    assert dispatcher.is_busy(sid), "session should be busy mid-turn"
    assert dispatcher.current_role_for(sid) == "team_admin"

    # /status while busy.
    frame_status = text_frame("stat1", "/status", userid="U1")
    await adapter._handle_msg_callback(frame_status)
    s = reply_text(sent)
    assert s and "正在执行" in s and "team_admin" in s, f"got {s!r}"
    print("  ✓ /status busy reply:", s)

    # Cancel the bg turn so the test can proceed.
    assert await adapter._cancel_running_turn(sid)
    try:
        await asyncio.wait_for(bg_task, timeout=2)
    except (asyncio.CancelledError, Exception):
        pass
    await asyncio.sleep(0.05)

    assert not dispatcher.is_busy(sid), "session should be idle after cancel"

    # /status while idle.
    sent.clear()
    frame_status2 = text_frame("stat2", "/状态", userid="U1")
    await adapter._handle_msg_callback(frame_status2)
    s2 = reply_text(sent)
    assert s2 and "空闲" in s2, f"got {s2!r}"
    print("  ✓ /status idle reply:", s2)


async def test_new_resets_history_keeps_role_and_workspace():
    settings = load_open_settings()
    llm = ScriptedLLM([reply("第一轮答复")])
    dispatcher = build_dispatcher(settings, llm)
    adapter, sent = build_adapter(settings, dispatcher)
    sid = "wecom-single-BOT-U1"

    # One real turn to seed history.
    s1 = CapturingStream()
    msg = IncomingMessage(
        session_id=sid, chat_type=ChatType.SINGLE, user_id="U1",
        text="hello", msg_id="m1", bot_id="BOT",
        content_blocks=[{"type": "text", "text": "hello"}],
    )
    await dispatcher.handle(msg, s1)
    assert s1.final and "第一轮" in s1.final
    sess = await dispatcher.sessions.get_or_create(sid)
    agent = sess.agents_by_role["team_admin"]
    assert len(agent.history) >= 2, "history should have user+assistant"
    # Drop a marker file in the workspace to verify /new keeps workspace files.
    marker = sess.cwd / "marker.txt"
    marker.write_text("kept", encoding="utf-8")
    assert marker.exists()

    # /刷新 (alias of /new)
    sent.clear()
    frame_new = text_frame("new1", "/刷新", userid="U1")
    await adapter._handle_msg_callback(frame_new)
    r = reply_text(sent)
    assert r and "已重置" in r and "team_admin" in r, f"got {r!r}"
    print("  ✓ /new reply:", r)

    # In-memory agents cleared.
    sess2 = await dispatcher.sessions.get_or_create(sid)
    assert "team_admin" not in sess2.agents_by_role, "agents_by_role should be cleared"
    assert sess2.restored_histories == {}, "restored_histories should be cleared"
    # current_role preserved.
    assert sess2.current_role == "team_admin", sess2.current_role
    # Workspace file preserved.
    assert marker.exists() and marker.read_text() == "kept", "workspace file must survive /new"
    # session.json on disk has empty histories.
    from chat_team.session.persistence import load_state
    state = load_state(sess2.cwd)
    assert state and state.get("histories") == {}, f"on-disk histories={state.get('histories')!r}"
    assert state.get("current_role") == "team_admin"
    print("  ✓ /new cleared history, kept current_role + workspace file")


async def test_new_refuses_when_busy():
    settings = load_open_settings()
    llm = ScriptedLLM([])  # blocks forever
    dispatcher = build_dispatcher(settings, llm)
    adapter, sent = build_adapter(settings, dispatcher)
    sid = "wecom-single-BOT-U1"

    bg = asyncio.create_task(adapter._handle_msg_callback(text_frame("c1", "hi")))
    await asyncio.sleep(0.05)
    assert dispatcher.is_busy(sid)

    sent.clear()
    await adapter._handle_msg_callback(text_frame("n1", "/new"))
    r = reply_text(sent)
    assert r and "执行任务" in r and "/stop" in r, f"got {r!r}"
    print("  ✓ /new refused while busy:", r)

    await adapter._cancel_running_turn(sid)
    try:
        await asyncio.wait_for(bg, timeout=2)
    except (asyncio.CancelledError, Exception):
        pass


async def test_stop_cancels_running_turn_and_drains_queue():
    settings = load_open_settings()
    llm = ScriptedLLM([])  # blocks forever
    dispatcher = build_dispatcher(settings, llm)
    adapter, sent = build_adapter(settings, dispatcher)
    sid = "wecom-group-G1"

    # Start a blocking turn in a GROUP chat (need @bot for commands there,
    # but the initial chat message doesn't need @bot — it's just a normal msg).
    bg = asyncio.create_task(
        adapter._handle_msg_callback(text_frame("c1", "hi", chattype="group", chatid="G1"))
    )
    await asyncio.sleep(0.05)
    assert dispatcher.is_busy(sid)

    # /停止 (alias of /stop) via @bot in the group.
    sent.clear()
    frame_stop = text_frame("s1", "@BOT /停止", chattype="group", chatid="G1")
    await adapter._handle_msg_callback(frame_stop)
    r = reply_text(sent)
    assert r and "已中止" in r, f"got {r!r}"
    print("  ✓ /stop reply:", r)

    # The bg task should have ended (cancelled).
    try:
        await asyncio.wait_for(bg, timeout=2)
    except (asyncio.CancelledError, Exception):
        pass
    await asyncio.sleep(0.05)
    assert not dispatcher.is_busy(sid), "session should be idle after /stop"

    # History rollback: the agent's half-appended user message must have been
    # rolled back by the except-BaseException path so the next turn is clean.
    sess = await dispatcher.sessions.get_or_create(sid)
    if "team_admin" in sess.agents_by_role:
        ag = sess.agents_by_role["team_admin"]
        assert len(ag.history) == 0, f"history should be rolled back, got {len(ag.history)}: {[m.role for m in ag.history]}"
        print("  ✓ history rolled back to empty after /stop")


async def test_stop_when_idle_drains_queue():
    settings = load_open_settings()
    llm = ScriptedLLM([])
    dispatcher = build_dispatcher(settings, llm)
    adapter, sent = build_adapter(settings, dispatcher)
    sid = "wecom-single-BOT-U1"

    # No running turn.
    sent.clear()
    await adapter._handle_msg_callback(text_frame("s1", "/stop"))
    r = reply_text(sent)
    assert r and "没有正在执行" in r, f"got {r!r}"
    print("  ✓ /stop when idle:", r)


async def test_running_lists_busy_groups():
    settings = load_open_settings()
    llm = ScriptedLLM([])
    dispatcher = build_dispatcher(settings, llm)
    adapter, sent = build_adapter(settings, dispatcher)

    # Start two busy GROUP sessions + one busy PRIVATE session.
    g1 = "wecom-group-G1"
    g2 = "wecom-group-G2"
    p1 = "wecom-single-BOT-U1"
    bg_tasks = []
    for sid, chatid in [(g1, "G1"), (g2, "G2")]:
        bg_tasks.append(asyncio.create_task(
            adapter._handle_msg_callback(text_frame("c-" + chatid, "hi", chattype="group", chatid=chatid))
        ))
    bg_tasks.append(asyncio.create_task(
        adapter._handle_msg_callback(text_frame("c-U1", "hi", userid="U1"))
    ))
    await asyncio.sleep(0.05)
    assert dispatcher.is_busy(g1) and dispatcher.is_busy(g2) and dispatcher.is_busy(p1)

    # /running is private-only. Issue from a private chat (a different user
    # so it doesn't collide with the busy U1 session's worker).
    sent.clear()
    frame_run = text_frame("r1", "/running", userid="U2")
    await adapter._handle_msg_callback(frame_run)
    r = reply_text(sent)
    assert r and "2" in r and "wecom-group-G1" in r and "wecom-group-G2" in r, f"got {r!r}"
    # The private session must NOT be listed.
    assert "wecom-single" not in r, f"private session leaked into /running: {r!r}"
    print("  ✓ /running reply:\n   ", r.replace("\n", "\n    "))

    # /running in a GROUP chat should be refused.
    sent.clear()
    frame_run_group = text_frame("rg1", "@BOT /running", chattype="group", chatid="G9")
    await adapter._handle_msg_callback(frame_run_group)
    rg = reply_text(sent)
    assert rg and "私聊" in rg, f"group /running should be refused, got {rg!r}"
    print("  ✓ /running refused in group:", rg)

    # Cleanup.
    for sid in (g1, g2, p1):
        await adapter._cancel_running_turn(sid)
    for t in bg_tasks:
        try:
            await asyncio.wait_for(t, timeout=2)
        except (asyncio.CancelledError, Exception):
            pass


async def main() -> None:
    await test_match_slash_command()
    await test_status_idle_and_busy()
    await test_new_resets_history_keeps_role_and_workspace()
    await test_new_refuses_when_busy()
    await test_stop_cancels_running_turn_and_drains_queue()
    await test_stop_when_idle_drains_queue()
    await test_running_lists_busy_groups()
    print("\nALL SLASH-COMMAND SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
