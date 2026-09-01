"""Smoke for the 2026-09-01 tool-loop incident fixes:

1. Identical-args SUCCESS circuit breaker: the same (tool, args) call
   succeeding N times within one turn force-breaks the loop with a fallback
   reply (production: 97 identical OCR calls in one turn).
2. Cap-hit + breaker fallbacks append a closing assistant message so the
   next user turn can't resume the abandoned loop.
3. The error breaker (identical ToolErrors) still works and also closes.
4. Dispatcher progress heartbeat stops when the stream reports itself
   expired (status() -> False), and keeps running for legacy None-returning
   streams.
5. WeComStreamHandle: an 846608 ack marks the stream expired; push/status
   no-op, finish() skips the closing stream frame but still sends markdown.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_loop_guards_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.agent.agent import Agent
from chat_team.agent.tools.base import Tool, ToolContext, ToolRegistry, ToolError
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


class CapturingStream:
    def __init__(self):
        self.statuses, self.final = [], None
    async def push(self, chunk, *, append=True): pass
    async def status(self, note): self.statuses.append(note)
    async def finish(self, text): self.final = text


class ScriptedLLM(LLMProvider):
    """Returns queued responses in order."""

    def __init__(self, replies: list[CompletionResponse]):
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if not self.replies:
            return CompletionResponse(
                message=ChatMessage(role="assistant", content="(script exhausted)"),
                finish_reason="stop",
            )
        return self.replies.pop(0)


class EchoTool(Tool):
    name = "echo"
    description = "always succeeds"
    parameters = {"type": "object", "properties": {"x": {"type": "string"}}}
    async def run(self, ctx: ToolContext, **kwargs):
        return f"echo:{kwargs.get('x')}"


class ParEchoTool(EchoTool):
    name = "par_echo"
    parallel_safe = True


class FailingTool(Tool):
    name = "fail"
    description = "always raises ToolError"
    parameters = {"type": "object", "properties": {"x": {"type": "string"}}}
    async def run(self, ctx: ToolContext, **kwargs):
        raise ToolError("boom")


def call_tool(name: str, args: dict, call_id: str = "tc") -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(
            role="assistant", content="",
            tool_calls=[ToolCall(id=f"{call_id}-{name}", name=name, arguments=args)],
        ),
        finish_reason="tool_calls",
    )


async def make_agent(settings, llm, tools, sid):
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create(sid)
    return Agent(role=roles.get("team_admin"), session=sess,
                 settings=settings, llm=llm, tools=tools)


# --------------------------------------------------------------------------
# 1. identical-args success breaker (serial path)
# --------------------------------------------------------------------------

async def test_success_breaker_serial():
    print("== test 1: identical-args success breaker (serial) ==")
    settings = load_settings()
    settings.llm.repeated_success_break_threshold = 3

    tools = ToolRegistry()
    tools.register(EchoTool())
    # LLM keeps issuing the exact same call; breaker must trip on the 3rd.
    llm = ScriptedLLM([call_tool("echo", {"x": "a"}, f"tc{i}") for i in range(10)])
    agent = await make_agent(settings, llm, tools, "sess-succ-serial")

    reply = await agent.handle("帮我处理", CapturingStream())
    assert reply.startswith("操作已停止"), f"unexpected reply: {reply!r}"
    assert "echo" in reply and "3 次" in reply, reply
    assert llm.calls == 3, f"expected breaker after 3 LLM calls, got {llm.calls}"
    # Closing assistant message appended so next turn can't resume the loop.
    assert agent.history[-1].role == "assistant", agent.history[-1].role
    assert agent.history[-1].content == reply
    print(f"  ✓ breaker tripped after {llm.calls} identical successes; history closed")


# --------------------------------------------------------------------------
# 2. identical-args success breaker (parallel path)
# --------------------------------------------------------------------------

async def test_success_breaker_parallel():
    print("== test 2: identical-args success breaker (parallel batch) ==")
    settings = load_settings()
    settings.llm.repeated_success_break_threshold = 4

    tools = ToolRegistry()
    tools.register(ParEchoTool())
    # One identical call per batch → trips on the 4th batch.
    llm = ScriptedLLM([call_tool("par_echo", {"x": "a"}, f"tc{i}") for i in range(10)])
    agent = await make_agent(settings, llm, tools, "sess-succ-par")

    reply = await agent.handle("帮我处理", CapturingStream())
    assert reply.startswith("操作已停止"), f"unexpected reply: {reply!r}"
    assert llm.calls == 4, f"expected breaker after 4 calls, got {llm.calls}"
    assert agent.history[-1].role == "assistant"
    assert agent.history[-1].content == reply
    print(f"  ✓ parallel-path breaker tripped after {llm.calls} identical successes")


# --------------------------------------------------------------------------
# 3. cap-hit appends closing assistant message; next turn sees it
# --------------------------------------------------------------------------

async def test_cap_hit_closing_message():
    print("== test 3: cap-hit fallback closes the transcript ==")
    settings = load_settings()
    settings.llm.max_tool_loops_per_turn = 3
    settings.llm.repeated_success_break_threshold = 100   # don't interfere

    tools = ToolRegistry()
    tools.register(EchoTool())
    # Distinct args each loop so the success breaker stays quiet.
    llm = ScriptedLLM([call_tool("echo", {"x": f"v{i}"}) for i in range(10)])
    agent = await make_agent(settings, llm, tools, "sess-cap")

    reply = await agent.handle("开始", CapturingStream())
    assert reply == "(已达到工具循环上限,本轮未给出最终答复)", reply
    assert llm.calls == 3, llm.calls
    assert agent.history[-1].role == "assistant"
    assert agent.history[-1].content == reply
    # Next turn: history must present the fallback as the prior assistant
    # answer (no dangling tool round), then the new user message.
    llm.replies = [CompletionResponse(
        message=ChatMessage(role="assistant", content="恢复正常"),
        finish_reason="stop",
    )]
    reply2 = await agent.handle("状态", CapturingStream())
    assert reply2 == "恢复正常", reply2
    roles_seq = [m.role for m in agent.history]
    assert roles_seq[-3:] == ["assistant", "user", "assistant"], roles_seq
    assert agent.history[-3].content == "(已达到工具循环上限,本轮未给出最终答复)"
    print("  ✓ cap-hit reply persisted as assistant msg; next turn well-formed")


# --------------------------------------------------------------------------
# 4. error breaker regression: still trips, now also closes the transcript
# --------------------------------------------------------------------------

async def test_error_breaker_still_works():
    print("== test 4: identical ToolError breaker (regression) ==")
    settings = load_settings()
    settings.llm.repeated_success_break_threshold = 100

    tools = ToolRegistry()
    tools.register(FailingTool())
    llm = ScriptedLLM([call_tool("fail", {"x": "a"}, f"tc{i}") for i in range(10)])
    agent = await make_agent(settings, llm, tools, "sess-err-break")

    reply = await agent.handle("帮我处理", CapturingStream())
    assert reply.startswith("操作未完成"), reply
    assert "fail" in reply and "boom" in reply, reply
    assert llm.calls == 3, llm.calls
    assert agent.history[-1].role == "assistant"
    assert agent.history[-1].content == reply
    print("  ✓ error breaker unchanged; fallback now closes the transcript")


# --------------------------------------------------------------------------
# 5. progress heartbeat stops on expired stream; legacy None stays alive
# --------------------------------------------------------------------------

class ExpiringStream:
    """status() returns True once, then False (stream expired)."""
    def __init__(self):
        self.n = 0
    async def status(self, note):
        self.n += 1
        return self.n <= 1


class LegacyStream:
    async def status(self, note):
        return None


async def test_heartbeat_stops_on_expired():
    print("== test 5: heartbeat stops when the stream expires ==")
    settings = load_settings()
    settings.session.progress_status_delay_seconds = 0.0
    settings.session.progress_status_interval_seconds = 0.05

    sessions = SessionManager(settings)
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    disp = Dispatcher(settings, sessions, roles, ToolRegistry(),
                      ScriptedLLM([]))

    stream = ExpiringStream()
    stop = asyncio.Event()
    task = asyncio.create_task(disp._progress_heartbeat(stream, stop))
    await asyncio.wait_for(task, timeout=2.0)     # must RETURN on its own
    assert stream.n == 2, f"expected 2 pushes then stop, got {stream.n}"
    assert not stop.is_set()
    print(f"  ✓ heartbeat exited by itself after status() returned False ({stream.n} pushes)")

    legacy = LegacyStream()
    stop2 = asyncio.Event()
    task2 = asyncio.create_task(disp._progress_heartbeat(legacy, stop2))
    await asyncio.sleep(0.2)
    assert not task2.done(), "legacy None-returning stream must keep heartbeat alive"
    stop2.set()
    await asyncio.wait_for(task2, timeout=2.0)
    print("  ✓ legacy None-returning status() keeps heartbeat running until stop")


# --------------------------------------------------------------------------
# 6. WeCom stream handle: 846608 marks expired, pushers stop, finish skips
# --------------------------------------------------------------------------

async def test_wecom_stream_expired():
    print("== test 6: WeComStreamHandle expiry via 846608 ack ==")
    from chat_team.adapters.wecom import WeComBotAdapter, WeComStreamHandle

    settings = load_settings()
    adapter = WeComBotAdapter(settings, bot_id="b", role_name="r")

    sent: list[dict] = []
    async def fake_write(payload): sent.append(payload)
    adapter._enqueue_write = fake_write          # type: ignore[method-assign]

    h = WeComStreamHandle(adapter, req_id="req-1")
    assert adapter._active_streams.get("req-1") is h

    # Healthy stream: status pushes and reports alive.
    assert await h.status("处理中") is True
    assert len(sent) == 1 and sent[0]["body"]["msgtype"] == "stream"

    # WeCom rejects a frame with 846608 → handle flips to expired.
    adapter._dispatch_ack({"headers": {"req_id": "req-1"},
                           "errcode": 846608, "errmsg": "expired"})
    assert h.expired is True
    assert await h.status("还在吗") is False      # heartbeat would stop here
    await h.push("more text")                     # silently dropped
    assert len(sent) == 1, "no further stream frames after expiry"

    # finish(): no closing stream frame, but the markdown reply still goes.
    await h.finish("最终答复")
    assert len(sent) == 2, sent
    assert sent[1]["body"]["msgtype"] == "markdown"
    assert "req-1" not in adapter._active_streams
    print("  ✓ 846608 marks stream expired; push/status no-op; finish sends markdown only")

    # Unrelated acks / other req_ids don't flip the flag.
    h2 = WeComStreamHandle(adapter, req_id="req-2")
    adapter._dispatch_ack({"headers": {"req_id": "req-other"},
                           "errcode": 846608, "errmsg": "expired"})
    adapter._dispatch_ack({"headers": {"req_id": "req-2"},
                           "errcode": 0, "errmsg": "ok"})
    assert h2.expired is False
    print("  ✓ unrelated/ok acks leave other streams alive")


async def main():
    await test_success_breaker_serial()
    await test_success_breaker_parallel()
    await test_cap_hit_closing_message()
    await test_error_breaker_still_works()
    await test_heartbeat_stops_on_expired()
    await test_wecom_stream_expired()
    print("\nALL LOOP-GUARD SMOKES PASSED")


if __name__ == "__main__":
    asyncio.run(main())
