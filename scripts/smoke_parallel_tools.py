"""Smoke tests for parallel tool-call dispatch.

Verifies that when an LLM returns multiple tool_calls in one assistant
message, the agent dispatches them concurrently when ALL target
``parallel_safe`` tools (currently only McpProxyTool), and falls back to
serial execution otherwise.

Run:
    python scripts/smoke_parallel_tools.py

Uses mocks — no real MCP servers, no real LLM.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_parallel_smoke"
os.environ.setdefault("OPENAI_API_KEY", "test")

from chat_team.agent.agent import Agent
from chat_team.agent.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
)
from chat_team.config import load_settings
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from chat_team.mcp.proxy_tool import McpProxyTool
from chat_team.roles.config import Role
from chat_team.session.notebook import Notebook
from chat_team.session.session import Session
from chat_team.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeMcpTool:
    name: str = "ping"
    description: str = "ping"
    inputSchema: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
    })


@dataclass
class FakeTextContent:
    type: str = "text"
    text: str = ""


@dataclass
class FakeCallToolResult:
    content: list = field(default_factory=list)
    isError: bool = False


class FakeMcpSession:
    """Records call order and supports a delay to test concurrency."""

    def __init__(
        self,
        result_text: str = "pong",
        delay: float = 0.0,
        error: Exception | None = None,
    ):
        self._result_text = result_text
        self._delay = delay
        self._error = error
        self.calls: list[tuple[str, dict | None, float]] = []  # name, args, start_time
        self.call_completions: list[tuple[str, float]] = []  # name, end_time

    async def call_tool(self, name: str, arguments: dict | None = None, **kwargs) -> FakeCallToolResult:
        self.calls.append((name, arguments, time.monotonic()))
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        self.call_completions.append((name, time.monotonic()))
        if self._error:
            raise self._error
        return FakeCallToolResult(content=[FakeTextContent(text=self._result_text)])


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedLLM exhausted")
        return self._responses.pop(0)


class CapturingStream:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    async def push(self, chunk: str, *, append: bool = True) -> None:
        pass

    async def status(self, note: str) -> None:
        self.statuses.append(note)

    async def finish(self, final_text: str) -> None:
        pass


class SerialSleepTool(Tool):
    """A non-parallel-safe tool that sleeps — used to verify serial fallback."""

    def __init__(self, name: str = "serial_sleep", delay: float = 0.3):
        self.name = name
        self.description = "sleeps then returns ok"
        self.parameters = {"type": "object", "properties": {}}
        self._delay = delay

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        await asyncio.sleep(self._delay)
        return f"{self.name}:ok"


def reply(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def multi_call(calls: list[tuple[str, dict, str]]) -> CompletionResponse:
    """Build an assistant response with multiple tool_calls."""
    tool_calls = [ToolCall(id=cid, name=name, arguments=args)
                  for name, args, cid in calls]
    return CompletionResponse(
        message=ChatMessage(role="assistant", content="", tool_calls=tool_calls),
        finish_reason="tool_calls",
    )


def make_session(sid: str, cwd: Path) -> Session:
    nb_path = cwd / ".chat_team" / "notebook.md"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    return Session(session_id=sid, cwd=cwd, current_role="test", notebook=Notebook(nb_path))


def make_agent(
    home: Path,
    settings,
    reg: ToolRegistry,
    llm: LLMProvider,
    role_tools: list[str],
    mcp_servers: list[str] | None = None,
) -> Agent:
    role = Role(
        name="test",
        display_name="Test",
        description="",
        system_prompt="You are a test agent.",
        tools=role_tools,
        mcp_servers=mcp_servers or [],
    )
    sess = make_session("s1", home)
    return Agent(
        role=role, session=sess, settings=settings,
        llm=llm, tools=reg, skills=SkillRegistry({}),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parallel_safe_default_false():
    """Tool base class: parallel_safe defaults to False."""
    assert Tool.parallel_safe is False
    assert SerialSleepTool().parallel_safe is False
    print("  Tool.parallel_safe default False: OK")


def test_mcp_proxy_parallel_safe_true():
    """McpProxyTool.parallel_safe is True."""
    assert McpProxyTool.parallel_safe is True
    proxy = McpProxyTool("srv", FakeMcpTool(), FakeMcpSession())
    assert proxy.parallel_safe is True
    print("  McpProxyTool.parallel_safe True: OK")


async def test_parallel_dispatch_concurrent():
    """3 MCP tool_calls with 0.3s delay each → total ~0.3s, not ~0.9s."""
    home = Path("/tmp/chat_team_parallel_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("")
    settings = load_settings()

    session = FakeMcpSession(result_text="pong", delay=0.3)
    reg = ToolRegistry()
    reg.register(McpProxyTool("srv", FakeMcpTool(name="ping"), session))

    llm = ScriptedLLM([
        multi_call([
            ("mcp__srv__ping", {}, "tc-1"),
            ("mcp__srv__ping", {}, "tc-2"),
            ("mcp__srv__ping", {}, "tc-3"),
        ]),
        reply("done"),
    ])
    agent = make_agent(home, settings, reg, llm, role_tools=[], mcp_servers=["srv"])

    stream = CapturingStream()
    t0 = time.monotonic()
    result = await agent.handle("ping 3 times", stream)
    elapsed = time.monotonic() - t0

    assert result == "done"
    # Parallel: ~0.3s. Serial would be ~0.9s. Allow generous margin.
    assert elapsed < 0.7, f"expected parallel (<0.7s), got {elapsed:.2f}s"
    assert len(session.calls) == 3
    # All 3 calls should start at roughly the same time (within 50ms)
    start_times = [c[2] for c in session.calls]
    spread = max(start_times) - min(start_times)
    assert spread < 0.05, f"calls not concurrent, start spread={spread:.3f}s"
    # Status pushed once for the whole batch
    assert len(stream.statuses) == 1
    assert "mcp__srv__ping" in stream.statuses[0]
    print(f"  parallel dispatch (3×0.3s = {elapsed:.2f}s): OK")


async def test_serial_fallback_when_unsafe_present():
    """Mixed batch (2 MCP + 1 serial_sleep) → serial, total ~0.9s."""
    home = Path("/tmp/chat_team_parallel_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("")
    settings = load_settings()

    session = FakeMcpSession(result_text="pong", delay=0.3)
    reg = ToolRegistry()
    reg.register(McpProxyTool("srv", FakeMcpTool(name="ping"), session))
    reg.register(SerialSleepTool("serial_sleep", delay=0.3))

    llm = ScriptedLLM([
        multi_call([
            ("mcp__srv__ping", {}, "tc-1"),
            ("serial_sleep", {}, "tc-2"),
            ("mcp__srv__ping", {}, "tc-3"),
        ]),
        reply("done"),
    ])
    agent = make_agent(home, settings, reg, llm,
                       role_tools=["serial_sleep"], mcp_servers=["srv"])

    stream = CapturingStream()
    t0 = time.monotonic()
    result = await agent.handle("mixed batch", stream)
    elapsed = time.monotonic() - t0

    assert result == "done"
    # Serial: 3 × 0.3s = 0.9s. Allow margin.
    assert elapsed >= 0.8, f"expected serial (>=0.8s), got {elapsed:.2f}s"
    assert elapsed < 1.2
    # 3 separate status pushes (serial pushes one per call)
    assert len(stream.statuses) == 3
    print(f"  serial fallback for mixed batch ({elapsed:.2f}s): OK")


async def test_serial_fallback_all_unsafe():
    """All-unsafe batch → serial."""
    home = Path("/tmp/chat_team_parallel_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("")
    settings = load_settings()

    reg = ToolRegistry()
    reg.register(SerialSleepTool("serial_a", delay=0.2))
    reg.register(SerialSleepTool("serial_b", delay=0.2))

    llm = ScriptedLLM([
        multi_call([
            ("serial_a", {}, "tc-1"),
            ("serial_b", {}, "tc-2"),
        ]),
        reply("done"),
    ])
    agent = make_agent(home, settings, reg, llm,
                       role_tools=["serial_a", "serial_b"])

    stream = CapturingStream()
    t0 = time.monotonic()
    result = await agent.handle("all unsafe", stream)
    elapsed = time.monotonic() - t0

    assert result == "done"
    assert elapsed >= 0.35, f"expected serial (>=0.35s), got {elapsed:.2f}s"
    print(f"  serial fallback all-unsafe ({elapsed:.2f}s): OK")


async def test_parallel_history_order_preserved():
    """Parallel results appended to history in tool_calls order."""
    home = Path("/tmp/chat_team_parallel_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("")
    settings = load_settings()

    # Different delays per call to maximize chance of reordering if buggy.
    # Use separate sessions per tool to control delays.
    class MultiDelaySession:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments=None, **kwargs):
            self.calls.append((name, arguments))
            idx = len(self.calls) - 1
            delays = [0.3, 0.05, 0.2]  # 2nd finishes first
            await asyncio.sleep(delays[idx % 3])
            return FakeCallToolResult(content=[FakeTextContent(text=f"result-{idx}")])

    ms = MultiDelaySession()
    reg = ToolRegistry()
    reg.register(McpProxyTool("srv", FakeMcpTool(name="ping"), ms))

    llm = ScriptedLLM([
        multi_call([
            ("mcp__srv__ping", {}, "tc-1"),
            ("mcp__srv__ping", {}, "tc-2"),
            ("mcp__srv__ping", {}, "tc-3"),
        ]),
        reply("done"),
    ])
    agent = make_agent(home, settings, reg, llm, role_tools=[], mcp_servers=["srv"])

    stream = CapturingStream()
    await agent.handle("test order", stream)

    # Find the tool messages in history
    tool_msgs = [m for m in agent.history if m.role == "tool"]
    assert len(tool_msgs) == 3
    # Order must match tool_calls order: tc-1, tc-2, tc-3
    assert tool_msgs[0].tool_call_id == "tc-1"
    assert tool_msgs[1].tool_call_id == "tc-2"
    assert tool_msgs[2].tool_call_id == "tc-3"
    print("  parallel history order preserved: OK")


async def test_parallel_error_isolation():
    """One MCP tool erroring doesn't cancel others; all results recorded."""
    home = Path("/tmp/chat_team_parallel_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("")
    settings = load_settings()

    class MixedSession:
        def __init__(self):
            self.call_count = 0

        async def call_tool(self, name, arguments=None, **kwargs):
            self.call_count += 1
            n = self.call_count
            if n == 2:
                # 2nd call errors
                raise RuntimeError("boom on call 2")
            await asyncio.sleep(0.1)
            return FakeCallToolResult(content=[FakeTextContent(text=f"ok-{n}")])

    ms = MixedSession()
    reg = ToolRegistry()
    reg.register(McpProxyTool("srv", FakeMcpTool(name="ping"), ms))

    llm = ScriptedLLM([
        multi_call([
            ("mcp__srv__ping", {}, "tc-1"),
            ("mcp__srv__ping", {}, "tc-2"),  # will error
            ("mcp__srv__ping", {}, "tc-3"),
        ]),
        reply("handled errors"),
    ])
    agent = make_agent(home, settings, reg, llm, role_tools=[], mcp_servers=["srv"])

    stream = CapturingStream()
    result = await agent.handle("test isolation", stream)

    assert result == "handled errors"
    tool_msgs = [m for m in agent.history if m.role == "tool"]
    assert len(tool_msgs) == 3, f"expected 3 tool msgs, got {len(tool_msgs)}"
    # tc-2 should have an error result
    tc2_msg = next(m for m in tool_msgs if m.tool_call_id == "tc-2")
    assert "tool_error" in tc2_msg.content or "boom" in tc2_msg.content, \
        f"tc-2 should be error, got: {tc2_msg.content!r}"
    # tc-1 and tc-3 should have ok results
    tc1_msg = next(m for m in tool_msgs if m.tool_call_id == "tc-1")
    tc3_msg = next(m for m in tool_msgs if m.tool_call_id == "tc-3")
    assert "ok" in tc1_msg.content, f"tc-1 should be ok, got: {tc1_msg.content!r}"
    assert "ok" in tc3_msg.content, f"tc-3 should be ok, got: {tc3_msg.content!r}"
    print("  parallel error isolation: OK")


async def test_parallel_single_call_still_works():
    """Single MCP tool_call in a batch → still dispatched correctly."""
    home = Path("/tmp/chat_team_parallel_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("")
    settings = load_settings()

    session = FakeMcpSession(result_text="solo-pong")
    reg = ToolRegistry()
    reg.register(McpProxyTool("srv", FakeMcpTool(name="ping"), session))

    llm = ScriptedLLM([
        multi_call([("mcp__srv__ping", {}, "tc-1")]),
        reply("solo done"),
    ])
    agent = make_agent(home, settings, reg, llm, role_tools=[], mcp_servers=["srv"])

    stream = CapturingStream()
    result = await agent.handle("single call", stream)

    assert result == "solo done"
    assert len(session.calls) == 1
    tool_msgs = [m for m in agent.history if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "solo-pong" in tool_msgs[0].content
    print("  parallel single call: OK")


async def test_unknown_tool_in_batch_falls_serial():
    """If a batch contains an unknown tool name, _is_parallel_safe returns
    False, so the batch runs serially — preserving the original ToolError
    behavior for unknown tools."""
    home = Path("/tmp/chat_team_parallel_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("")
    settings = load_settings()

    session = FakeMcpSession(result_text="pong")
    reg = ToolRegistry()
    reg.register(McpProxyTool("srv", FakeMcpTool(name="ping"), session))

    llm = ScriptedLLM([
        multi_call([
            ("mcp__srv__ping", {}, "tc-1"),
            ("nonexistent_tool", {}, "tc-2"),
        ]),
        reply("done"),
    ])
    agent = make_agent(home, settings, reg, llm, role_tools=[], mcp_servers=["srv"])

    stream = CapturingStream()
    result = await agent.handle("unknown in batch", stream)

    assert result == "done"
    tool_msgs = [m for m in agent.history if m.role == "tool"]
    assert len(tool_msgs) == 2
    # tc-2 should have a tool_error for unknown tool
    tc2_msg = next(m for m in tool_msgs if m.tool_call_id == "tc-2")
    assert "tool_error" in tc2_msg.content or "unknown" in tc2_msg.content.lower(), \
        f"tc-2 should be error, got: {tc2_msg.content!r}"
    print("  unknown tool in batch → serial fallback: OK")


async def main() -> None:
    print("=== Parallel tool dispatch smoke tests ===")

    test_parallel_safe_default_false()
    test_mcp_proxy_parallel_safe_true()
    await test_parallel_dispatch_concurrent()
    await test_serial_fallback_when_unsafe_present()
    await test_serial_fallback_all_unsafe()
    await test_parallel_history_order_preserved()
    await test_parallel_error_isolation()
    await test_parallel_single_call_still_works()
    await test_unknown_tool_in_batch_falls_serial()

    print("\nALL PARALLEL TOOL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
