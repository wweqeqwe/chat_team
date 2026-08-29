"""Smoke coverage for prompt-prefix cache preservation.

Pure Python / scripted LLM:
* notebook writes never rewrite the leading system prompt;
* notebook and handoff updates are appended with the next user turn;
* compaction extends the exact last agent request (messages + tools).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_prompt_cache_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.agent.agent import Agent
from chat_team.agent.compactor import maybe_compact
from chat_team.agent.tools.base import ToolRegistry
from chat_team.agent.tools.notebook_tools import NotebookWriteTool
from chat_team.config import load_settings
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
    async def push(self, chunk: str, *, append: bool = True) -> None:
        pass

    async def status(self, note: str) -> None:
        pass

    async def finish(self, final_text: str) -> None:
        pass


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if request.call_kind == "compactor":
            return reply("缓存感知压缩摘要")
        if not self.responses:
            raise RuntimeError("ScriptedLLM exhausted")
        return self.responses.pop(0)


class FailOnceLLM(LLMProvider):
    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise RuntimeError("simulated transient failure")
        return reply("重试成功")


def reply(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def tool_call(name: str, arguments: dict, call_id: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id=call_id, name=name, arguments=arguments),
            ],
        ),
        finish_reason="tool_calls",
    )


async def build_agent(
    sid: str,
    llm: LLMProvider,
    *,
    with_notebook_write: bool = False,
) -> Agent:
    settings = load_settings()
    role = RoleRegistry.load(settings.paths.user_roles_dir).get("team_admin")
    tools = ToolRegistry()
    if with_notebook_write:
        tools.register(NotebookWriteTool())
        role.tools = ["notebook_write"]
    session = await SessionManager(settings).get_or_create(sid)
    return Agent(
        role=role,
        session=session,
        settings=settings,
        llm=llm,
        tools=tools,
    )


async def test_notebook_write_keeps_system_prefix() -> None:
    print("== notebook write keeps system prefix ==")
    llm = ScriptedLLM([
        tool_call("notebook_write", {"key": "order-1", "value": "done"}, "tc1"),
        reply("写入完成"),
    ])
    agent = await build_agent("notebook-tool-loop", llm, with_notebook_write=True)
    await agent.handle("记录工单", CapturingStream())
    first, second = llm.requests
    assert first.messages[0] == second.messages[0]
    assert second.messages[:len(first.messages)] == first.messages
    assert "order-1" not in (second.messages[0].content or "")
    print("  PASS")


async def test_notebook_update_is_appended_once() -> None:
    print("== notebook revision becomes one append-only update ==")
    llm = ScriptedLLM([reply("收到"), reply("继续")])
    agent = await build_agent("notebook-external-update", llm)
    original_system = agent._build_system_messages()[0]
    agent.session.notebook.write("shared-fact", "来自其他角色")
    assert agent._build_system_messages()[0] == original_system

    await agent.handle("读取共享进度", CapturingStream())
    first_user = llm.requests[0].messages[-1]
    assert first_user.role == "user"
    assert "团队记事本更新" in (first_user.content or "")
    assert "shared-fact" in (first_user.content or "")
    assert llm.requests[0].messages[0] == original_system

    await agent.handle("再继续", CapturingStream())
    second_user = llm.requests[1].messages[-1]
    assert "团队记事本更新" not in (second_user.content or "")
    assert llm.requests[1].messages[:len(llm.requests[0].messages)] == \
        llm.requests[0].messages
    print("  PASS")


async def test_notebook_update_reaches_existing_peer_agent() -> None:
    print("== notebook update reaches an already-started peer agent ==")
    writer_llm = ScriptedLLM([
        tool_call(
            "notebook_write",
            {"key": "cross-bot-order", "value": "机器人A写入"},
            "tc-cross",
        ),
        reply("A写入完成"),
    ])
    reader_llm = ScriptedLLM([reply("B已看到更新")])
    # Build B first so its stable system snapshot/revision is intentionally
    # older than the write performed by A.  Separate SessionManager instances
    # mirror solo mode while both notebooks point at the same workspace file.
    reader = await build_agent("cross-bot", reader_llm)
    reader_system = reader._build_system_messages()[0]
    writer = await build_agent(
        "cross-bot", writer_llm, with_notebook_write=True,
    )

    await writer.handle("A记录共享工单", CapturingStream())
    await reader.handle("B继续处理", CapturingStream())

    reader_request = reader_llm.requests[0]
    assert reader_request.messages[0] == reader_system
    assert "cross-bot-order" not in (reader_request.messages[0].content or "")
    assert "团队记事本更新" in (reader_request.messages[-1].content or "")
    assert "cross-bot-order" in (reader_request.messages[-1].content or "")
    print("  PASS")


async def test_notebook_update_retry_is_not_duplicated() -> None:
    print("== failed turn restores one notebook update for retry ==")
    llm = FailOnceLLM()
    agent = await build_agent("notebook-retry", llm)
    agent.session.notebook.write("retry-fact", "只通知一次")

    try:
        await agent.handle("首次请求", CapturingStream())
    except RuntimeError as exc:
        assert "simulated transient failure" in str(exc)
    else:
        raise AssertionError("first request should fail")

    await agent.handle("重试请求", CapturingStream())
    retry_user = llm.requests[-1].messages[-1]
    assert isinstance(retry_user.content, str)
    assert retry_user.content.count("[团队记事本更新]") == 1
    assert "retry-fact" in retry_user.content
    print("  PASS")


async def test_handoff_note_is_prefix_extension() -> None:
    print("== handoff note is a prefix extension ==")
    llm = ScriptedLLM([reply("第一轮完成"), reply("接手完成")])
    agent = await build_agent("handoff-prefix", llm)
    await agent.handle("第一轮", CapturingStream())
    first = llm.requests[0]
    agent.queue_context_note("[交接备忘] 已确认客户资料，请继续。")
    await agent.handle("继续开单", CapturingStream())
    second = llm.requests[1]
    assert second.messages[:len(first.messages)] == first.messages
    assert "交接备忘" in (second.messages[-1].content or "")
    assert len([m for m in second.messages if m.role == "system"]) == 1
    print("  PASS")


async def test_compactor_extends_last_request() -> None:
    print("== compactor extends exact last agent request ==")
    llm = ScriptedLLM([
        reply("第一轮回答 " + "a" * 120),
        reply("第二轮回答 " + "b" * 120),
    ])
    agent = await build_agent("compactor-prefix", llm)
    agent.role.llm.history_token_budget = 99999
    await agent.handle("第一轮问题 " + "x" * 120, CapturingStream())
    await agent.handle("第二轮问题 " + "y" * 120, CapturingStream())
    base = agent.last_completion_request
    assert base is not None

    agent.role.llm.history_token_budget = 20
    assert await maybe_compact(agent, llm)
    compact = llm.requests[-1]
    assert compact.call_kind == "compactor"
    assert compact.messages[:len(base.messages)] == base.messages
    assert compact.tools == base.tools
    assert compact.messages[-1].role == "system"
    assert "系统维护任务：历史压缩" in (
        compact.messages[-1].content or ""
    )
    print("  PASS")


async def main() -> None:
    await test_notebook_write_keeps_system_prefix()
    await test_notebook_update_is_appended_once()
    await test_notebook_update_reaches_existing_peer_agent()
    await test_notebook_update_retry_is_not_duplicated()
    await test_handoff_note_is_prefix_extension()
    await test_compactor_extends_last_request()
    print("\nALL PROMPT-CACHE SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
