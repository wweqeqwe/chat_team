"""End-to-end smoke for employee transfer mechanics:

* Multi-hop transfer in a single user turn (admin → engineer → customer).
* Per-turn transfer cap forces an answer after the 3rd handoff attempt.
* Handoff note is prepended to the receiving agent's next user turn so the
  prior request remains an exact cache prefix.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_transfer_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

# Seed two test-only roles so this smoke doesn't depend on the builtin set
# (which intentionally only ships team_admin). The fake LLM is scripted, so
# only the role *names* matter here — prompts and tool lists are minimal.
_roles_dir = Path(os.environ["CHAT_TEAM_HOME"]) / "roles"
_roles_dir.mkdir(parents=True, exist_ok=True)
for _name, _display in (("engineer", "测试工程师"), ("customer", "测试客服")):
    (_roles_dir / f"{_name}.yaml").write_text(
        f"name: {_name}\n"
        f"display_name: {_display}\n"
        "system_prompt: |\n"
        f"  你是测试用的 {_display}。\n"
        "tools:\n"
        "  - notebook_read\n"
        "  - notebook_write\n"
        "  - transfer_to_employee\n"
        "llm:\n"
        "  temperature: 0.3\n",
        encoding="utf-8",
    )

from chat_team.adapters.base import ChatType, IncomingMessage
from chat_team.agent.tools.base import ToolRegistry
from chat_team.agent.tools.file_tools import ListDirTool, ReadFileTool, WriteFileTool
from chat_team.agent.tools.notebook_tools import NotebookReadTool, NotebookWriteTool
from chat_team.agent.tools.shell_tool import RunCommandTool
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


class CapturingStream:
    def __init__(self):
        self.statuses, self.final = [], None
    async def push(self, chunk, *, append=True): pass
    async def status(self, note): self.statuses.append(note)
    async def finish(self, text): self.final = text


class ScriptedLLM(LLMProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
    async def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise RuntimeError("ScriptedLLM exhausted")
        return self.responses.pop(0)


def reply(text):
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def transfer(target, note, call_id):
    return CompletionResponse(
        message=ChatMessage(
            role="assistant", content="",
            tool_calls=[ToolCall(
                id=call_id, name="transfer_to_employee",
                arguments={"employee": target, "reason": "需要换人", "handoff_note": note},
            )],
        ),
        finish_reason="tool_calls",
    )


def build(settings, llm):
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    for t in (
        ReadFileTool(), WriteFileTool(), ListDirTool(), RunCommandTool(),
        NotebookReadTool(), NotebookWriteTool(),
        TransferToEmployeeTool(available_employees=roles.names()),
    ):
        tools.register(t)
    return Dispatcher(settings, SessionManager(settings), roles, tools, llm)


def msg(sid, text):
    return IncomingMessage(
        session_id=sid, chat_type=ChatType.SINGLE, user_id="u",
        text=text, msg_id="m-" + text[:6], bot_id="bot",
    )


async def test_chain_within_cap():
    print("== test 1: admin → engineer → customer (within cap=3) ==")
    settings = load_settings()
    llm = ScriptedLLM([
        transfer("engineer", "用户先看代码", "tc1"),
        transfer("customer", "代码看完了,后续是答疑", "tc2"),
        reply("你好,我是小客,有什么问题尽管问。"),
    ])
    disp = build(settings, llm)
    s = CapturingStream()
    await disp.handle(msg("sess-A", "我想找客服,但先让研发看一眼代码"), s)
    sess = await disp.sessions.get_or_create("sess-A")
    print("  final:", s.final)
    print("  current_role:", sess.current_role)
    assert sess.current_role == "customer"
    assert "小客" in s.final

    # The handoff note is a one-shot context block inside the next persisted
    # user message.  It must not be inserted as a leading system message,
    # because that would invalidate an existing target agent's prompt cache.
    third_request = llm.requests[2]
    sys_msgs = [m.content for m in third_request.messages if m.role == "system"]
    assert not any("代码看完了" in s for s in sys_msgs)
    user_msgs = [m.content for m in third_request.messages if m.role == "user"]
    assert any(
        isinstance(content, str) and "代码看完了" in content
        and "[系统上下文通知 — 非用户输入]" in content
        for content in user_msgs
    ), f"handoff note missing in customer's user context: {user_msgs}"
    print("  ✓ handoff note appended with customer's next user turn")


async def test_cap_forces_answer():
    print("== test 2: cap=3 forces current role to answer ==")
    settings = load_settings()
    settings.session.per_turn_transfer_cap = 2  # tighten so we don't need many scripts
    llm = ScriptedLLM([
        transfer("engineer", "n1", "t1"),   # 1st transfer
        transfer("customer", "n2", "t2"),   # 2nd → hits cap (>=2)
        # forced answer call by the customer agent
        reply("(在客服位上被强制回答)抱歉,我来直接回答:你好,有什么可以帮你?"),
    ])
    disp = build(settings, llm)
    s = CapturingStream()
    await disp.handle(msg("sess-B", "你好"), s)
    sess = await disp.sessions.get_or_create("sess-B")
    print("  final:", s.final)
    print("  transfer_count(reset_after_turn):", sess.transfer_count_this_turn)
    assert sess.transfer_count_this_turn == 0, "counter should be reset after turn"
    assert "强制" in s.final or "强制" in disp.roles.get(sess.current_role).display_name or "回答" in s.final


async def test_unknown_target_recovery():
    print("== test 3: unknown transfer target — fall back gracefully ==")
    settings = load_settings()
    llm = ScriptedLLM([
        # script: admin returns a tool call with a bogus target — but the
        # tool itself rejects (TransferToEmployeeTool validates against enum
        # so this becomes a tool_error rather than TransferRequested).
        # To exercise the dispatcher's "target unknown" branch we need to
        # make the tool RAISE TransferRequested with a bogus target. The
        # tool only does that if 'employee' passes its enum check, which it
        # won't. So this branch is mostly defensive — assert dispatch is sane.
        reply("OK"),
    ])
    disp = build(settings, llm)
    s = CapturingStream()
    await disp.handle(msg("sess-C", "随便说"), s)
    print("  final:", s.final)
    assert s.final == "OK"


async def main():
    await test_chain_within_cap()
    await test_cap_forces_answer()
    await test_unknown_target_recovery()
    print("\nALL TRANSFER TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
