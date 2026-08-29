"""End-to-end smoke for stage 6:

* Compactor: when an agent's history exceeds its token budget, summarize the
  prefix in place. Boundary must land on a user message (no orphaned tool
  messages).
* Persistence: after each turn, ``Dispatcher`` schedules a debounced flush.
  ``PersistenceManager.flush_now`` and SessionManager restoration round-trip
  current_role + per-role histories on a brand-new SessionManager.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_compact_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

# Seed a test-only `engineer` role so the smoke doesn't depend on the builtin
# set (which intentionally only ships team_admin). Used by the persistence
# round-trip test that transfers admin → engineer.
_roles_dir = Path(os.environ["CHAT_TEAM_HOME"]) / "roles"
_roles_dir.mkdir(parents=True, exist_ok=True)
(_roles_dir / "engineer.yaml").write_text(
    "name: engineer\n"
    "display_name: 测试工程师\n"
    "system_prompt: |\n"
    "  你是测试用的研发同事。\n"
    "tools:\n"
    "  - notebook_read\n"
    "  - notebook_write\n"
    "  - transfer_to_employee\n"
    "llm:\n"
    "  temperature: 0.2\n"
    "  history_token_budget: 16000\n",
    encoding="utf-8",
)

from chat_team.adapters.base import ChatType, IncomingMessage
from chat_team.agent.agent import Agent
from chat_team.agent.compactor import count_tokens, maybe_compact
from chat_team.agent.tools.base import ToolRegistry
from chat_team.agent.tools.transfer_tool import TransferToEmployeeTool
from chat_team.config import load_settings
from chat_team.dispatcher import Dispatcher
from chat_team.llm.base import (
    ChatMessage,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager
from chat_team.session.persistence import (
    PersistenceManager,
    _deserialize_message,
    _serialize_message,
    load_state,
    restored_histories,
)


class CapturingStream:
    def __init__(self):
        self.statuses, self.final = [], None
    async def push(self, chunk, *, append=True): pass
    async def status(self, note): self.statuses.append(note)
    async def finish(self, text): self.final = text


class ScriptedLLM(LLMProvider):
    """Dispenses scripted CompletionResponses; used for non-summary calls.

    For sterile or cache-aware summarize calls, returns a canned summary.
    Otherwise pops from the queue.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        is_sterile_compactor = (
            request.messages
            and request.messages[0].role == "system"
            and "会话历史压缩器" in (request.messages[0].content or "")
        )
        is_cache_aware_compactor = any(
            m.role == "system"
            and isinstance(m.content, str)
            and "[系统维护任务：历史压缩]" in m.content
            for m in request.messages
        )
        if is_sterile_compactor or is_cache_aware_compactor:
            return CompletionResponse(
                message=ChatMessage(role="assistant", content="(压缩摘要) 用户主要诉求与历史决策的要点。"),
                finish_reason="stop",
            )
        if not self.responses:
            raise RuntimeError("ScriptedLLM exhausted")
        return self.responses.pop(0)


def reply(text):
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def msg(sid, text):
    return IncomingMessage(
        session_id=sid, chat_type=ChatType.SINGLE, user_id="u",
        text=text, msg_id="m-" + text[:6], bot_id="bot",
    )


def build_dispatcher(settings, llm, persistence=None):
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    tools.register(TransferToEmployeeTool(available_employees=roles.names()))
    sessions = SessionManager(settings)
    return sessions, Dispatcher(settings, sessions, roles, tools, llm, persistence=persistence)


# --------------------------------------------------------------------------
# 1. Compactor: prefix gets summarised, boundary lands on a user message.
# --------------------------------------------------------------------------

async def test_compactor_prefix_summarised():
    print("== test 1: compactor summarises old prefix ==")
    settings = load_settings()
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    tools.register(TransferToEmployeeTool(available_employees=roles.names()))
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-compact")

    role = roles.get("team_admin")
    role.llm.history_token_budget = 50            # tighten so we trip the cap
    llm = ScriptedLLM([])                         # only summarize call expected
    agent = Agent(role=role, session=sess, settings=settings, llm=llm, tools=tools)

    # 10 user→assistant turns with chunky content → blow past the 50-token budget.
    for i in range(10):
        agent.history.append(ChatMessage(role="user", content=f"用户第 {i} 次问题: " + "x" * 50))
        agent.history.append(ChatMessage(role="assistant", content=f"回答 {i}: " + "y" * 50))

    before_tokens = count_tokens(agent.history)
    before_len = len(agent.history)
    print(f"  pre-compact: {before_len} msgs, {before_tokens} tokens")
    assert before_tokens > 50

    did = await maybe_compact(agent, llm)
    assert did, "compaction should have run"
    print(f"  post-compact: {len(agent.history)} msgs, {count_tokens(agent.history)} tokens")

    # Head must be a system summary; second message must be a user message
    # (boundary always lands on user) so the LLM doesn't see an orphan tool.
    assert agent.history[0].role == "system"
    assert "历史摘要" in agent.history[0].content
    assert agent.history[1].role == "user"
    # The target-token policy keeps as many complete recent turns as fit.
    # With this deliberately tiny budget, only the latest complete turn fits.
    assert len(agent.history) == 3, f"unexpected length {len(agent.history)}"
    assert "用户第 9 次问题" in (agent.history[1].content or "")
    assert "回答 9" in (agent.history[2].content or "")


# --------------------------------------------------------------------------
# 2. Compactor: short history stays untouched.
# --------------------------------------------------------------------------

async def test_compactor_skipped_when_under_budget():
    print("== test 2: compactor skipped under budget ==")
    settings = load_settings()
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-skip")
    role = roles.get("team_admin")
    role.llm.history_token_budget = 99999
    llm = ScriptedLLM([])
    agent = Agent(role=role, session=sess, settings=settings, llm=llm, tools=tools)
    agent.history.append(ChatMessage(role="user", content="hi"))
    agent.history.append(ChatMessage(role="assistant", content="hello"))
    did = await maybe_compact(agent, llm)
    assert not did
    assert len(agent.history) == 2
    print("  ok — no compaction")


async def test_compactor_six_turns_still_compacts():
    print("== test 2b: exactly 6 user turns can still compact ==")
    settings = load_settings()
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-six-turns")

    role = roles.get("team_admin")
    role.llm.history_token_budget = 50
    llm = ScriptedLLM([])
    agent = Agent(role=role, session=sess, settings=settings, llm=llm, tools=tools)

    # Exactly 6 user turns, but each turn is large enough to exceed budget.
    for i in range(6):
        agent.history.append(ChatMessage(role="user", content=f"用户问题 {i}: " + "x" * 80))
        agent.history.append(ChatMessage(role="assistant", content=f"回答 {i}: " + "y" * 80))

    assert count_tokens(agent.history) > 50
    did = await maybe_compact(agent, llm)
    assert did, "compaction should run even when user turns == KEEP_LAST_USER_TURNS"
    assert agent.history[0].role == "system"
    assert "历史摘要" in (agent.history[0].content or "")
    assert agent.history[1].role == "user"
    # Tiny target → summary + latest complete user/assistant turn.
    assert len(agent.history) == 3, f"unexpected length {len(agent.history)}"
    assert "用户问题 5" in (agent.history[1].content or "")
    assert "回答 5" in (agent.history[2].content or "")
    print("  ok — compacted to target while keeping latest complete turn")


async def test_cache_aware_compactor_extends_last_agent_request():
    print("== test 2c: compactor extends exact last agent request ==")
    settings = load_settings()
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    tools.register(TransferToEmployeeTool(available_employees=roles.names()))
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("sess-cache-aware")
    role = roles.get("team_admin")
    role.llm.history_token_budget = 99999
    llm = ScriptedLLM([
        reply("第一轮回答。" + "a" * 100),
        reply("第二轮回答。" + "b" * 100),
    ])
    agent = Agent(
        role=role, session=sess, settings=settings, llm=llm, tools=tools,
    )
    stream = CapturingStream()
    await agent.handle("第一轮问题。" + "x" * 100, stream)
    await agent.handle("第二轮问题。" + "y" * 100, stream)

    base_request = agent.last_completion_request
    assert base_request is not None
    base_messages = list(base_request.messages)
    base_tools = list(base_request.tools)

    role.llm.history_token_budget = 20
    did = await maybe_compact(agent, llm)
    assert did
    compact_request = llm.requests[-1]
    assert compact_request.call_kind == "compactor"
    assert compact_request.messages[:len(base_messages)] == base_messages
    assert compact_request.tools == base_tools
    assert compact_request.model == base_request.model
    assert compact_request.temperature == base_request.temperature
    assert compact_request.reasoning_effort == base_request.reasoning_effort
    assert compact_request.messages[-1].role == "system"
    assert "[系统维护任务：历史压缩]" in (
        compact_request.messages[-1].content or ""
    )
    assert agent.history[0].role == "system"
    assert agent.history[1].role == "user"
    print("  ok — previous request is an exact message/tool prefix")


# --------------------------------------------------------------------------
# 3. Persistence: a turn snapshot survives a brand-new SessionManager.
# --------------------------------------------------------------------------

async def test_persistence_round_trip():
    print("== test 3: persistence round-trip via flush_now ==")
    settings = load_settings()
    settings.session.persistence_debounce_seconds = 0.05  # snappy for test

    llm = ScriptedLLM([
        # turn 1: admin handles greeting directly
        reply("你好,我是小管。"),
        # turn 2: admin transfers to engineer with a tool call
        CompletionResponse(
            message=ChatMessage(
                role="assistant", content="",
                tool_calls=[ToolCall(
                    id="tc1", name="transfer_to_employee",
                    arguments={"employee": "engineer", "reason": "讨论代码", "handoff_note": "用户想看代码"},
                )],
            ),
            finish_reason="tool_calls",
        ),
        # engineer answers
        reply("好的,我是小研,你想看哪部分代码?"),
    ])
    persistence = PersistenceManager(settings)
    sessions, disp = build_dispatcher(settings, llm, persistence=persistence)

    # turn 1
    s1 = CapturingStream()
    await disp.handle(msg("sess-persist", "你好"), s1)
    print("  turn1 final:", s1.final)

    # turn 2 — transfer
    s2 = CapturingStream()
    await disp.handle(msg("sess-persist", "我想找研发"), s2)
    print("  turn2 final:", s2.final)

    sess = await sessions.get_or_create("sess-persist")
    assert sess.current_role == "engineer"
    persistence.flush_now(sess)

    # Verify file exists and has expected shape
    state = load_state(sess.cwd)
    assert state is not None
    assert state["current_role"] == "engineer"
    assert "team_admin" in state["histories"]
    assert "engineer" in state["histories"]
    print("  ✓ session.json written; roles:", list(state["histories"].keys()))

    # Build a brand-new SessionManager pointing at the same workspace_root —
    # it should restore current_role and per-role histories from disk.
    sessions2 = SessionManager(settings)
    sess2 = await sessions2.get_or_create("sess-persist")
    assert sess2.current_role == "engineer"
    assert "team_admin" in sess2.restored_histories
    assert "engineer" in sess2.restored_histories
    admin_history = sess2.restored_histories["team_admin"]
    assert any(m.role == "user" and "你好" in (m.content or "") for m in admin_history)
    print("  ✓ restored to fresh SessionManager:",
          sess2.current_role, {r: len(h) for r, h in sess2.restored_histories.items()})

    # Wire the restored Session into a new Dispatcher and run another turn —
    # the engineer agent should pick up its prior history.
    llm2 = ScriptedLLM([reply("我接着之前的对话继续帮你看代码。")])
    persistence2 = PersistenceManager(settings)
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    tools = ToolRegistry()
    tools.register(TransferToEmployeeTool(available_employees=roles.names()))
    disp2 = Dispatcher(settings, sessions2, roles, tools, llm2, persistence=persistence2)
    s3 = CapturingStream()
    await disp2.handle(msg("sess-persist", "继续"), s3)
    print("  turn3 final:", s3.final)
    eng_agent = sess2.agents_by_role["engineer"]
    # restored history (3 messages from turn 2) + new turn (user + assistant) = 5
    assert any(
        m.role == "assistant" and "我是小研" in (m.content or "")
        for m in eng_agent.history
    ), "engineer should have its prior assistant message"
    assert "继续" in eng_agent.history[-2].content
    print("  ✓ engineer agent resumed with prior history")


# --------------------------------------------------------------------------
# 4. Persistence: debounced flush actually fires after the delay.
# --------------------------------------------------------------------------

async def test_persistence_debounced_fires():
    print("== test 4: debounced schedule actually flushes ==")
    settings = load_settings()
    settings.session.persistence_debounce_seconds = 0.1
    llm = ScriptedLLM([reply("ok")])
    persistence = PersistenceManager(settings)
    sessions, disp = build_dispatcher(settings, llm, persistence=persistence)
    s = CapturingStream()
    await disp.handle(msg("sess-debounce", "你好"), s)
    sess = await sessions.get_or_create("sess-debounce")
    state_path = sess.cwd / ".chat_team" / "session.json"
    assert not state_path.exists(), "should not be flushed instantly"
    await asyncio.sleep(0.25)
    assert state_path.exists(), "debounced flush should have fired"
    print("  ✓ session.json appeared after debounce window")


async def test_persistence_list_content_round_trip():
    print("== test 5: list content round-trips through persistence ==")
    blocks = [
        {"type": "text", "text": "看这两张"},
        {"type": "image", "path": "./inbox/a.jpg"},
        {"type": "image", "path": "./inbox/b.png"},
        {"type": "text", "text": "对比一下"},
    ]
    msg = ChatMessage(role="user", content=blocks)
    d = _serialize_message(msg)
    assert isinstance(d["content"], list)
    assert d["content"][1] == {"type": "image", "path": "./inbox/a.jpg"}

    # JSON survives the round-trip
    import json
    redeserialized = _deserialize_message(json.loads(json.dumps(d)))
    assert isinstance(redeserialized.content, list)
    assert len(redeserialized.content) == 4
    assert redeserialized.content[1]["type"] == "image"
    assert redeserialized.content[3]["text"] == "对比一下"
    print("  ✓ list content survives serialize → JSON → deserialize")

    # Legacy string-content still loads as string
    legacy = {"role": "user", "content": "你好"}
    legacy_msg = _deserialize_message(legacy)
    assert isinstance(legacy_msg.content, str)
    assert legacy_msg.content == "你好"
    print("  ✓ legacy string-content session.json loads unchanged")


async def main():
    await test_compactor_prefix_summarised()
    await test_compactor_skipped_when_under_budget()
    await test_compactor_six_turns_still_compacts()
    await test_cache_aware_compactor_extends_last_agent_request()
    await test_persistence_round_trip()
    await test_persistence_debounced_fires()
    await test_persistence_list_content_round_trip()
    print("\nALL STAGE-6 SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
