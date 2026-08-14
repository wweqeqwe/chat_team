"""History compaction: when an agent's history exceeds its token budget,
summarize the early portion via the LLM and replace it with a single
``[历史摘要]`` system message at the head.

Constraints respected:
* Keep window always starts at a ``user`` message — we never split an
  ``assistant(tool_calls)`` + corresponding ``tool`` message pair.
* Token counting uses ``tiktoken`` (``cl100k_base`` — close enough for the
  gpt-4 family); falls back to char/4 when tiktoken is unavailable.
* Compaction is best-effort: if the LLM call fails, we leave history
  untouched so the next turn still works.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Iterable

import tiktoken

from ..adapters.base import blocks_to_text
from ..llm.base import ChatMessage, CompletionRequest, LLMProvider

if TYPE_CHECKING:
    from .agent import Agent

log = logging.getLogger(__name__)

KEEP_LAST_USER_TURNS = 6                # upper bound: keep up to this many recent user→answer cycles verbatim
PER_MESSAGE_OVERHEAD_TOKENS = 4         # OpenAI roughly charges ~3-4 framing tokens per msg

_ENCODING: tiktoken.Encoding | None = None


def _enc() -> tiktoken.Encoding | None:
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:                                     # noqa: BLE001
            log.warning("tiktoken encoding unavailable; falling back to char/4")
            _ENCODING = None
    return _ENCODING


def _msg_token_repr(m: ChatMessage) -> str:
    s = blocks_to_text(m.content)
    if m.tool_calls:
        s += json.dumps(
            [{"name": c.name, "args": c.arguments} for c in m.tool_calls],
            ensure_ascii=False,
        )
    if m.tool_call_id:
        s += m.tool_call_id
    return s


def count_tokens(messages: Iterable[ChatMessage]) -> int:
    enc = _enc()
    total = 0
    for m in messages:
        s = _msg_token_repr(m)
        if enc is not None:
            total += len(enc.encode(s))
        else:
            total += max(1, len(s) // 4)
        total += PER_MESSAGE_OVERHEAD_TOKENS
    return total


def _find_keep_boundary(history: list[ChatMessage], keep_user_turns: int) -> int:
    user_indices = [i for i, m in enumerate(history) if m.role == "user"]
    # Keep at least one most-recent user turn intact, but still compact when
    # a short history (e.g. exactly 6 turns) already blows past budget.
    if len(user_indices) <= 1:
        return 0
    keep = min(keep_user_turns, len(user_indices) - 1)
    return user_indices[-keep]


def _render_for_summary(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        prefix = f"[{m.role}]"
        if m.name:
            prefix += f" ({m.name})"
        body = blocks_to_text(m.content).strip()
        if body:
            lines.append(f"{prefix} {body}")
        if m.tool_calls:
            for tc in m.tool_calls:
                args = json.dumps(tc.arguments, ensure_ascii=False)
                lines.append(f"  ↳ tool_call {tc.name}({args})")
    return "\n".join(lines)


async def _summarize(
    prefix: list[ChatMessage],
    llm: LLMProvider,
    model: str,
    *,
    agent: "Agent",
) -> str:
    body = _render_for_summary(prefix)
    request = CompletionRequest(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "你是会话历史压缩器。请把下列对话摘要为不超过 300 字的中文要点,"
                    "需保留:用户目标、关键决策与事实、已完成的动作、待办事项、"
                    "重要的工具调用结论。省略寒暄、重复内容、可在记事本中查到的明细。"
                    "输出仅是要点本身,不要加任何前后缀。"
                ),
            ),
            ChatMessage(role="user", content=body),
        ],
        model=model,
        temperature=0.0,
        reasoning_effort=(agent.settings.llm.chat.reasoning_effort or "").strip() or None,
        session_id=agent.session.session_id,
        session_uuid=agent.session.session_uuid,
        role_name=agent.role.name,
        call_kind="compactor",
        debug_log_dir=agent.session.cwd / ".chat_team" / "llm",
    )
    resp = await llm.complete(request)
    return (resp.message.content or "").strip()


async def maybe_compact(agent: "Agent", llm: LLMProvider) -> bool:
    """If history exceeds budget, summarize the prefix in place. Returns True iff compaction ran."""
    budget = (
        agent.role.llm.history_token_budget
        or agent.settings.llm.chat.history_token_budget
    )
    if budget <= 0:
        return False
    tokens = count_tokens(agent.history)
    if tokens <= budget:
        return False

    boundary = _find_keep_boundary(agent.history, KEEP_LAST_USER_TURNS)
    if boundary <= 0:
        log.info(
            "role=%s over budget (%d > %d) but has <=1 user turn; nothing safe to compact",
            agent.role.name, tokens, budget,
        )
        return False

    prefix = agent.history[:boundary]
    suffix = agent.history[boundary:]
    if not prefix:
        return False

    model = agent.role.llm.model or agent.settings.llm.chat.model
    try:
        summary = await _summarize(prefix, llm, model, agent=agent)
    except Exception:                                         # noqa: BLE001
        log.exception("summarize failed for role=%s; leaving history intact", agent.role.name)
        return False
    if not summary:
        log.warning("summary came back empty; leaving history intact")
        return False

    new_head = ChatMessage(
        role="system",
        content=(
            f"[历史摘要 — 由系统压缩,原始 {len(prefix)} 条消息 / {tokens} tokens]\n"
            f"{summary}"
        ),
    )
    agent.history = [new_head] + suffix
    log.info(
        "compacted role=%s: %d msgs → %d msgs (kept %d trailing)",
        agent.role.name, len(prefix) + len(suffix), len(agent.history), len(suffix),
    )
    return True
