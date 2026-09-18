"""LLM provider abstraction. Concrete providers wrap a vendor SDK
and translate to/from the dataclasses below."""
from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..adapters.base import ContentBlock

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    # ``str`` for assistant / tool / plain user messages; ``list[ContentBlock]``
    # for multi-modal user messages (text + image blocks). Provider, compactor,
    # and persistence all branch on ``isinstance(content, list)``.
    content: str | list[ContentBlock] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == "tool"
    name: str | None = None


@dataclass
class ToolSpec:
    """JSON-Schema description of a tool, fed to the LLM."""
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class CompletionRequest:
    messages: list[ChatMessage]
    tools: list[ToolSpec] = field(default_factory=list)
    # Optional per-request override.  Keep ``tools`` unchanged when a
    # maintenance request must reuse the same cached request prefix; passing
    # "none" tells compatible providers not to generate tool calls.
    tool_choice: str | None = None
    model: str = ""
    temperature: float = 0.3
    max_tokens: int | None = None
    # Optional reasoning depth (for models/providers that support it).
    reasoning_effort: str | None = None
    image_detail: str | None = None  # "low" | "high" | "auto"; provider applies to every image block
    image_base_dir: Any | None = None  # base for resolving relative image paths (typically session.cwd)
    # Optional debug-log context. When ``debug_log_dir`` is set AND the
    # provider was constructed with ``debug_log_enabled=True``, the
    # provider writes one JSON file per call. All four are optional so
    # legacy/test call sites stay valid.
    session_id: str | None = None
    # Stable conversation UUID sent as the ``x-session-id`` HTTP header when
    # the provider makes an upstream model request. Kept separate from the
    # internal WeCom-derived session_id for privacy and backward compatibility.
    session_uuid: str | None = None
    role_name: str | None = None
    call_kind: str | None = None  # "agent" | "compactor" | "vision"
    debug_log_dir: Path | None = None
    # Optional live callback for provider-side streaming. Provider passes the
    # cumulative assistant text seen so far (not just the latest delta).
    stream_text_callback: Callable[[str], Awaitable[None]] | None = None


@dataclass
class CompletionResponse:
    message: ChatMessage          # assistant message (may contain tool_calls)
    finish_reason: str            # "stop" | "tool_calls" | "length" | ...
    raw: Any | None = None


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
