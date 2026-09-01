"""Bot adapter abstraction. Concrete adapters translate platform-specific
events to the internal ``IncomingMessage`` / ``OutgoingMessage`` shapes."""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Protocol, TypedDict


class ChatType(str, Enum):
    SINGLE = "single"
    GROUP = "group"


class ContentBlock(TypedDict, total=False):
    """Ordered piece of multi-modal user content. ``text`` blocks carry a
    string; ``image`` blocks carry a workspace-relative path (e.g.
    ``./inbox/20260523-foo.jpg``). Any other ``type`` is reserved for
    future use and renders as a ``[未支持]`` placeholder via
    :func:`blocks_to_text`."""
    type: str
    text: str
    path: str


def blocks_to_text(content: str | list[ContentBlock] | None) -> str:
    """Flatten a content-block list to a single string. Image blocks render
    as ``[图:<basename>]``. Used for logging, dedup keys, stream previews,
    compactor token counting, and summary input."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text") or "")
        elif btype == "image":
            path = block.get("path") or ""
            parts.append(f"[图:{os.path.basename(path)}]" if path else "[图]")
        else:
            parts.append(f"[未支持:{btype}]")
    return "\n".join(p for p in parts if p)


def coalesce_text_blocks(blocks: Iterable[ContentBlock]) -> list[ContentBlock]:
    """Merge adjacent text blocks; drop empty text blocks. Image blocks
    pass through untouched and preserve order."""
    out: list[ContentBlock] = []
    for block in blocks:
        if block.get("type") == "text":
            text = (block.get("text") or "").strip()
            if not text:
                continue
            if out and out[-1].get("type") == "text":
                merged = (out[-1].get("text") or "") + "\n" + text
                out[-1] = {"type": "text", "text": merged}
            else:
                out.append({"type": "text", "text": text})
        else:
            out.append(dict(block))
    return out


@dataclass
class IncomingMessage:
    """Normalised inbound message handed to the dispatcher."""
    session_id: str
    chat_type: ChatType
    user_id: str           # sender's wxid
    text: str              # flattened textual rendering (via blocks_to_text)
    msg_id: str            # platform message id (used for dedup)
    bot_id: str            # which bot received this
    raw: dict[str, Any] = field(default_factory=dict)
    chat_id: str | None = None      # group chat id, None for single
    user_name: str | None = None    # display name if available
    reply_token: Any | None = None  # adapter-specific (e.g. WeCom req_id)
    content_blocks: list[ContentBlock] = field(default_factory=list)


@dataclass
class OutgoingMessage:
    text: str
    finish: bool = True


class StreamHandle(Protocol):
    """Live handle to push streaming updates back to the user.

    The adapter creates one per inbound message that warrants streaming.
    Implementations are expected to throttle / coalesce internally.
    """

    async def push(self, chunk: str, *, append: bool = True) -> None: ...
    async def status(self, note: str) -> bool | None:
        """Push a transient status note.

        Implementations SHOULD return ``False`` once the stream is dead /
        platform-expired so long-running pushers (progress heartbeat) can
        stop. Returning ``None`` (legacy fakes) is treated as alive.
        """
        ...
    async def finish(self, final_text: str) -> None: ...
    async def send_image(self, path: Path, *, filename: str | None = None) -> None: ...
    async def send_file(self, path: Path, *, filename: str | None = None) -> None: ...


MessageHandler = Callable[[IncomingMessage, StreamHandle], Awaitable[None]]


class BotAdapter(abc.ABC):
    """Lifecycle: ``await connect()`` → registers handler → ``await run()`` blocks."""

    @abc.abstractmethod
    def set_handler(self, handler: MessageHandler) -> None: ...

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def run(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...
