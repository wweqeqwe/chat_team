"""Tool ABC + ToolContext + ToolRegistry.

Tools are pure logic; they receive a ``ToolContext`` so they have access
to the session's working dir / notebook / settings without globals.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...llm.base import ToolSpec

if TYPE_CHECKING:                       # avoid circular imports at runtime
    from ...adapters.base import StreamHandle
    from ...config import Settings
    from ...llm.base import LLMProvider
    from ...session.session import Session


class ToolError(Exception):
    """Raised by a tool to signal a recoverable error returned to the LLM."""


class TransferRequested(Exception):
    """Raised by transfer_to_employee — caught by the dispatcher to switch role.

    Carries the requested target plus the structured handoff note.
    """

    def __init__(self, target: str, reason: str, handoff_note: str):
        super().__init__(f"transfer to {target}")
        self.target = target
        self.reason = reason
        self.handoff_note = handoff_note


@dataclass
class ToolContext:
    cwd: Path
    session: "Session"
    settings: "Settings"
    stream: "StreamHandle | None" = None
    # ``llm`` is None only in test contexts that mock the registry directly;
    # all production tool invocations carry the agent's provider so tools that
    # need vision/text completion (e.g. describe_image) can run.
    llm: "LLMProvider | None" = None
    # Optional separate provider for vision/OCR calls (describe_image).
    # When set, vision-aware tools (describe_image) use this instead of ``llm``.
    # None means "fall back to llm".
    vision_llm: "LLMProvider | None" = None


class Tool(abc.ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    # ``parallel_safe=True`` marks a tool as safe to invoke concurrently with
    # other parallel-safe tools within the same agent turn (when the LLM emits
    # multiple tool_calls in one assistant message). The agent dispatches such
    # batches via ``asyncio.gather`` instead of the default serial loop.
    #
    # A tool is parallel-safe when it is read-only, side-effect-free, and does
    # not mutate shared session state (cwd files, notebook, agent.history).
    # MCP proxy tools qualify by default: their state lives in the remote
    # server and the MCP client multiplexes requests by ID without a global
    # lock. Local tools that write files or run shell commands must stay
    # ``parallel_safe=False`` (the default) to avoid races.
    parallel_safe: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    @abc.abstractmethod
    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        """Return a string result fed back to the LLM as a tool message."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"tool missing name: {tool!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def specs_for(self, names: list[str]) -> list[ToolSpec]:
        return [self._tools[n].spec() for n in names if n in self._tools]


def stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)
