"""MCP server configuration dataclass."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Default per-call timeout for MCP tool invocations. MCP servers that hang
# (e.g. an upstream vision API returning 504 after 60-90s) would otherwise
# block the agent's tool loop indefinitely, freezing the WeCom stream at
# "正在处理,请稍候...". This ceiling turns a hung MCP call into a ToolError
# the LLM can recover from. Override in config.yaml via
# ``mcp.tool_timeout_seconds``. Set to 0 to disable the timeout entirely.
DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0


@dataclass
class McpServerConfig:
    """One MCP server entry from config.yaml ``mcp.servers``."""

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""

    def validate(self) -> None:
        if not self.name or not _NAME_RE.match(self.name):
            raise ValueError(
                f"mcp server name must match [a-zA-Z0-9_-]+: {self.name!r}"
            )
        if "__" in self.name:
            raise ValueError(
                f"mcp server name must not contain '__': {self.name!r}"
            )
        has_command = bool(self.command)
        has_url = bool(self.url)
        if has_command == has_url:
            raise ValueError(
                f"mcp server {self.name!r}: set exactly one of 'command' (stdio) or 'url' (sse)"
            )
