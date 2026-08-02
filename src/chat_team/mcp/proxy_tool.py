"""McpProxyTool — wraps a single MCP server tool as a local Tool subclass."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..agent.tools.base import Tool, ToolContext, ToolError
from .config import DEFAULT_TOOL_TIMEOUT_SECONDS

log = logging.getLogger(__name__)


class McpProxyTool(Tool):
    """Bridges an MCP tool into the local ToolRegistry.

    Registered as ``mcp__<server>__<tool>`` so role YAMLs (via
    ``mcp_servers:``) or explicit ``tools:`` entries can reference it.
    """

    def __init__(self, server_name: str, mcp_tool: Any, session: Any) -> None:
        self.name = f"mcp__{server_name}__{mcp_tool.name}"
        self.description = mcp_tool.description or ""
        self.parameters = mcp_tool.inputSchema or {"type": "object", "properties": {}}
        self.server_name = server_name
        self._session = session
        self._remote_name: str = mcp_tool.name

    def _resolve_timeout(self, ctx: ToolContext) -> float:
        """Resolve the per-call timeout, in seconds. 0 means no timeout.

        Reads ``settings.mcp.tool_timeout_seconds`` when available; falls back
        to DEFAULT_TOOL_TIMEOUT_SECONDS (60s) in test contexts where
        ``ctx.settings`` is None.
        """
        settings = getattr(ctx, "settings", None)
        if settings is None:
            return DEFAULT_TOOL_TIMEOUT_SECONDS
        try:
            return float(settings.mcp.tool_timeout_seconds)
        except (AttributeError, TypeError, ValueError):
            return DEFAULT_TOOL_TIMEOUT_SECONDS

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        timeout = self._resolve_timeout(ctx)
        try:
            if timeout > 0:
                result = await asyncio.wait_for(
                    self._session.call_tool(self._remote_name, kwargs or None),
                    timeout=timeout,
                )
            else:
                result = await self._session.call_tool(self._remote_name, kwargs or None)
        except asyncio.TimeoutError as exc:
            # A hung MCP server (e.g. upstream API stuck returning 504 after
            # 60-90s) would otherwise freeze the agent's tool loop and the
            # WeCom stream. Surface it as a recoverable ToolError so the LLM
            # can switch to a fallback tool instead of retrying forever.
            raise ToolError(
                f"MCP tool {self.name} timed out after {timeout:.1f}s"
            ) from exc
        except Exception as exc:
            raise ToolError(f"MCP tool {self.name} failed: {exc}") from exc

        if result.isError:
            parts = [c.text for c in result.content if hasattr(c, "text")]
            raise ToolError("\n".join(parts) or "MCP tool returned an error")

        parts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif hasattr(item, "data") and hasattr(item, "mimeType"):
                parts.append(f"[image: {item.mimeType}]")
            else:
                parts.append(str(item))
        return "\n".join(parts) or "(no output)"
