"""McpProxyTool — wraps a single MCP server tool as a local Tool subclass."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..agent.tools.base import Tool, ToolContext, ToolError
from .config import DEFAULT_TOOL_TIMEOUT_SECONDS

DEFAULT_TOOL_RESULT_MAX_BYTES = 8192

log = logging.getLogger(__name__)


def _resolve_workspace_image_path(cwd: Path, supplied: str) -> str:
    """Resolve common LLM workspace aliases to an existing sandboxed file."""
    direct = Path(supplied)
    if direct.is_absolute() and direct.is_file():
        return str(direct.resolve())

    relative: str | None = None
    for prefix in ("/workspace/", "/mnt/data/"):
        if supplied.startswith(prefix):
            relative = supplied.removeprefix(prefix)
            break
    else:
        if supplied.startswith("./"):
            relative = supplied.removeprefix("./")
        elif not direct.is_absolute():
            relative = supplied

    if not relative:
        return supplied

    workspace = cwd.resolve()
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return supplied

    return str(candidate) if candidate.is_file() else supplied


def _truncate_result(text: str, max_bytes: int) -> str:
    """Truncate ``text`` to at most ``max_bytes`` of UTF-8, appending a
    notice when truncation happened.

    The notice carries original/kept/dropped byte counts so the LLM can
    see *how much* data was elided and react (e.g. re-query with a
    ``filter_spec`` + smaller ``limit`` instead of dumping the whole
    sheet again). Mirrors ``shell_tool._truncate`` semantics: encode →
    slice on the byte boundary → decode with errors="ignore" so we never
    split a multi-byte UTF-8 char mid-codepoint.

    ``max_bytes <= 0`` disables truncation (returns ``text`` unchanged).
    """
    if max_bytes <= 0:
        return text
    encoded = text.encode("utf-8", errors="replace")
    total = len(encoded)
    if total <= max_bytes:
        return text
    kept = encoded[:max_bytes].decode("utf-8", errors="ignore")
    dropped = total - max_bytes
    notice = (
        f"\n[truncated: original {total} bytes, "
        f"kept {max_bytes} bytes, dropped {dropped} bytes]"
    )
    return kept + notice


class McpProxyTool(Tool):
    """Bridges an MCP tool into the local ToolRegistry.

    Registered as ``mcp__<server>__<tool>`` so role YAMLs (via
    ``mcp_servers:``) or explicit ``tools:`` entries can reference it.

    ``parallel_safe=True`` because MCP tool state lives in the remote server
    and the MCP client multiplexes concurrent ``call_tool`` requests by ID
    without a global lock. The agent can dispatch a batch of MCP tool_calls
    concurrently via ``asyncio.gather``.
    """
    parallel_safe: bool = True

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

    def _resolve_max_bytes(self, ctx: ToolContext) -> int:
        """Resolve the per-call result size cap, in bytes. 0 disables it.

        Reads ``settings.mcp.tool_result_max_bytes`` when available; falls
        back to DEFAULT_TOOL_RESULT_MAX_BYTES (8192) in test contexts
        where ``ctx.settings`` is None or the field is missing (older
        fakes that only set ``tool_timeout_seconds``).
        """
        settings = getattr(ctx, "settings", None)
        if settings is None:
            return DEFAULT_TOOL_RESULT_MAX_BYTES
        try:
            return int(settings.mcp.tool_result_max_bytes)
        except (AttributeError, TypeError, ValueError):
            return DEFAULT_TOOL_RESULT_MAX_BYTES

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        timeout = self._resolve_timeout(ctx)
        arguments = dict(kwargs)
        image = arguments.get("image")
        if isinstance(image, str):
            resolved_image = _resolve_workspace_image_path(ctx.cwd, image)
            if resolved_image != image:
                log.debug(
                    "resolved MCP image path for %s: %s -> %s",
                    self.name,
                    image,
                    resolved_image,
                )
                arguments["image"] = resolved_image

        try:
            if timeout > 0:
                result = await asyncio.wait_for(
                    self._session.call_tool(self._remote_name, arguments or None),
                    timeout=timeout,
                )
            else:
                result = await self._session.call_tool(self._remote_name, arguments or None)
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
        text = "\n".join(parts) or "(no output)"
        # Cap the size before it lands in agent.history — an MCP server
        # can return megabytes in one shot (e.g. a 1000-row smartsheet
        # dump), and once it's in history it inflates *every* subsequent
        # LLM request and can't be compacted when the session has <=1
        # user turn. See compactor._find_keep_boundary for the dead-end.
        return _truncate_result(text, self._resolve_max_bytes(ctx))
