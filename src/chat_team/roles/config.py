"""Role config dataclass + YAML loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RoleLLMConfig:
    model: str = ""                      # falls back to settings.llm.chat.model
    temperature: float | None = None     # falls back to settings.llm.chat.temperature
    history_token_budget: int | None = None
    image_detail: str | None = None      # "low" | "high" | "auto"; falls back to settings.llm.vision.image_detail
    # "tool" → inbound images become placeholder text blocks (no pre-OCR).
    # "direct" → pass image blocks straight to the provider (high-fidelity
    # multi-turn visual chat).
    # None → fall back to settings.llm.vision.strategy.
    vision_strategy: str | None = None


@dataclass
class RoleMcpToolFilter:
    """Per-role, per-server MCP tool allow/deny policy."""

    mode: str                          # "whitelist" or "blacklist"
    tools: list[str] = field(default_factory=list)


@dataclass
class Role:
    name: str
    display_name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    mcp_tools: dict[str, RoleMcpToolFilter] = field(default_factory=dict)
    llm: RoleLLMConfig = field(default_factory=RoleLLMConfig)
    welcome_message: str | None = None   # used for enter_chat events

    @classmethod
    def from_yaml(cls, path: Path) -> "Role":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"role yaml must be a mapping: {path}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Role":
        llm_raw = raw.get("llm") or {}
        llm = RoleLLMConfig(
            model=llm_raw.get("model", ""),
            temperature=llm_raw.get("temperature"),
            history_token_budget=llm_raw.get("history_token_budget"),
            image_detail=llm_raw.get("image_detail"),
            vision_strategy=llm_raw.get("vision_strategy"),
        )
        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("role yaml missing required 'name'")
        skills_raw = raw.get("skills") or []
        if not isinstance(skills_raw, list) or not all(isinstance(s, str) for s in skills_raw):
            raise ValueError("role yaml 'skills' must be a list of strings")
        mcp_servers_raw = raw.get("mcp_servers") or []
        if not isinstance(mcp_servers_raw, list) or not all(
            isinstance(s, str) for s in mcp_servers_raw
        ):
            raise ValueError("role yaml 'mcp_servers' must be a list of strings")
        mcp_tools_raw = raw.get("mcp_tools") or {}
        if not isinstance(mcp_tools_raw, dict):
            raise ValueError("role yaml 'mcp_tools' must be a mapping")
        mcp_tools: dict[str, RoleMcpToolFilter] = {}
        mcp_servers = list(mcp_servers_raw)
        for server_name, filter_raw in mcp_tools_raw.items():
            if server_name not in mcp_servers:
                raise ValueError(
                    "role yaml 'mcp_tools' references a server not listed in "
                    f"'mcp_servers': {server_name!r}"
                )
            if not isinstance(filter_raw, dict):
                raise ValueError(
                    f"role yaml 'mcp_tools.{server_name}' must be a mapping"
                )
            mode = filter_raw.get("mode")
            if mode not in ("whitelist", "blacklist"):
                raise ValueError(
                    f"role yaml 'mcp_tools.{server_name}.mode' must be "
                    "'whitelist' or 'blacklist'"
                )
            filter_tools_raw = filter_raw.get("tools") or []
            if not isinstance(filter_tools_raw, list) or not all(
                isinstance(tool_name, str) and tool_name
                for tool_name in filter_tools_raw
            ):
                raise ValueError(
                    f"role yaml 'mcp_tools.{server_name}.tools' must be a "
                    "list of non-empty tool-name strings"
                )
            if len(filter_tools_raw) != len(set(filter_tools_raw)):
                raise ValueError(
                    f"role yaml 'mcp_tools.{server_name}.tools' contains duplicates"
                )
            mcp_tools[server_name] = RoleMcpToolFilter(
                mode=mode,
                tools=list(filter_tools_raw),
            )
        return cls(
            name=name,
            display_name=raw.get("display_name", name),
            description=raw.get("description", ""),
            system_prompt=raw.get("system_prompt", "").strip(),
            tools=list(raw.get("tools") or []),
            skills=list(skills_raw),
            mcp_servers=list(mcp_servers_raw),
            mcp_tools=mcp_tools,
            llm=llm,
            welcome_message=raw.get("welcome_message"),
        )
