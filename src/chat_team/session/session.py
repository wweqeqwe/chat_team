"""Session: per-conversation state.

Holds the working directory, the current employee in charge, the per-role
conversation histories, the shared notebook, an asyncio lock for serialised
turn processing, and a pending handoff note (consumed on the next turn).
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .notebook import Notebook

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..llm.base import ChatMessage


@dataclass
class PendingHandoff:
    from_role: str
    to_role: str
    reason: str
    note: str


@dataclass
class Session:
    session_id: str
    cwd: Path
    current_role: str
    notebook: Notebook
    # Stable opaque identifier for this conversation. ``session_id`` is
    # derived from the WeCom chat/user identity and is also used as the
    # workspace key; this UUID is safe to send to upstream LLM gateways in
    # the ``x-session-id`` request header and survives process restarts.
    session_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    agents_by_role: dict[str, "Agent"] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_handoff: PendingHandoff | None = None
    transfer_count_this_turn: int = 0
    # Histories loaded from disk on startup; consumed by Dispatcher when an
    # Agent for that role is first instantiated this session.
    restored_histories: dict[str, list["ChatMessage"]] = field(default_factory=dict)
    state_filename: str = "session.json"

    def reset_turn_counters(self) -> None:
        self.transfer_count_this_turn = 0

    def metadata_dir(self) -> Path:
        return self.cwd / ".chat_team"
