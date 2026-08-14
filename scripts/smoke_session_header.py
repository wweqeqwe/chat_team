"""Smoke test for the stable per-conversation ``x-session-id`` header."""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

home = Path("/tmp/chat_team_smoke_session_header")
shutil.rmtree(home, ignore_errors=True)
os.environ["CHAT_TEAM_HOME"] = str(home)
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from chat_team.config import load_settings
from chat_team.llm.base import ChatMessage, CompletionRequest
from chat_team.llm.openai_provider import OpenAIChatCompletionProvider
from chat_team.session.manager import SessionManager
from chat_team.session.persistence import PersistenceManager


class _FakeMessage:
    content = "ok"
    tool_calls = None


class _FakeChoice:
    message = _FakeMessage()
    finish_reason = "stop"


class _FakeCompletion:
    choices = [_FakeChoice()]
    usage = None


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeCompletion()


async def test_header_is_sent() -> None:
    provider = OpenAIChatCompletionProvider(
        api_key="test-key",
        use_streaming=False,
    )
    fake = _FakeCompletions()
    provider._clients[0].chat.completions = fake
    provider._client = provider._clients[0]
    await provider.complete(CompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
        model="gpt-4o-mini",
        session_id="wecom-group-chat-1",
        session_uuid="019c0000-0000-7000-8000-000000000001",
    ))
    assert fake.kwargs["extra_headers"] == {
        "x-session-id": "019c0000-0000-7000-8000-000000000001",
    }, fake.kwargs
    print("  ✓ x-session-id is added to the upstream model request")


async def test_uuid_survives_persistence() -> None:
    settings = load_settings()
    persistence = PersistenceManager(settings)
    first_manager = SessionManager(settings, persistence=persistence)
    first = await first_manager.get_or_create("stable-session")
    persistence.flush_now(first)

    second_manager = SessionManager(settings, persistence=PersistenceManager(settings))
    second = await second_manager.get_or_create("stable-session")
    assert first.session_uuid == second.session_uuid
    print("  ✓ session UUID survives session.json reload")


async def main() -> None:
    await test_header_is_sent()
    await test_uuid_survives_persistence()
    print("\nALL SESSION-HEADER SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
