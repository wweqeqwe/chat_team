"""Smoke tests for the P0 fixes batch.

Covers six independent items:

1. WebSocket reconnect — `_run_forever` reconnects on transient errors
   and exits cleanly on `close()`.
2. `disconnected_event` flags `_connection_dead` only (NOT `_shutdown`).
3. `run_command` subprocess does not inherit secrets-bearing env vars.
4. `SessionManager` LRU evicts past the cap, flushing the victim first.
5. Janitor unlinks old files in inbox/runs/llm subdirs on first touch.
6. LLM retry: transient errors and empty completions are retried; tool calls
   with empty text remain valid; non-transient errors are raised.

All tests are pure-Python: no live WS, no live LLM, no network.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_p0_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.adapters.wecom import WeComBotAdapter
from chat_team.agent.tools.base import ToolContext
from chat_team.agent.tools.shell_tool import RunCommandTool, _scrub_env
from chat_team.config import load_settings
from chat_team.llm.base import CompletionRequest, ChatMessage
from chat_team.llm.openai_provider import OpenAIChatCompletionProvider, _strip_thinking_markup
from chat_team.session.manager import SessionManager
from chat_team.session.persistence import PersistenceManager


# --------------------------------------------------------------------------
# P0-1: reconnect loop
# --------------------------------------------------------------------------

async def test_run_forever_reconnects_after_transient_failure():
    print("== test 1: run_forever reconnects after one failure ==")
    settings = load_settings()
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="S")

    calls = {"open": 0, "serve": 0, "shutdown_signaled": False}

    async def fake_open():
        calls["open"] += 1
        if calls["open"] == 1:
            raise ConnectionRefusedError("simulated transient failure")
        # Second open: success, adapter is "running".
        adapter._connection_dead = asyncio.Event()

    async def fake_serve():
        calls["serve"] += 1
        # First successful serve: shut down immediately so loop exits.
        adapter._shutdown.set()
        calls["shutdown_signaled"] = True
        adapter._connection_dead.set()
        await adapter._connection_dead.wait()

    async def fake_tear_down():
        pass

    adapter._open_connection = fake_open                 # type: ignore[assignment]
    adapter._serve_one_connection = fake_serve           # type: ignore[assignment]
    adapter._tear_down_connection = fake_tear_down       # type: ignore[assignment]

    # Patch sleep so the test doesn't actually wait the backoff.
    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *args, **kwargs):
        # Minimal cooperative yield; ignore the actual delay value.
        await real_sleep(0)

    asyncio.sleep = fast_sleep                          # type: ignore[assignment]
    try:
        await asyncio.wait_for(adapter.run_forever(), timeout=2.0)
    finally:
        asyncio.sleep = real_sleep                      # type: ignore[assignment]

    assert calls["open"] == 2, f"expected 2 connect attempts, got {calls['open']}"
    assert calls["serve"] == 1, f"expected 1 successful serve, got {calls['serve']}"
    assert calls["shutdown_signaled"]
    print("  ✓ open called twice, served once, exited via shutdown")


async def test_close_stops_run_forever():
    print("== test 2: close() prevents further reconnect attempts ==")
    settings = load_settings()
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="S")

    open_calls = {"n": 0}

    async def fake_open():
        open_calls["n"] += 1
        adapter._connection_dead = asyncio.Event()

    async def fake_serve():
        # Simulate connection dying without setting shutdown.
        adapter._connection_dead.set()

    async def fake_tear_down():
        pass

    adapter._open_connection = fake_open                 # type: ignore[assignment]
    adapter._serve_one_connection = fake_serve           # type: ignore[assignment]
    adapter._tear_down_connection = fake_tear_down       # type: ignore[assignment]

    # Schedule a close() shortly after run_forever starts.
    async def call_close_soon():
        await asyncio.sleep(0)
        await asyncio.sleep(0)        # let one open+serve cycle finish first
        await adapter.close()

    asyncio.create_task(call_close_soon())

    real_sleep = asyncio.sleep

    async def fast_sleep(delay, *a, **k):
        await real_sleep(0)

    asyncio.sleep = fast_sleep                          # type: ignore[assignment]
    try:
        await asyncio.wait_for(adapter.run_forever(), timeout=2.0)
    finally:
        asyncio.sleep = real_sleep                      # type: ignore[assignment]
    assert adapter._shutdown.is_set()
    # Some small number of cycles is fine, but it must terminate, not loop.
    assert open_calls["n"] < 10, f"loop didn't exit: {open_calls}"
    print(f"  ✓ run_forever exited cleanly after close() (cycles={open_calls['n']})")


# --------------------------------------------------------------------------
# P0-2: env scrub
# --------------------------------------------------------------------------

async def test_env_scrub_unit():
    print("== test 3: _scrub_env drops secrets, keeps benign vars ==")
    raw = {
        "PATH": "/usr/bin",
        "HOME": "/Users/x",
        "LANG": "en_US.UTF-8",
        "OPENAI_API_KEY": "sk-leak-me",
        "OPENAI_BASE_URL": "https://oai.example",
        "WECOM_SECRET": "do-not-leak",
        "WECOM_BOT_ID": "also-private",
        "ANTHROPIC_API_KEY": "claude-key",
        "CHAT_TEAM_HOME": "/tmp/x",
        "MY_DB_PASSWORD": "p@ss",
        "GH_TOKEN": "gh_xxx",
        "SOME_SECRET_value": "v",
        "USER_CREDENTIAL_FILE": "/etc/cred",
        "BENIGN_VAR": "ok",
    }
    scrubbed = _scrub_env(raw, extra_drop=[])
    assert "PATH" in scrubbed and "HOME" in scrubbed and "LANG" in scrubbed
    assert "BENIGN_VAR" in scrubbed
    for leaked in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "WECOM_SECRET", "WECOM_BOT_ID",
        "ANTHROPIC_API_KEY", "CHAT_TEAM_HOME", "MY_DB_PASSWORD", "GH_TOKEN",
        "SOME_SECRET_value", "USER_CREDENTIAL_FILE",
    ):
        assert leaked not in scrubbed, f"leak: {leaked}"
    # extra_drop pulls a benign var too
    scrubbed2 = _scrub_env(raw, extra_drop=["BENIGN_VAR"])
    assert "BENIGN_VAR" not in scrubbed2 and "PATH" in scrubbed2
    print("  ✓ _scrub_env enforces deny-list + extra_drop")


async def test_run_command_env_scrubbed():
    print("== test 4: run_command subprocess does not see secrets ==")
    settings = load_settings()
    sessions = SessionManager(settings)
    sess = await sessions.get_or_create("p0-env-scrub")
    ctx = ToolContext(cwd=sess.cwd, session=sess, settings=settings)

    # Seed env vars that MUST NOT reach the child.
    os.environ["OPENAI_API_KEY"] = "SECRET_LEAK_OPENAI"
    os.environ["WECOM_SECRET"] = "SECRET_LEAK_WECOM"
    os.environ["MY_DB_TOKEN"] = "SECRET_LEAK_TOKEN"
    os.environ["BENIGN_FOR_TEST"] = "BENIGN_OK"
    try:
        out = await RunCommandTool().run(
            ctx,
            command=(
                "printf '%s|%s|%s|%s|%s\\n' "
                "\"${OPENAI_API_KEY:-MISSING}\" "
                "\"${WECOM_SECRET:-MISSING}\" "
                "\"${MY_DB_TOKEN:-MISSING}\" "
                "\"${BENIGN_FOR_TEST:-MISSING}\" "
                "\"${PATH:+HAVE_PATH}\""
            ),
        )
    finally:
        for k in ("OPENAI_API_KEY", "WECOM_SECRET", "MY_DB_TOKEN", "BENIGN_FOR_TEST"):
            os.environ.pop(k, None)
    print("  output:", out.splitlines()[-1])
    # The body is the last line of the tool output (header is line 1, --- on line 2).
    body = out.splitlines()[-1]
    assert "SECRET_LEAK_OPENAI" not in body, body
    assert "SECRET_LEAK_WECOM" not in body, body
    assert "SECRET_LEAK_TOKEN" not in body, body
    assert "BENIGN_OK" in body, body          # benign var passes through
    assert "HAVE_PATH" in body, body          # PATH still set
    assert body.count("MISSING") == 3
    print("  ✓ secrets blocked, PATH + benign vars passed through")


# --------------------------------------------------------------------------
# P0-3: SessionManager LRU
# --------------------------------------------------------------------------

async def test_session_manager_lru_evicts_and_flushes():
    print("== test 5: SessionManager LRU evicts oldest + flushes to disk ==")
    settings = load_settings()
    settings.session.max_in_memory_sessions = 3
    persistence = PersistenceManager(settings)
    mgr = SessionManager(settings, persistence=persistence)

    # Seed 3 sessions; mutate "a"'s state so we can verify flush on eviction.
    s_a = await mgr.get_or_create("user-a")
    s_a.current_role = "alpha_role"
    await mgr.get_or_create("user-b")
    await mgr.get_or_create("user-c")
    assert len(mgr.known_sessions()) == 3

    # Adding a 4th should evict user-a (LRU).
    await mgr.get_or_create("user-d")
    assert "user-a" not in mgr.known_sessions(), mgr.known_sessions()
    assert len(mgr.known_sessions()) == 3

    # session.json for user-a must reflect alpha_role after eviction-time flush.
    cwd_a = mgr.workspace_for("user-a")
    state_path = cwd_a / ".chat_team" / "session.json"
    assert state_path.exists(), "evicted session was not flushed"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["current_role"] == "alpha_role", persisted

    # Re-touch user-a → reloaded from disk with current_role intact.
    s_a2 = await mgr.get_or_create("user-a")
    assert s_a2.current_role == "alpha_role"
    assert s_a2 is not s_a              # genuinely a fresh object
    print("  ✓ LRU evicts oldest, flushes session.json, transparent reload")


async def test_session_manager_recency_updates_on_hit():
    print("== test 6: get_or_create on existing session updates recency ==")
    settings = load_settings()
    settings.session.max_in_memory_sessions = 2
    mgr = SessionManager(settings)

    await mgr.get_or_create("recent-a")
    await mgr.get_or_create("recent-b")
    await mgr.get_or_create("recent-a")          # bump "recent-a" to MRU
    await mgr.get_or_create("recent-c")          # should evict "recent-b"
    assert "recent-a" in mgr.known_sessions()
    assert "recent-c" in mgr.known_sessions()
    assert "recent-b" not in mgr.known_sessions()
    print("  ✓ MRU touch protects recent-a; oldest (recent-b) evicted")


# --------------------------------------------------------------------------
# P0-4: janitor
# --------------------------------------------------------------------------

async def test_janitor_unlinks_old_files():
    print("== test 7: janitor sweeps stale files on first session touch ==")
    settings = load_settings()
    settings.cleanup.max_age_days = 7
    settings.cleanup.sweep_interval_hours = 0.0    # always sweep on touch
    mgr = SessionManager(settings)

    # Pre-create the workspace + dirs so we can plant aged files BEFORE
    # the session is materialised.
    sid = "p0-janitor-test"
    cwd = mgr.workspace_for(sid)
    (cwd / "inbox").mkdir(parents=True, exist_ok=True)
    (cwd / ".chat_team" / "runs").mkdir(parents=True, exist_ok=True)
    (cwd / ".chat_team" / "llm").mkdir(parents=True, exist_ok=True)

    now = time.time()
    old_ts = now - 30 * 86400        # 30 days old → should be deleted
    fresh_ts = now - 60              # 1 min old → must survive

    old_files = [
        cwd / "inbox" / "old.jpg",
        cwd / ".chat_team" / "runs" / "old.log",
        cwd / ".chat_team" / "llm" / "old.json",
    ]
    fresh_files = [
        cwd / "inbox" / "fresh.jpg",
        cwd / ".chat_team" / "runs" / "fresh.log",
        cwd / ".chat_team" / "llm" / "fresh.json",
    ]
    for p in old_files + fresh_files:
        p.write_bytes(b"x")
    for p in old_files:
        os.utime(p, (old_ts, old_ts))
    for p in fresh_files:
        os.utime(p, (fresh_ts, fresh_ts))

    await mgr.get_or_create(sid)                 # triggers sweep

    for p in old_files:
        assert not p.exists(), f"stale file not unlinked: {p}"
    for p in fresh_files:
        assert p.exists(), f"fresh file was deleted: {p}"
    print("  ✓ 3 stale files removed; 3 fresh files preserved")


async def test_janitor_throttled_by_interval():
    print("== test 8: janitor honors sweep_interval_hours (no double sweep) ==")
    settings = load_settings()
    settings.cleanup.max_age_days = 7
    settings.cleanup.sweep_interval_hours = 24.0  # never re-sweep within 24h
    mgr = SessionManager(settings)
    sid = "p0-janitor-throttle"
    cwd = mgr.workspace_for(sid)
    (cwd / "inbox").mkdir(parents=True, exist_ok=True)
    await mgr.get_or_create(sid)                  # first touch → sweep baseline

    # Plant a stale file AFTER the first sweep. A second touch within the
    # throttle window should NOT delete it.
    stale = cwd / "inbox" / "stale_after_baseline.jpg"
    stale.write_bytes(b"x")
    os.utime(stale, (time.time() - 30 * 86400,) * 2)
    await mgr.get_or_create(sid)
    assert stale.exists(), "throttled sweep ran when it shouldn't have"
    print("  ✓ sweep_interval throttle holds")


# --------------------------------------------------------------------------
# P0-6: LLM retry
# --------------------------------------------------------------------------

class _FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class _FakeMessage:
    def __init__(self, content="ok", tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = list(tool_calls or [])


class _FakeChoice:
    def __init__(self, content="retried-ok", tool_calls=None, finish_reason="stop"):
        self.message = _FakeMessage(content, tool_calls)
        self.finish_reason = finish_reason


class _FakeCompletion:
    def __init__(self, content="retried-ok", tool_calls=None, finish_reason="stop"):
        self.choices = [_FakeChoice(content, tool_calls, finish_reason)]
        self.usage = _FakeUsage()


class _FakeChatCompletionsCreator:
    """Mimic ``client.chat.completions.create`` with a scripted sequence
    of exceptions then a successful completion."""
    def __init__(self, errors_then_success: list[BaseException]):
        self._errors = list(errors_then_success)
        self.call_count = 0

    async def create(self, **kwargs):
        self.call_count += 1
        if self._errors:
            err = self._errors.pop(0)
            raise err
        return _FakeCompletion()


class _ScriptedCompletionCreator:
    """Return scripted completions in order."""

    def __init__(self, completions: list[_FakeCompletion]):
        self._completions = list(completions)
        self.call_count = 0

    async def create(self, **kwargs):
        self.call_count += 1
        if not self._completions:
            raise AssertionError("unexpected extra completion call")
        return self._completions.pop(0)


def _install_fake_client(provider, fake):
    """Hot-swap provider._client.chat.completions with our scripted fake."""
    class _Chat:
        completions = fake
    provider._client.chat = _Chat()


async def test_llm_retry_on_transient_error():
    print("== test 9: LLM retries on APIConnectionError ==")
    from openai import APIConnectionError

    provider = OpenAIChatCompletionProvider(
        api_key="test", max_retries=3, retry_initial_delay=0.0,
    )
    fake = _FakeChatCompletionsCreator(
        errors_then_success=[
            APIConnectionError(request=None),                  # type: ignore[arg-type]
            APIConnectionError(request=None),                  # type: ignore[arg-type]
        ],
    )
    _install_fake_client(provider, fake)

    req = CompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    resp = await provider.complete(req)
    assert resp.message.content == "retried-ok", resp.message
    assert fake.call_count == 3, f"expected 3 attempts, got {fake.call_count}"
    print(f"  ✓ retried twice, succeeded on attempt {fake.call_count}")


def _http_response(status: int):
    import httpx
    return httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "https://api.openai.example/v1/chat/completions"),
    )


async def test_llm_exhausts_retries():
    print("== test 10: LLM raises after exhausting retries ==")
    from openai import RateLimitError

    provider = OpenAIChatCompletionProvider(
        api_key="test", max_retries=2, retry_initial_delay=0.0,
    )
    # 2 attempts, both fail.
    fake = _FakeChatCompletionsCreator(
        errors_then_success=[
            RateLimitError("429", response=_http_response(429), body=None),
            RateLimitError("429", response=_http_response(429), body=None),
        ],
    )
    _install_fake_client(provider, fake)

    req = CompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    try:
        await provider.complete(req)
    except RateLimitError:
        pass
    else:
        raise AssertionError("RateLimitError should have been raised")
    assert fake.call_count == 2, fake.call_count
    print("  ✓ retries exhausted, original exception bubbled up")


async def test_llm_no_retry_on_4xx():
    print("== test 11: LLM does NOT retry on BadRequestError ==")
    from openai import BadRequestError

    provider = OpenAIChatCompletionProvider(
        api_key="test", max_retries=5, retry_initial_delay=0.0,
    )
    fake = _FakeChatCompletionsCreator(
        errors_then_success=[
            BadRequestError("400", response=_http_response(400), body=None),
        ],
    )
    _install_fake_client(provider, fake)

    req = CompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    try:
        await provider.complete(req)
    except BadRequestError:
        pass
    else:
        raise AssertionError("BadRequestError should have been raised")
    assert fake.call_count == 1, f"expected single attempt, got {fake.call_count}"
    print("  ✓ non-transient 4xx raised on first attempt, no retry")


async def test_llm_retries_empty_completion():
    print("== test 12: LLM retries empty no-tool completions ==")
    for use_streaming in (True, False):
        provider = OpenAIChatCompletionProvider(
            api_key="test",
            max_retries=3,
            retry_initial_delay=0.0,
            use_streaming=use_streaming,
        )
        fake = _ScriptedCompletionCreator([
            _FakeCompletion(content=""),
            _FakeCompletion(content=" \n\t"),
            _FakeCompletion(content="recovered"),
        ])
        _install_fake_client(provider, fake)

        req = CompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o-mini",
            temperature=0.0,
        )
        resp = await provider.complete(req)
        assert resp.message.content == "recovered", resp.message
        assert fake.call_count == 3, fake.call_count
    print("  ✓ streaming and non-streaming both retry until visible text")


async def test_llm_rejects_exhausted_empty_completions():
    print("== test 13: LLM raises after empty completions exhaust retries ==")
    provider = OpenAIChatCompletionProvider(
        api_key="test", max_retries=2, retry_initial_delay=0.0,
    )
    fake = _ScriptedCompletionCreator([
        _FakeCompletion(content=""),
        _FakeCompletion(content=" "),
    ])
    _install_fake_client(provider, fake)

    req = CompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    try:
        await provider.complete(req)
    except RuntimeError as exc:
        assert type(exc).__name__ == "_EmptyCompletionError", type(exc).__name__
    else:
        raise AssertionError("empty completions should not be returned as success")
    assert fake.call_count == 2, fake.call_count
    print("  ✓ exhausted empty responses raise instead of ending the turn")


async def test_llm_accepts_tool_call_without_text():
    print("== test 14: LLM accepts a tool call with empty text ==")

    class _Function:
        name = "lookup"
        arguments = '{"plate":"粤A12345"}'

    class _ToolCall:
        id = "call-1"
        function = _Function()

    provider = OpenAIChatCompletionProvider(
        api_key="test", max_retries=3, retry_initial_delay=0.0,
    )
    fake = _ScriptedCompletionCreator([
        _FakeCompletion(
            content="",
            tool_calls=[_ToolCall()],
            finish_reason="tool_calls",
        ),
    ])
    _install_fake_client(provider, fake)

    req = CompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    resp = await provider.complete(req)
    assert resp.message.content == ""
    assert len(resp.message.tool_calls) == 1
    assert resp.message.tool_calls[0].name == "lookup"
    assert fake.call_count == 1, fake.call_count
    print("  ✓ valid tool-only response succeeds without retry")

async def test_strip_thinking_markup_wrappers():
    print("== test 15: reasoning wrappers strip to visible answer, non-wrappers survive ==")
    short_open = "\x3c" + "think" + "\x3e"
    short_close = "\x3c" + "/think" + "\x3e"
    long_open = "\x3c" + "thinking" + "\x3e"
    long_close = "\x3c" + "/thinking" + "\x3e"
    assert _strip_thinking_markup(short_open + " 推理 " + short_close + "\n\n你好") == "你好"
    assert _strip_thinking_markup(long_open + " 推理 " + long_close + "\n你好") == "你好"
    assert _strip_thinking_markup(short_open + " 只有推理没有正文") == ""
    assert _strip_thinking_markup(long_open + " 只有推理没有正文") == ""
    assert _strip_thinking_markup("\x3c车牌\x3e 浙E·P671T") == "\x3c车牌\x3e 浙E·P671T"
    assert _strip_thinking_markup("普通回复") == "普通回复"
    assert _strip_thinking_markup("") == ""
    print("  ✓ short/long wrappers stripped; unrelated tags and plain text preserved")

# --------------------------------------------------------------------------

async def main():
    await test_run_forever_reconnects_after_transient_failure()
    await test_close_stops_run_forever()
    await test_env_scrub_unit()
    await test_run_command_env_scrubbed()
    await test_session_manager_lru_evicts_and_flushes()
    await test_session_manager_recency_updates_on_hit()
    await test_janitor_unlinks_old_files()
    await test_janitor_throttled_by_interval()
    await test_llm_retry_on_transient_error()
    await test_llm_exhausts_retries()
    await test_llm_no_retry_on_4xx()
    await test_llm_retries_empty_completion()
    await test_llm_rejects_exhausted_empty_completions()
    await test_llm_accepts_tool_call_without_text()
    await test_strip_thinking_markup_wrappers()
    print("\nALL P0 SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
