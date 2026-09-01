"""Offline smoke test for the private-chat reply policy.

Covers:
  * PrivateChatConfig.allows() for all 4 modes + invalid-mode fail-open
  * YAML parsing & validation in load_settings (invalid mode → 'open')
  * The actual gate in WeComBotAdapter._handle_msg_callback:
        - single chat blocked when policy says no
        - single chat allowed otherwise
        - group chat always passes through (independent of policy)
  * blocked_reply: empty → silent, non-empty → single text frame queued
  * enter_chat welcome is also gated for single chats

Run: ``python3 scripts/smoke_private_chat_policy.py`` — no network, no LLM.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_pc_smoke"
shutil.rmtree(os.environ["CHAT_TEAM_HOME"], ignore_errors=True)

from chat_team.adapters.base import ChatType, IncomingMessage  # noqa: E402
from chat_team.adapters.wecom import WeComBotAdapter  # noqa: E402
from chat_team.config import PrivateChatConfig, load_settings  # noqa: E402


# ------------------------- unit: PrivateChatConfig.allows -------------------

def test_allows_open():
    pc = PrivateChatConfig(mode="open")
    assert pc.allows("anyone") is True
    assert pc.allows("") is True                       # even unknown senders


def test_allows_closed():
    pc = PrivateChatConfig(mode="closed", whitelist=["vip"], blacklist=["vip"])
    # closed ignores the lists entirely
    assert pc.allows("vip") is False
    assert pc.allows("anyone") is False


def test_allows_blacklist():
    pc = PrivateChatConfig(mode="blacklist", blacklist=["eve", "spammer"])
    assert pc.allows("eve") is False
    assert pc.allows("spammer") is False
    assert pc.allows("alice") is True
    # empty blacklist = allow everyone
    assert PrivateChatConfig(mode="blacklist").allows("anyone") is True


def test_allows_whitelist():
    pc = PrivateChatConfig(mode="whitelist", whitelist=["alice", "bob"])
    assert pc.allows("alice") is True
    assert pc.allows("bob") is True
    assert pc.allows("eve") is False
    # empty whitelist = block everyone (foot-gun documented in template)
    assert PrivateChatConfig(mode="whitelist").allows("anyone") is False


def test_allows_invalid_mode_fails_closed():
    """An unrecognised mode must fall back to whitelist semantics (default-deny).

    A typo in config must NOT accidentally expose the bot to everyone. The
    maintainer just adds their own userid to ``whitelist`` once to recover.
    Note ``load_settings`` lowercases ``mode`` first, so 'WHITELIST' would
    still be recognised as the canonical 'whitelist'; this test pins the
    behaviour of ``PrivateChatConfig`` itself when handed something genuinely
    unrecognised (e.g. an abbreviation typo).
    """
    pc = PrivateChatConfig(mode="wl", whitelist=["x"])            # unrecognised
    assert pc.allows("x") is True                                 # whitelist still consulted
    assert pc.allows("anyone") is False
    pc = PrivateChatConfig(mode="garbage")
    assert pc.allows("anyone") is False                           # default-deny


def test_default_dataclass_is_whitelist_default_deny():
    """A bare ``PrivateChatConfig()`` is default-deny — matches the
    template's documented behaviour and what an upgrade with no
    ``private_chat`` block gets."""
    pc = PrivateChatConfig()
    assert pc.mode == "whitelist"
    assert pc.allows("anyone") is False
    assert pc.allows("") is False
    # default blocked_reply is non-empty so users can self-discover their userid
    assert "{userid}" in pc.blocked_reply, "default blocked_reply must contain {userid} placeholder"


# ------------------------- unit: YAML parsing ------------------------------

def test_yaml_parse_and_validate(tmp_home: Path):
    (tmp_home / "config.yaml").write_text(
        """
private_chat:
  mode: AllowList  # typo
  whitelist: [alice, ' ', bob]  # whitespace entries stripped
  blacklist: [eve, '']
  blocked_reply: 暂未开放
""",
        encoding="utf-8",
    )
    s = load_settings()
    # 'AllowList' is not a valid mode → coerces to default 'whitelist'
    assert s.private_chat.mode == "whitelist", s.private_chat.mode
    # whitespace-only entries are stripped
    assert s.private_chat.whitelist == ["alice", "bob"]
    assert s.private_chat.blacklist == ["eve"]
    assert s.private_chat.blocked_reply == "暂未开放"
    # whitelist mode is now in effect → only alice/bob pass
    assert s.private_chat.allows("alice")
    assert not s.private_chat.allows("eve")


def test_yaml_valid_whitelist_mode(tmp_home: Path):
    (tmp_home / "config.yaml").write_text(
        """
private_chat:
  mode: whitelist
  whitelist: [vip_user]
""",
        encoding="utf-8",
    )
    s = load_settings()
    assert s.private_chat.mode == "whitelist"
    assert s.private_chat.allows("vip_user")
    assert not s.private_chat.allows("outsider")


def test_yaml_omitted_defaults_whitelist(tmp_home: Path):
    # No private_chat block at all → default-deny (matches the template).
    # This is the documented upgrade-breaking behaviour: an upgrade that
    # doesn't opt in goes silent on private chats until the maintainer
    # adds mode: open or populates whitelist.
    s = load_settings()
    assert s.private_chat.mode == "whitelist"
    assert not s.private_chat.allows("anyone")


# ------------------------- integration: adapter gate -----------------------

def _single_msg_frame(msgid: str, userid: str, *, bot_id: str = "BOT") -> dict:
    return {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": f"rq-{msgid}"},
        "body": {
            "msgid": msgid,
            "aibotid": bot_id,
            "chattype": "single",
            "from": {"userid": userid},
            "msgtype": "text",
            "text": {"content": "hi"},
        },
    }


def _group_msg_frame(msgid: str, userid: str) -> dict:
    return {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": f"rq-{msgid}"},
        "body": {
            "msgid": msgid,
            "aibotid": "BOT",
            "chatid": "CHAT1",
            "chattype": "group",
            "from": {"userid": userid},
            "msgtype": "text",
            "text": {"content": "@bot hi"},
        },
    }


class _RecordingHandler:
    """Stand-in for Dispatcher.handle. Records every invocation."""

    def __init__(self) -> None:
        self.calls: list[IncomingMessage] = []

    async def __call__(self, msg: IncomingMessage, stream) -> None:
        self.calls.append(msg)
        # Match the production contract: handler must finish the stream.
        await stream.finish("ok")


def _drain(adapter: WeComBotAdapter) -> list[dict]:
    out = []
    while not adapter._write_queue.empty():
        out.append(adapter._write_queue.get_nowait())
    return out


async def test_gate_blocks_single_chat_blocked_user():
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(
        mode="blacklist", blacklist=["banned"],
        blocked_reply="您已被屏蔽",
    )
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")
    h = _RecordingHandler()
    adapter.set_handler(h)

    await adapter._handle_msg_callback(_single_msg_frame("m1", "banned"))

    assert h.calls == [], "blocked user must NOT reach the handler"
    frames = _drain(adapter)
    # Expecting exactly two frames: the finished stream close ("处理完成。")
    # followed by the blocked_reply as a markdown message — the standard
    # finish() shape WeCom honours for aibot_msg_callback replies.
    assert len(frames) == 2, frames
    f = frames[0]
    assert f["cmd"] == "aibot_respond_msg"
    assert f["body"]["msgtype"] == "stream"
    assert f["body"]["stream"]["finish"] is True
    assert f["headers"]["req_id"] == "rq-m1"
    md = frames[1]
    assert md["cmd"] == "aibot_respond_msg"
    assert md["body"]["msgtype"] == "markdown"
    assert md["body"]["markdown"]["content"] == "您已被屏蔽"
    assert md["headers"]["req_id"] == "rq-m1"


async def test_blocked_reply_userid_placeholder_substituted():
    """The {userid} placeholder is replaced with the sender's WeCom userid.

    This is the recommended self-service pattern: a user without their own
    userid (which the WeCom client never shows) sends one private chat,
    gets back the string they need to hand to the maintainer to be added
    to whitelist. No log access required.
    """
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(
        mode="whitelist", whitelist=["someone_else"],
        blocked_reply="私聊未开放。你的账号是 {userid},请联系管理员。",
    )
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")
    adapter.set_handler(_RecordingHandler())

    await adapter._handle_msg_callback(_single_msg_frame("m1", "alice"))

    frames = _drain(adapter)
    assert len(frames) == 2
    assert frames[0]["body"]["msgtype"] == "stream"
    assert frames[0]["body"]["stream"]["finish"] is True
    assert frames[1]["body"]["msgtype"] == "markdown"
    # {userid} substituted with the sender's userid ("alice"); assert on
    # the substituted fragment rather than the full sentence so a wording
    # tweak in the test fixture can't break this assertion.
    assert "alice" in frames[1]["body"]["markdown"]["content"], frames[1]


async def test_blocked_reply_unknown_placeholder_rendered_literally():
    """A typo like {foo} must NOT crash the reply path — render verbatim."""
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(
        mode="closed",
        blocked_reply="hi {user} (your id is {userid}) and {{escaped}}",
    )
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")
    adapter.set_handler(_RecordingHandler())

    # {{escaped}} is a literal-brace escape under str.format — that's fine,
    # we just need to confirm the call doesn't raise and still emits a frame.
    await adapter._handle_msg_callback(_single_msg_frame("m1", "bob"))

    frames = _drain(adapter)
    assert len(frames) == 2
    assert frames[1]["body"]["msgtype"] == "markdown"
    content = frames[1]["body"]["markdown"]["content"]
    assert "your id is bob" in content
    assert "hi {user}" in content          # unknown placeholder preserved literally


async def test_gate_blocks_single_chat_silent_when_blocked_reply_empty():
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(mode="closed", blocked_reply="")  # explicit silent
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")
    h = _RecordingHandler()
    adapter.set_handler(h)

    await adapter._handle_msg_callback(_single_msg_frame("m1", "anyone"))

    assert h.calls == []
    assert _drain(adapter) == [], "empty blocked_reply → silent drop, no frames"


async def test_gate_passes_single_chat_allowed_user():
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(mode="whitelist", whitelist=["vip"])
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")
    h = _RecordingHandler()
    adapter.set_handler(h)

    await adapter._handle_msg_callback(_single_msg_frame("m1", "vip"))

    assert len(h.calls) == 1
    assert h.calls[0].user_id == "vip"


async def test_gate_group_chat_always_passes():
    """Group chats must NOT be filtered by private_chat policy at all."""
    settings = load_settings()
    # Strictest possible policy — would block everyone in single chat:
    settings.private_chat = PrivateChatConfig(mode="whitelist", whitelist=["no_one"])
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")
    h = _RecordingHandler()
    adapter.set_handler(h)

    await adapter._handle_msg_callback(_group_msg_frame("m1", "outsider"))

    assert len(h.calls) == 1, "group chat must bypass private_chat policy"
    assert h.calls[0].chat_type == ChatType.GROUP


async def test_gate_open_mode_passes_everyone():
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(mode="open", blacklist=["x"], whitelist=["y"])
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")
    h = _RecordingHandler()
    adapter.set_handler(h)

    await adapter._handle_msg_callback(_single_msg_frame("m1", "totally_random"))
    assert len(h.calls) == 1


async def test_gate_skipped_before_thinking_frame():
    """Critical: a blocked user must NOT see the 思考中… spinner.

    The 思考中 frame is a ``finish=false`` intermediate stream frame that
    WeComStreamHandle sends via ``_send_frame("思考中…")`` — but only AFTER
    the handler is invoked, which never happens for a blocked chat. So any
    stream frame a blocked chat produces must be the single ``finish=true``
    blocked_reply delivery, never the spinner. (When blocked_reply is empty
    there are no frames at all — covered by the silent-drop test above.)
    """
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(mode="closed")  # default reply non-empty
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")
    adapter.set_handler(_RecordingHandler())

    await adapter._handle_msg_callback(_single_msg_frame("m1", "anyone"))

    for f in _drain(adapter):
        if f.get("body", {}).get("msgtype") == "stream":
            # The only allowed stream frame is the final blocked_reply.
            assert f["body"]["stream"]["finish"] is True, \
                "blocked chat must not enqueue a 思考中 (finish=false) frame"
            assert "思考中" not in f["body"]["stream"]["content"], \
                "blocked chat must not show the spinner content"


# ------------------------- integration: enter_chat welcome gate ------------

def _enter_chat_event_frame(req_id: str, chattype: str, userid: str) -> dict:
    return {
        "cmd": "aibot_event_callback",
        "headers": {"req_id": req_id},
        "body": {
            "msgid": "",
            "aibotid": "BOT",
            "chattype": chattype,
            "from": {"userid": userid},
            "event": {"eventtype": "enter_chat"},
        },
    }


async def test_enter_chat_welcome_blocked_for_single_chat_blocked_user():
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(mode="closed")
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")

    await adapter._handle_event_callback(_enter_chat_event_frame("r1", "single", "u"))

    # No welcome frame should be queued.
    for f in _drain(adapter):
        assert f.get("cmd") != "aibot_respond_welcome_msg"


async def test_enter_chat_welcome_allowed_for_group():
    settings = load_settings()
    settings.private_chat = PrivateChatConfig(mode="closed")  # would block single
    adapter = WeComBotAdapter(settings, bot_id="BOT", secret="s")

    await adapter._handle_event_callback(_enter_chat_event_frame("r1", "group", "u"))

    frames = _drain(adapter)
    assert any(f.get("cmd") == "aibot_respond_welcome_msg" for f in frames), frames


# ------------------------- runner -------------------------------------------

def _set_home(home: Path) -> None:
    os.environ["CHAT_TEAM_HOME"] = str(home)
    # load_settings caches nothing, but be explicit so each test sees its own.
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    import tempfile

    # --- pure allows() tests ---
    test_allows_open()
    test_allows_closed()
    test_allows_blacklist()
    test_allows_whitelist()
    test_allows_invalid_mode_fails_closed()
    test_default_dataclass_is_whitelist_default_deny()

    # --- YAML parsing tests (each gets its own CHAT_TEAM_HOME) ---
    tmp = Path(tempfile.mkdtemp(prefix="pc_smoke_"))
    try:
        _set_home(tmp); test_yaml_parse_and_validate(tmp)
        _set_home(tmp); test_yaml_valid_whitelist_mode(tmp)
        _set_home(tmp); test_yaml_omitted_defaults_whitelist(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- adapter gate tests (use the global CHAT_TEAM_HOME seeded above) ---
    # Re-seed to a clean home so load_settings() works.
    _set_home(Path("/tmp/chat_team_pc_smoke"))

    await test_gate_blocks_single_chat_blocked_user()
    await test_blocked_reply_userid_placeholder_substituted()
    await test_blocked_reply_unknown_placeholder_rendered_literally()
    await test_gate_blocks_single_chat_silent_when_blocked_reply_empty()
    await test_gate_passes_single_chat_allowed_user()
    await test_gate_group_chat_always_passes()
    await test_gate_open_mode_passes_everyone()
    await test_gate_skipped_before_thinking_frame()
    await test_enter_chat_welcome_blocked_for_single_chat_blocked_user()
    await test_enter_chat_welcome_allowed_for_group()

    print("ALL PRIVATE-CHAT POLICY TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
