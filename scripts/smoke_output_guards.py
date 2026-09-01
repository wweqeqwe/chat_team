"""Smoke tests for the runaway-output guards.

Covers the two protections against a model "looping" on its own output:

1. **Degenerate-repetition detector** (``llm/repetition.py``) — catches a
   tail that is N verbatim copies of one block, at any block size or
   alignment, without tripping on legitimate tables/lists.
2. **Provider integration** — streaming and non-streaming calls abort as
   soon as repetition is detected and are retried via the normal retryable
   path (a fresh sample breaks the loop); tool-call argument streams are
   watched too.
3. **max_tokens backstop** — ``llm.chat.max_tokens`` (default 16384) is
   parsed from config and forwarded as the ``max_tokens`` kwarg; 0 means
   "no cap" (kwarg omitted).

All pure-Python: no live LLM, no network.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

home = Path("/tmp/chat_team_smoke_output_guards")
shutil.rmtree(home, ignore_errors=True)
home.mkdir(parents=True, exist_ok=True)
os.environ["CHAT_TEAM_HOME"] = str(home)

from chat_team.config import load_settings
from chat_team.llm.base import ChatMessage, CompletionRequest
from chat_team.llm.openai_provider import (
    OpenAIChatCompletionProvider,
    _DegenerateOutputError,
    _RETRYABLE_EXCEPTIONS,
)
from chat_team.llm.repetition import find_degenerate_repeat

LOOP_PHRASE = "好的，我来处理。"          # 8-char repeating unit
NORMAL_REPLY = "你好，已为您查询完毕，车牌浙E293F1。"


# --------------------------------------------------------------------------- #
# Detector unit tests
# --------------------------------------------------------------------------- #

def test_detector_hits() -> None:
    assert find_degenerate_repeat("前缀。" + LOOP_PHRASE * 20) is not None
    assert find_degenerate_repeat(LOOP_PHRASE * 40) is not None
    # 23-char unit, alignment offset by a leading char
    unit23 = "这是一段二十一个字符的重复块内容哦，测试用的。"
    assert find_degenerate_repeat("x" + unit23 * 10) is not None
    # ~50-char paragraph unit
    unit50 = "长段落复读测试内容，这里需要多写一点文字才行啊，继续凑字数中……"
    assert find_degenerate_repeat("开头" + unit50 * 6) is not None
    assert find_degenerate_repeat("ab" * 500) is not None
    print("ok  detector catches loops of various block sizes/alignments")


def test_detector_no_false_positives() -> None:
    assert find_degenerate_repeat("") is None
    assert find_degenerate_repeat("短短短") is None
    assert find_degenerate_repeat(NORMAL_REPLY) is None
    # markdown table with DISTINCT rows
    table = "| 项目 | 数量 |\n|---|---|\n" + "".join(
        f"| 项目{i} | {i} |\n" for i in range(30)
    )
    assert find_degenerate_repeat(table) is None
    # a few identical short lines (list) must not trip
    assert find_degenerate_repeat("5+5环保洗车\n" * 4) is None
    assert find_degenerate_repeat("| 更换机油 | 1 |\n" * 5) is None
    # repeated sentence structure with varying content
    assert find_degenerate_repeat(
        "已确认：车牌浙E293F1。已确认：客户白女士。已确认：手机号无误。"
        "其余字段识别正常，准备进入下一步查询F6档案，然后按流程修改企微备注。"
    ) is None
    print("ok  detector stays quiet on tables/lists/normal prose")


def test_detector_is_retryable() -> None:
    assert _DegenerateOutputError in _RETRYABLE_EXCEPTIONS
    print("ok  _DegenerateOutputError is in _RETRYABLE_EXCEPTIONS")


# --------------------------------------------------------------------------- #
# Fake OpenAI client (streaming + non-streaming)
# --------------------------------------------------------------------------- #

class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta=None, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


class _TCFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCallDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _TCFunction(name=name, arguments=arguments)


class _FakeStream:
    """Async iterator of chunks that also tracks aclose()."""

    def __init__(self, chunks: list):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):
        self.closed = True


def content_stream(parts: list[str]) -> _FakeStream:
    chunks = [_Chunk([_Choice(_Delta(content=p))]) for p in parts]
    chunks.append(_Chunk([_Choice(_Delta(), finish_reason="stop")]))
    return _FakeStream(chunks)


class _FakeMessage:
    def __init__(self, text):
        self.content = text
        self.tool_calls = None


class _FakeCompletionChoice:
    def __init__(self, text):
        self.message = _FakeMessage(text)
        self.finish_reason = "stop"


class _FakeCompletion:
    def __init__(self, text):
        self.choices = [_FakeCompletionChoice(text)]
        self.usage = None


class _ScriptedCreate:
    """Script entries: _FakeStream | _FakeCompletion | str | BaseException."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls = 0
        self.last_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, str):
                return _FakeCompletion(item)
            return item
        return _FakeCompletion("default")


def _provider(fake: _ScriptedCreate, *, streaming: bool) -> OpenAIChatCompletionProvider:
    p = OpenAIChatCompletionProvider(
        api_key="unused", max_retries=3, retry_initial_delay=0.0,
        use_streaming=streaming,
    )

    class _Chat:
        pass

    ch = _Chat()
    ch.completions = fake
    p._clients[0].chat = ch
    p._client = p._clients[0]
    return p


def _req() -> CompletionRequest:
    return CompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="xmz-text-model", temperature=0.0,
    )


# --------------------------------------------------------------------------- #
# Provider integration tests
# --------------------------------------------------------------------------- #

async def test_streaming_degenerate_aborts_and_retries() -> None:
    print("== streaming: degenerate loop aborts early, retry succeeds ==")
    bad = content_stream(["正常开头。"] + [LOOP_PHRASE] * 80)   # ~645 chars, loops
    good = content_stream([NORMAL_REPLY])
    fake = _ScriptedCreate([bad, good])
    p = _provider(fake, streaming=True)
    resp = await p.complete(_req())
    assert resp.message.content == NORMAL_REPLY, resp.message.content
    assert fake.calls == 2, fake.calls
    assert bad.closed is True, "degenerate stream was not closed on abort"
    assert len(resp.message.content) > 0
    print(f"  ✓ aborted degenerate stream (closed={bad.closed}), retry returned clean reply")


async def test_streaming_tool_args_degenerate() -> None:
    print("== streaming: degenerate tool-call arguments abort and retry ==")
    pad = '"pad": "xxxxxxxxxxxxxxxx", '
    arg_chunks = ['{"mobile": '] + [pad] * 40 + ['"exact": true}']
    chunks = [_Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, id="c1", name="f")]))])]
    chunks += [
        _Chunk([_Choice(_Delta(tool_calls=[_ToolCallDelta(0, arguments=a)]))])
        for a in arg_chunks
    ]
    bad = _FakeStream(chunks)
    good = content_stream([NORMAL_REPLY])
    fake = _ScriptedCreate([bad, good])
    p = _provider(fake, streaming=True)
    resp = await p.complete(_req())
    assert resp.message.content == NORMAL_REPLY
    assert fake.calls == 2, fake.calls
    assert bad.closed is True
    print("  ✓ degenerate tool-args stream aborted, retry returned clean reply")


async def test_nonstream_degenerate_retries() -> None:
    print("== non-streaming: degenerate completion retries ==")
    fake = _ScriptedCreate([
        _FakeCompletion("开头。" + LOOP_PHRASE * 60),
        _FakeCompletion(NORMAL_REPLY),
    ])
    p = _provider(fake, streaming=False)
    resp = await p.complete(_req())
    assert resp.message.content == NORMAL_REPLY
    assert fake.calls == 2, fake.calls
    print("  ✓ degenerate non-stream completion retried once, clean reply")


async def test_streaming_normal_reply_untouched() -> None:
    print("== streaming: normal long reply passes the detector ==")
    # A long, structured, non-repetitive reply must not trip the guard.
    lines = [f"{i}. 检查项目{i}：状态正常，数值为 {i * 7}。" for i in range(40)]
    fake = _ScriptedCreate([content_stream(lines)])
    p = _provider(fake, streaming=True)
    resp = await p.complete(_req())
    assert resp.message.content == "".join(lines)
    assert fake.calls == 1, fake.calls
    print("  ✓ 40-line structured reply delivered without false positive")


async def test_max_tokens_kwarg() -> None:
    print("== max_tokens kwarg forwarding ==")
    fake = _ScriptedCreate([NORMAL_REPLY])
    p = _provider(fake, streaming=False)
    req = _req()
    req.max_tokens = 16384
    await p.complete(req)
    assert fake.last_kwargs.get("max_tokens") == 16384, fake.last_kwargs.get("max_tokens")
    fake2 = _ScriptedCreate([NORMAL_REPLY])
    p2 = _provider(fake2, streaming=False)
    req2 = _req()                       # max_tokens=None by default
    await p2.complete(req2)
    assert "max_tokens" not in fake2.last_kwargs, fake2.last_kwargs.get("max_tokens")
    print("  ✓ max_tokens forwarded when set, omitted when None")


# --------------------------------------------------------------------------- #
# Settings + agent wiring
# --------------------------------------------------------------------------- #

def test_settings_default_and_parse() -> None:
    print("== settings: default 16384 + yaml override ==")
    settings = load_settings()          # seeds a fresh default config
    assert settings.llm.chat.max_tokens == 16384, settings.llm.chat.max_tokens
    cfg = home / "config.yaml"
    text = cfg.read_text(encoding="utf-8")
    text = text.replace(
        "model:", "max_tokens: 2048\n    model:", 1,
    )
    cfg.write_text(text, encoding="utf-8")
    settings2 = load_settings()
    assert settings2.llm.chat.max_tokens == 2048, settings2.llm.chat.max_tokens
    print("  ✓ default 16384; yaml override to 2048 parsed")


def test_agent_max_tokens_helper() -> None:
    print("== agent: _max_tokens helper ==")
    from chat_team.agent.agent import Agent

    class _NS:
        pass

    fake = _NS()
    fake.settings = _NS()
    fake.settings.llm = _NS()
    fake.settings.llm.chat = _NS()
    fake.settings.llm.chat.max_tokens = 0
    assert Agent._max_tokens(fake) is None
    fake.settings.llm.chat.max_tokens = 16384
    assert Agent._max_tokens(fake) == 16384
    fake.settings.llm.chat.max_tokens = "4096"
    assert Agent._max_tokens(fake) == 4096
    fake.settings.llm.chat.max_tokens = None
    assert Agent._max_tokens(fake) is None
    print("  ✓ 0/None → uncapped, positive int/str → cap")


async def main() -> None:
    test_detector_hits()
    test_detector_no_false_positives()
    test_detector_is_retryable()
    await test_streaming_degenerate_aborts_and_retries()
    await test_streaming_tool_args_degenerate()
    await test_nonstream_degenerate_retries()
    await test_streaming_normal_reply_untouched()
    await test_max_tokens_kwarg()
    test_settings_default_and_parse()
    test_agent_max_tokens_helper()
    print("\nALL OUTPUT-GUARD SMOKES PASSED")


if __name__ == "__main__":
    asyncio.run(main())
