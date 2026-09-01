"""OpenAI Chat Completion provider."""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import os
import random
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from ..adapters.base import blocks_to_text
from . import debug_logger
from . import http_debug_logger
from .repetition import CHECK_STEP_CHARS, find_degenerate_repeat
from .base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from .image_cache import ImageDataURICache, default_cache
from .key_rotation import SessionKeyRouter

log = logging.getLogger(__name__)

_OPENCODE_USER_AGENT = "OpenCode/1.0"


class _EmptyCompletionError(RuntimeError):
    """The model completed without visible text or tool calls."""


class _DegenerateOutputError(RuntimeError):
    """Model output collapsed into verbatim repetition (generation loop).

    Raised mid-stream (or after a non-stream completion) when the tail of
    the output becomes many exact copies of one block.  Retryable: a fresh
    sample almost always breaks the loop.
    """


_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    _EmptyCompletionError,
    _DegenerateOutputError,
)


_THINKING_PAIRS: tuple[tuple[str, str], ...] = (
    ("<thinking>", "</thinking>"),
    # xmz-text-model returns a short think wrapper without the "ing".
    # Keep the long form first for unambiguous prefix matching.
    ("<think>", "</think>"),
)

def _strip_thinking_markup(content: str) -> str:
    """Remove provider-wrapped reasoning text.

    Upstream providers wrap reasoning in either
    "<thinking>...</thinking>" or the shorter " thinking... response" and place
    the visible answer after it. Keep only the answer so reasoning never shows
    to users or leaks into history.

    An unmatched reasoning opener means only reasoning was produced with no
    answer; strip it to empty so the provider's empty-completion retry asks
    again. Content that merely starts with a non-reasoning "<" tag is left as
    is.
    """
    if not content:
        return content
    stripped = content.lstrip()
    for open_tag, close_tag in _THINKING_PAIRS:
        if stripped.startswith(open_tag):
            if close_tag in content:
                return content.split(close_tag, 1)[1].strip()
            return ""
    return content


@dataclass
class _HttpLogContext:
    dir_path: Path
    session_id: str | None
    role_name: str | None
    call_kind: str | None
    call_id: str
    request_index: int = 0


_http_log_ctx: contextvars.ContextVar[_HttpLogContext | None] = contextvars.ContextVar(
    "chat_team_http_log_ctx",
    default=None,
)


def _content_as_string(content: Any) -> str:
    """Coerce ChatMessage.content to a string. Image blocks render as
    `[图:<basename>]`; used for tool / assistant / system messages."""
    if isinstance(content, list):
        return blocks_to_text(content)
    return content or ""


def _expand_user_content(
    content: Any,
    *,
    image_detail: str | None,
    image_base_dir: Path | str | None,
    cache: ImageDataURICache,
) -> str | list[dict[str, Any]]:
    """User-message content expansion. Returns a flat string when content
    is a string OR when the list contains only text blocks (so we never
    confuse the API with single-element content arrays). Otherwise returns
    the OpenAI multi-part content list with ``image_url`` data URIs."""
    if not isinstance(content, list):
        return content or ""

    has_image = any(b.get("type") == "image" for b in content)
    if not has_image:
        return blocks_to_text(content)

    detail = image_detail or "high"
    parts: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text") or ""
            if not text:
                continue
            parts.append({"type": "text", "text": text})
            continue
        if btype == "image":
            rel = block.get("path") or ""
            if not rel:
                continue
            abs_path = (
                rel
                if os.path.isabs(rel)
                else os.path.join(str(image_base_dir or "."), rel)
            )
            uri = cache.get(abs_path)
            if uri is None:
                # Defensive fallback: degrade to a text block so the rest of the
                # turn proceeds. Distinguishes missing vs. oversize via a probe.
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    size = -1
                tag = "已丢失" if size < 0 else "过大,已省略"
                parts.append({
                    "type": "text",
                    "text": f"[图:{os.path.basename(rel)}({tag})]",
                })
                continue
            parts.append({
                "type": "image_url",
                "image_url": {"url": uri, "detail": detail},
            })
            continue
        # Unknown block type: render as text placeholder.
        parts.append({"type": "text", "text": f"[未支持:{btype}]"})

    if not parts:
        return ""
    # If after fallback nothing image-shaped remains, collapse to flat string.
    if not any(p.get("type") == "image_url" for p in parts):
        return "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
    return parts


def _to_openai_messages(
    messages: list[ChatMessage],
    *,
    image_detail: str | None = None,
    image_base_dir: Path | str | None = None,
    cache: ImageDataURICache | None = None,
) -> list[dict[str, Any]]:
    cache = cache or default_cache()
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": _content_as_string(m.content),
            })
            continue
        if m.role == "user":
            content = _expand_user_content(
                m.content,
                image_detail=image_detail,
                image_base_dir=image_base_dir,
                cache=cache,
            )
        else:
            content = _content_as_string(m.content)
        msg: dict[str, Any] = {"role": m.role, "content": content}
        if m.role == "assistant" and m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]
        if m.name:
            msg["name"] = m.name
        out.append(msg)
    return out


def _to_openai_tools(specs) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters or {"type": "object", "properties": {}},
            },
        }
        for s in specs
    ]


class OpenAIChatCompletionProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        *,
        api_keys: list[str] | None = None,
        key_rotation_enabled: bool = True,
        key_rotation_idle_seconds: float = 600.0,
        debug_log_enabled: bool = False,
        http_debug_log_enabled: bool = False,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
        use_streaming: bool = True,
    ):
        # Build the effective key list. ``api_keys`` (plural) takes priority
        # when non-empty; otherwise fall back to the single ``api_key`` for
        # full backward compatibility.
        keys: list[str] = [k for k in (api_keys or []) if k]
        if not keys:
            keys = [api_key]
        if not keys[0]:
            raise ValueError(
                "OpenAIChatCompletionProvider requires at least one api_key "
                "(pass api_key=... or api_keys=[...])"
            )
        self._keys = keys
        self._http_debug_log_enabled = http_debug_log_enabled
        event_hooks: dict[str, list[Any]] | None = None
        if http_debug_log_enabled:
            event_hooks = {
                "request": [self._on_http_request],
                "response": [self._on_http_response],
            }
        # One shared httpx connection pool for all keys — N AsyncOpenAI
        # clients reuse it so we don't multiply connection counts by N.
        # max_retries=0 disables the SDK's own retry layer so our outer loop
        # (which now also iterates keys on failure) is the single source of
        # truth for retry policy.
        self._http_client = httpx.AsyncClient(
            timeout=request_timeout_seconds,
            event_hooks=event_hooks,
        )
        self._clients: list[AsyncOpenAI] = [
            AsyncOpenAI(
                api_key=k,
                base_url=base_url or None,
                http_client=self._http_client,
                default_headers={"User-Agent": _OPENCODE_USER_AGENT},
                timeout=request_timeout_seconds,
                max_retries=0,
            )
            for k in keys
        ]
        # ``self._client`` kept as the index-0 client for any legacy/test
        # code path that reads it directly (monkey-patched create() etc.).
        self._client = self._clients[0]
        # Round-robin binder is only active when there is >1 key AND the
        # maintainer hasn't disabled it. With a single key everything goes
        # to index 0 and there's zero per-call overhead.
        if len(keys) > 1 and key_rotation_enabled:
            self._router: SessionKeyRouter | None = SessionKeyRouter(
                len(keys), idle_reset_seconds=key_rotation_idle_seconds,
            )
        else:
            self._router = None
        self._debug_log_enabled = debug_log_enabled
        self._max_retries = max(1, int(max_retries))
        self._retry_initial_delay = max(0.0, float(retry_initial_delay))
        self._use_streaming = bool(use_streaming)


    def apply_runtime_overrides(
        self,
        *,
        debug_log_enabled: bool,
        use_streaming: bool,
        max_retries: int,
        retry_initial_delay: float,
    ) -> list[str]:
        """Hot-apply the four runtime knobs that aren't baked into the client.

        ``api_key`` / ``base_url`` / ``request_timeout_seconds`` /
        ``http_debug_log_enabled`` are baked into the constructed
        ``AsyncOpenAI`` + ``httpx.AsyncClient`` at ``__init__`` time and cannot
        be swapped without rebuilding the client — the reloader reports those
        as ``requires_restart``. The four knobs here are plain attributes the
        provider reads per call, so they're safe to mutate live.

        Returns the list of knobs whose value actually changed.
        """
        changed: list[str] = []
        if self._debug_log_enabled != debug_log_enabled:
            self._debug_log_enabled = debug_log_enabled
            changed.append("debug_log_enabled")
        if self._use_streaming != use_streaming:
            self._use_streaming = use_streaming
            changed.append("use_streaming")
        new_max = max(1, int(max_retries))
        if self._max_retries != new_max:
            self._max_retries = new_max
            changed.append("max_retries")
        new_delay = max(0.0, float(retry_initial_delay))
        if self._retry_initial_delay != new_delay:
            self._retry_initial_delay = new_delay
            changed.append("retry_initial_delay")
        return changed

    @staticmethod
    def _build_tool_calls_from_deltas(
        tool_calls_by_index: dict[int, dict[str, str]],
    ) -> list[ToolCall]:
        out: list[ToolCall] = []
        for idx in sorted(tool_calls_by_index):
            item = tool_calls_by_index[idx]
            raw_args = item.get("arguments", "")
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            out.append(ToolCall(
                id=item.get("id") or f"tool_call_{idx}",
                name=item.get("name") or "",
                arguments=args,
            ))
        return out

    @staticmethod
    async def _quiet_close_stream(stream: Any) -> None:
        """Best-effort close of an async stream after an early abort."""
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close is None:
            return
        try:
            res = close()
            if inspect.isawaitable(res):
                await res
        except Exception:                                          # noqa: BLE001
            log.debug("closing aborted LLM stream failed", exc_info=True)

    async def _complete_with_streaming(
        self,
        kwargs: dict[str, Any],
        stream_text_callback=None,
        *,
        client: AsyncOpenAI | None = None,
    ) -> tuple[ChatMessage, str, dict[str, Any] | None, Any | None]:
        # Some tests monkey-patch create() with a non-stream fake object. If
        # the returned value is not async-iterable, treat it as non-stream.
        c = client or self._client
        maybe_stream = await c.chat.completions.create(**kwargs, stream=True)
        if not hasattr(maybe_stream, "__aiter__"):
            completion = maybe_stream
            choice = completion.choices[0]
            msg = choice.message
            tool_calls: list[ToolCall] = []
            for tc in (msg.tool_calls or []):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            usage = None
            if getattr(completion, "usage", None) is not None:
                try:
                    usage = completion.usage.model_dump()
                except Exception:                                 # noqa: BLE001
                    usage = None
            return (
                ChatMessage(role="assistant", content=_strip_thinking_markup(msg.content or ""), tool_calls=tool_calls),
                choice.finish_reason or "stop",
                usage,
                completion,
            )

        content_parts: list[str] = []
        content_acc = ""          # running text for degenerate-repeat detection
        checked_content_len = 0   # last content length checked by the detector
        args_checked: dict[int, int] = {}   # per-tool-call checked args length
        tool_calls_by_index: dict[int, dict[str, str]] = {}
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        raw_last_chunk: Any | None = None

        async for chunk in maybe_stream:
            raw_last_chunk = chunk
            if getattr(chunk, "usage", None) is not None:
                try:
                    usage = chunk.usage.model_dump()
                except Exception:                                 # noqa: BLE001
                    usage = None
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            part = getattr(delta, "content", None)
            if part:
                content_parts.append(part)
                content_acc += part
                if len(content_acc) - checked_content_len >= CHECK_STEP_CHARS:
                    checked_content_len = len(content_acc)
                    pat = find_degenerate_repeat(content_acc)
                    if pat is not None:
                        await self._quiet_close_stream(maybe_stream)
                        raise _DegenerateOutputError(
                            "streaming content degenerated into repetition "
                            f"(pattern {pat!r}); aborting stream to retry"
                        )
                if stream_text_callback is not None:
                    display = _strip_thinking_markup("".join(content_parts))
                    if display:
                        try:
                            await stream_text_callback(display)
                        except Exception:                             # noqa: BLE001
                            log.debug("stream_text_callback failed", exc_info=True)
            for tc in (getattr(delta, "tool_calls", None) or []):
                idx = getattr(tc, "index", None)
                if idx is None:
                    idx = len(tool_calls_by_index)
                state = tool_calls_by_index.setdefault(idx, {
                    "id": "",
                    "name": "",
                    "arguments": "",
                })
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    state["id"] = tc_id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        state["name"] = fn_name
                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        state["arguments"] += fn_args
                        acc = state["arguments"]
                        if len(acc) - args_checked.get(idx, 0) >= CHECK_STEP_CHARS:
                            args_checked[idx] = len(acc)
                            pat = find_degenerate_repeat(acc)
                            if pat is not None:
                                await self._quiet_close_stream(maybe_stream)
                                raise _DegenerateOutputError(
                                    "tool-call arguments degenerated into "
                                    f"repetition (pattern {pat!r}); aborting "
                                    "stream to retry"
                                )

        return (
            ChatMessage(
                role="assistant",
                content=_strip_thinking_markup("".join(content_parts)),
                tool_calls=self._build_tool_calls_from_deltas(tool_calls_by_index),
            ),
            finish_reason,
            usage,
            raw_last_chunk,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        token = None
        if self._http_debug_log_enabled and request.debug_log_dir is not None:
            token = _http_log_ctx.set(_HttpLogContext(
                dir_path=request.debug_log_dir.parent / "llm_http",
                session_id=request.session_id,
                role_name=request.role_name,
                call_kind=request.call_kind,
                call_id=secrets.token_hex(8),
            ))
        messages_payload = _to_openai_messages(
            request.messages,
            image_detail=request.image_detail,
            image_base_dir=request.image_base_dir,
        )
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages_payload,
            "temperature": request.temperature,
        }
        if request.session_uuid:
            # ``extra_headers`` is per API call, unlike AsyncOpenAI's
            # ``default_headers``. This keeps one stable conversation UUID
            # on every agent/tool/vision/compactor request without leaking
            # the internal WeCom-derived session_id.
            kwargs["extra_headers"] = {
                "x-session-id": request.session_uuid,
            }
        if request.tools:
            kwargs["tools"] = _to_openai_tools(request.tools)
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        effort = (request.reasoning_effort or "").strip()
        if effort:
            kwargs["reasoning_effort"] = effort

        t0 = time.monotonic()
        attempts = 0
        completion = None
        response_msg: ChatMessage | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        raw_obj: Any | None = None
        last_exc: Exception | None = None
        # ---- per-(session,role) key binding + multi-key retry ----
        # On failure we don't disable keys (maintainer's rule): we try the
        # bound key first, then every other key in round-robin order, each
        # retried up to ``max_retries`` times. The binding itself never
        # moves on failure — only idle-reset advances it — so the next
        # turn for this ``(session, role)`` reuses the same bound key
        # (cache-friendly) unless it has since gone idle.
        bound_idx = (
            self._router.select(request.session_id, request.role_name)
            if self._router is not None else 0
        )
        n_clients = len(self._clients)
        key_order = [(bound_idx + i) % n_clients for i in range(n_clients)]
        succeeded = False
        try:
            for key_idx in key_order:
                client = self._clients[key_idx]
                key_attempts = 0
                for attempt in range(self._max_retries):
                    attempts += 1
                    key_attempts += 1
                    try:
                        candidate_completion = None
                        if self._use_streaming:
                            (
                                candidate_response_msg,
                                candidate_finish_reason,
                                candidate_usage,
                                candidate_raw_obj,
                            ) = (
                                await self._complete_with_streaming(
                                    kwargs,
                                    stream_text_callback=request.stream_text_callback,
                                    client=client,
                                )
                            )
                        else:
                            candidate_completion = await client.chat.completions.create(**kwargs)
                            choice = candidate_completion.choices[0]
                            msg = choice.message
                            candidate_tool_calls: list[ToolCall] = []
                            for tc in (msg.tool_calls or []):
                                try:
                                    args = json.loads(tc.function.arguments or "{}")
                                except json.JSONDecodeError:
                                    args = {"_raw": tc.function.arguments}
                                candidate_tool_calls.append(
                                    ToolCall(
                                        id=tc.id,
                                        name=tc.function.name,
                                        arguments=args,
                                    )
                                )
                            candidate_response_msg = ChatMessage(
                                role="assistant",
                                content=_strip_thinking_markup(msg.content or ""),
                                tool_calls=candidate_tool_calls,
                            )
                            _pat = find_degenerate_repeat(
                                _content_as_string(candidate_response_msg.content)
                            )
                            if _pat is None:
                                for tc in (msg.tool_calls or []):
                                    _pat = find_degenerate_repeat(
                                        tc.function.arguments or ""
                                    )
                                    if _pat is not None:
                                        break
                            if _pat is not None:
                                raise _DegenerateOutputError(
                                    "completion degenerated into repetition "
                                    f"(pattern {_pat!r}); retrying"
                                )
                            candidate_finish_reason = choice.finish_reason or "stop"
                            candidate_usage = None
                            if getattr(candidate_completion, "usage", None) is not None:
                                try:
                                    candidate_usage = candidate_completion.usage.model_dump()
                                except Exception:                         # noqa: BLE001
                                    candidate_usage = None
                            candidate_raw_obj = candidate_completion

                        if (
                            not candidate_response_msg.tool_calls
                            and not _content_as_string(candidate_response_msg.content).strip()
                        ):
                            raise _EmptyCompletionError(
                                "model returned no visible content or tool calls "
                                f"(finish_reason={candidate_finish_reason!r})"
                            )

                        completion = candidate_completion
                        response_msg = candidate_response_msg
                        finish_reason = candidate_finish_reason
                        usage = candidate_usage
                        raw_obj = candidate_raw_obj
                        succeeded = True
                        last_exc = None
                        break
                    except _RETRYABLE_EXCEPTIONS as exc:
                        last_exc = exc
                        if attempt >= self._max_retries - 1:
                            # this key's retry budget is spent — fall
                            # through to the next key (if any).
                            if key_idx != key_order[-1]:
                                log.warning(
                                    "LLM call failed on key %d (%s) after "
                                    "%d/%d attempts; trying next key",
                                    key_idx, type(exc).__name__,
                                    key_attempts, self._max_retries,
                                )
                            break
                        delay = self._retry_initial_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        log.warning(
                            "LLM call failed on key %d (%s) attempt %d/%d; "
                            "retrying in %.2fs",
                            key_idx, type(exc).__name__,
                            key_attempts, self._max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    except Exception as exc:                          # noqa: BLE001
                        # Non-retryable (401/403/etc): don't keep hammering
                        # this key — jump to the next one. We never mark a
                        # key permanently bad (maintainer's "don't disable"
                        # rule); it stays eligible for future bindings.
                        last_exc = exc
                        if key_idx != key_order[-1]:
                            log.warning(
                                "LLM call hit non-retryable error on key %d "
                                "(%s); trying next key",
                                key_idx, type(exc).__name__,
                            )
                        break
                if succeeded:
                    break
        finally:
            if token is not None:
                _http_log_ctx.reset(token)

        if completion is None and response_msg is None:
            latency_ms = (time.monotonic() - t0) * 1000.0
            self._maybe_write_log(
                request,
                messages_payload=messages_payload,
                response_message=None,
                finish_reason=None,
                usage=None,
                latency_ms=latency_ms,
                error=repr(last_exc) if last_exc else "unknown",
                attempts=attempts,
            )
            assert last_exc is not None
            raise last_exc

        latency_ms = (time.monotonic() - t0) * 1000.0
        if response_msg is None:
            assert completion is not None
            choice = completion.choices[0]
            msg = choice.message
            tool_calls: list[ToolCall] = []
            for tc in (msg.tool_calls or []):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            response_msg = ChatMessage(
                role="assistant",
                content=_strip_thinking_markup(msg.content or ""),
                tool_calls=tool_calls,
            )
            finish_reason = choice.finish_reason or "stop"
            if getattr(completion, "usage", None) is not None:
                try:
                    usage = completion.usage.model_dump()
                except Exception:                                 # noqa: BLE001
                    usage = None
            raw_obj = completion

        if (finish_reason or "") == "length":
            log.warning(
                "LLM response truncated at the output cap "
                "(finish_reason=length); consider raising llm.chat.max_tokens"
            )
        serialised_response = {
            "role": "assistant",
            "content": response_msg.content,
        }
        if response_msg.tool_calls:
            serialised_response["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response_msg.tool_calls
            ]
        self._maybe_write_log(
            request,
            messages_payload=messages_payload,
            response_message=serialised_response,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            error=None,
            attempts=attempts,
        )
        return CompletionResponse(
            message=response_msg,
            finish_reason=finish_reason or "stop",
            raw=raw_obj,
        )

    async def _on_http_request(self, request: httpx.Request) -> None:
        ctx = _http_log_ctx.get()
        if ctx is None:
            return
        try:
            body_bytes = request.content if request.content else b""
        except Exception:                                          # noqa: BLE001
            body_bytes = None
        ctx.request_index += 1
        request.extensions["chat_team_http_request_index"] = ctx.request_index
        http_debug_logger.write_http_request_log(
            dir_path=ctx.dir_path,
            session_id=ctx.session_id,
            role_name=ctx.role_name,
            call_kind=ctx.call_kind,
            call_id=ctx.call_id,
            request_index=ctx.request_index,
            method=request.method,
            url=str(request.url),
            headers_raw=list(request.headers.raw),
            body_bytes=body_bytes,
        )

    async def _on_http_response(self, response: httpx.Response) -> None:
        ctx = _http_log_ctx.get()
        if ctx is None:
            return
        request = response.request
        try:
            body_bytes = await response.aread()
        except Exception:                                          # noqa: BLE001
            body_bytes = None
        request_index = request.extensions.get("chat_team_http_request_index")
        if not isinstance(request_index, int) or request_index <= 0:
            ctx.request_index += 1
            request_index = ctx.request_index
        http_debug_logger.write_http_response_log(
            dir_path=ctx.dir_path,
            session_id=ctx.session_id,
            role_name=ctx.role_name,
            call_kind=ctx.call_kind,
            call_id=ctx.call_id,
            request_index=request_index,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            reason_phrase=response.reason_phrase or "",
            headers_raw=list(response.headers.raw),
            body_bytes=body_bytes,
        )

    def _maybe_write_log(
        self,
        request: CompletionRequest,
        *,
        messages_payload: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        finish_reason: str | None,
        usage: dict[str, Any] | None,
        latency_ms: float,
        error: str | None,
        attempts: int = 1,
    ) -> None:
        if not self._debug_log_enabled:
            return
        if request.debug_log_dir is None:
            return
        debug_logger.write_call_log(
            dir_path=request.debug_log_dir,
            session_id=request.session_id,
            role_name=request.role_name,
            call_kind=request.call_kind,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            reasoning_effort=request.reasoning_effort,
            tool_names=[s.name for s in request.tools],
            messages_payload=messages_payload,
            response_message=response_message,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            error=error,
            attempts=attempts,
        )
