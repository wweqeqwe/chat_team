"""Vision-as-tool: turn workspace images into text via an LLM call.

The agent can call :class:`DescribeImageTool` to inspect one or more images
it already sees as placeholders in chat history (for example ``[图:./inbox/x]``).
Same image + prompt + detail + model is OCR'd at most once per process thanks
to :class:`ImageDescriptionCache`.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from ...llm.base import ChatMessage, CompletionRequest, LLMProvider
from ...llm.image_cache import MAX_INLINE_BYTES, default_cache as image_default_cache
from ...llm.image_description_cache import ImageDescriptionCache, default_cache
from .base import Tool, ToolContext, ToolError
from .file_tools import _resolve_under

log = logging.getLogger(__name__)

DEFAULT_TOOL_PROMPT = (
    "请详细描述这张图片的内容,包括其中的可见文字(若有)以及画面要素。"
    "对文字部分尽量原样转写,不要总结、不要翻译。"
)
MAX_IMAGES_PER_CALL = 16
DEFAULT_DESCRIBE_CONCURRENCY = 8


def _read_failure_reason(abs_path: str) -> str | None:
    """Returns a short reason string if the file can't / shouldn't be sent
    to the vision model; ``None`` if the file is fine.

    Oversized images are *not* rejected here — the ``ImageDataURICache``
    handles them by auto-resizing (or the ``reject`` strategy). We only
    check for missing / empty files.
    """
    try:
        size = os.path.getsize(abs_path)
    except OSError:
        return "文件不存在或无法读取"
    if size <= 0:
        return "文件为空"
    return None


async def _describe_one(
    abs_path: str,
    *,
    prompt: str,
    detail: str,
    reasoning_effort: str | None = None,
    llm: LLMProvider,
    model: str,
    image_base_dir: str,
    cache: ImageDescriptionCache,
    session_id: str | None = None,
    session_uuid: str | None = None,
    role_name: str | None = None,
    debug_log_dir: Path | None = None,
) -> str:
    fail = _read_failure_reason(abs_path)
    if fail is not None:
        return f"[读取失败:{fail}]"

    cached = cache.get(abs_path, detail=detail, model=model, prompt=prompt)
    if cached is not None:
        return cached

    request = CompletionRequest(
        messages=[
            ChatMessage(role="user", content=[
                {"type": "text", "text": prompt},
                {"type": "image", "path": abs_path},
            ]),
        ],
        model=model,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
        image_detail=detail,
        image_base_dir=image_base_dir,
        session_id=session_id,
        session_uuid=session_uuid,
        role_name=role_name,
        call_kind="vision",
        debug_log_dir=debug_log_dir,
    )
    try:
        resp = await llm.complete(request)
    except Exception as err:                                  # noqa: BLE001
        log.exception("describe_image LLM call failed for %s", abs_path)
        return f"[读取失败:视觉模型调用异常 {type(err).__name__}]"
    text = (resp.message.content or "").strip()
    if not text:
        return "[读取失败:视觉模型返回空内容]"
    cache.put(abs_path, text, detail=detail, model=model, prompt=prompt)
    return text


async def describe_images(
    paths: list[str],
    *,
    prompt: str,
    detail: str,
    reasoning_effort: str | None = None,
    llm: LLMProvider,
    model: str,
    image_base_dir: str,
    cache: ImageDescriptionCache | None = None,
    max_concurrency: int = DEFAULT_DESCRIBE_CONCURRENCY,
    session_id: str | None = None,
    session_uuid: str | None = None,
    role_name: str | None = None,
    debug_log_dir: Path | None = None,
) -> list[str]:
    """Describe a batch of images, returning one description per input path
    in the same order. Each image is its own LLM call (bounded concurrency)
    so a single bad image doesn't poison the rest, and the cache makes repeats
    free.

    ``paths`` are expected to be **absolute** and already sandbox-validated.
    """
    if not paths:
        return []
    cache = cache or default_cache()
    limit = max(1, min(int(max_concurrency), len(paths)))
    sem = asyncio.Semaphore(limit)

    async def _run_one(path: str) -> str:
        async with sem:
            return await _describe_one(
                path,
                prompt=prompt,
                detail=detail,
                reasoning_effort=reasoning_effort,
                llm=llm,
                model=model,
                image_base_dir=image_base_dir,
                cache=cache,
                session_id=session_id,
                session_uuid=session_uuid,
                role_name=role_name,
                debug_log_dir=debug_log_dir,
            )

    return await asyncio.gather(*(_run_one(p) for p in paths))


class DescribeImageTool(Tool):
    name = "describe_image"
    description = (
        "对工作区里的图片调用视觉模型生成文字描述/OCR 结果。"
        "适合在 agent 已经看到 [图:path] 占位但默认摘要不够,"
        "想要换一个 prompt 再问、或者用更高的 detail 重新扫描时使用。"
        "只接受相对当前工作目录的路径(例如 ./inbox/xxx.jpg);"
        "支持多图并发,一次最多 16 张。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1 到 16 个相对工作目录的图片路径",
                "minItems": 1,
                "maxItems": MAX_IMAGES_PER_CALL,
            },
            "prompt": {
                "type": "string",
                "description": (
                    "可选;描述图片时给视觉模型的指令。默认让模型详细描述图片"
                    "并原样转写文字。"
                ),
            },
            "detail": {
                "type": "string",
                "enum": ["low", "high", "auto"],
                "description": "可选;视觉细节级别,默认 high。",
            },
        },
        "required": ["paths"],
    }

    async def run(self, ctx: ToolContext, **kwargs: Any) -> str:
        if ctx.llm is None and ctx.vision_llm is None:
            raise ToolError("describe_image requires an LLM provider in ToolContext")
        llm = ctx.vision_llm or ctx.llm
        raw_paths = kwargs.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ToolError("paths must be a non-empty list of strings")
        if len(raw_paths) > MAX_IMAGES_PER_CALL:
            raise ToolError(f"paths supports at most {MAX_IMAGES_PER_CALL} images per call")
        rel_paths: list[str] = []
        abs_paths: list[str] = []
        for rel in raw_paths:
            if not isinstance(rel, str):
                raise ToolError("paths entries must be strings")
            target = _resolve_under(ctx.cwd, rel)        # raises ToolError on escape
            rel_paths.append(rel)
            abs_paths.append(str(target))

        prompt = kwargs.get("prompt") or DEFAULT_TOOL_PROMPT
        if not isinstance(prompt, str):
            raise ToolError("prompt must be a string")
        detail = kwargs.get("detail") or ctx.settings.llm.vision.image_detail
        if detail not in ("low", "high", "auto"):
            raise ToolError("detail must be one of low|high|auto")

        model = ctx.settings.llm.vision.model or ctx.settings.llm.chat.model
        reasoning_effort = (ctx.settings.llm.vision.reasoning_effort or "").strip() or None

        descriptions = await describe_images(
            abs_paths,
            prompt=prompt,
            detail=detail,
            reasoning_effort=reasoning_effort,
            llm=llm,
            model=model,
            image_base_dir=str(ctx.cwd),
            max_concurrency=min(DEFAULT_DESCRIBE_CONCURRENCY, len(abs_paths)),
            session_id=ctx.session.session_id,
            session_uuid=getattr(ctx.session, "session_uuid", None),
            role_name=ctx.session.current_role,
            debug_log_dir=ctx.cwd / ".chat_team" / "llm",
        )
        sections = [
            f"[图:{rel}]\n{desc}"
            for rel, desc in zip(rel_paths, descriptions)
        ]
        return "\n\n".join(sections)
