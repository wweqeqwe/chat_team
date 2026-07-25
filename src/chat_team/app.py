"""Application bootstrap: wires settings + tools + roles + dispatcher + adapter."""
from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import os
import shutil
import signal
import sys

from .adapters.base import BotAdapter
from .agent.tools.base import Tool, ToolRegistry
from .agent.tools.describe_image import DescribeImageTool
from .agent.tools.file_tools import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from .agent.tools.media_tools import SendFileTool, SendImageTool
from .agent.tools.notebook_tools import (
    NotebookDeleteTool,
    NotebookReadTool,
    NotebookWriteTool,
)
from .agent.tools.shell_tool import RunCommandTool
from .agent.tools.skill_tools import SkillReadFileTool, SkillTool
from .agent.tools.transfer_tool import TransferToEmployeeTool
from .config import Settings, load_settings
from .out_rotator import OutFileRotator
from .daemon import daemonize_and_run, reload_daemon, stop_daemon
from .dispatcher import Dispatcher
from .llm.base import LLMProvider
from .llm.image_cache import configure_default_cache
from .llm.openai_provider import OpenAIChatCompletionProvider
from .roles.registry import RoleRegistry
from .session.manager import SessionManager
from .session.persistence import PersistenceManager
from .paths import resolve_home
from .reload import Reloader
from .skills.registry import SkillRegistry

log = logging.getLogger(__name__)


def build_tool_registry(
    roles: RoleRegistry,
    skills: SkillRegistry | None = None,
    extra_tools: list[Tool] | None = None,
    *,
    include_transfer: bool = True,
) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ReadFileTool())
    reg.register(WriteFileTool())
    reg.register(EditFileTool())
    reg.register(ListDirTool())
    reg.register(GlobTool())
    reg.register(GrepTool())
    reg.register(RunCommandTool(skills=skills, roles=roles))
    reg.register(NotebookReadTool())
    reg.register(NotebookWriteTool())
    reg.register(NotebookDeleteTool())
    reg.register(SendImageTool())
    reg.register(SendFileTool())
    reg.register(DescribeImageTool())
    if include_transfer:
        reg.register(TransferToEmployeeTool(available_employees=roles.names()))
    if skills is not None and skills.names():
        reg.register(SkillTool(skills=skills, roles=roles))
        reg.register(SkillReadFileTool(skills=skills, roles=roles))
    for tool in extra_tools or []:
        reg.register(tool)
    return reg


def _resolve_main_keys(settings: Settings) -> list[str]:
    """Resolved, de-duplicated-empty main key list (llm.api_keys wins)."""
    keys = [k for k in (settings.llm.api_keys or []) if k]
    if not keys:
        single = settings.llm.api_key or os.environ.get("OPENAI_API_KEY", "")
        keys = [single] if single else []
    return keys


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm.provider != "openai":
        raise NotImplementedError(f"llm provider not supported yet: {settings.llm.provider}")
    api_keys = _resolve_main_keys(settings)
    if not api_keys:
        raise RuntimeError(
            "LLM API key missing — set llm.api_key or llm.api_keys in "
            "~/.chat_team/config.yaml or the OPENAI_API_KEY environment variable"
        )
    api_key = api_keys[0]
    base_url = settings.llm.base_url or os.environ.get("OPENAI_BASE_URL") or None
    kr = settings.llm.key_rotation
    return OpenAIChatCompletionProvider(
        api_key=api_key,
        api_keys=api_keys,
        base_url=base_url,
        key_rotation_enabled=kr.enabled,
        key_rotation_idle_seconds=kr.idle_reset_seconds,
        debug_log_enabled=settings.llm.debug_log_enabled,
        http_debug_log_enabled=settings.llm.http_debug_log_enabled,
        use_streaming=settings.llm.use_streaming,
        request_timeout_seconds=settings.llm.request_timeout_seconds,
        max_retries=settings.llm.max_retries,
        retry_initial_delay=settings.llm.retry_initial_delay,
    )


def build_vision_llm_provider(settings: Settings, main_llm: LLMProvider) -> LLMProvider:
    """Return an LLM provider for vision/OCR calls.

    Credential resolution mirrors the main provider: ``llm.vision.api_keys``
    (plural) wins when non-empty, otherwise ``llm.vision.api_key`` /
    ``OPENAI_VISION_API_KEY`` fall back to the *main* key list. When the
    resolved vision key list + base_url are identical to the main
    provider's, ``main_llm`` is returned as-is — this shares not only the
    httpx connection pool but also the per-``(session, role)`` key router,
    so vision OCR calls reuse the same key the chat turns are using.
    """
    main_api_keys = _resolve_main_keys(settings)
    main_base_url = settings.llm.base_url or os.environ.get("OPENAI_BASE_URL") or None

    vision_api_keys = [k for k in (settings.llm.vision.api_keys or []) if k]
    if not vision_api_keys:
        vision_single = (
            settings.llm.vision.api_key
            or os.environ.get("OPENAI_VISION_API_KEY", "")
        )
        if vision_single:
            vision_api_keys = [vision_single]
        else:
            vision_api_keys = list(main_api_keys)
    vision_base_url = (
        settings.llm.vision.base_url
        or os.environ.get("OPENAI_VISION_BASE_URL", "")
        or main_base_url
    ) or None

    # Reuse the main provider when the resolved key list + base_url are
    # identical — shares the connection pool AND the session→key router.
    if vision_api_keys == main_api_keys and vision_base_url == main_base_url:
        return main_llm

    if not vision_api_keys:
        # Defensive: main had no keys either (build_llm_provider would have
        # already raised, but be safe for direct callers in tests).
        raise RuntimeError(
            "vision LLM API key missing — set llm.vision.api_key or "
            "llm.vision.api_keys, or OPENAI_VISION_API_KEY"
        )

    log.info(
        "vision LLM uses separate credentials (base_url=%s, keys=%d)",
        vision_base_url or "(default)",
        len(vision_api_keys),
    )
    vision_api_key = vision_api_keys[0]
    kr = settings.llm.key_rotation
    return OpenAIChatCompletionProvider(
        api_key=vision_api_key,
        api_keys=vision_api_keys,
        base_url=vision_base_url,
        key_rotation_enabled=kr.enabled,
        key_rotation_idle_seconds=kr.idle_reset_seconds,
        debug_log_enabled=settings.llm.debug_log_enabled,
        http_debug_log_enabled=settings.llm.http_debug_log_enabled,
        use_streaming=settings.llm.use_streaming,
        request_timeout_seconds=settings.llm.request_timeout_seconds,
        max_retries=settings.llm.max_retries,
        retry_initial_delay=settings.llm.retry_initial_delay,
    )


def warn_if_uv_missing(roles: RoleRegistry) -> None:
    """WARN when a Python-capable role is loaded but `uv` isn't on PATH.

    Triggered by the same `{skill, run_command}` combination that injects the
    PEP 723 convention into the agent's system prompt (see ``agent.py``).
    Non-fatal: roles that don't run Python skills keep working.
    """
    if shutil.which("uv"):
        return
    if not any({"skill", "run_command"}.issubset(set(r.tools)) for r in roles.all()):
        return
    log.warning(
        "skill+run_command roles loaded but `uv` not on PATH; "
        "Python skills will fail. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    )


def build_dispatcher(
    settings: Settings,
    extra_tools: list[Tool] | None = None,
) -> Dispatcher:
    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    skills = SkillRegistry.load(settings.paths.user_skills_dir)
    warn_if_uv_missing(roles)
    tools = build_tool_registry(roles, skills, extra_tools=extra_tools)
    persistence = PersistenceManager(settings)
    sessions = SessionManager(settings, persistence=persistence)
    llm = build_llm_provider(settings)
    vision_llm = build_vision_llm_provider(settings, llm)
    return Dispatcher(
        settings, sessions, roles, tools, llm,
        skills=skills, persistence=persistence,
        vision_llm=vision_llm,
    )


def configure_logging(settings: Settings, *, file_only: bool = False) -> None:
    """(Re)configure root logging.

    Idempotent: clears existing root handlers before re-adding so a hot
    reload (SIGHUP → Reloader) doesn't accumulate duplicate handlers and
    multi-log every line. Safe to call at startup and on every reload.
    """
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.paths.logs_dir / "chat_team.log"
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    rotating = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=settings.logging.max_bytes,
        backupCount=settings.logging.backup_count,
        encoding="utf-8",
    )
    rotating.setFormatter(fmt)
    handlers: list[logging.Handler] = [rotating]
    if not file_only:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        handlers.append(sh)
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


# Grace period after a SIGTERM/SIGINT before we hard-exit the process.
# In practice the WS adapter's ``_tear_down_connection`` and the MCP stdio
# transport's anyio task group can both hang on a CancelledError, refusing to
# unwind. Without this backstop, ``--stop`` (and systemd ``stop``) would block
# until they escalate to SIGKILL after their own timeout. Five seconds is
# generous for the persistence flush (debounced writes already in
# session.json) and short enough to feel snappy.
_HARD_EXIT_GRACE_SECONDS = 5.0


def _force_exit_after_grace(task: "asyncio.Future", delay: float) -> None:
    """Schedule a process exit if ``task`` hasn't finished within ``delay``.

    Uses ``os._exit`` deliberately: it skips ``asyncio.run``'s own teardown
    (``shutdown_asyncgens``, executor shutdown), which is exactly where the
    MCP/anyio hang lives. Child MCP servers die on their own when this
    process's stdio pipe closes; persistence is already on disk except for
    at most the last debounce window (``persistence_debounce_seconds``).
    """
    import os

    def _force() -> None:
        if task.done():
            return
        log.warning(
            "main task did not finish within %.1fs of shutdown signal; "
            "forcing process exit (last persistence debounce may be lost)",
            delay,
        )
        # Flush log handlers so the warning above actually lands on disk.
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:  # noqa: BLE001
                pass
        os._exit(0)

    asyncio.get_running_loop().call_later(delay, _force)


async def _run_with_shutdown(main_awaitable, *, on_sighup=None) -> None:
    """Run ``main_awaitable`` until it finishes or a shutdown signal arrives.

    On SIGTERM/SIGINT the awaited task is cancelled, which lets the caller's
    ``finally`` block (adapter.close, persistence flush, MCP teardown) execute
    normally. This is what makes ``--stop`` and systemd's SIGTERM shut the bot
    down gracefully instead of dropping in-flight session state.

    As a backstop, if the cancelled task doesn't unwind within
    ``_HARD_EXIT_GRACE_SECONDS`` (some adapter/MCP teardown paths can hang on
    CancelledError), we force the process to exit. This keeps ``--stop`` and
    systemd responsive at the cost of at most the last persistence debounce.

    ``on_sighup`` is an optional zero-arg callable (the ``Reloader.reload``
    method) invoked when the process receives SIGHUP. Unlike SIGTERM/SIGINT,
    SIGHUP does NOT cancel the main task — it triggers an in-place hot reload
    of config/roles/skills/team.md without dropping the WebSocket connection.
    This is what ``chat-team --reload`` and ``kill -HUP <pid>`` activate.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(main_awaitable)

    def _on_signal() -> None:
        log.info("shutdown signal received, cancelling main task")
        if not task.done():
            task.cancel()
        # Backstop: don't let a hung teardown block shutdown indefinitely.
        _force_exit_after_grace(task, _HARD_EXIT_GRACE_SECONDS)

    def _on_sighup() -> None:
        if on_sighup is None:
            log.info("SIGHUP received but no reloader wired; ignoring")
            return
        log.info("SIGHUP received, starting hot reload")
        try:
            report = on_sighup()
            # report may be a CombinedReloadReport (has summary) or None.
            msg = report.summary() if hasattr(report, "summary") else "done"
            log.info("hot reload result: %s", msg)
        except Exception:  # noqa: BLE001
            log.exception("hot reload failed")

    installed: list[int] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal)
            installed.append(sig)
        except NotImplementedError:
            pass  # non-Unix / non-main-thread
    sighup_installed = False
    if on_sighup is not None and hasattr(signal, "SIGHUP"):
        try:
            loop.add_signal_handler(signal.SIGHUP, _on_sighup)
            sighup_installed = True
        except (NotImplementedError, AttributeError):
            pass  # non-Unix / non-main-thread

    try:
        await task
    except asyncio.CancelledError:
        log.info("shutdown complete")
    finally:
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass
        if sighup_installed:
            try:
                loop.remove_signal_handler(signal.SIGHUP)
            except (NotImplementedError, RuntimeError):
                pass


async def _run_solo(
    settings: Settings,
    mcp_tools: list[Tool],
    mcp_manager,
    adapter_factory,
) -> None:
    """Solo mode: one process, N bots, each pinned to one role."""
    if not settings.bots:
        raise RuntimeError("mode=solo but no bots configured in config.yaml")

    configure_default_cache(
        max_inline_bytes=settings.llm.vision.max_inline_bytes,
        oversized_image=settings.llm.vision.oversized_image,
        resize_long_side=settings.llm.vision.resize_long_side,
        resize_quality=settings.llm.vision.resize_quality,
    )

    roles = RoleRegistry.load(settings.paths.user_roles_dir)
    skills = SkillRegistry.load(settings.paths.user_skills_dir)
    warn_if_uv_missing(roles)
    llm = build_llm_provider(settings)
    vision_llm = build_vision_llm_provider(settings, llm)
    persistence = PersistenceManager(settings)

    adapters: list[tuple[BotAdapter, Dispatcher]] = []
    for bot_cfg in settings.bots:
        if not roles.has(bot_cfg.name):
            raise RuntimeError(
                f"solo bot '{bot_cfg.name}' references unknown role; "
                f"available: {roles.names()}"
            )
        tools = build_tool_registry(
            roles, skills, extra_tools=mcp_tools, include_transfer=False,
        )
        sessions = SessionManager(
            settings, persistence=persistence, solo_role=bot_cfg.name,
        )
        dispatcher = Dispatcher(
            settings, sessions, roles, tools, llm,
            skills=skills, persistence=persistence,
            vision_llm=vision_llm, fixed_role=bot_cfg.name,
        )
        adapter: BotAdapter = adapter_factory(
            settings, sessions.workspace_for,
            bot_id=bot_cfg.bot_id, secret=bot_cfg.secret,
            role_name=bot_cfg.name,
        )
        adapter.set_handler(dispatcher.handle)
        adapters.append((adapter, dispatcher))
        log.info("solo bot '%s' configured (bot_id=%s)", bot_cfg.name, bot_cfg.bot_id[:8] + "...")

    reloader = Reloader(
        settings, [d for _, d in adapters], reconfigure_logging=configure_logging,
    )
    out_rotator = OutFileRotator(
        settings.paths.logs_dir / "chat_team.out",
        max_bytes=settings.logging.out_max_bytes,
        backup_count=settings.logging.out_backup_count,
        check_interval_seconds=settings.logging.out_check_interval_seconds,
    )
    out_rotator.start()
    try:
        await _run_with_shutdown(
            asyncio.gather(*[a.run_forever() for a, _ in adapters]),
            on_sighup=reloader.reload,
        )
    finally:
        await out_rotator.stop()
        for a, _ in adapters:
            await a.close()
        all_sessions: list = []
        for _, d in adapters:
            all_sessions.extend(d.sessions.all_sessions())
        await persistence.flush_all(all_sessions)
        await mcp_manager.close_all()


async def _async_main(adapter_factory) -> None:
    from .mcp.client import McpClientManager

    settings = load_settings()
    configure_logging(settings)
    configure_default_cache(
        max_inline_bytes=settings.llm.vision.max_inline_bytes,
        oversized_image=settings.llm.vision.oversized_image,
        resize_long_side=settings.llm.vision.resize_long_side,
        resize_quality=settings.llm.vision.resize_quality,
    )
    log.info("chat_team starting; home=%s mode=%s", settings.paths.home, settings.mode)

    mcp_manager = McpClientManager()
    mcp_tools: list[Tool] = []
    if settings.mcp.servers:
        mcp_tools = await mcp_manager.connect_all(settings.mcp.servers)
        log.info(
            "MCP: %d tool(s) from %d server(s)",
            len(mcp_tools),
            len(settings.mcp.servers),
        )

    if settings.mode == "solo":
        await _run_solo(settings, mcp_tools, mcp_manager, adapter_factory)
        return

    dispatcher = build_dispatcher(settings, extra_tools=mcp_tools)
    bot = settings.bots[0] if settings.bots else None
    adapter: BotAdapter = adapter_factory(
        settings, dispatcher.sessions.workspace_for,
        bot_id=bot.bot_id if bot else None,
        secret=bot.secret if bot else None,
    )
    adapter.set_handler(dispatcher.handle)
    reloader = Reloader(
        settings, [dispatcher], reconfigure_logging=configure_logging,
    )
    # Rotate the daemon's stdout/stderr file (chat_team.out) so it can't grow
    # unbounded. No-op in foreground mode (-f): the file isn't written there,
    # but the reaper stat()s a non-existent file and exits immediately. In
    # solo mode the reaper is started below in _run_solo instead so each bot's
    # lifecycle is self-contained.
    out_rotator = OutFileRotator(
        settings.paths.logs_dir / "chat_team.out",
        max_bytes=settings.logging.out_max_bytes,
        backup_count=settings.logging.out_backup_count,
        check_interval_seconds=settings.logging.out_check_interval_seconds,
    )
    out_rotator.start()
    try:
        # Prefer run_forever (handles transient WS loss); fall back to the
        # one-shot connect+run for adapters that don't implement it.
        runner = getattr(adapter, "run_forever", None)
        if runner is not None:
            await _run_with_shutdown(runner(), on_sighup=reloader.reload)
        else:
            async def _one_shot() -> None:
                await adapter.connect()
                await adapter.run()

            await _run_with_shutdown(_one_shot(), on_sighup=reloader.reload)
    finally:
        await out_rotator.stop()
        await adapter.close()
        if dispatcher.persistence is not None:
            await dispatcher.persistence.flush_all(dispatcher.sessions.all_sessions())
        await mcp_manager.close_all()


def run() -> None:
    """Entry point for ``chat-team`` / ``python main.py``.

    By default the bot runs as a background daemon (survives SSH disconnect).
    Use ``--foreground`` to run in the current terminal (for systemd/supervisor
    or debugging), or ``--stop`` to terminate a running daemon.
    """
    parser = argparse.ArgumentParser(
        prog="chat-team",
        description="WeCom AI Bot — virtual employee team.",
    )
    parser.add_argument(
        "-f", "--foreground",
        action="store_true",
        help="run in the foreground (default: background daemon)",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="stop a running background daemon",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="hot-reload config.yaml / team.md / roles / skills in a running "
             "daemon (sends SIGHUP); the bot keeps serving without restart",
    )
    args = parser.parse_args()

    from .adapters.wecom import WeComBotAdapter   # lazy import

    home = resolve_home()
    pid_path = home / "chat_team.pid"
    out_path = home / "logs" / "chat_team.out"

    if args.stop:
        sys.exit(stop_daemon(pid_path))

    if args.reload:
        sys.exit(reload_daemon(pid_path))

    if args.foreground:
        asyncio.run(_async_main(WeComBotAdapter))
        return

    daemonize_and_run(
        out_path, pid_path,
        lambda: asyncio.run(_async_main(WeComBotAdapter)),
    )
