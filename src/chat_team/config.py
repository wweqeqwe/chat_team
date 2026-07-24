"""Load configuration from ``~/.chat_team/config.yaml``."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

import yaml
from dotenv import load_dotenv

from .mcp.config import McpServerConfig
from .paths import Paths, init_home


@dataclass
class SessionConfig:
    msgid_lru_size: int = 500
    persistence_debounce_seconds: float = 10.0
    per_turn_transfer_cap: int = 3
    # UX: while the agent is still working, periodically push a transient
    # status line so WeCom users don't stare at a silent chat window.
    progress_status_enabled: bool = True
    progress_status_delay_seconds: float = 1.5
    progress_status_interval_seconds: float = 2.5
    progress_status_text: str = "正在处理,请稍候..."
    # Hard cap on Sessions held in memory. When exceeded, the LRU entry is
    # flushed to session.json and evicted; the user transparently reloads
    # from disk on next message. Without this, every distinct user × bot
    # leaks a Session forever.
    max_in_memory_sessions: int = 1000


@dataclass
class NotebookConfig:
    max_bytes: int = 4096


@dataclass
class ToolsConfig:
    file_read_max_bytes: int = 1_048_576
    file_write_max_bytes: int = 1_048_576
    shell_timeout_seconds: int = 30
    shell_output_max_bytes: int = 8192
    # Extra env-var names dropped from the run_command subprocess on top of
    # the built-in deny-list (OPENAI_*, WECOM_*, ANTHROPIC_*, CHAT_TEAM_*,
    # plus anything containing KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL).
    shell_env_extra_drop: list[str] = field(default_factory=list)


@dataclass
class CleanupConfig:
    # Files older than this in inbox/, .chat_team/runs/, .chat_team/llm/
    # are unlinked on the lazy per-session sweep.
    max_age_days: int = 14
    # Minimum gap between two sweeps of the same session. Sweep is triggered
    # by get_or_create; this throttles thrash on chatty sessions.
    sweep_interval_hours: float = 6.0
    sweep_subdirs: list[str] = field(
        default_factory=lambda: ["inbox", ".chat_team/runs", ".chat_team/llm"]
    )


@dataclass
class KeyRotationConfig:
    """Multi-API-key rotation settings (see ``llm/api_keys``).

    When ``llm.api_keys`` has more than one entry, the OpenAI provider binds
    each ``(session_id, role_name)`` pair to a single key (round-robin) and
    reuses it for the life of that binding so the upstream prefix-cache stays
    warm. A binding is released after ``idle_reset_seconds`` with no activity;
    the next request for that pair advances the round-robin pointer (it does
    *not* re-pick the just-released key). Rotation is in-memory only — not
    persisted, not hot-reloadable (keys are baked into constructed clients).
    """
    enabled: bool = True
    idle_reset_seconds: float = 600.0


@dataclass
class LLMConfig:
    provider: str = "openai"
    api_key: str = ""
    api_keys: list[str] = field(default_factory=list)
    base_url: str = ""
    key_rotation: "KeyRotationConfig" = field(default_factory=lambda: KeyRotationConfig())
    chat: "LLMChatConfig" = field(default_factory=lambda: LLMChatConfig())
    vision: "LLMVisionConfig" = field(default_factory=lambda: LLMVisionConfig())
    debug_log_enabled: bool = False
    http_debug_log_enabled: bool = False
    use_streaming: bool = True
    request_timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_initial_delay: float = 1.0
    max_tool_loops_per_turn: int = 16


@dataclass
class LLMChatConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    history_token_budget: int = 12000
    # Optional reasoning depth for chat turns. Keep empty to let provider/model
    # defaults decide.
    reasoning_effort: str = ""


@dataclass
class LLMVisionConfig:
    model: str = ""
    strategy: str = "tool"
    image_detail: str = "high"
    reasoning_effort: str = ""
    api_key: str = ""
    api_keys: list[str] = field(default_factory=list)
    base_url: str = ""
    # Image size limits and resize behaviour for vision payloads.
    # When a raw image exceeds max_inline_bytes:
    #   "resize" → auto downscale + re-encode as JPEG (default)
    #   "reject" → replace with [图:xxx(过大,已省略)] placeholder
    max_inline_bytes: int = 6 * 1024 * 1024       # 6 MB raw file size ceiling
    oversized_image: str = "resize"                # resize | reject
    resize_long_side: int = 2048                   # max pixel dimension after resize
    resize_quality: int = 85                       # JPEG quality 1-95


@dataclass
class LoggingConfig:
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass
class McpConfig:
    servers: list[McpServerConfig] = field(default_factory=list)



@dataclass
class AdminConfig:
    """Backend management web UI (``chat-team-admin``).

    A standalone process serving an HTTPS admin panel: view chat_team
    service status, restart/reload it, inspect server disk usage + the
    chat_team home footprint, tail logs. Auth is account+password with a
    login page; sessions live in-memory inside the admin process. The admin
    process never touches the bot process directly — it shells out to
    ``systemctl`` for restart/reload and reads the filesystem for the rest.

    See ``chat_team.admin`` for the standalone entry point.
    """
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8443
    # Empty → default to ~/.chat_team/admin/{cert,key}.pem (run
    # ``chat-team-admin init-certs`` once to generate the self-signed pair).
    tls_cert: str = ""
    tls_key: str = ""
    # Session idle timeout (seconds). Default 8h = a workday.
    session_idle_seconds: float = 28800.0
    # Per-IP failed-login budget in a sliding 5-minute window. 5 attempts;
    # the 6th inside the window returns 429.
    login_rate_limit_per_5min: int = 5
    # Empty → ~/.chat_team/logs/admin.log
    audit_log_path: str = ""

@dataclass
class PrivateChatConfig:
    """Policy for whether the bot replies to single (private) chats.

    Group chats always pass through; only single chats are gated.

    Modes:
      * ``open``      — reply to everyone
      * ``closed``    — reply to no one
      * ``blacklist`` — reply to everyone except ``blacklist``
      * ``whitelist`` — reply only to ``whitelist`` (DEFAULT — see below)

    Default mode is ``whitelist`` with an empty list, i.e. default-deny:
    a brand-new install or an upgrade that doesn't add a ``private_chat``
    block will NOT reply to any private chat until the maintainer explicitly
    opts in (either by switching ``mode`` or by populating ``whitelist``).
    Group chats are always served regardless.

    IMPORTANT for upgrades: pre-feature deployments replied to every private
    chat. After this change, the bot will go silent on private chats until
    you add ``private_chat: {mode: open}`` (or a populated whitelist). The
    ``team_admin`` role can still be reached via group chats and the
    ``chat-team-boss`` CLI (which bypasses this gate entirely).

    ``blocked_reply`` is sent (as a single text message) to a user whose
    private chat was blocked. Empty string = silent drop. Note: this only
    gates one-to-one single chats; group chats are unaffected.
    """
    mode: str = "whitelist"
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    # Default is NON-empty: a default-deny policy that silently drops would
    # trap users in a dead-end (they can't learn their own userid to ask the
    # maintainer to whitelist them). The {userid} placeholder is substituted
    # by the adapter with the blocked sender's WeCom userid, so a brand-new
    # install self-documents the onboarding path without any config.
    blocked_reply: str = "私聊未开放。你的企业微信账号是 {userid},请联系管理员将其加入白名单。"

    def allows(self, user_id: str) -> bool:
        """True if this user_id should receive a reply in a single chat."""
        if self.mode == "open":
            return True
        if self.mode == "closed":
            return False
        if self.mode == "blacklist":
            return user_id not in self.blacklist
        # "whitelist" or any unrecognised value → fail closed (default-deny).
        # A typo in config.yaml can't accidentally expose the bot to everyone;
        # the maintainer simply adds their own userid to whitelist once.
        return user_id in self.whitelist


@dataclass
class BotConfig:
    """One WeCom bot entry. Solo mode requires ``name`` (= role); team mode doesn't."""
    bot_id: str
    secret: str
    name: str = ""


@dataclass
class Settings:
    paths: Paths
    workspace_root: Path
    default_role: str = "team_admin"
    mode: str = "team"  # "team" (multi-role transfer) | "solo" (one bot = one role)
    bots: list[BotConfig] = field(default_factory=list)
    log_level: str = "INFO"
    session: SessionConfig = field(default_factory=SessionConfig)
    notebook: NotebookConfig = field(default_factory=NotebookConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    private_chat: PrivateChatConfig = field(default_factory=PrivateChatConfig)
    admin: AdminConfig = field(default_factory=AdminConfig)
    team_profile: str = ""


def _coerce(target: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if hasattr(target, key):
            setattr(target, key, value)


# --------------------------------------------------------------------------
# Hot reload
# --------------------------------------------------------------------------
# ``reload_settings`` mutates the *existing* ``Settings`` instance in place so
# every component that already holds a ``self.settings`` reference (Dispatcher,
# SessionManager, PersistenceManager, Agent, WeComBotAdapter, compactor) sees
# the new values on its next read — without re-wiring the whole graph.
#
# Not everything can be hot-reloaded: anything that baked itself into a
# long-lived OS-level resource at startup (WebSocket connections for
# ``bots``, the team/solo ``mode`` switch, MCP subprocesses/SSE sessions, the
# OpenAI SDK client's ``api_key``/``base_url``/``timeout``/http-debug hooks,
# and the image-cache singleton's resize knobs) is reported back as
# ``requires_restart`` instead of being silently swapped. The caller (the
# SIGHUP handler in ``app.py``) logs these so the maintainer knows a restart
# is still needed for those fields.
#
# Fields that ARE hot-reloadable are read "per turn" or "per message" by the
# runtime (e.g. ``private_chat`` is consulted on every inbound message,
# ``llm.chat.model`` on every agent turn, ``team_profile`` on every system
# prompt rebuild), so swapping them in the shared ``Settings`` instance is
# sufficient — no re-wiring required.


# Top-level Settings fields whose values are determined by *live OS resources*
# (open sockets, spawned subprocesses, constructed SDK clients) and therefore
# cannot be swapped without tearing those resources down. ``reload_settings``
# detects divergence in these and reports it instead of applying it.
_REQUIRES_RESTART_TOP = frozenset({"mode", "bots", "workspace_root", "mcp"})

# Nested ``settings.llm`` fields that the live ``OpenAIChatCompletionProvider``
# has already baked into its constructed ``AsyncOpenAI`` client / ``httpx``
# client. The provider's ``apply_runtime_overrides`` covers the *other* llm
# knobs; these stay restart-only.
_LLM_REQUIRES_RESTART = frozenset({
    "api_key", "api_keys", "base_url", "key_rotation",
    "request_timeout_seconds", "http_debug_log_enabled",
})

# Nested ``settings.llm.vision`` fields that the process-level
# ``ImageDataURICache`` singleton baked in at startup. ``configure_default_cache``
# is re-invoked on reload for the *other* vision cache knobs, but api_key /
# base_url feed the vision provider construction.
_VISION_REQUIRES_RESTART = frozenset({"api_key", "api_keys", "base_url"})


@dataclass
class ReloadReport:
    """Result of a ``reload_settings`` call.

    ``applied`` lists top-level/nested field paths whose new value was written
    into the live ``Settings`` (and will be picked up on the next read).
    ``requires_restart`` lists field paths whose value diverged but cannot be
    applied hot — the maintainer must ``--stop`` and restart for those.
    ``errors`` is non-empty only if the YAML/file read itself failed (in which
    case nothing was applied).
    """
    applied: list[str] = field(default_factory=list)
    requires_restart: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.errors:
            return "reload FAILED: " + "; ".join(self.errors)
        parts = []
        if self.applied:
            parts.append("applied=[" + ", ".join(self.applied) + "]")
        if self.requires_restart:
            parts.append("requires_restart=[" + ", ".join(self.requires_restart) + "]")
        return "reload OK; " + (" ".join(parts) if parts else "no changes")


def _parse_raw(paths: Paths) -> dict[str, Any]:
    """Read config.yaml → dict. Missing/unparseable → empty dict (with a log)."""
    if not paths.config_yaml.exists():
        return {}
    try:
        loaded = yaml.safe_load(paths.config_yaml.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        _log.exception("config.yaml parse failed during reload; keeping old config")
        raise
    return loaded if isinstance(loaded, dict) else {}


def _build_bots(raw: dict[str, Any]) -> list[BotConfig]:
    bots: list[BotConfig] = []
    if isinstance(raw.get("bots"), list):
        for b in raw["bots"]:
            if isinstance(b, dict) and b.get("bot_id") and b.get("secret"):
                bots.append(BotConfig(
                    bot_id=str(b["bot_id"]),
                    secret=str(b["secret"]),
                    name=str(b.get("name", "")),
                ))
    return bots


def _build_mcp(raw: dict[str, Any]) -> McpConfig:
    if not isinstance(raw.get("mcp"), dict):
        return McpConfig()
    servers_raw = raw["mcp"].get("servers")
    if not isinstance(servers_raw, dict):
        return McpConfig()
    mcp_servers: list[McpServerConfig] = []
    for srv_name, srv_cfg in servers_raw.items():
        if not isinstance(srv_cfg, dict):
            continue
        cfg = McpServerConfig(
            name=str(srv_name),
            command=srv_cfg.get("command", ""),
            args=list(srv_cfg.get("args") or []),
            env=dict(srv_cfg.get("env") or {}),
            url=srv_cfg.get("url", ""),
        )
        try:
            cfg.validate()
            mcp_servers.append(cfg)
        except ValueError:
            _log.warning("skipping invalid mcp server %r", srv_name, exc_info=True)
    return McpConfig(servers=mcp_servers)


def _build_private_chat(raw: dict[str, Any]) -> PrivateChatConfig:
    pc_raw = raw.get("private_chat")
    if not isinstance(pc_raw, dict):
        return PrivateChatConfig()
    valid_modes = {"open", "closed", "blacklist", "whitelist"}
    mode = str(pc_raw.get("mode", "whitelist")).strip().lower()
    if mode not in valid_modes:
        _log.warning(
            "private_chat.mode=%r is not one of %s; defaulting to 'whitelist'",
            pc_raw.get("mode"), sorted(valid_modes),
        )
        mode = "whitelist"
    return PrivateChatConfig(
        mode=mode,
        whitelist=[str(u).strip() for u in (pc_raw.get("whitelist") or []) if str(u).strip()],
        blacklist=[str(u).strip() for u in (pc_raw.get("blacklist") or []) if str(u).strip()],
        blocked_reply=str(pc_raw.get("blocked_reply", "") or ""),
    )


def _build_settings_from_raw(raw: dict[str, Any], paths: Paths) -> Settings:
    """Construct a brand-new ``Settings`` from a parsed raw dict + paths.

    Shared by ``load_settings`` (initial load) and ``reload_settings`` (which
    builds a temp instance from this, then copies safe fields into the live
    one in place).
    """
    workspace_root_raw = raw.get("workspace_root", "workspaces")
    workspace_root = Path(workspace_root_raw)
    if not workspace_root.is_absolute():
        workspace_root = paths.home / workspace_root

    settings = Settings(
        paths=paths,
        workspace_root=workspace_root,
        default_role=raw.get("default_role", "team_admin"),
        mode=raw.get("mode", "team"),
        bots=_build_bots(raw),
        log_level=raw.get("log_level", "INFO"),
    )
    if isinstance(raw.get("session"), dict):
        _coerce(settings.session, raw["session"])
    if isinstance(raw.get("notebook"), dict):
        _coerce(settings.notebook, raw["notebook"])
    if isinstance(raw.get("tools"), dict):
        _coerce(settings.tools, raw["tools"])
    if isinstance(raw.get("llm"), dict):
        llm_raw = raw["llm"]
        assert isinstance(llm_raw, dict)
        _coerce(settings.llm, {
            k: v for k, v in llm_raw.items() if k not in {"chat", "vision", "key_rotation"}
        })
        if isinstance(llm_raw.get("chat"), dict):
            _coerce(settings.llm.chat, llm_raw["chat"])
        if isinstance(llm_raw.get("vision"), dict):
            _coerce(settings.llm.vision, llm_raw["vision"])
        if isinstance(llm_raw.get("key_rotation"), dict):
            _coerce(settings.llm.key_rotation, llm_raw["key_rotation"])
    if isinstance(raw.get("logging"), dict):
        _coerce(settings.logging, raw["logging"])
    if isinstance(raw.get("cleanup"), dict):
        _coerce(settings.cleanup, raw["cleanup"])
    settings.private_chat = _build_private_chat(raw)
    settings.mcp = _build_mcp(raw)
    if isinstance(raw.get("admin"), dict):
        _coerce(settings.admin, raw["admin"])

    # Backward compat: no bots in YAML → synthesize from WECOM_* env vars.
    if not settings.bots:
        env_bot_id = os.environ.get("WECOM_BOT_ID", "")
        env_secret = os.environ.get("WECOM_SECRET", "")
        if env_bot_id and env_secret:
            settings.bots = [BotConfig(bot_id=env_bot_id, secret=env_secret)]

    if paths.team_md.exists():
        settings.team_profile = paths.team_md.read_text(encoding="utf-8").strip()
    else:
        settings.team_profile = ""
    workspace_root.mkdir(parents=True, exist_ok=True)
    return settings


def load_settings(paths: Paths | None = None) -> Settings:
    paths = paths or init_home()
    load_dotenv(paths.dotenv, override=False)
    raw = _parse_raw(paths)
    return _build_settings_from_raw(raw, paths)


def _values_differ(a: Any, b: Any) -> bool:
    """True if ``a`` and ``b`` should be treated as different for reload diffing.

    Uses ``!=`` first (so ``10`` and ``10.0`` compare equal — YAML ints and
    dataclass float defaults shouldn't trip a false "changed"). Falls back to
    ``repr`` comparison for types whose ``!=`` raises or returns a non-bool
    (e.g. some numpy-style objects); we don't ship any, but the guard keeps
    the diff robust against future field types.
    """
    try:
        ne = (a != b)
    except Exception:  # noqa: BLE001
        return repr(a) != repr(b)
    if isinstance(ne, bool):
        return ne
    # Truthy non-bool (e.g. array-like) → fall back to repr.
    return repr(a) != repr(b)


def _apply_safe_nested(dst: Any, src: Any, prefix: str, restart_keys: frozenset[str],
                       report: ReloadReport) -> None:
    """Copy each field from ``src`` into ``dst`` (both dataclasses), diffing.

    Fields in ``restart_keys`` are *not* copied — divergence is reported as
    requires_restart instead. Everything else is copied and, if it changed,
    appended to ``report.applied`` as ``prefix.field``.
    """
    for fname in dst.__dataclass_fields__:
        old = getattr(dst, fname)
        new = getattr(src, fname)
        if fname in restart_keys:
            if _values_differ(old, new):
                report.requires_restart.append(f"{prefix}.{fname}")
            continue
        if _values_differ(old, new):
            setattr(dst, fname, new)
            report.applied.append(f"{prefix}.{fname}")


def reload_settings(settings: Settings) -> ReloadReport:
    """Re-read ``config.yaml`` + ``team.md`` and mutate ``settings`` in place.

    Returns a ``ReloadReport`` describing what was applied vs. what needs a
    restart. On a YAML parse error, nothing is mutated and ``report.errors``
    is populated.

    The ``settings.paths`` object is reused as-is (it's tied to the home dir
    which doesn't change). Structural fields (``mode``, ``bots``,
    ``workspace_root``, ``mcp``) are compared but NOT applied — they require
    a process restart because they're bound to live sockets/subprocesses.
    """
    report = ReloadReport()
    paths = settings.paths
    try:
        raw = _parse_raw(paths)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"config.yaml: {exc!r}")
        return report
    fresh = _build_settings_from_raw(raw, paths)

    # --- structural (restart-only) top-level fields -------------------------
    for fname in _REQUIRES_RESTART_TOP:
        old = getattr(settings, fname)
        new = getattr(fresh, fname)
        if _values_differ(old, new):
            report.requires_restart.append(fname)

    # --- safe top-level scalars --------------------------------------------
    for fname in ("default_role", "log_level", "team_profile"):
        old = getattr(settings, fname)
        new = getattr(fresh, fname)
        if _values_differ(old, new):
            setattr(settings, fname, new)
            report.applied.append(fname)

    # --- safe nested dataclasses -------------------------------------------
    _apply_safe_nested(settings.session, fresh.session, "session",
                       restart_keys=frozenset(), report=report)
    _apply_safe_nested(settings.notebook, fresh.notebook, "notebook",
                       restart_keys=frozenset(), report=report)
    _apply_safe_nested(settings.tools, fresh.tools, "tools",
                       restart_keys=frozenset(), report=report)
    _apply_safe_nested(settings.logging, fresh.logging, "logging",
                       restart_keys=frozenset(), report=report)
    _apply_safe_nested(settings.cleanup, fresh.cleanup, "cleanup",
                       restart_keys=frozenset(), report=report)

    # private_chat is a single dataclass instance referenced by the adapter;
    # replace the whole object so ``adapter.settings.private_chat`` picks up
    # the new mode/whitelist on the next message.
    if settings.private_chat != fresh.private_chat:
        settings.private_chat = fresh.private_chat
        report.applied.append("private_chat")

    # admin panel — host/port/tls_cert/tls_key/enabled are baked into the
    # live aiohttp.web listener at admin-process startup; report them as
    # restart-only so the maintainer knows to restart chat-team-admin.
    _apply_safe_nested(settings.admin, fresh.admin, "admin",
                       restart_keys=frozenset({"enabled", "host", "port", "tls_cert", "tls_key"}),
                       report=report)

    # llm: chat sub-config is fully safe (read per turn).
    _apply_safe_nested(settings.llm.chat, fresh.llm.chat, "llm.chat",
                       restart_keys=frozenset(), report=report)
    # llm: vision sub-config — api_key/base_url are restart-only (vision
    # provider is already constructed); the rest (strategy/image_detail/
    # oversized_image/resize_*/max_inline_bytes/model) we copy. The cache
    # singleton is re-configured by the caller (app.Reloader) for the resize
    # knobs.
    _apply_safe_nested(settings.llm.vision, fresh.llm.vision, "llm.vision",
                       restart_keys=_VISION_REQUIRES_RESTART, report=report)
    # llm: top-level — api_key/base_url/request_timeout/http_debug are
    # restart-only (baked into AsyncOpenAI client); debug_log_enabled /
    # use_streaming / max_retries / retry_initial_delay are picked up by the
    # provider's apply_runtime_overrides (called by app.Reloader).
    _apply_safe_nested(settings.llm, fresh.llm, "llm",
                       restart_keys=_LLM_REQUIRES_RESTART, report=report)

    return report
