# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A WeCom (企业微信) AI Bot that fronts a team of role-differentiated "virtual employees". The only builtin role is `team_admin` (the front-desk/receptionist); every other role is user-defined as a YAML in `~/.chat_team/roles/`. Sample non-builtin roles (`research_engineer`, `customer_service`) live under `docs/examples/roles/` for users to copy. Two deployment modes: **team** (default — one bot, multi-role transfer via `transfer_to_employee`) and **solo** (one bot per role, multiple bots in one process, shared notebook for cross-bot communication). Each chat session gets its own working directory; sessions are fully isolated.

## Commands

```bash
# Run the bot (long-connection WeCom WebSocket).
# Default runs as a background daemon (double-fork, survives SSH disconnect):
#   stdout/stderr → ~/.chat_team/logs/chat_team.out   (prints + uncaught tracebacks)
#   PID           → ~/.chat_team/chat_team.pid        (removed on clean exit)
# First run seeds ~/.chat_team/{config.yaml,roles,workspaces,logs,state}.
python main.py                # background daemon (default)
python main.py -f             # foreground — use this under systemd/supervisor or for debugging
python main.py --stop         # stop the background daemon (SIGTERM, then SIGKILL after 10s)
# Same -f / --stop flags work on the `chat-team` console script.

# Smoke tests — all are pure Python, no LLM/network. Run individually.
python scripts/smoke_dispatch.py                  # dispatcher + agent + tools (scripted LLM)
python scripts/smoke_transfer.py                  # multi-hop transfer + cap + unknown target
python scripts/smoke_tools.py                     # write/read/list/run_command + sandbox
python scripts/smoke_wecom_parse.py               # adapter parse + LRU + stream frame shape
python scripts/smoke_compaction_persistence.py    # tiktoken compaction + session.json round-trip
python scripts/smoke_media_events.py              # AES-256-CBC decrypt + enter_chat/disconnected
python scripts/smoke_team_profile.py              # team.md injection + compactor isolation
python scripts/smoke_boss.py                      # CLI boss agent + team_tools
python scripts/smoke_describe_image.py            # describe_images() cache + DescribeImageTool sandbox
python scripts/smoke_vision_shim.py               # eager OCR shim (image blocks → text in user content)
python scripts/smoke_llm_debug_log.py             # per-call LLM debug log: redaction + write + seq
python scripts/smoke_skills.py                    # SkillRegistry + skill / skill_read_file tools + system-prompt TOC
python scripts/smoke_critical_fixes.py            # persistence-race snapshot + agent turn rollback + dispatcher post_turn on _run_turn error
python scripts/smoke_p0_fixes.py                  # reconnect + env scrub + LRU + janitor + LLM retry
python scripts/smoke_split_llm.py                 # vision_llm vs chat_llm split: build_vision_llm_provider + DescribeImageTool routing
python scripts/smoke_p0_round2.py                 # _bg_tasks strong-ref + concurrent get_or_create + eviction off event loop + finish-before-post_turn
python scripts/smoke_mcp.py                       # MCP proxy tool + config parsing + agent integration
python scripts/smoke_solo.py                      # solo mode: per-bot dispatchers + shared notebook + isolated persistence
python scripts/smoke_private_chat_policy.py     # private_chat policy: open/closed/blacklist/whitelist + adapter gate + enter_chat
python scripts/smoke_key_rotation.py            # llm.api_keys round-robin: (session,role) binding + idle reset + multi-key retry + single-key compat
python scripts/smoke_slash_commands.py            # /new /stop /status (group+private) + /running (private-only) + history rollback on /stop
python scripts/smoke_admin.py                    # backend admin panel: pbkdf2/session/CSRF/rate-limit/audit + aiohttp HTTP flow

# Conversational team-setup CLI (not the WeCom bot — see "Boss agent" below).
chat-team-boss

# Print all tools registered in the main runtime, for hand-authored role YAMLs.
chat-team-tools          # or: python -m chat_team.list_tools

# Backend admin panel (HTTPS web UI, separate process — see "Admin panel" below).
chat-team-admin serve            # start the admin web server (default if no subcommand)
chat-team-admin init-certs       # generate a self-signed TLS cert pair to ~/.chat_team/admin/
chat-team-admin add-user <name>  # create/update an admin login user (interactive password)
```

There is no test framework — smokes are async `main()` scripts that print and assert. Add new smokes alongside the existing ones; they all set `CHAT_TEAM_HOME=/tmp/...` and `shutil.rmtree` it at startup so they don't pollute the real `~/.chat_team`.

## Runtime directory

All persistent state lives under `~/.chat_team/` (override with `CHAT_TEAM_HOME`):

```
~/.chat_team/
  config.yaml              # all configuration + credentials (chmod 0600); defaults written on first run
  .env                     # (legacy fallback) env vars still loaded if present; new installs don't create this
  team.md                  # global team profile; injected into every agent's system prompt
  roles/                   # user-defined role YAMLs override builtins by name
  skills/                  # user-defined skill dirs (<name>/SKILL.md + aux files) override builtins by name
  workspaces/<sid>/        # one per chat session
    inbox/                 # decrypted inbound media lands here
    .chat_team/
      session.json         # current_role + per-role histories (debounced, atomic)
      notebook.md          # shared "team whiteboard", ## key blocks, 4KB cap
      notebook.index.json  # updated_at sidecar
      runs/<ts>.log        # full shell stdout (tool returns truncated)
      llm/<ts>-<seq>-<role>-<kind>.json  # per-LLM-call debug record (when llm.debug_log_enabled)
  chat_team.pid            # PID of the background daemon (only in default background mode)
  logs/                    # all rotating: chat_team.log + chat_team.out + admin.log (see Log rotation)
  state/                   # cross-session bits (currently empty, reserved)
```

`paths.sanitize_session_id` is the only function that maps `session_id → directory name`. Both `SessionManager.workspace_for` and `WeComBotAdapter._save_media` go through it (the adapter via the `workspace_resolver` callback wired in `app.py`). Don't recompute paths anywhere else.

## Layered architecture

```
WeComBotAdapter   ── parses WS frames, manages stream replies, decrypts media
        ↓ IncomingMessage
Dispatcher        ── owns session.lock, transfer loop, post-turn compact + persist
        ↓ session_id
SessionManager    ── workspace_for / get_or_create; restores from session.json
        ↓
Session           ── cwd, current_role, agents_by_role, notebook, lock,
                     pending_handoff, transfer_count_this_turn, restored_histories
        ↓
Agent (per role)  ── owns this role's history; runs chat+tool loop
        ↓
Tool              ── ToolContext(cwd, session, settings); sandboxed I/O
```

**One Agent = one role × one session.** Histories are NEVER shared across roles. Cross-role facts go through `notebook.md` (a Markdown file with `## key` blocks) — agents see only the TOC injected into their system prompt, and fetch values via `notebook_read`.

### Boss agent (CLI-only)

`src/chat_team/boss.py` is a separate entry point (`chat-team-boss`) that reuses `Agent` + `LLMProvider` + a fresh `ToolRegistry` of `team_tools`. It runs a stdin/stdout chat loop so the user can shape `~/.chat_team/team.md` and `~/.chat_team/roles/*.yaml` conversationally instead of hand-editing YAML.

Key isolation points:
- **`BOSS_ROLE` is hardcoded in `boss.py`** and is NOT registered in `RoleRegistry`. The WeCom-side `transfer_to_employee` enum and `enter_chat` flow never see it.
- **Boss tools** (`agent/tools/team_tools.py`) deliberately bypass the cwd sandbox and operate on absolute paths under `settings.paths.user_roles_dir` / `settings.paths.team_md`. They are NOT registered in `app.build_tool_registry` and never reach the dispatcher.
- **`list_available_tools`** dynamically calls `app.build_tool_registry(RoleRegistry({}))` to enumerate the *main-runtime* tool catalog so the boss recommends valid `tools:` names — picks up any new tool without editing the boss.
- **`write_role`** validates input (yaml parse → `Role.from_dict` → name match → atomic write) before touching disk; bad YAML returns `ToolError` and the LLM self-corrects on retry.
- **Confirmation is by-prompt, not by-CLI**: the boss's system prompt requires it to paste the full proposed YAML/markdown to the user and ask "是否确认写入?" before invoking any write tool. There is no separate y/N gate.
- **No persistence**: each `chat-team-boss` invocation starts a fresh chat. The durable state lives in the role YAMLs and `team.md` it edits.

## Admin panel (`chat-team-admin`, separate process)

`src/chat_team/admin/` is a **standalone** HTTPS web panel (`chat-team-admin`)
running in its own systemd unit (`scripts/chat-team-admin.service`). It is
deliberately decoupled from the bot runtime — no shared imports of
`Dispatcher`/`Agent`/`Adapter`. Status comes from `systemctl`, disk from
`os.statvfs` + `du`, log tail from `journalctl`/file. The bot and the panel
can be restarted independently.

Layout:
- `admin/auth.py` — `UserStore` (pbkdf2_sha256, 600k iters, OWASP 2023; tolerates
  `None` user with a timing-equal dummy pbkdf2 to mitigate username enumeration),
  `SessionStore` (in-memory `{sid: {username, expires_at, csrf_token, ip}}`,
  sliding-window expiry), `LoginRateLimiter` (per-IP sliding 5min, counts
  *failed* logins only — the right password never trips), `AuditLogger`
  (one line per audit event to `~/.chat_team/logs/admin.log`, rotated via
  `RotatingFileHandler` with `admin.audit_log_max_bytes` × `admin.audit_log_backup_count`,
  ~25 MB ceiling).
- `admin/inspect.py` — framework-agnostic sync helpers wrapped in
  `asyncio.to_thread` for use from aiohttp routes. `get_service_status_sync`
  parses `systemctl show` (ActiveState/SubState/MainPID/ActiveEnterTimestamp/
  MemoryCurrent), falls back to `ps` + `/proc/<pid>` RSS when systemctl is
  unavailable. `_du_subdir_sync` walks `~/.chat_team/{logs,workspaces,state,...}`
  via `du -sb` (Python fallback), plus per-session top-N. `CachedDiskInspector`
  caches the result 30s so refresh-spamming doesn't hammer the disk.
- `admin/cli.py` — `add-user` (interactive `getpass`, ≥8 chars, mismatch
  rejected, atomic write to users.json), `init-certs` (RSA-2048 self-signed,
  1 year, via `cryptography`), `serve` (loads settings + validates cert/users
  exist + runs `aiohttp.web.run_app(..., ssl_context=...)`).
- `admin/server.py` — `build_app(settings)` wires aiohttp routes +
  `require_auth` middleware (HTML routes 302 /login, /api/* → 401). CSRF is
  double-submit (X-CSRF-Token header == csrf cookie AND == server-side session
  token, constant-time compared). On_startup spawns a session-sweeper task
  (every 5min); the callback MUST be async because aiohttp's `aiosignal.send`
  awaits the receiver's return value — a sync callback returning None raises
  `TypeError: object NoneType can't be used in 'await' expression`.

Auth flow: `POST /login` (form) → `UserStore.verify` (timing-safe) → on
success `SessionStore.create` returns `(sid, csrf)` → both set as cookies
(session: `HttpOnly+Secure+SameSite=Strict`; csrf: `Secure+SameSite=Strict`
but NOT HttpOnly so the dashboard JS can read it for the X-CSRF-Token header).
Failed logins recorded in `LoginRateLimiter` AND `AuditLogger`; success clears
the IP's failure window. `POST /api/{restart,reload}` require a valid session
AND a matching CSRF header; both shell out to `systemctl` (no PolKit needed
because the unit runs as root).

Config: `Settings.admin: AdminConfig` (enabled/host/port/tls_cert/tls_key/
session_idle_seconds/login_rate_limit_per_5min/audit_log_path). All fields are
marked `requires_restart` in `reload_settings` because they're baked into the
live aiohttp listener at admin-process startup — `chat-team --reload` (the
bot's SIGHUP) does NOT pick them up; restart `chat-team-admin` instead.

`build_app` is a pure function so `scripts/smoke_admin.py` exercises the full
HTTP flow via `aiohttp.test_utils.TestClient` over plain HTTP (TLS skipped in
tests). Critical smoke-test detail: `TestClient` follows redirects by default,
so the login POST must use `allow_redirects=False` to read `Set-Cookie` off
the 302 directly — otherwise the cookie lands in the client's jar instead of
in `r.cookies`, and every subsequent `headers={"Cookie": ...}` test would
send literal `"None"` and 500 the request serializer.

## Non-obvious mechanics

**Transfer flow.** `transfer_to_employee` raises `TransferRequested` (a special exception, not a `ToolError`). It bubbles from tool → agent → dispatcher. Before re-raising, the agent appends a synthetic `tool` message (`"[transferred] target=X"`) to its OWN history so the dangling `tool_calls` is closed — otherwise reopening that role later breaks OpenAI history validation. The dispatcher catches it, increments `transfer_count_this_turn`, and either:
- transfers (sets `current_role`, queues a `PendingHandoff`, re-loops with the SAME user_text on the new agent), or
- forces the current agent to answer (cap reached / unknown target) by injecting a synthetic system note.

**Handoff notes are one-shot.** `agent.queue_system_note` puts a string into `pending_system_inject`; `_build_system_messages` emits it as a system message and clears the buffer. It is NOT persisted to `agent.history` — that prevents re-injection every turn. If you want it persistent, you need a different mechanism.

**Post-turn pipeline.** `Dispatcher._post_turn` runs `compactor.maybe_compact(agent)` for every agent in the session, then `persistence.schedule(session)`. Both happen INSIDE the session lock to avoid races with the next inbound message. Compaction may make an LLM call.

**Compaction boundary.** `_find_keep_boundary` always lands on a `user` message — never split an `assistant(tool_calls)` + `tool` pair, otherwise the next OpenAI request 400s. Default keeps the last 6 user turns verbatim and replaces the rest with one `[历史摘要]` system message at the head of `agent.history`. Token budget is `role.llm.history_token_budget` falling back to `settings.llm.default_history_token_budget`.

**History restoration is lazy.** `SessionManager.get_or_create` reads `session.json` on first touch, populates `session.restored_histories` and `session.current_role`. The dispatcher's `_agent_for` consumes (pops) the entry only when that role is materialised — that way restoring a session for which an unused role had a long history doesn't pay any cost until that role is actually needed.

**WS write serialization.** Three cooperating asyncio tasks: reader / heartbeat (30s `ping`) / writer. The writer is the only thing that calls `ws.send` — everything else `await self._enqueue_write(payload)`. This prevents stream frames from interleaving with heartbeats or two replies stomping on each other. Stream pushes are throttled to `STREAM_PUSH_MIN_INTERVAL=1.0s` except `finish=true`, which always sends.

**msgid dedup.** WeCom can replay callbacks; `_LRU(500)` per-adapter dedups both `aibot_msg_callback` and `aibot_event_callback` by `msgid`.

**Group `@bot` stripping.** `_strip_mention_from_first_text` is applied inside `_resolve_inbound_blocks` to the **first text block of the current message ONLY** — never to a quote interior, and never to non-leading text blocks. This is a behavioural change from the pre-vision adapter (which used `_MENTION_RE` on the joined string and would strip any leading-position `@x ` regardless of which fragment it came from). The `[image, text("@bot hi")]` ordering is now stripped where it previously wasn't — intentional improvement. `_MENTION_RE` itself is still exported for the boss + tests.

**Vision content blocks.** The adapter no longer flattens inbound user content to a string. `IncomingMessage.content_blocks: list[ContentBlock]` carries an ordered mix of `{"type":"text","text":...}` and `{"type":"image","path":"./inbox/<file>"}` items; `inbound.text` is its `blocks_to_text` rendering (`[图:<basename>]` for images), kept for logging/dedup/stream previews. `ChatMessage.content` is `str | list[ContentBlock]`; only **user** messages ever carry list content. Persistence stores list content verbatim in `session.json` (legacy string content reads through unchanged). The compactor and OpenAI provider both branch on `isinstance(content, list)` and use `blocks_to_text` for any string-only path (token counting, summary input, tool/assistant/system messages). Out of scope: `file` / `video` / `voice` always degrade to text placeholders even inside vision turns; tool-result strings never become list content. **In the default `tool` vision strategy, the dispatcher pre-processes `content_blocks` through `vision_shim.apply_vision_strategy` before reaching the agent — image blocks are replaced with `[图:rel]\n<desc>` text, so what lands in `agent.history` is a flat string. List content only flows into history for roles that explicitly opt into `vision_strategy: direct`.**

**Vision strategy (eager OCR shim).** Default `settings.llm.vision.strategy = "tool"` runs every inbound image through `describe_images()` *before* `agent.handle` — pre-OCR'd descriptions land in the user message text, so `agent.history` is text-only and the compactor counts real token weight. The agent never sees raw images in tool mode (except via the `describe_image` tool when it wants a different prompt). Why eager-not-lazy: agent reliability ("forgot to call the tool") is eliminated, OCR-heavy multi-turn workloads see ~6× token savings, and the per-turn LLM call upcharge is repaid by cross-turn caching. **`direct` strategy** keeps the original list content and falls back to in-context vision (the pre-shim behaviour) — set `llm.vision_strategy: direct` on a role YAML for high-fidelity visual chat (e.g. art/diagram analysis). Invalid `vision_strategy` values silently fall back to the settings default with a warning. `default_eager_prompt` (in `settings.llm`) is the prompt fed to OCR — defaults to OCR-priority with a fallback short caption; users can override per-deployment in `~/.chat_team/config.yaml`. `default_eager_detail` defaults to `"high"` because `low` can't read small text.

**Split vision / chat providers.** 视觉/OCR 调用可以走独立的 API 端点。凭证在 `config.yaml` 的 `llm.vision.api_key` / `llm.vision.base_url` 配置，留空则回落至 `llm.api_key` / `llm.base_url`；也可通过 `OPENAI_VISION_API_KEY` / `OPENAI_VISION_BASE_URL` 环境变量设置（config.yaml 优先）。`llm.vision.model` 指定视觉模型名称（留空则复用 `llm.chat.model`）。当凭证与主模型相同时复用同一 `LLMProvider` 实例（不多开连接）。`app.build_vision_llm_provider` 处理此逻辑；`Dispatcher` 持有 `self._vision_llm` 并传给急切 OCR shim（`apply_vision_strategy`）和 `Agent` 构造（`describe_image` 工具经 `ToolContext.vision_llm` 使用）。Compactor 始终用聊天模型。

**Multi-API-key rotation (prefix-cache friendly).** `llm.api_keys: [sk-a, sk-b, ...]` (and `llm.vision.api_keys` for the vision provider) lets a single process rotate across N API keys. The unit of binding is `(session_id, role_name)` — *not* the workspace: different roles in the same session get **different** keys (their histories are isolated, so there's no shared prefix cache to lose). On the first request for a pair the round-robin pointer advances (`SessionKeyRouter`, `llm/key_rotation.py`) and that key is reused for every subsequent turn of the same pair so the upstream prefix cache stays warm. A binding is released after `llm.key_rotation.idle_reset_seconds` (default 600s, **10 min**) of no activity; the next request for that pair **continues the rotation** (advances the pointer — it does NOT re-pick the just-released key). Bindings are **in-memory only**: not persisted across restart, not hot-reloadable (`api_keys` / `key_rotation` are `requires_restart`, baked into constructed `AsyncOpenAI` clients at startup).

  **Failure handling — try every key, never disable.** On a failed call the provider iterates the bound key first, then every other key in round-robin order, each retried up to `llm.max_retries` (default 3) times → up to N×3 attempts before the last exception bubbles. Retryable errors (`APITimeoutError` / `APIConnectionError` / `RateLimitError` / `InternalServerError`) are retried with backoff on the *same* key first; non-retryable errors (401/403/etc) jump to the next key immediately (no 3× hammer). **No key is ever permanently disabled** — a key that fails stays eligible for future bindings. The binding itself never moves on failure: even if the call succeeds on a *different* key, the next turn for that `(session, role)` reuses the original bound key (unless it has since gone idle). This matches the maintainer's rule: "once bound, always this key; only idle reset advances; failures never permanently disable."

  **Vision shares the router.** When vision credentials are identical to the main provider's, `build_vision_llm_provider` returns `main_llm` itself — sharing not just the httpx connection pool but also the `SessionKeyRouter`, so eager OCR / `describe_image` calls reuse the same key the chat turns are using. Separate vision keys (`llm.vision.api_keys`) get their own router. All call paths already pass `session_id` + `role_name` on `CompletionRequest`, so the agent/compactor/vision-shim/describe-image all route through the same binding with zero call-site changes. Single-key installs (no `api_keys`, just `api_key`) get one client and `router=None` — identical behaviour to before, full backward compat.

**Image description cache.** `chat_team.llm.image_description_cache.ImageDescriptionCache` is a process-level LRU keyed by `(abs_path, mtime_ns, size, detail, model, prompt)` → description text. Caps: `MAX_ENTRIES=128`, `MAX_TOTAL_BYTES≈1MB`. Module-level singleton via `default_cache()`; same image with same prompt+detail+model is OCR'd exactly once across roles, sessions, and turns within a process. Different prompt or different file mtime/size invalidates. The cache is shared by both the eager shim AND the `describe_image` tool, so an agent re-querying with a custom prompt only pays for prompts not already cached.

**Image base64 cache.** `chat_team.llm.image_cache.ImageDataURICache` is a module-level LRU keyed by `(abs_path, mtime_ns, size, resize_long_side, resize_quality)` → `data:image/<mime>;base64,...`. Caps: `MAX_ENTRIES=32`, `MAX_TOTAL_BYTES≈32MB`. Per-image `max_inline_bytes` defaults to 6MB (raw — base64 ≈ 8MB, leaves headroom under OpenAI's ~10MB request limit); configurable via `config.yaml` `llm.vision.max_inline_bytes`. Missing file → `None`. Oversize behaviour depends on `llm.vision.oversized_image`: `"resize"` (default) auto-downscales the image so the longest dimension ≤ `resize_long_side` pixels (default 2048) and re-encodes as JPEG at `resize_quality` (default 85), then serves the resized data URI; `"reject"` returns `None` (text placeholder). Pillow is required for resize; without it, oversized images degrade to placeholders with a WARNING. RGBA images are composited onto white before JPEG re-encoding. The provider degrades missing/unrecoverable files to `[图:<name>(已丢失)]` / `(过大,已省略)` text blocks so a single bad image doesn't fail the whole turn. MIME map reuses `wecom_media.sniff_extension` (jpg/png/gif/webp; everything else → image/jpeg). `configure_default_cache()` in `app.py` wires the cache singleton to settings at startup.

**Quote (引用) flattening.** WeCom's `quote` field (sibling of `msgtype` on the body) can be text / image / mixed and is handled recursively by the same `_flatten_payload` that handles the current message. The resolver wraps the quote sequence between two text blocks `[引用开始]` / `[引用结束 — 以下为本条新消息]` and prepends them before the current-message blocks; `coalesce_text_blocks` then merges adjacent text spans. The @bot strip is applied to the current-side blocks BEFORE the quote sequence is concatenated, so a quote whose first item is `@bot` is never touched.

**`image_detail` knob.** Defaults to `settings.llm.vision.image_detail = "high"` (~1600 tokens per 1024² image on `gpt-4o`-class models). **Only matters in `vision_strategy: direct` mode** — in the default `tool` mode, the agent never receives raw images, so `image_detail` on the agent's role is moot. The eager shim itself uses `settings.llm.default_eager_detail` (also `"high"` by default). Override per role with `llm.image_detail: low|high|auto`. The provider stamps `detail` on every `image_url` part it builds; `image_base_dir` is plumbed from `session.cwd` so `./inbox/<file>` paths in history resolve correctly. Compactor token-counting drift (placeholder vs. real vision tokens) is no longer an issue in the default mode because history is text — only direct-mode roles still need to budget for it.

**Media decryption.** Each `image`/`file`/`video` payload carries its own per-URL `aeskey` (NOT the global EncodingAESKey from registration). Decode base64 → 32 bytes (AES-256). IV = first 16 bytes. AES-CBC + PKCS#7. The download URL is valid for 5 minutes — fetch immediately. Files land in `<cwd>/inbox/<ts>-<msgid>-<idx>.<ext>` (the `idx` suffix prevents same-second collisions on multi-image bursts inside one `mixed` message); extension comes from magic-byte sniffing (jpg/png/gif/webp/pdf/zip/mp4) or the msgtype default. The adapter's `workspace_resolver` callback (wired from `SessionManager.workspace_for`) decides where they go.

**Tool sandbox.** `_resolve_under(cwd, rel)` rejects absolute paths and `..`, then double-checks via `os.path.realpath` + `os.path.commonpath`. `list_dir` hides anything starting with `.chat_team` so the LLM doesn't see internal metadata. `run_command` runs through `bash -c` with `cwd=ctx.cwd`, hard timeout from settings, output truncated to `shell_output_max_bytes` (full log to `.chat_team/runs/<ts>-<rand>.log` so the LLM can re-read via `read_file` if needed).

**Private chat policy.** `private_chat` in `config.yaml` controls whether the bot replies to 1:1 single chats. Group chats always pass through — only `chattype == "single"` is gated. Four modes: `open` (reply to everyone — the *old* pre-feature behaviour), `closed` (no one), `blacklist` (everyone except `blacklist`), `whitelist` (only `whitelist`). **The default is `whitelist` with an empty list (default-deny).** That means a brand-new install OR an upgrade that doesn't add a `private_chat` block will NOT reply to any private chat until the maintainer explicitly opts in via `mode: open` or by populating `whitelist`. **This is a breaking change on upgrade** — pre-feature deployments replied to every private chat. The maintainer recovers by either setting `mode: open` (back to old behaviour) or adding their own `from.userid` to `whitelist`. Group chats and the `chat-team-boss` CLI are unaffected (the boss runs locally and never goes through the adapter). Identifier is the WeCom `from.userid` (i.e. `IncomingMessage.user_id`) — NOT the member's name and NOT their phone number. End users can't see their own userid in the WeCom client, so the maintainer needs to obtain it first. Three ways: (1) **self-service via `blocked_reply`**: the reply template supports a `{userid}` placeholder that's substituted with the sender's userid — set `blocked_reply: "你的账号是 {userid}"`, a colleague sends one private chat, and learns the exact string to hand back. (2) **Log scrape**: the gate logs `private chat blocked: user=<userid> mode=whitelist` at INFO to `~/.chat_team/logs/chat_team.log`. (3) **WeCom admin console**: 通讯录 → member → 账号. `blocked_reply` with an unknown placeholder (e.g. `{foo}`) renders literally instead of raising so a config typo can't take the reply path down. Invalid `mode` values are normalised: the YAML loader lowercases first, and anything not in the four-value enum falls back to `whitelist` with a WARNING (fail-closed so a typo can't accidentally expose the bot; the maintainer just adds their userid once). `blocked_reply` (string) is sent as a single `aibot_respond_msg` text frame to a blocked user; empty string = silent drop. The gate lives in `WeComBotAdapter._handle_msg_callback` BEFORE the `思考中…` stream frame is queued (so a blocked user never sees the spinner) and also in `_handle_event_callback` for `enter_chat` welcomes in single chats. Blocked single chats never create a `Session`, never reach the dispatcher — the gate is at the adapter boundary. `PrivateChatConfig.allows(user_id)` is the single source of truth and is unit-tested in `scripts/smoke_private_chat_policy.py`.

**Team profile injection.** `~/.chat_team/team.md` is read once by `load_settings` into `settings.team_profile` (stripped); when non-empty, `Agent._build_system_messages` splices it as a `[团队信息]` block alongside the role prompt and meta lines. Empty/missing file → no block, behaviour unchanged. The compactor's `_summarize` uses its own sterile system prompt (`compactor.py:100-107`) and is intentionally NOT touched. Hot-reloadable: `chat-team --reload` (or `kill -HUP <pid>`) re-reads `team.md` into `settings.team_profile` in place; the next turn's system-prompt rebuild picks it up without a restart. See "Hot reload" below.

**LLM debug log.** Opt-in: set `llm.debug_log_enabled: true` in `~/.chat_team/config.yaml` (default off — one file per call piles up fast and transcripts can carry sensitive user content, so production must stay off). When on, every call into `OpenAIChatCompletionProvider.complete` writes a JSON file to `<workspace>/.chat_team/llm/<ts>-<seq>-<role>-<kind>.json`. The record carries the full request payload (messages + tools + model + temperature + max_tokens), the response (content + tool_calls + finish_reason + usage from `completion.usage.model_dump()`), and `latency_ms`. Three `call_kind` values: `agent` (main turn), `compactor` (post-turn summary), `vision` (eager OCR shim + `describe_image` tool). Failures write the same file with `error=repr(exc)` and `response=null` before re-raising. **Base64 image data URIs are redacted** to `[redacted: <mime> <bytes> bytes]` via `chat_team.llm.debug_logger.redact_messages` — files stay grep-able. Per-session monotonic `seq` (process-local dict keyed by `session_id`) keeps filenames sortable when the millisecond clock collides. The provider's `_maybe_write_log` reuses the exact `messages_payload` it built for OpenAI (no re-serialisation), so what you see in the log is what the API saw. Writes are best-effort: a write failure is logged at WARNING and the call still returns normally.

**Log rotation (all log files are bounded).** Three logs live under ``~/.chat_team/logs/``, all rotated so the directory can't grow without bound:

  - ``chat_team.log`` (app log via Python logging) — ``RotatingFileHandler`` with ``logging.max_bytes`` (default 10 MB) × ``logging.backup_count`` (default 5); ~60 MB ceiling. Hot-reloadable.
  - ``chat_team.out`` (daemon stdout/stderr captured via ``os.open + dup2`` in ``chat_team.daemon.daemonize_and_run``) — a Python handler can't intercept an OS-level FD, so an in-process asyncio **copytruncate reaper** (``chat_team.out_rotator.OutFileRotator``, started in ``_async_main`` / ``_run_solo``) ``stat()``s the file every ``logging.out_check_interval_seconds`` (default 300 s) and, when the size crosses ``logging.out_max_bytes`` (default 10 MB), shifts ``.N → .N+1`` (oldest deleted), ``shutil.copy2``s the live file into ``.1``, then ``os.truncate(0)``s the live file. The open ``O_APPEND`` FD stays valid — every ``write(2)`` seeks to the current EOF before writing, so after truncation the daemon's next stdout/stderr write resumes at byte 0 of the same inode (no FD re-open, no lost writes). ``requires_restart`` because the reaper task is created at daemon startup with the then-current thresholds.
  - ``admin.log`` (admin panel audit log) — ``AuditLogger`` wraps a ``RotatingFileHandler`` with ``admin.audit_log_max_bytes`` (default 5 MB) × ``admin.audit_log_backup_count`` (default 5); ~25 MB ceiling. ``requires_restart`` (handler constructed once at admin-process startup).

Defaults bound the three log files to ~145 MB combined (``chat_team.log`` ~60 MB + ``chat_team.out`` ~60 MB + ``admin.log`` ~25 MB). Session workspace files (``inbox/``, ``.chat_team/runs/``, ``.chat_team/llm/``) are bounded separately by the janitor (``cleanup.max_age_days`` = 14, see ``SessionManager._maybe_sweep``). Setting any ``max_bytes``/``backup_count`` to ``0`` disables rotation for that file (not recommended). Smoke: ``scripts/smoke_out_rotator.py`` covers the copytruncate reaper's FD-preservation property; ``scripts/smoke_admin.py`` covers the audit log rotation.

**Log rotation (all log files are bounded).** Three logs live under ``~/.chat_team/logs/``, all rotated so the directory can't grow without bound:

  - ``chat_team.log`` (app log via Python logging) — ``RotatingFileHandler`` with ``logging.max_bytes`` (default 10 MB) × ``logging.backup_count`` (default 5); ~60 MB ceiling. Hot-reloadable.
  - ``chat_team.out`` (daemon stdout/stderr captured via ``os.open + dup2`` in ``chat_team.daemon.daemonize_and_run``) — a Python handler can't intercept an OS-level FD, so an in-process asyncio **copytruncate reaper** (``chat_team.out_rotator.OutFileRotator``, started in ``_async_main`` / ``_run_solo``) ``stat()``s the file every ``logging.out_check_interval_seconds`` (default 300 s) and, when the size crosses ``logging.out_max_bytes`` (default 10 MB), shifts ``.N → .N+1`` (oldest deleted), ``shutil.copy2``s the live file into ``.1``, then ``os.truncate(0)``s the live file. The open ``O_APPEND`` FD stays valid — every ``write(2)`` seeks to the current EOF before writing, so after truncation the daemon's next stdout/stderr write resumes at byte 0 of the same inode (no FD re-open, no lost writes). ``requires_restart`` because the reaper task is created at daemon startup with the then-current thresholds.
  - ``admin.log`` (admin panel audit log) — ``AuditLogger`` wraps a ``RotatingFileHandler`` with ``admin.audit_log_max_bytes`` (default 5 MB) × ``admin.audit_log_backup_count`` (default 5); ~25 MB ceiling. ``requires_restart`` (handler constructed once at admin-process startup).

Defaults bound the three log files to ~145 MB combined (``chat_team.log`` ~60 MB + ``chat_team.out`` ~60 MB + ``admin.log`` ~25 MB). Session workspace files (``inbox/``, ``.chat_team/runs/``, ``.chat_team/llm/``) are bounded separately by the janitor (``cleanup.max_age_days`` = 14, see ``SessionManager._maybe_sweep``). Setting any ``max_bytes``/``backup_count`` to ``0`` disables rotation for that file (not recommended). Smoke: ``scripts/smoke_out_rotator.py`` covers the copytruncate reaper's FD-preservation property; ``scripts/smoke_admin.py`` covers the audit log rotation.

## Slash commands (chat-side, 企业微信端)

Four user-facing slash commands live entirely in
`WeComBotAdapter._handle_msg_callback` — intercepted **after** the
`private_chat` gate but **before** `_enqueue_inbound_turn`, so they bypass
the per-session inbound queue entirely. A slash command never creates a
`Session`, never enters the dispatcher, and never makes an LLM call: it
replies with one `aibot_respond_msg` stream frame (via
`WeComStreamHandle.finish`) and returns.

| Command | Scope | Behaviour |
|---|---|---|
| `/new` | group + private | Clear every role's conversation history for this session, **preserve `current_role` and all workspace files** (`inbox/`, `.chat_team/runs/`, `.chat_team/llm/`, `notebook.md`). Refuses with "请先 /stop" if a turn is in flight — the user must explicitly stop a running task before resetting. |
| `/stop` | group + private | Cancel the running inbound-worker task for this session (`asyncio.Task.cancel()`) **and** drain the per-session inbound queue (`drain_pending_turns`) so any turns queued behind the cancelled one don't immediately re-fire. Idle → "当前没有正在执行的任务". |
| `/status` | group + private | "🟢 正在执行任务（角色: X）" or "⚪ 当前空闲。" Reads `Dispatcher._busy_sessions` — never touches `session.lock`. |
| `/running` | **private only** | Enumerate `Dispatcher.busy_group_sessions()` (filters by the `wecom-group-` session-id prefix) and reply with the count + the chatid list. Sent in a group → refused with "/running 仅在私聊中可用". |

### Trigger rules

- **Group chats** require an `@bot` mention before the command
  (`@bot /new`), matching the existing group-@bot contract for normal
  conversation. A bare `/new` in a group is treated as ordinary user text
  and forwarded to the agent. This prevents a member typing "/new"
  mid-sentence from resetting the session.
- **Private chats** need no `@bot` (the `private_chat` gate already
  controls who can reach the slash layer; default-deny whitelist means a
  brand-new install answers no slash commands from private chats either
  until the maintainer opens the gate).
- Word-boundary anchored: `/newton` does **not** match `/new`. Case-
  insensitive. Only `msgtype == "text"` is considered — images/mixed/
  voice can never be slash commands.

### Why `/stop` is safe (history rollback)

`asyncio.CancelledError` is `BaseException`, not `Exception`. The agent's
tool loop has `try: ... except TransferRequested: raise; except BaseException:
del self.history[pre_turn_len:]; raise` — so a `/stop` mid-LLM-call or
mid-tool-loop rolls back everything the agent appended this turn (the
pending user message, any half-finished `assistant(tool_calls)`, any
unanswered tool results). Without this, the next turn would either replay
a dangling user message or 400 the OpenAI request with an
`assistant(tool_calls)` whose tool replies never landed. The dispatcher's
own `_handle_locked` `finally` resets turn counters; the outer `handle`
`finally` clears the busy-state entry — so `/status` immediately after
`/stop` reports idle.

### Busy-state tracking (`Dispatcher._busy_sessions`)

`Dispatcher.handle` sets `_busy_sessions[session_id] = current_role`
**before** acquiring `session.lock` (so `/status` and `/running` see an
accurate picture even when a turn is queued behind a previous turn waiting
on the lock) and clears it in a `finally` that runs even on
`CancelledError`. `is_busy` / `current_role_for` / `busy_group_sessions`
are pure dict reads — they never take `session.lock`, so they can't
deadlock against a long-running turn.

### `reset_session_history` semantics

Clears `session.agents_by_role` (forces `_agent_for` to re-materialise on
the next turn from the now-empty `restored_histories`), clears
`session.restored_histories`, then writes `session.json` atomically via
`persistence.write_atomic` with `histories == {}` and the **same**
`current_role`. Any pending debounced flush is cancelled first so it
can't clobber the reset with a stale snapshot. Does NOT touch
`notebook.md` — the team whiteboard persists across `/new` (it's a shared
cross-role fact store, not conversation history).

### Adapter ↔ Dispatcher coupling

The adapter reaches the dispatcher via `self._handler.__self__` (the bound
method's instance) — `_dispatcher_ref()` duck-types for the four command
methods and returns `None` if the handler is a test fake. Slash commands
degrade to "命令不可用（未连接调度器）" when no dispatcher is wired, so
existing test fakes that register a plain coroutine keep working unchanged.

**Key files:** `adapters/wecom.py` (`_match_slash_command`,
`_handle_slash_command`, `drain_pending_turns`, `_cancel_running_turn`,
`_reply_slash`, `_dispatcher_ref`, the `_SLASH_CMD_RE` interception block
in `_handle_msg_callback`), `dispatcher.py` (`_busy_sessions`,
`is_busy`, `current_role_for`, `busy_group_sessions`,
`reset_session_history`, the `handle`→`_handle_locked` split for busy-state
cleanup), `agent/agent.py` (`except BaseException` rollback),
`scripts/smoke_slash_commands.py` (8-case coverage).

## Hot reload (`chat-team --reload` / SIGHUP)

Most config can be changed without dropping the WebSocket. Send SIGHUP to the
daemon (`chat-team --reload`, or `kill -HUP $(cat ~/.chat_team/chat_team.pid)`)
and the running process re-reads `config.yaml` + `team.md` + `roles/*.yaml` +
`skills/*/` and applies the changes **in place** — no reconnect, no session
loss. The result is logged to `~/.chat_team/logs/chat_team.log` as
`hot reload result: ...`.

**What is hot-reloadable** (mutated on the live `Settings` / registries,
picked up on the next read):

- `private_chat.*` — read on every inbound message; whitelist/blacklist/mode
  changes take effect on the next message.
- `session.*` (transfer cap, progress text/interval, debounce,
  `max_in_memory_sessions`), `notebook.max_bytes`, `cleanup.*`,
  `log_level` / `logging.*` — re-read on next use; logging is reconfigured
  immediately (handlers cleared + re-added, no duplicate lines).
- `llm.chat.{model,temperature,reasoning_effort,history_token_budget}`,
  `llm.vision.{image_detail,strategy,model,oversized_image,resize_*,
  max_inline_bytes}` — read per turn; the image-cache singleton is recreated
  with the new resize knobs.
- `llm.{debug_log_enabled,use_streaming,max_retries,retry_initial_delay}` —
  pushed onto the live `OpenAIChatCompletionProvider` via
  `apply_runtime_overrides` (chat + vision providers).
- `team.md` (`team_profile`) — next turn's system-prompt rebuild picks it up.
- `default_role` — affects new sessions only (existing sessions keep their role).
- `roles/*.yaml` — `RoleRegistry.reload_in_place` atomically swaps the dict;
  `transfer_to_employee`'s enum is refreshed; every live in-memory `Agent`'s
  `role` attribute is swapped to the reloaded `Role` (history untouched). New
  roles become transferable immediately; deleted roles leave existing agents
  running with their frozen role until the session ends (counted as
  `agents_role_orphaned`).
- `skills/*/` — `SkillRegistry.reload_in_place` mirrors the above.

**What requires a restart** (reported as `requires_restart`, NOT applied —
these are bound to live OS resources that can't be swapped mid-flight):

- `mode` (team↔solo), `bots[].{bot_id,secret}` (open WebSocket connections),
  `workspace_root`, `mcp.servers` (subprocess/SSE lifecycle).
- `llm.{api_key,api_keys,base_url,key_rotation,request_timeout_seconds,http_debug_log_enabled}` —
  baked into the constructed `AsyncOpenAI` + `httpx` client at startup; swap
  them by editing `config.yaml` then `--stop && chat-team`.
- `llm.vision.{api_key,api_keys,base_url}` — same (vision provider is constructed once).

**Trigger options.** `chat-team --reload` reads the pid file and sends SIGHUP.
`kill -HUP <pid>` works directly. The SIGHUP handler is **non-cancelling**:
unlike SIGTERM/SIGINT it does NOT touch the main task, so in-flight turns and
the WebSocket are untouched. Concurrency is best-effort atomic at the
attribute level (GIL): a turn in flight may see pre- or post-reload config
for a single read but never corrupted state; we deliberately do not acquire
every session lock (hundreds) for a reload.

**Bad YAML is safe.** If `config.yaml` fails to parse, `reload_settings`
populates `report.errors`, mutates nothing, and logs `reload FAILED` — the
running process keeps its previous config. Fix the YAML and `--reload` again.

**Key files:** `config.py` (`reload_settings`, `ReloadReport`),
`roles/registry.py` + `skills/registry.py` (`reload_in_place`),
`agent/tools/transfer_tool.py` (`update_employees`),
`llm/openai_provider.py` (`apply_runtime_overrides`),
`reload.py` (`Reloader` orchestrator + `CombinedReloadReport`),
`app.py` (SIGHUP wiring in `_run_with_shutdown` + `--reload` CLI),
`daemon.py` (`reload_daemon`), `scripts/smoke_reload.py` (coverage).

## Adding a role / tool

**Role** — drop a YAML in `src/chat_team/roles/builtin/` (committed) or `~/.chat_team/roles/` (user override, takes precedence). Required fields: `name`, `system_prompt`, `tools` (a subset of registered tool names). Optional: `display_name`, `welcome_message` (used for `enter_chat`), `llm.{model,temperature,history_token_budget,image_detail,vision_strategy}`, `mcp_servers` (list of MCP server names from `config.yaml`). `vision_strategy: tool|direct` overrides the global default — set `direct` on a role that needs raw images in context (e.g. art critique). No code changes needed — `RoleRegistry.load` picks it up and `transfer_to_employee`'s enum is rebuilt from `roles.names()`. Hot-reloadable: `chat-team --reload` re-scans `roles/`, atomically swaps `RoleRegistry`'s internal dict, refreshes the `transfer_to_employee` enum, and swaps `agent.role` on every live in-memory agent (history preserved) so the next turn's prompt rebuild picks up new prompts/tools. Deleted roles: running agents keep their frozen role until the session ends (counted as `agents_role_orphaned`). See "Hot reload" below.

**Tool** — subclass `Tool` (`src/chat_team/agent/tools/base.py`), set `name`/`description`/`parameters` (JSON schema), implement `async run(ctx, **kwargs)`. Register in `app.build_tool_registry`. Reference it in role YAMLs that should expose it. Raise `ToolError` for recoverable failures (returned to the LLM as a tool message); raise `TransferRequested` only if you're implementing role-switch semantics. If your tool needs to call the LLM (e.g. vision/embedding), read `ctx.llm` — `ToolContext.llm` is wired to `agent.llm` so the tool reuses the same provider configuration.

**Skill** — a no-code capability pack: drop a directory at `~/.chat_team/skills/<name>/` containing `SKILL.md` (YAML frontmatter + markdown body) plus optional auxiliary files. Frontmatter must have `name` (must equal the directory name) and `description` (single line preferred — multi-line works but only the first line lands in the system-prompt TOC). Body is whatever instructions you want the agent to follow when invoked. To expose a skill to a role, list it under `skills:` in the role YAML AND include `skill` (and optionally `skill_read_file`) in `tools:`. The agent sees a `[可用 skills] - name: description` block in its system prompt, fetches the body via `skill(name=...)`, and reads aux files via `skill_read_file(skill=..., path=...)`. `SkillRegistry.load` mirrors `RoleRegistry.load`: builtin (`src/chat_team/skills/builtin/`) first, user dir overrides by name. Malformed skills (missing/invalid frontmatter, name/dir mismatch, missing SKILL.md) are logged at WARNING and skipped — one bad dir won't break the rest. Per-role gating happens twice: once when rendering the TOC (filtered to `role.skills ∩ registry.names()`) and again at tool invocation in `SkillTool.run`. The `enum` on the JSON-schema parameters is the full registry (one tool instance for all roles), so the runtime check is the real gate. Hot-reloadable via `chat-team --reload` (SIGHUP) — `SkillRegistry.reload_in_place` atomically swaps the internal dict, so existing `SkillTool`/`SkillReadFileTool` references see new/changed skills on the next call. See "Hot reload" below.

**Python deps for skills (uv + PEP 723).** SKILL.md format is deliberately kept 100% compatible with community skills (frontmatter only carries `name` + `description`), so per-skill dependency declarations are out of scope. Instead: when a role's `tools` contains **both** `skill` and `run_command`, `Agent._build_system_messages` splices in the `PYTHON_UV_CONVENTION` block (`agent/agent.py`) — it tells the agent to write Python scripts with PEP 723 inline metadata (`# /// script\n# dependencies = [...]\n# ///`) and run them via `uv run script.py`. `uv` resolves deps into its global content-addressed env cache (`~/.cache/uv/environments-v2/<hash>/`), shared across sessions/roles/workspaces with zero per-workspace state. `app.warn_if_uv_missing` logs a WARNING on startup if any loaded role would need `uv` but it isn't on PATH; the bot still runs (non-Python skills unaffected). Why not per-workspace venv: 100 sessions × 100MB site-packages is wasteful in a multi-tenant bot. Why not `pip_install` tool: reactive install-on-import-fail costs a full LLM turn per missed dep.

## MCP (Model Context Protocol)

MCP 让角色无需写 Python 即可使用外部工具服务器。用户在 `config.yaml` 声明 MCP 服务器，在角色 YAML 的 `mcp_servers` 字段引用，启动时自动发现并注册工具。

**配置。** `config.yaml` 新增 `mcp:` 节，字典风格（与 Claude Desktop 等 MCP 客户端一致）：

```yaml
mcp:
  servers:
    filesystem:                          # 服务器名 — 角色 YAML 中引用
      command: npx                       # stdio transport: 启动子进程
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    github:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:                               # 传给子进程的额外环境变量
        GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
    remote_api:
      url: http://localhost:8080/sse     # SSE transport: 连接远程服务
```

Transport 自动推断：有 `command` → stdio，有 `url` → SSE，二选一。服务器名须匹配 `[a-zA-Z0-9_-]+` 且不含 `__`（双下划线用于注册名分隔）。

**角色引用。** 在角色 YAML 中添加 `mcp_servers` 字段，列出要使用的服务器名：

```yaml
name: developer
tools: [read_file, write_file, run_command, transfer_to_employee]
mcp_servers: [filesystem, github]        # 该角色可使用这两个 MCP 服务器的所有工具
```

**注册名。** MCP 工具注册为 `mcp__<server>__<tool>`（如 `mcp__filesystem__read_file`），符合 OpenAI function name regex。`Agent._effective_tool_names()` 在每轮调用 LLM 时自动展开 `role.mcp_servers` 为具体工具名，与内置工具合并后传给 `ToolRegistry.specs_for()`。

**生命周期。** `app._async_main` 在 `build_dispatcher` 之前调用 `McpClientManager.connect_all()` 连接所有配置的服务器、发现工具、创建 `McpProxyTool` 实例，然后通过 `build_tool_registry(extra_tools=...)` 注入 `ToolRegistry`。`finally` 块调用 `close_all()` 关闭连接。单个服务器连接失败记 WARNING 并跳过，不阻塞启动。

**工具调用。** `McpProxyTool.run()` 调用 MCP SDK 的 `session.call_tool()`，结果中的 `TextContent` 直接取 `.text`，`ImageContent` 降级为 `[image: mime]` 占位符。MCP 返回 `isError=True` 或底层异常均包装为 `ToolError`，走已有的 agent 错误处理流程。

**关键文件：** `src/chat_team/mcp/config.py`（配置 dataclass）、`src/chat_team/mcp/client.py`（`McpClientManager` 生命周期）、`src/chat_team/mcp/proxy_tool.py`（`McpProxyTool(Tool)` 桥接）。

**限制。** 当前只支持 MCP Tools，不支持 Resources / Prompts。不支持热加载——修改 MCP 配置需重启 bot（MCP 子进程/SSE 连接的 lifecycle 不能在运行中安全替换；`chat-team --reload` 会把 `mcp.servers` 的变更报为 `requires_restart` 而非应用）。`chat-team-tools` CLI 不列出 MCP 工具（它们是运行时动态发现的）。

## Solo mode (一 bot 一角色)

`config.yaml` 设置 `mode: solo` + `bots:` 列表即可启用。一个进程内为每个 bot 开一条 WebSocket 连接，各自绑定一个角色，不注册 `transfer_to_employee` 工具。

```yaml
mode: solo
bots:
  - name: research_engineer   # 必须匹配 roles/ 下的角色名
    bot_id: "BOT_ID_1"
    secret: "SECRET_1"
  - name: customer_service
    bot_id: "BOT_ID_2"
    secret: "SECRET_2"
```

**关键设计点：**

- **每 bot 独立 Dispatcher + SessionManager**（`Dispatcher(fixed_role=...)` 跳过 transfer 循环）。
- **共享 notebook**：同一群聊 → 同一 workspace 目录 → 各 bot 的 `Notebook` 实例指向同一个 `notebook.md`。asyncio 单线程保证 read→modify→write 不会被打断。
- **隔离持久化**：每 bot 写 `session-{role}.json`（`Session.state_filename`），互不冲突。
- **系统提示调整**：solo 模式下隔离规则改为"其他机器人各自维护自己的对话,你们通过团队记事本共享事实"。
- **向后兼容**：`mode` 默认 `team`，所有新增参数都有默认值，现有部署零改动。

**关键文件：** `config.py`（`BotConfig`）、`app.py`（`_run_solo`）、`dispatcher.py`（`_run_turn_solo`）、`session/manager.py`（`solo_role`）、`session/persistence.py`（参数化 filename）。

## Things to not break

- **Don't put system messages into `agent.history`.** System content is rebuilt every turn by `Agent._build_system_messages` (role prompt + notebook TOC + pending one-shot injects). The history list is for `user`/`assistant`/`tool` only. The compactor's summary head is the sole exception and is intentional — but `_find_keep_boundary` must continue to return 0 when the head is already a system summary, otherwise we'd compact a compaction.
- **Don't call `ws.send` directly from anywhere except the writer task.** Use `_enqueue_write`.
- **Don't share notebooks or histories across sessions.** Each `Session` instance owns its own `Notebook` pointed at its own `notebook.md`; SessionManager keys by raw `session_id` (sanitization is path-only).
- **Don't catch `TransferRequested` in tools.** Only `Agent.handle` (which closes the dangling tool_call) and `Dispatcher._run_turn` (which acts on it) should see it.
- **Don't put `BOSS_ROLE` into `roles/builtin/` or any user role dir.** It would be picked up by `RoleRegistry.load` and leak into `transfer_to_employee`'s enum + WeCom's enter_chat flow. Boss must stay hardcoded in `boss.py` and registered nowhere.
- **Don't add boss-side tools (`team_tools.py`) to `app.build_tool_registry`.** They sidestep the cwd sandbox by design and would let any WeCom-side role overwrite arbitrary files under `~/.chat_team/`.
