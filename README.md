# chat_team

[中文文档](./README.zh-CN.md)

WeCom (Enterprise WeChat) AI Bot — supports two deployment modes:

- **Team Mode** (`mode: team`, default): A single bot fronts an extensible "virtual employee team". The backend dynamically transfers sessions to the most appropriate role as needed.
- **Solo Mode** (`mode: solo`): One bot = one role. Multiple bots run in the same process and share information via a shared notebook.

**Only `team_admin` (receptionist/admin) is built-in.** Other roles are user-defined via YAML files in `~/.chat_team/roles/`. Sample roles (`research_engineer`, `customer_service`) are provided in `docs/examples/roles/` for reference. Each chat session has its own isolated working directory.

## Features

- **WebSocket Long Connection**: No public callback URL required; stays online with application-layer heartbeat.
- **Streaming Replies**: Users see a "thinking..." placeholder that updates progressively until `finish=true`.
- **Role = YAML File**: Add roles without modifying dispatch code; place in `~/.chat_team/roles/` to override built-ins.
- **Session-Level Tool Sandbox**: File I/O and shell commands are restricted to `~/.chat_team/workspaces/<sid>/`. Path traversal, absolute paths, and symlink escapes are rejected. Shell output exceeding thresholds is truncated, with full logs saved for review.
- **Shared Notebook**: A "team whiteboard" across employees — Markdown file with `## key` blocks. History only sees the TOC; values are fetched on-demand via `notebook_read` to avoid context pollution.
- **Auto Compaction**: Independent token budget per role. When exceeded, early messages are summarized by LLM and prepended to history.
- **Persistence**: `session.json` is debounced (10s) and written atomically. Session history and current active role survive restarts.
- **Media Handling**: Images/files/videos are decrypted using per-message AES-256-CBC keys and saved to `<cwd>/inbox/`. Agents receive text pointers.
- **Eager Image OCR**: Under default `vision_strategy=tool`, inbound images are **automatically OCR'd upfront**. Descriptions are injected as `[image:relative_path]\n<description>` into user messages, keeping agent history text-only and token consumption predictable. Raw images only enter context when a role explicitly sets `vision_strategy: direct`.
- **Solo Mode**: One bot per role, multiple WebSocket connections in one process. Bots in the same group chat share a notebook while maintaining isolated conversation histories (`session-{role}.json`).

## Installation

Requires Python ≥ 3.11. Recommended to install in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Re-run `pip install -e .` only when modifying `pyproject.toml`, adding non-Python resources, or rebuilding the venv. Source code changes don't require reinstallation (editable install symlinks `src/chat_team` into `site-packages`).

**Optional: Install `uv`** (`curl -LsSf https://astral.sh/uv/install.sh | sh`). When a role has both `skill` and `run_command` tools, the system injects PEP 723 + `uv run` conventions into its prompt. Agent-written Python scripts can automatically fetch third-party dependencies without polluting the host environment. The bot runs without `uv`, but community skills requiring third-party Python libraries won't work — a WARNING is logged at startup.

## Configuration

On first launch, default configuration is generated under `~/.chat_team/`:

```
~/.chat_team/
  config.yaml      # Global parameters + credentials (chmod 0600); auto-generated on first run
  team.md          # Global team profile; injected into every employee's system prompt each turn if non-empty; requires restart after editing
  roles/           # User-defined role YAMLs; same-name overrides built-ins
  workspaces/      # One subdirectory per session
  logs/            # chat_team.log + chat_team.out + admin.log, all rotating (see Log rotation in AGENTS.md)
  state/           # Cross-session state reservation
```

Fill in your WeCom BotID/Secret and OpenAI API key in `~/.chat_team/config.yaml`:

```yaml
bots:
  - bot_id: "..."
    secret: "..."

llm:
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"   # Can be replaced with internal proxy / vLLM / Ollama gateway
```

Credentials can also be set via environment variables (`WECOM_BOT_ID` / `WECOM_SECRET` / `OPENAI_API_KEY` / `OPENAI_BASE_URL`); values in `config.yaml` take precedence.

For testing, use `CHAT_TEAM_HOME=/tmp/chat_team_dev` to relocate the entire root directory and avoid polluting the real environment.

Model-related configuration in `config.yaml` is split into two groups:

```yaml
llm:
  chat:
    model: gpt-4o-mini
    temperature: 0.3
    history_token_budget: 12000
    reasoning_effort: ""   # low | medium | high, empty = model default
  vision:
    model: ""              # Empty = reuse chat.model
    strategy: tool         # tool | direct
    image_detail: high
    reasoning_effort: ""   # low | medium | high, empty = model default
```

### Solo Mode Configuration

Don't need `team_admin` as a front-desk router? Want each bot to focus on a single role? Switch in `config.yaml`:

```yaml
mode: solo
bots:
  - name: research_engineer    # Must match a role name under roles/
    bot_id: "YOUR_BOT_ID_1"   # BotID from WeCom admin console
    secret: "YOUR_SECRET_1"
  - name: customer_service
    bot_id: "YOUR_BOT_ID_2"
    secret: "YOUR_SECRET_2"
```

Startup command remains unchanged (`python main.py`). In Solo mode:
- Each bot only responds using its bound role; `transfer_to_employee` is never triggered.
- Multiple bots in the same group share facts via `notebook_write` / `notebook_read`.
- Conversation histories are isolated per bot (stored separately as `session-{role}.json`).

### LLM Debug Logging

When troubleshooting issues like "wrong tool routing / compactor summary drift / weird vision OCR results", the limited info in `chat_team.log` may not suffice. Set `llm.debug_log_enabled` to `true` in `~/.chat_team/config.yaml` to log **complete request + response** as JSON files for every OpenAI call:

```
~/.chat_team/workspaces/<sid>/.chat_team/llm/<timestamp>-<seq>-<role>-<kind>.json
```

Three `kind` values: `agent` (main role turn) / `compactor` (compression call) / `vision` (image OCR shim and `describe_image` tool). Each record includes `messages` / `tools` / `model` / `temperature` / `reasoning_effort` / `response` / `finish_reason` / `usage` (token counts) / `latency_ms`. On failure, an additional `error=repr(exc)` field is added with `response=null`.

- **Image base64 data is automatically redacted** to `[redacted: <mime> <bytes> bytes]`; logs are safe to grep.
- **Disabled by default** — multiple calls per turn + compaction + OCR accumulate files quickly. **Do not enable in production**; messages contain raw user conversations with privacy concerns.
- Write failures only produce a WARNING in `chat_team.log` and do not affect the main flow.

Set back to `false` and restart after debugging.

### Team Profile (`team.md`)

`~/.chat_team/team.md` is free-form markdown text. If non-empty, it is read at startup and injected as a `[Team Info]` block into **every turn** of every virtual employee's system prompt — ensuring all roles know which company/team they serve without duplicating boilerplate in each role YAML.

- Edits **require restarting** `chat-team` to take effect (configuration is loaded once at startup).
- To disable, simply empty the file; behavior is identical to "not configured".
- Keep content under 300 words to avoid inflating per-turn API token consumption.

## Running

```bash
python main.py
# Equivalent to:
chat-team
```

After `pip install -e .`, the `chat-team` command is registered (via `[project.scripts]` in `pyproject.toml`) for single-command distribution.

Normal startup should show:

```
INFO chat_team.app | chat_team starting; home=/Users/<you>/.chat_team
INFO chat_team.adapters.wecom | connecting to wss://openws.work.weixin.qq.com
INFO chat_team.adapters.wecom | subscribe ok: {...errcode: 0...}
```

You can then @mention the bot or start a private chat in WeCom. Logs are written to both stderr and `~/.chat_team/logs/chat_team.log` (rotating 10 MB × 5 files); the daemon's stdout/stderr land in `chat_team.out` (rotated in-process by the OutFileRotator copytruncate reaper, default 10 MB × 5); admin audit events go to `admin.log` (rotating 5 MB × 5). All three are bounded — see "Log rotation" in `AGENTS.md`.

## `chat-team-boss` (Configuration Assistant)

Prefer not to hand-write YAML? After `pip install -e .`, run:

```bash
chat-team-boss
```

This launches a **CLI conversational interface**: describe the virtual employees/team profile you want in natural language, and the boss agent reads/writes `~/.chat_team/team.md` and `~/.chat_team/roles/*.yaml` for you. It will paste the full proposed YAML/markdown for your review and explicitly ask "Confirm write?" before persisting to disk.

- **Sessions are not persisted**: Each invocation starts fresh; true "memory" lives in the files written to disk.
- **Not connected to WeCom**: The boss role exists only in CLI; it **does not** appear in the `transfer_to_employee` enum and is never triggered by enter_chat.
- **Done when ready**: Exit with `Ctrl+D` or `/quit`, then start the main bot with `chat-team` for changes to take effect (role registry is loaded once at main process startup).

Requires `OPENAI_API_KEY` (same as main bot; read from `llm.api_key` in `~/.chat_team/config.yaml` or environment variable).

## List Available Tools

When hand-writing role YAMLs, the `tools:` field must reference tool names registered in the main process. To see the current list of available tools:

```bash
chat-team-tools
# Or without installing console script: python -m chat_team.list_tools
```

Outputs tools sorted by name with a one-line description. This list shares the same source as `list_available_tools` in `chat-team-boss` — new tools become visible in both automatically without manual maintenance.

## Adding a Role

Don't want to hand-write? Use `chat-team-boss` (see previous section) to generate it. Manual example:

```yaml
# ~/.chat_team/roles/data_analyst.yaml
name: data_analyst
display_name: Data Analyst
description: Handles SQL, reporting, and data exploration requests.
system_prompt: |
  You are a data analyst named "Xiao Shu". Before answering, use read_file to check sample data in the current directory.
tools:
  - read_file
  - list_dir
  - run_command
  - notebook_read
  - notebook_write
  - transfer_to_employee
welcome_message: Hello, I'm Xiao Shu, your data analyst. Feel free to ask any data-related questions.
llm:
  model: gpt-4o-mini
  temperature: 0.2
  history_token_budget: 16000
```

No restart needed — actually, yes, restart is required since the role registry loads at process startup. But **no code changes** are needed, and no re-installation via `pip install` is necessary. The `transfer_to_employee` target enum will automatically include `data_analyst` on next startup.

## Adding a Tool

Subclass `Tool` under `src/chat_team/agent/tools/`, implement `async run(ctx, **kwargs)`, register it in `app.build_tool_registry` via `reg.register(...)`, and reference the name in role YAML `tools:`. Raise `ToolError` for recoverable errors (returned to LLM as a tool message).

## Smoke Tests (No LLM / Network Dependency)

```bash
python scripts/smoke_dispatch.py                  # Dispatcher + Agent + Tools
python scripts/smoke_transfer.py                  # Multi-hop transfer + cap + unknown target
python scripts/smoke_tools.py                     # File / Shell + Sandbox
python scripts/smoke_wecom_parse.py               # Adapter parsing + LRU + Stream frame shape
python scripts/smoke_compaction_persistence.py    # Tiktoken compaction + session.json round-trip
python scripts/smoke_media_events.py              # AES-256-CBC + enter_chat / disconnected
python scripts/smoke_llm_debug_log.py             # Debug log: image base64 redaction + file write + per-session seq
python scripts/smoke_solo.py                      # Solo mode: independent dispatcher + shared notebook + isolated persistence
```

Each smoke test points `CHAT_TEAM_HOME` to `/tmp/...` and runs `rmtree` at startup, making repeated runs safe.

## Extensibility: MCP & Skills

### MCP (Model Context Protocol)

Integrate external tool servers without writing Python. Declare servers in `config.yaml`, reference them in role YAML, and tools are auto-discovered and registered at startup.

**Configuration Example** (`~/.chat_team/config.yaml`):

```yaml
mcp:
  servers:
    filesystem:                          # Server name referenced in role YAML
      command: npx                       # stdio transport
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    github:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
    remote_api:
      url: http://localhost:8080/sse     # SSE transport
```

Transport is auto-detected: `command` → stdio, `url` → SSE. Server names must match `[a-zA-Z0-9_-]+` and cannot contain `__`.

**Role Reference** (Role YAML):

```yaml
name: developer
tools: [read_file, write_file, run_command, transfer_to_employee]
mcp_servers: [filesystem, github]        # This role can use all tools from these two servers
```

MCP tools are registered as `mcp__<server>__<tool>` (e.g., `mcp__filesystem__read_file`). Agents invoke them using the original tool name without needing to know the prefix.

**Limitations**: Currently only supports MCP Tools (not Resources / Prompts); MCP config changes require bot restart; `chat-team-tools` CLI does not list MCP tools (dynamically discovered at runtime).

### Skills (Code-Free Capability Packs)

Add instruction-based capabilities to roles via drop-in directories without writing Python tools.

**Directory Structure**: `~/.chat_team/skills/<name>/SKILL.md` + optional auxiliary files.

**SKILL.md Format**:

```markdown
---
name: data_analysis          # Must equal directory name
description: SQL query and report generation guide   # Single-line description injected into system prompt TOC
---

# Data Analyst Skill

When users request data analysis, follow these steps:
1. Use `list_dir` to check available data files
2. Execute pandas scripts via `run_command`
...
```

**Enable in Role**: Declare both `skills:` and `tools: [skill, skill_read_file]` in role YAML:

```yaml
name: data_analyst
skills: [data_analysis]
tools: [skill, skill_read_file, read_file, run_command, transfer_to_employee]
```

Agents see `[Available skills] - data_analysis: SQL query and report generation guide` in their system prompt. They fetch full instructions via `skill(name="data_analysis")` and read auxiliary files via `skill_read_file(skill="data_analysis", path="template.sql")`.

**Python Dependencies**: When a role has both `skill` and `run_command` tools, the system automatically injects PEP 723 + `uv run` conventions. Agent-written Python scripts can declare inline dependencies resolved by `uv`'s global cache without polluting the host environment. Non-Python skills work normally even without `uv` installed.

**Note**: Skills do not support hot reloading; restart the bot after adding or modifying skills.

## FAQ

- **Why aren't my changes to `team.md` / Role YAML / Skills / MCP config taking effect?**
  All configurations are loaded once at startup. Restart `chat-team` after modifications.

- **How do bots perceive notebook updates from each other in Solo mode?**
  There is no active notification mechanism currently. After Bot A writes to the notebook, Bot B must explicitly call `notebook_read` in its next turn to fetch the latest values. For strong synchronization, consider agreeing in prompts to explicitly remind others to check after updating critical facts.

- **How to troubleshoot MCP tool call failures?**
  Check WARNING logs in `~/.chat_team/logs/chat_team.log` to confirm server connection status. For stdio transport, verify the `command` can be executed manually in terminal. For SSE, ensure the URL is reachable and returns a valid event stream.

- **What if image OCR results are inaccurate?**
  The default eager prompt prioritizes text extraction. Customize OCR instructions via `llm.vision.default_eager_prompt` in `config.yaml`; or have the agent call the `describe_image` tool with a custom prompt for specific images.

- **Vision `direct` mode consumes too many tokens?**
  `direct` mode injects raw image base64 into context; a single 1024² image costs ~1600 tokens. Enable only for roles requiring high-fidelity visual analysis (e.g., charts, artwork), and increase `llm.history_token_budget` accordingly.

- **Why doesn't `chat-team-tools` show MCP tools?**
  This is expected behavior. MCP tools are dynamically discovered at runtime; the CLI only lists statically registered built-in tools. Actual available tools depend on agent runtime.

## Design Notes

For detailed architecture, subtle mechanics, and pitfalls to avoid, see [`CLAUDE.md`](./CLAUDE.md). Written for Claude Code, but also for humans — it clearly explains "why things are done this way".

## Known Limitations (v1)

- **Single instance per BotID**. WeCom enforces "new connection kicks old"; the same BotID cannot run multiple replicas simultaneously (they will kick each other). Solo mode allows managing multiple different BotIDs in one process, but each BotID still supports only one connection.
- **OpenAI Chat Completion only**. Anthropic / Gemini support planned via `LLMProvider` subclass extension.
- **Media upload supports images / files only** (`send_image` / `send_file` tools via `aibot_upload_media_init/chunk/finish`); voice / video not yet supported.
- **No hot-reload for configuration**. Changes to `team.md`, Role YAML, Skills, or MCP config require bot restart.
- **`chat-team-tools` CLI does not list MCP tools**. MCP tools are dynamically discovered at runtime; CLI only shows statically registered built-in tools.
- **Vision `direct` mode consumes significantly more tokens than `tool` mode**. Raw image base64 injection costs ~1600 tokens per 1024² image; default `tool` mode converts to text via upfront OCR, reducing token overhead by ~6×. Increase `history_token_budget` when enabling `direct`.
- **No cross-bot active notifications in Solo mode**. Other bots won't automatically detect notebook updates; rely on prompt agreements or manual `notebook_read`.
- **MCP supports Tools only**. Resources and Prompts are not yet implemented; individual server connection failures log a WARNING and skip without blocking startup, but corresponding MCP tools for that role will be unavailable.

## Protocols

- Protocol Specifications: `docs/wechat_bot_api.md`, `docs/wechat_bot_接收消息.md`
