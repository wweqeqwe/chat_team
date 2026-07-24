# chat_team

企业微信 (WeCom) 智能机器人 —— 支持两种部署模式:

- **团队模式** (`mode: team`,默认):一个机器人后面隐藏一支可扩展的”虚拟员工团队”,
  后端按需把会话交接给最合适的角色。
- **独立模式** (`mode: solo`):一个机器人 = 一个角色,多角色就多个 bot,同一进程内运行,
  通过共享 notebook 交换信息。

**内置只有 `team_admin`**(团队管理员/前台),其他岗位由你按公司业务在
`~/.chat_team/roles/` 下自己写 YAML 添加 —— `docs/examples/roles/` 下放了
“研发工程师 / 客服专员”两份示例可以照抄起步。每个会话都有独立工作目录,
会话之间完全隔离。

## 特性

- **长连接 (WebSocket) 模式**:无需公网回调地址,启动即在线,自带应用层心跳。
- **流式回复**:用户先看到“思考中…”占位,过程中持续刷新,最终 `finish=true` 收尾。
- **角色 = 一份 YAML**:加角色不用改派发代码;放进 `~/.chat_team/roles/` 即可覆盖内置。
- **会话级工具沙箱**:文件读写 / shell 都被限制在 `~/.chat_team/workspaces/<sid>/` 内,
  路径穿越、绝对路径、symlink 逃逸全部拒绝;shell 输出超阈值会截断,原文落盘可回查。
- **共享 notebook**:跨员工的“团队白板”—— Markdown 文件,`## key` 分块,
  历史里只看到 TOC,值由 `notebook_read` 工具按需取,避免上下文污染。
- **自动压缩**:每个角色独立 token 预算,超阈值时 LLM 摘要早期消息插回历史头。
- **持久化**:`session.json` debounce 10s 原子刷盘,重启后会话历史和当前在岗员工照旧。
- **媒体落地**:image / file / video 通过每条消息独立的 AES-256-CBC `aeskey` 解密后
  自动入 `<cwd>/inbox/`,Agent 收到一条文本指针。
- **图像自动前置 OCR**：默认 `vision_strategy=tool` 下，入站图片会**自动前置 OCR**，
  描述文本以 `[图:相对路径]\n<描述>` 形式注入用户消息，agent 历史保持纯文本，
  token 消耗可控；仅当角色显式设置 `vision_strategy: direct` 时，原始图片才进入上下文。
- **Solo 模式**:一 bot 一角色,同进程多 WebSocket 连接;同一群聊的多 bot 共享 notebook,
  对话历史各自隔离(`session-{role}.json`)。

## 安装

需要 Python ≥ 3.11。建议在虚拟环境中安装:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

只在 `pyproject.toml` 改动 / 增加非 Python 资源 / 重建 venv 时才需要再次 `pip install -e .`;
普通改源码不需要重装(editable 安装把 `src/chat_team` 软链进了 `site-packages`)。

**可选: 装 `uv`**(`curl -LsSf https://astral.sh/uv/install.sh | sh`)。当某个角色
同时拥有 `skill` 和 `run_command` 工具时,系统会在它的 prompt 里注入一段 PEP 723 +
`uv run` 约定;agent 写出来的 Python 脚本可以自动拉第三方依赖,不污染宿主环境。
没装 `uv` 也能跑,但社区里依赖第三方 Python 库的 skill 就跑不动了 —— 启动时会 WARN。

## 配置

首次启动会在 `~/.chat_team/` 下生成默认配置:

```
~/.chat_team/
  config.yaml      # 全局参数 + 凭证 (chmod 0600);首次启动自动生成
  team.md          # 全局团队画像;非空时每轮注入到所有员工的 system prompt;改后需重启
  roles/           # 用户自定义角色 YAML,同名优先于内置
  workspaces/      # 每个会话一个子目录
  logs/            # chat_team.log,RotatingFileHandler
  state/           # 跨会话状态保留位
```

把企微机器人后台拿到的 BotID / Secret 和 OpenAI 密钥填进 `~/.chat_team/config.yaml`:

```yaml
bots:
  - bot_id: "..."
    secret: "..."

llm:
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"   # 替换成内部代理 / vLLM / Ollama 网关也行
```

凭证也可通过同名环境变量设置（`WECOM_BOT_ID` / `WECOM_SECRET` / `OPENAI_API_KEY` / `OPENAI_BASE_URL`），config.yaml 中的值优先。

测试时可用 `CHAT_TEAM_HOME=/tmp/chat_team_dev` 把整个根目录搬走,避免污染真实环境。

`config.yaml` 中模型相关配置已拆分为两组:

```yaml
llm:
  chat:
    model: gpt-4o-mini
    temperature: 0.3
    history_token_budget: 12000
    reasoning_effort: ""   # low | medium | high,留空=模型默认
  vision:
    model: ""              # 空=复用 chat.model
    strategy: tool         # tool | direct
    image_detail: high
    reasoning_effort: ""   # low | medium | high,留空=模型默认
```

### Solo 模式配置

不需要 `team_admin` 做前台路由、希望每个机器人专注一个角色?在 `config.yaml` 里切换:

```yaml
mode: solo
bots:
  - name: research_engineer    # 对应 roles/ 下的角色名
    bot_id: "YOUR_BOT_ID_1"   # 企微后台拿到的 BotID
    secret: "YOUR_SECRET_1"
  - name: customer_service
    bot_id: "YOUR_BOT_ID_2"
    secret: "YOUR_SECRET_2"
```

启动命令不变(`python main.py`)。Solo 模式下:
- 每个 bot 只会用自己绑定的角色回答,不会触发 `transfer_to_employee`
- 多个 bot 在同一群里时,通过 `notebook_write` / `notebook_read` 共享事实
- 各 bot 的对话历史互不影响(分别存储在 `session-{角色名}.json`)

### 大模型调用调试日志

排查"工具路由错了 / compactor 摘要变样 / 视觉 OCR 返回奇怪结果"这类问题时,光看
`chat_team.log` 那点信息往往不够。把 `~/.chat_team/config.yaml` 里的
`llm.debug_log_enabled` 设成 `true`,每次调 OpenAI 都会把**完整的 request + response**
落成 JSON 文件:

```
~/.chat_team/workspaces/<sid>/.chat_team/llm/<时间戳>-<序号>-<角色>-<kind>.json
```

`kind` 三种:`agent`(角色主轮)/ `compactor`(压缩调用)/ `vision`(图片 OCR shim 和
`describe_image` 工具)。每条记录包含 `messages` / `tools` / `model` / `temperature` /
`reasoning_effort` / `response` / `finish_reason` / `usage`(token 用量)/ `latency_ms`,失败时多一项
`error=repr(exc)`、`response=null`。

- **图片 base64 自动脱敏**为 `[redacted: <mime> <字节数> bytes]`,日志可以放心 grep。
- **默认关闭**——一轮多次调用 + 压缩 + OCR 会快速堆出大量文件,生产**不要打开**,
  并且 messages 里有用户对话原文,有隐私顾虑。
- 写入失败只会在 `chat_team.log` 出一条 WARNING,不会影响主流程。

排查完调回 `false` 重启即可。

### 团队画像 (`team.md`)

`~/.chat_team/team.md` 是自由 markdown 文本,非空时启动会原样读入,在每个虚拟员工
**每一轮**的 system prompt 里以 `[团队信息]` 块注入 —— 让所有角色都知道自己服务于
哪个公司/团队,而不必在每个角色 YAML 里复制相同的话术。

- 修改后**需要重启** `chat-team` 才能生效(配置启动时一次性加载)。
- 不需要这个能力时把整个文件清空即可,行为完全等价于"未配置"。
- 建议正文不超过 300 字,过长会推高每轮 API token 消耗。

## 运行

```bash
python main.py
# 等价于:
chat-team
```

`pip install -e .` 之后会注册 `chat-team` 命令(由 `pyproject.toml` 的
`[project.scripts]` 提供),分发时一条命令即可。

正常启动应看到:

```
INFO chat_team.app | chat_team starting; home=/Users/<you>/.chat_team
INFO chat_team.adapters.wecom | connecting to wss://openws.work.weixin.qq.com
INFO chat_team.adapters.wecom | subscribe ok: {...errcode: 0...}
```

之后就可以在企微里 @ 机器人或单聊。日志同时打到 stderr 和 `~/.chat_team/logs/chat_team.log`
(按 10MB × 5 份轮转)。

## `chat-team-boss`(配置助手)

不愿手写 YAML?`pip install -e .` 之后跑:

```bash
chat-team-boss
```

会进入一个**命令行对话**:你用中文说想要什么样的虚拟员工/团队画像,boss 替你读写
`~/.chat_team/team.md` 和 `~/.chat_team/roles/*.yaml`。它会先把待写入的完整 YAML/markdown
贴给你看、明确询问"是否确认写入",得到肯定回复后再落盘。

- **会话不持久化**:每次启动从零开始;真正的"记忆"是落到磁盘上的那些文件。
- **不上企微**:boss 角色仅在 CLI,**不会**出现在 `transfer_to_employee` 的员工枚举里,
  也不会被 enter_chat 触发。
- **改完即可**:`Ctrl+D` 或 `/quit` 退出,然后用 `chat-team` 启动主机器人即可生效
  (角色注册表在主进程启动时一次性加载)。

需要 `OPENAI_API_KEY`（同主机器人,从 `~/.chat_team/config.yaml` 的 `llm.api_key` 或环境变量读取）。

## 后台管理界面 (`chat-team-admin`)

独立的 HTTPS 后台管理 web 界面,用来**查看服务状态、重启/热重载 chat_team、
查看服务器磁盘占用、查看日志**。与主 bot 进程完全独立 —— bot 重启不影响管理界面,
管理界面重启也不影响 bot。生产级部署:TLS 自管 + 账号密码登录 + CSRF + 登录限流 + 审计日志。

**包含的能力:**

- 浏览器打开网页 → 登录页 → 输账号密码进入 dashboard
- 服务状态卡:运行中/未运行徽章、PID、运行时长、内存占用
- 磁盘卡:所在分区总/已用/可用 + chat_team 总占用 + 各子目录(logs/workspaces/state)细分
- 会话列表:`~/.chat_team/workspaces/<sid>` 按占用排序(文件系统视角,top 20)
- 操作按钮:**热重载**(`systemctl reload` = SIGHUP,不打断 WebSocket)、
  **重启**(`systemctl restart`)
- 日志 tail:chat_team 日志(`journalctl -u chat-team` 或 `~/.chat_team/logs/chat_team.log`)、
  admin 操作日志(`~/.chat_team/logs/admin.log`),每 5 秒自动刷新

### 装机步骤

```bash
# 1. 在 ~/.chat_team/config.yaml 把 admin.enabled 改成 true (其余字段用默认即可):
#    admin:
#      enabled: true

# 2. 生成自签 TLS 证书 (写到 ~/.chat_team/admin/{cert,key}.pem):
chat-team-admin init-certs

# 3. 创建第一个登录账号 (交互式输密码,长度 >= 8):
chat-team-admin add-user admin

# 4. 装 systemd unit 并启动:
cp scripts/chat-team-admin.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now chat-team-admin

# 5. 浏览器访问 https://<server-ip>:8443/
#    自签证书浏览器会有"不安全"警告,点"高级 → 继续访问"即可。
systemctl status chat-team-admin     # 看运行状态
journalctl -u chat-team-admin -f     # 看实时日志
```

### 子命令

```bash
chat-team-admin serve           # 启动 web server (无参数时默认)
chat-team-admin init-certs      # 生成自签证书 (--force 覆盖)
chat-team-admin add-user <name>  # 创建/更新账号 (交互式输密码)
```

### 鉴权细节

- **账号存储**:`~/.chat_team/admin/users.json`,密码用 `pbkdf2_sha256` (600000 iterations,OWASP 2023 推荐)。
- **会话**:登录成功后服务器颁发 32 字节随机 session id,内存存储,cookie 走
  `HttpOnly + Secure + SameSite=Strict`,默认 8 小时空闲超时。
- **CSRF**:重启/热重载的 POST 接口必须带 `X-CSRF-Token` 头(同源 cookie 里有 csrf token,
  dashboard 内置 JS 自动塞头)。坏 token → 403。
- **登录限流**:同一 IP 5 分钟内 5 次失败 → 第 6 次 429(在密码检查之前就拦截,
  避免 pbkdf2 被暴力打满 CPU)。成功登录清零。
- **审计日志**:`~/.chat_team/logs/admin.log`,每次 `login_success`/`login_failed`/
  `login_blocked`/`restart`/`reload`/`logout` 写一行(时间 + 用户 + IP + UA)。

### 生产加固建议

公网部署时,**仅靠账号密码** 是基线,推荐再叠加一层:

- **升级到 Let's Encrypt 证书**(有域名时,消除浏览器自签警告):
  ```bash
  certbot certonly --standalone -d admin.yourdomain.com
  # 在 config.yaml 里改:
  # admin:
  #   tls_cert: /etc/letsencrypt/live/admin.yourdomain.com/fullchain.pem
  #   tls_key:  /etc/letsencrypt/live/admin.yourdomain.com/privkey.pem
  systemctl reload chat-team-admin   # 或 restart
  ```
- **上 nginx 反代做 IP allowlist**(防御纵深,token 泄露也进不来):
  在 nginx vhost 里 `allow <你的IP>; deny all;` + `proxy_pass https://127.0.0.1:8443`。
- **云厂商防火墙限白名单源 IP**(简单粗暴但有效)。

### 配置项 (config.yaml 的 `admin:` 块)

完整字段及默认值见 `~/.chat_team/config.yaml` 末尾的注释段。关键项:
`enabled` (默认 false)、`host` (默认 0.0.0.0)、`port` (默认 8443)、
`tls_cert`/`tls_key` (空 → `~/.chat_team/admin/{cert,key}.pem`)、
`session_idle_seconds` (默认 28800=8h)、`login_rate_limit_per_5min` (默认 5)。

## 看可用工具清单

手写 role YAML 时,`tools:` 字段只能填主进程注册过的工具名。要看一份当前真实可用的清单:

```bash
chat-team-tools
# 或不安装 console script: python -m chat_team.list_tools
```

会按工具名排序、附第一行简介。这个清单和 `chat-team-boss` 里的 `list_available_tools`
工具同源 —— 加新工具后两边都自动可见,无需手动维护。

## 加一个角色

不想手写?用 `chat-team-boss`(见上一节)让它替你写。手写示例:

```yaml
# ~/.chat_team/roles/data_analyst.yaml
name: data_analyst
display_name: 数据分析师
description: 处理 SQL、报表、数据探索类需求。
system_prompt: |
  你是数据分析师,名字叫"小数"。回答前先用 read_file 看一下当前目录的样本。
tools:
  - read_file
  - list_dir
  - run_command
  - notebook_read
  - notebook_write
  - transfer_to_employee
welcome_message: 你好,我是数据分析师小数,有数据相关的问题随时问我。
llm:
  model: gpt-4o-mini
  temperature: 0.2
  history_token_budget: 16000
```

不需要重启 —— 不,要重启,角色注册表是进程启动时加载的。但**不需要改任何代码**,
也不需要重新 `pip install`。`transfer_to_employee` 的目标 enum 会在下次启动时自动包含 `data_analyst`。

## 加一个工具

`src/chat_team/agent/tools/` 下子类化 `Tool`,实现 `async run(ctx, **kwargs)`,
然后在 `app.build_tool_registry` 里 `reg.register(...)`,最后在角色 YAML 的 `tools:`
里引用名字即可。可恢复错误抛 `ToolError`(会作为 tool 消息回给 LLM)。

## 烟囱测试 (不依赖 LLM / 网络)

```bash
python scripts/smoke_dispatch.py                  # 派发器 + Agent + 工具
python scripts/smoke_transfer.py                  # 多跳交接 + 上限 + 未知目标
python scripts/smoke_tools.py                     # 文件 / shell + 沙箱
python scripts/smoke_wecom_parse.py               # adapter 解析 + LRU + stream 帧形
python scripts/smoke_compaction_persistence.py    # tiktoken 压缩 + session.json 往返
python scripts/smoke_media_events.py              # AES-256-CBC + enter_chat / disconnected
python scripts/smoke_llm_debug_log.py             # 调试日志: image base64 脱敏 + 文件落盘 + per-session seq
python scripts/smoke_solo.py                      # solo 模式: 独立 dispatcher + 共享 notebook + 隔离持久化
python scripts/smoke_admin.py                    # 后台管理界面: pbkdf2/session/CSRF/限流/审计 + HTTP 流程
```

每个 smoke 都把 `CHAT_TEAM_HOME` 指到 `/tmp/...` 并在启动时 `rmtree`,所以反复跑安全。

## 扩展能力：MCP 与 Skills

### MCP（Model Context Protocol）

无需写 Python 即可接入外部工具服务器。在 `config.yaml` 声明服务器，在角色 YAML 引用，启动时自动发现并注册工具。

**配置示例**（`~/.chat_team/config.yaml`）：

```yaml
mcp:
  servers:
    filesystem:                          # 服务器名，角色 YAML 中引用
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

Transport 自动推断：有 `command` → stdio，有 `url` → SSE。服务器名须匹配 `[a-zA-Z0-9_-]+` 且不含 `__`。

**角色引用**（角色 YAML）：

```yaml
name: developer
tools: [read_file, write_file, run_command, transfer_to_employee]
mcp_servers: [filesystem, github]        # 该角色可使用这两个服务器的所有工具
```

MCP 工具注册名为 `mcp__<server>__<tool>`（如 `mcp__filesystem__read_file`），agent 调用时无需感知前缀，直接使用原始工具名即可。

**限制**：当前仅支持 MCP Tools，不支持 Resources / Prompts；修改 MCP 配置需重启 bot；`chat-team-tools` CLI 不列出 MCP 工具（运行时动态发现）。

### Skills（无代码能力包）

通过 drop-in 目录为角色添加指令型能力，无需编写 Python 工具。

**目录结构**：`~/.chat_team/skills/<name>/SKILL.md` + 可选辅助文件。

**SKILL.md 格式**：

```markdown
---
name: data_analysis          # 必须等于目录名
description: SQL 查询与报表生成指南   # 单行描述，注入系统提示 TOC
---

# 数据分析师 Skill

当用户请求数据分析时，请遵循以下步骤：
1. 先用 `list_dir` 查看可用数据文件
2. 使用 `run_command` 执行 pandas 脚本
...
```

**角色启用**：在角色 YAML 中同时声明 `skills:` 和 `tools: [skill, skill_read_file]`：

```yaml
name: data_analyst
skills: [data_analysis]
tools: [skill, skill_read_file, read_file, run_command, transfer_to_employee]
```

Agent 会在系统提示中看到 `[可用 skills] - data_analysis: SQL 查询与报表生成指南`，通过 `skill(name="data_analysis")` 获取完整指令，通过 `skill_read_file(skill="data_analysis", path="template.sql")` 读取辅助文件。

**Python 依赖**：当角色同时拥有 `skill` 和 `run_command` 工具时，系统自动注入 PEP 723 + `uv run` 约定，agent 编写的 Python 脚本可声明内联依赖并由 `uv` 全局缓存解析，不污染宿主环境。未安装 `uv` 时非 Python skill 仍可正常使用。

**注意**：Skill 不支持热加载，新增或修改后需重启 bot。

## 常见问题

- **修改了 `team.md` / 角色 YAML / Skills / MCP 配置为何没生效？**
  所有配置均在启动时一次性加载，修改后需重启 `chat-team` 才能生效。

- **Solo 模式下 bot 之间如何感知对方更新了 notebook？**
  当前无主动通知机制。Bot A 写入 notebook 后，Bot B 需在下轮对话中主动调用 `notebook_read` 才能获取最新值。如需强同步，建议在 prompt 中约定关键事实更新后显式提醒对方查阅。

- **MCP 工具调用失败如何排查？**
  检查 `~/.chat_team/logs/chat_team.log` 中的 WARNING 日志，确认服务器连接状态；若为 stdio transport，检查 `command` 是否可在终端手动执行；若为 SSE，确认 URL 可达且返回合法事件流。

- **图片 OCR 结果不准确怎么办？**
  默认 eager prompt 优先提取文字。可在 `config.yaml` 中调整 `llm.vision.default_eager_prompt` 自定义 OCR 指令；或对特定图片让 agent 调用 `describe_image` 工具传入定制 prompt 重新识别。

- **Vision `direct` 模式 token 消耗过高？**
  `direct` 模式将原始图片 base64 注入上下文，单张 1024² 图片约消耗 1600 tokens。建议仅在需要高保真视觉分析（如图表、艺术作品）的角色上启用，并相应调大 `llm.history_token_budget`。

- **`chat-team-tools` 看不到 MCP 工具？**
  这是预期行为。MCP 工具在运行时动态发现，CLI 仅列出静态注册的内置工具。实际可用工具以 agent 运行时为准。

## 设计要点

详细架构、不显眼的细节、不要踩的地雷见 [`CLAUDE.md`](./CLAUDE.md)。给 Claude Code 看的,
也给人看 —— 它把"为什么这样写"明明白白讲了一遍。

## 已知限制 (v1)

- **每个 BotID 单实例**。企微"新连接踢旧",同一 BotID 不能多副本启动,会互踢。
  Solo 模式下一个进程可管多个不同 BotID,但每个 BotID 仍只能有一条连接。
- **只支持 OpenAI Chat Completion**。Anthropic / Gemini 通过 `LLMProvider` 子类后续扩展。
- **媒体回传仅支持图片 / 文件**(`send_image` / `send_file` 工具,走
  `aibot_upload_media_init/chunk/finish`);voice / video 暂不支持。
- **不支持配置热加载**。修改 `team.md`、角色 YAML、Skills、MCP 配置后均需重启 bot 才能生效。
- **`chat-team-tools` CLI 不列出 MCP 工具**。MCP 工具为运行时动态发现，CLI 仅展示静态注册的内置工具。
- **Vision `direct` 模式 token 消耗显著高于 `tool` 模式**。原始图片 base64 注入上下文，单张 1024² 图片约 1600 tokens；默认 `tool` 模式因前置 OCR 转为纯文本，token 开销降低约 6 倍。启用 `direct` 时需相应调大 `history_token_budget`。
- **Solo 模式无跨 bot 主动通知**。notebook 更新后其他 bot 不会自动感知，需依赖 prompt 约定或手动 `notebook_read`。
- **MCP 仅支持 Tools**。Resources 和 Prompts 暂未实现；单个服务器连接失败仅记 WARNING 并跳过，不阻塞启动，但该角色对应 MCP 工具将不可用。

## 协议

- 协议规约: `docs/wechat_bot_api.md`, `docs/wechat_bot_接收消息.md`
