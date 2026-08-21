# 角色 YAML 示例

这个目录里的 YAML 不会被打包,也不会被 `RoleRegistry` 自动加载 —— 它们只是参考样例,
让你照着写自己的虚拟员工。

内置角色只有 `team_admin`(团队管理员/前台),用来接待用户、识别意图、按需把会话
交给其他同事。其他岗位都由你来定义,因为不同公司的"合适同事"完全不同。

## 怎么用这些示例

复制到本地 `~/.chat_team/roles/`,重启 `chat-team` 即可生效:

```bash
mkdir -p ~/.chat_team/roles
cp docs/examples/roles/research_engineer.yaml ~/.chat_team/roles/
cp docs/examples/roles/customer_service.yaml  ~/.chat_team/roles/
```

`transfer_to_employee` 工具的目标枚举会在启动时根据 `~/.chat_team/roles/` + 内置一起
重建,新角色加进来后,`team_admin` 就能转给它,不用改代码。

## 自己写一个角色

最少需要 `name` / `system_prompt` / `tools` 三个字段,其余可选:

```yaml
name: data_analyst              # 必填,唯一英文 id;会出现在 transfer_to_employee 枚举里
display_name: 数据分析师          # 可选,展示用
description: 处理 SQL / 报表 / 数据探索类需求。   # 可选

system_prompt: |
  你是数据分析师,名字叫"小数"。回答前先用 read_file 看一下当前目录的样本数据。
  ...

tools:                          # 必填,从 app.build_tool_registry 注册过的工具里挑
  - read_file
  - list_dir
  - run_command
  - notebook_read
  - notebook_write
  - transfer_to_employee

mcp_servers:                    # 可选,引用 config.yaml 中 mcp.servers 下的服务器名
  - filesystem                  # 该角色自动获得此 MCP 服务器暴露的所有工具

mcp_tools:                      # 可选,按角色、按 server 设置白名单/黑名单
  filesystem:
    mode: blacklist             # whitelist=只暴露 tools 列表;blacklist=屏蔽 tools 列表
    tools:
      - write_file              # 写原始 MCP 工具名,不写 mcp__filesystem__ 前缀

llm:                            # 可选,留空就用全局默认
  model: ""                     # 例如 "gpt-4o-mini"
  temperature: 0.3
  history_token_budget: 16000

welcome_message: |              # 可选,enter_chat 事件里 default_role 会用到
  你好,我是数据分析师小数,有数据相关的问题随时问我。
```

可用工具名以代码为准:`src/chat_team/app.py` 的 `build_tool_registry()`。
现成包括:`read_file` / `write_file` / `list_dir` / `run_command` /
`notebook_read` / `notebook_write` / `notebook_delete` / `transfer_to_employee`。

MCP 工具无需在 `tools:` 列表中单独列出 —— 只要在 `mcp_servers:` 中写服务器名,
该服务器的所有工具就会自动暴露给这个角色。MCP 服务器在 `~/.chat_team/config.yaml`
的 `mcp.servers` 节定义,参见 `config.yaml` 中的注释示例。
如需按角色限制,用 `mcp_tools`:未配置的 server 继续暴露全部工具;
`mode: whitelist` 只暴露列出的原始工具名,`mode: blacklist` 屏蔽列出的原始工具名。

如果同名 YAML 同时出现在 `~/.chat_team/roles/` 和 `src/chat_team/roles/builtin/`,
**用户目录的优先**。这意味着你也可以覆盖 `team_admin` 自己重写。
