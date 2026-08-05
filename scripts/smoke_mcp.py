"""Smoke tests for MCP (Model Context Protocol) support.

Run:
    python scripts/smoke_mcp.py

Uses mocks — no real MCP servers needed.
"""
from __future__ import annotations

import asyncio
import types
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CHAT_TEAM_HOME"] = "/tmp/chat_team_mcp_smoke"

from chat_team.adapters.base import ChatType, IncomingMessage
from chat_team.agent.agent import Agent
from chat_team.agent.tools.base import ToolContext, ToolError, ToolRegistry
from chat_team.agent.tools.notebook_tools import NotebookReadTool
from chat_team.config import McpConfig, Settings, load_settings
from chat_team.dispatcher import Dispatcher
from chat_team.llm.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
    ToolCall,
)
from chat_team.mcp.config import McpServerConfig
from chat_team.mcp.proxy_tool import McpProxyTool
from chat_team.roles.config import Role
from chat_team.roles.registry import RoleRegistry
from chat_team.session.manager import SessionManager
from chat_team.session.notebook import Notebook
from chat_team.session.session import Session
from chat_team.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeMcpTool:
    name: str = "get_weather"
    description: str = "Get weather for a city"
    inputSchema: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    })


@dataclass
class FakeTextContent:
    type: str = "text"
    text: str = ""


@dataclass
class FakeImageContent:
    type: str = "image"
    data: str = "base64data"
    mimeType: str = "image/png"


@dataclass
class FakeCallToolResult:
    content: list = field(default_factory=list)
    isError: bool = False


class FakeMcpSession:
    def __init__(
        self,
        result: FakeCallToolResult | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ):
        self._result = result or FakeCallToolResult(content=[FakeTextContent(text="sunny, 25C")])
        self._error = error
        self._delay = delay
        self.calls: list[tuple[str, dict | None]] = []

    async def call_tool(self, name: str, arguments: dict | None = None, **kwargs) -> FakeCallToolResult:
        self.calls.append((name, arguments))
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return self._result


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedLLM exhausted")
        return self._responses.pop(0)


def make_session(sid: str, cwd: Path) -> Session:
    nb_path = cwd / ".chat_team" / "notebook.md"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    return Session(session_id=sid, cwd=cwd, current_role="test", notebook=Notebook(nb_path))


class CapturingStream:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.final: str | None = None

    async def push(self, chunk: str, *, append: bool = True) -> None:
        pass

    async def status(self, note: str) -> None:
        self.statuses.append(note)

    async def finish(self, final_text: str) -> None:
        self.final = final_text


def reply(text: str) -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(role="assistant", content=text),
        finish_reason="stop",
    )


def tool_call(name: str, args: dict, call_id: str = "tc-1") -> CompletionResponse:
    return CompletionResponse(
        message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        ),
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_parsing():
    """McpServerConfig parsed from config.yaml dict format."""
    home = Path("/tmp/chat_team_mcp_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.setdefault("OPENAI_API_KEY", "test")

    cfg_yaml = home / "config.yaml"
    cfg_yaml.write_text("""\
mcp:
  servers:
    filesystem:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    github:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_TOKEN: ghp_xxx
    remote:
      url: http://localhost:8080/sse
""")
    settings = load_settings()
    assert len(settings.mcp.servers) == 3, f"expected 3, got {len(settings.mcp.servers)}"

    fs = settings.mcp.servers[0]
    assert fs.name == "filesystem"
    assert fs.command == "npx"
    assert fs.args == ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    assert fs.url == ""

    gh = settings.mcp.servers[1]
    assert gh.name == "github"
    assert gh.env == {"GITHUB_TOKEN": "ghp_xxx"}

    remote = settings.mcp.servers[2]
    assert remote.name == "remote"
    assert remote.url == "http://localhost:8080/sse"
    assert remote.command == ""

    print("  config parsing: OK")


def test_config_invalid_server_skipped():
    """Invalid MCP server entries are skipped with a warning."""
    home = Path("/tmp/chat_team_mcp_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.setdefault("OPENAI_API_KEY", "test")

    cfg_yaml = home / "config.yaml"
    cfg_yaml.write_text("""\
mcp:
  servers:
    bad__name:
      command: echo
    good:
      command: echo
    both_set:
      command: echo
      url: http://example.com
""")
    settings = load_settings()
    assert len(settings.mcp.servers) == 1, f"expected 1 valid, got {len(settings.mcp.servers)}"
    assert settings.mcp.servers[0].name == "good"
    print("  invalid server skipped: OK")


def test_config_no_mcp_section():
    """Missing mcp: section → empty servers list."""
    home = Path("/tmp/chat_team_mcp_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    (home / "config.yaml").write_text("default_role: team_admin\n")
    settings = load_settings()
    assert settings.mcp.servers == []
    print("  no mcp section: OK")


def test_proxy_tool_construction():
    """McpProxyTool name, description, parameters, and spec()."""
    mcp_tool = FakeMcpTool()
    session = FakeMcpSession()
    proxy = McpProxyTool(server_name="weather", mcp_tool=mcp_tool, session=session)

    assert proxy.name == "mcp__weather__get_weather"
    assert proxy.description == "Get weather for a city"
    assert proxy.parameters == mcp_tool.inputSchema
    assert proxy.server_name == "weather"

    spec = proxy.spec()
    assert spec.name == "mcp__weather__get_weather"
    assert spec.description == "Get weather for a city"
    assert spec.parameters == mcp_tool.inputSchema
    print("  proxy tool construction: OK")


async def test_proxy_tool_run():
    """McpProxyTool.run() calls session.call_tool and returns text."""
    session = FakeMcpSession(FakeCallToolResult(
        content=[FakeTextContent(text="sunny"), FakeTextContent(text="25C")],
    ))
    proxy = McpProxyTool("weather", FakeMcpTool(), session)
    ctx = ToolContext(cwd=Path("/tmp"), session=None, settings=None)  # type: ignore[arg-type]
    result = await proxy.run(ctx, city="Beijing")

    assert result == "sunny\n25C"
    assert session.calls == [("get_weather", {"city": "Beijing"})]
    print("  proxy tool run: OK")


async def test_proxy_tool_resolves_workspace_image_aliases():
    """Workspace-style image paths are resolved before an MCP call."""
    workspace = Path("/tmp/chat_team_mcp_smoke/workspace")
    image = workspace / "inbox" / "photo.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fake jpeg")

    session = FakeMcpSession()
    proxy = McpProxyTool(
        "agnes",
        FakeMcpTool(name="agnes_image_recognition"),
        session,
    )
    ctx = ToolContext(
        cwd=workspace,
        session=None,
        settings=None,
    )  # type: ignore[arg-type]

    for supplied_path in [
        "/workspace/inbox/photo.jpg",
        "/mnt/data/inbox/photo.jpg",
        "./inbox/photo.jpg",
        "inbox/photo.jpg",
    ]:
        await proxy.run(ctx, image=supplied_path, prompt="describe")

    expected = str(image.resolve())
    assert session.calls == [
        (
            "agnes_image_recognition",
            {"image": expected, "prompt": "describe"},
        )
    ] * 4
    print("  proxy tool workspace image aliases: OK")


async def test_proxy_tool_error_wrapping():
    """MCP call_tool exception → ToolError."""
    session = FakeMcpSession(error=ConnectionError("server down"))
    proxy = McpProxyTool("weather", FakeMcpTool(), session)
    ctx = ToolContext(cwd=Path("/tmp"), session=None, settings=None)  # type: ignore[arg-type]
    try:
        await proxy.run(ctx, city="Beijing")
        assert False, "should have raised ToolError"
    except ToolError as e:
        assert "server down" in str(e)
    print("  proxy tool error wrapping: OK")


async def test_proxy_tool_is_error_flag():
    """MCP result with isError=True → ToolError."""
    session = FakeMcpSession(FakeCallToolResult(
        content=[FakeTextContent(text="city not found")],
        isError=True,
    ))
    proxy = McpProxyTool("weather", FakeMcpTool(), session)
    ctx = ToolContext(cwd=Path("/tmp"), session=None, settings=None)  # type: ignore[arg-type]
    try:
        await proxy.run(ctx, city="???")
        assert False, "should have raised ToolError"
    except ToolError as e:
        assert "city not found" in str(e)
    print("  proxy tool isError flag: OK")


async def test_proxy_tool_image_content():
    """Image content in MCP result → placeholder text."""
    session = FakeMcpSession(FakeCallToolResult(
        content=[FakeImageContent()],
    ))
    proxy = McpProxyTool("vision", FakeMcpTool(name="screenshot"), session)
    ctx = ToolContext(cwd=Path("/tmp"), session=None, settings=None)  # type: ignore[arg-type]
    result = await proxy.run(ctx, city="x")
    assert "[image: image/png]" in result
    print("  proxy tool image content: OK")


class _FakeMcpSettings:
    """Minimal stand-in for Settings.mcp so proxy_tool can read the timeout
    without spinning up the full load_settings() machinery."""

    def __init__(self, tool_timeout_seconds: float):
        self.mcp = types.SimpleNamespace(tool_timeout_seconds=tool_timeout_seconds)


class _FakeSettings:
    def __init__(self, tool_timeout_seconds: float):
        self.mcp = types.SimpleNamespace(tool_timeout_seconds=tool_timeout_seconds)


async def test_proxy_tool_timeout_triggers_toolerror():
    """A slow MCP call exceeding the timeout raises ToolError, not hang."""
    session = FakeMcpSession(delay=5.0)
    proxy = McpProxyTool("weather", FakeMcpTool(), session)
    # 0.2s timeout — well under the 5s the fake call will sleep.
    ctx = ToolContext(
        cwd=Path("/tmp"), session=None,
        settings=_FakeSettings(tool_timeout_seconds=0.2),  # type: ignore[arg-type]
    )
    try:
        await proxy.run(ctx, city="Beijing")
        assert False, "should have raised ToolError on timeout"
    except ToolError as e:
        assert "timed out" in str(e).lower()
        assert "0.2s" in str(e)
    print("  proxy tool timeout → ToolError: OK")


async def test_proxy_tool_timeout_zero_disables():
    """tool_timeout_seconds=0 means no timeout — slow call completes."""
    session = FakeMcpSession(delay=0.1)
    proxy = McpProxyTool("weather", FakeMcpTool(), session)
    ctx = ToolContext(
        cwd=Path("/tmp"), session=None,
        settings=_FakeSettings(tool_timeout_seconds=0),  # type: ignore[arg-type]
    )
    result = await proxy.run(ctx, city="Beijing")
    assert result == "sunny, 25C"
    print("  proxy tool timeout=0 (disabled): OK")


async def test_proxy_tool_timeout_settings_none_uses_default():
    """When ctx.settings is None (test contexts), the default timeout applies
    but a fast call still succeeds — verifying the fallback path doesn't
    break normal calls."""
    session = FakeMcpSession()
    proxy = McpProxyTool("weather", FakeMcpTool(), session)
    ctx = ToolContext(cwd=Path("/tmp"), session=None, settings=None)  # type: ignore[arg-type]
    result = await proxy.run(ctx, city="Beijing")
    assert result == "sunny, 25C"
    print("  proxy tool settings=None fallback: OK")


def test_config_tool_timeout_parsing():
    """mcp.tool_timeout_seconds is read from config.yaml."""
    home = Path("/tmp/chat_team_mcp_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    (home / "config.yaml").write_text("""\
mcp:
  tool_timeout_seconds: 30
  servers:
    fs:
      command: /bin/true
""")
    settings = load_settings()
    assert settings.mcp.tool_timeout_seconds == 30.0, \
        f"expected 30.0, got {settings.mcp.tool_timeout_seconds}"
    print("  config tool_timeout_seconds parsing: OK")


def test_config_tool_timeout_default():
    """Absent tool_timeout_seconds falls back to 60.0."""
    home = Path("/tmp/chat_team_mcp_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    (home / "config.yaml").write_text("""\
mcp:
  servers:
    fs:
      command: /bin/true
""")
    settings = load_settings()
    assert settings.mcp.tool_timeout_seconds == 60.0, \
        f"expected 60.0 default, got {settings.mcp.tool_timeout_seconds}"
    print("  config tool_timeout_seconds default: OK")


def test_registry_names():
    """ToolRegistry.names() returns all registered tool names."""
    reg = ToolRegistry()
    reg.register(NotebookReadTool())
    session = FakeMcpSession()
    reg.register(McpProxyTool("weather", FakeMcpTool(), session))

    names = reg.names()
    assert "notebook_read" in names
    assert "mcp__weather__get_weather" in names
    print("  registry names: OK")


def test_effective_tool_names():
    """Agent._effective_tool_names() expands mcp_servers."""
    home = Path("/tmp/chat_team_mcp_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    (home / "config.yaml").write_text("")
    settings = load_settings()

    reg = ToolRegistry()
    reg.register(NotebookReadTool())
    session = FakeMcpSession()
    reg.register(McpProxyTool("weather", FakeMcpTool(name="get_weather"), session))
    reg.register(McpProxyTool("weather", FakeMcpTool(name="get_forecast"), session))
    reg.register(McpProxyTool("files", FakeMcpTool(name="read_file"), session))

    role = Role(
        name="test",
        display_name="Test",
        description="",
        system_prompt="you are a test",
        tools=["notebook_read"],
        mcp_servers=["weather"],
    )
    sess = make_session("s1", home)
    agent = Agent(
        role=role, session=sess, settings=settings,
        llm=ScriptedLLM([]), tools=reg, skills=SkillRegistry({}),
    )

    names = agent._effective_tool_names()
    assert "notebook_read" in names
    assert "mcp__weather__get_weather" in names
    assert "mcp__weather__get_forecast" in names
    assert "mcp__files__read_file" not in names, "files server not in mcp_servers"
    print("  effective tool names: OK")


def test_effective_tool_names_no_mcp():
    """No mcp_servers → only role.tools returned."""
    home = Path("/tmp/chat_team_mcp_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    (home / "config.yaml").write_text("")
    settings = load_settings()

    reg = ToolRegistry()
    reg.register(NotebookReadTool())
    session = FakeMcpSession()
    reg.register(McpProxyTool("weather", FakeMcpTool(), session))

    role = Role(name="t", display_name="T", description="", system_prompt="x",
                tools=["notebook_read"], mcp_servers=[])
    sess = make_session("s1", home)
    agent = Agent(role=role, session=sess, settings=settings,
                  llm=ScriptedLLM([]), tools=reg, skills=SkillRegistry({}))

    names = agent._effective_tool_names()
    assert names == ["notebook_read"]
    print("  effective tool names (no mcp): OK")


def test_role_mcp_servers_parsing():
    """Role.from_dict parses mcp_servers field."""
    role = Role.from_dict({
        "name": "dev",
        "system_prompt": "hi",
        "tools": ["read_file"],
        "mcp_servers": ["filesystem", "github"],
    })
    assert role.mcp_servers == ["filesystem", "github"]

    role2 = Role.from_dict({
        "name": "basic",
        "system_prompt": "hi",
    })
    assert role2.mcp_servers == []
    print("  role mcp_servers parsing: OK")


async def test_agent_invokes_mcp_tool():
    """Agent chat+tool loop calls McpProxyTool and returns result."""
    home = Path("/tmp/chat_team_mcp_smoke")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    (home / "config.yaml").write_text("")
    settings = load_settings()

    session = FakeMcpSession(FakeCallToolResult(
        content=[FakeTextContent(text="sunny, 25C")],
    ))
    reg = ToolRegistry()
    proxy = McpProxyTool("weather", FakeMcpTool(), session)
    reg.register(proxy)

    llm = ScriptedLLM([
        tool_call("mcp__weather__get_weather", {"city": "Beijing"}),
        reply("The weather in Beijing is sunny, 25C."),
    ])

    role = Role(
        name="assistant",
        display_name="Assistant",
        description="",
        system_prompt="You are a helpful assistant.",
        tools=[],
        mcp_servers=["weather"],
    )
    sess = make_session("s1", home)
    agent = Agent(
        role=role, session=sess, settings=settings,
        llm=llm, tools=reg, skills=SkillRegistry({}),
    )

    stream = CapturingStream()
    result = await agent.handle("What's the weather in Beijing?", stream)

    assert result == "The weather in Beijing is sunny, 25C."
    assert session.calls == [("get_weather", {"city": "Beijing"})]
    # Verify the MCP tool appeared in the LLM request's tools list
    assert any(t.name == "mcp__weather__get_weather" for t in llm.requests[0].tools)
    print("  agent invokes MCP tool: OK")


async def main() -> None:
    print("=== MCP smoke tests ===")

    test_config_parsing()
    test_config_invalid_server_skipped()
    test_config_no_mcp_section()
    test_proxy_tool_construction()
    await test_proxy_tool_run()
    await test_proxy_tool_resolves_workspace_image_aliases()
    await test_proxy_tool_error_wrapping()
    await test_proxy_tool_is_error_flag()
    await test_proxy_tool_image_content()
    await test_proxy_tool_timeout_triggers_toolerror()
    await test_proxy_tool_timeout_zero_disables()
    await test_proxy_tool_timeout_settings_none_uses_default()
    test_config_tool_timeout_parsing()
    test_config_tool_timeout_default()
    test_registry_names()
    test_effective_tool_names()
    test_effective_tool_names_no_mcp()
    test_role_mcp_servers_parsing()
    await test_agent_invokes_mcp_tool()

    print("\nALL MCP SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
