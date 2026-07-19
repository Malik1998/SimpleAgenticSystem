from mcp.shared.memory import create_connected_server_and_client_session

from simple_agentic_system.tools import MCPToolSource, ToolRegistry, serve_registry_as_mcp
from simple_agentic_system.tools.builtin import calculator, echo


async def test_registry_served_over_mcp_roundtrip():
    """This is how the sandbox reaches the same tools the main Agent uses: a
    ToolRegistry served as an MCP server, consumed by an MCPToolSource elsewhere.
    Uses mcp's in-memory client/server pairing — no subprocess, no Docker."""
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(echo)

    server = serve_registry_as_mcp(registry)
    async with create_connected_server_and_client_session(server) as session:
        remote_registry = ToolRegistry()
        names = await MCPToolSource(session).register_into(remote_registry)
        assert set(names) == {"calculator", "echo"}

        result = await remote_registry.call("calculator", {"a": 6, "b": 7, "op": "mul"})
        assert not result.is_error
        # calculator's params are typed "number" -> the MCP layer coerces int args to float
        assert result.output == "42.0"


async def test_mcp_tool_source_name_prefix():
    registry = ToolRegistry()
    registry.register(echo)

    server = serve_registry_as_mcp(registry)
    async with create_connected_server_and_client_session(server) as session:
        remote_tools = await MCPToolSource(session, name_prefix="remote_").list_tools()
        assert [t.name for t in remote_tools] == ["remote_echo"]
