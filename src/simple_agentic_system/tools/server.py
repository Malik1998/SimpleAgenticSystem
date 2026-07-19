from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from .registry import ToolRegistry
from .schema import signature_from_json_schema


def serve_registry_as_mcp(registry: ToolRegistry, *, name: str = "simple-agentic-system-tools") -> FastMCP:
    """Exposes every Tool currently in `registry` as an MCP tool on a FastMCP server, so
    the exact same tools the main Agent uses are reachable over the network — in
    particular from code running inside a sandbox container via sandbox/bridge/.
    """
    server = FastMCP(name)
    for exposed_tool in registry.list_tools():
        server.add_tool(
            _make_handler(registry, exposed_tool.name, exposed_tool.parameters),
            name=exposed_tool.name,
            description=exposed_tool.description,
        )
    return server


def _make_handler(registry: ToolRegistry, tool_name: str, parameters: dict[str, Any]) -> Callable[..., Any]:
    async def handler(**kwargs: Any) -> str:
        result = await registry.call(tool_name, kwargs)
        return result.as_text()

    handler.__name__ = tool_name
    handler.__signature__ = signature_from_json_schema(parameters)  # type: ignore[attr-defined]
    return handler
