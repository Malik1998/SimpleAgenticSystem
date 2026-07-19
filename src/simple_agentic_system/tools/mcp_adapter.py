from __future__ import annotations

from typing import Any

from mcp import ClientSession

from .base import ToolResult
from .registry import ToolRegistry


class MCPToolSource:
    """Wraps an already-initialized MCP ClientSession and exposes its tools as Tool
    instances that can be merged into a local ToolRegistry.

    Transport-agnostic on purpose: the caller connects the session (stdio, streamable
    HTTP, in-memory for tests, ...) and owns its lifecycle. This is how external/
    third-party MCP servers get pulled into our own ToolRegistry namespace.
    """

    def __init__(self, session: ClientSession, *, name_prefix: str = ""):
        self._session = session
        self._name_prefix = name_prefix

    async def list_tools(self) -> list["_RemoteMCPTool"]:
        result = await self._session.list_tools()
        return [_RemoteMCPTool(self._session, t, self._name_prefix) for t in result.tools]

    async def register_into(self, registry: ToolRegistry) -> list[str]:
        remote_tools = await self.list_tools()
        for remote_tool in remote_tools:
            registry.register(remote_tool)
        return [t.name for t in remote_tools]


class _RemoteMCPTool:
    def __init__(self, session: ClientSession, mcp_tool: Any, name_prefix: str):
        self._session = session
        self._remote_name = mcp_tool.name
        self.name = f"{name_prefix}{mcp_tool.name}"
        self.description = mcp_tool.description or ""
        self.parameters = mcp_tool.inputSchema or {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> ToolResult:
        result = await self._session.call_tool(self._remote_name, arguments=kwargs)
        text = "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
        return ToolResult(output=text, is_error=bool(result.isError))
