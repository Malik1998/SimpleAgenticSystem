from __future__ import annotations

from typing import Any

from ..llm.base import ToolSchema
from .base import Tool, ToolResult


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    """Single source of truth for callable tools. Used directly by the main Agent loop,
    and indirectly (via tools/server.py) by code running in the sandbox."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name) from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list_specs(self) -> list[ToolSchema]:
        return [
            ToolSchema(name=t.name, description=t.description, parameters=t.parameters)
            for t in self._tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        try:
            return await tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - normalize any tool failure into a ToolResult
            return ToolResult(is_error=True, error=str(exc))
