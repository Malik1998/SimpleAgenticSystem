from __future__ import annotations

from typing import Any

from ..llm.base import ToolSchema
from ..observability.tracer import NullTracer, Tracer
from .base import Tool, ToolResult


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    """Single source of truth for callable tools. Used directly by the main Agent loop,
    and indirectly (via tools/server.py) by code running in the sandbox."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._tracer = tracer or NullTracer()

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
        with self._tracer.span(f"tool:{name}", kind="tool", arguments=arguments) as span:
            try:
                result = await tool.run(**arguments)
            except Exception as exc:  # noqa: BLE001 - normalize any tool failure into a ToolResult
                span.set_error(str(exc))
                return ToolResult(is_error=True, error=str(exc))
            if result.is_error:
                span.set_error(result.error or "")
            else:
                span.set_output(result.output)
            return result
