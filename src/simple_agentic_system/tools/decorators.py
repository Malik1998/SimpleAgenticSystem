from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from .base import ToolResult
from .schema import schema_from_signature


class FunctionTool:
    """Adapts a plain sync/async function into a Tool."""

    def __init__(self, func: Callable[..., Any], *, name: str, description: str, parameters: dict[str, Any]):
        self._func = func
        self.name = name
        self.description = description
        self.parameters = parameters

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            if asyncio.iscoroutinefunction(self._func):
                output = await self._func(**kwargs)
            else:
                output = self._func(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize into ToolResult, like ToolRegistry.call does
            return ToolResult(is_error=True, error=str(exc))
        if isinstance(output, ToolResult):
            return output
        return ToolResult(output=output)


def tool(*, name: str | None = None, description: str | None = None) -> Callable[[Callable[..., Any]], FunctionTool]:
    """@tool decorator: wraps a function into a Tool, inferring its JSON schema from
    type hints and its description from the docstring unless overridden."""

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            func,
            name=name or func.__name__,
            description=description or (inspect.getdoc(func) or ""),
            parameters=schema_from_signature(func),
        )

    return decorator
