from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolResult:
    output: Any = None
    is_error: bool = False
    error: str | None = None

    def as_text(self) -> str:
        if self.is_error:
            return f"Error: {self.error}"
        if isinstance(self.output, str):
            return self.output
        return str(self.output)


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]
    """JSON schema for the keyword arguments accepted by `run`."""

    async def run(self, **kwargs: Any) -> ToolResult: ...
