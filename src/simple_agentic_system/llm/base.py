from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    """Set on role="tool" messages; ties the result back to the ToolCall.id it answers."""


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    provider: str = ""
    model: str = ""


class LLMProvider(Protocol):
    name: str

    async def complete(
        self, messages: list[LLMMessage], tools: list[ToolSchema], **kwargs: Any
    ) -> LLMResponse: ...
