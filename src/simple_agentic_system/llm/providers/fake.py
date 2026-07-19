from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..base import LLMMessage, LLMResponse, ToolSchema


class FakeLLMProvider:
    """Deterministic provider for tests/examples: either plays back a fixed script of
    responses, or delegates to a callable(messages, tools) -> LLMResponse."""

    name = "fake"

    def __init__(
        self,
        responses: list[LLMResponse] | Callable[[list[LLMMessage], list[ToolSchema]], LLMResponse],
    ):
        self._responses = responses
        self._index = 0

    async def complete(self, messages: list[LLMMessage], tools: list[ToolSchema], **kwargs: Any) -> LLMResponse:
        if callable(self._responses):
            return self._responses(messages, tools)
        if self._index >= len(self._responses):
            raise IndexError("FakeLLMProvider script exhausted")
        response = self._responses[self._index]
        self._index += 1
        return response
