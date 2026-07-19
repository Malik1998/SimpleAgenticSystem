from __future__ import annotations

from typing import Protocol

from ..llm.base import LLMMessage


class History(Protocol):
    async def add(self, message: LLMMessage) -> None: ...
    async def get_messages(self) -> list[LLMMessage]: ...
    async def clear(self) -> None: ...


class InMemoryHistory:
    def __init__(self) -> None:
        self._messages: list[LLMMessage] = []

    async def add(self, message: LLMMessage) -> None:
        self._messages.append(message)

    async def get_messages(self) -> list[LLMMessage]:
        return list(self._messages)

    async def clear(self) -> None:
        self._messages.clear()
