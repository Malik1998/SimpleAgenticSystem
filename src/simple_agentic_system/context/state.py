from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class StateEntry:
    value: Any
    summary: str | None = None


class StateStore(Protocol):
    def set(self, key: str, value: Any, *, summary: str | None = None) -> None: ...
    def get(self, key: str) -> Any | None: ...
    def delete(self, key: str) -> None: ...
    def describe(self) -> dict[str, str]:
        """Short human/LLM-readable summary per key — what PromptEnrichers inject,
        instead of dumping full (possibly large) values into the prompt."""
        ...


class InMemoryStateStore:
    def __init__(self) -> None:
        self._entries: dict[str, StateEntry] = {}

    def set(self, key: str, value: Any, *, summary: str | None = None) -> None:
        self._entries[key] = StateEntry(value=value, summary=summary)

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        return entry.value if entry else None

    def delete(self, key: str) -> None:
        self._entries.pop(key, None)

    def describe(self) -> dict[str, str]:
        return {key: entry.summary or _default_summary(entry.value) for key, entry in self._entries.items()}


def _default_summary(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 120 else f"{text[:117]}..."
