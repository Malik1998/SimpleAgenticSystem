from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class MemoryHit:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore(Protocol):
    async def add(self, text: str, *, metadata: dict[str, Any] | None = None) -> None: ...
    async def search(self, query: str, k: int = 5) -> list[MemoryHit]: ...


class NaiveMemoryStore:
    """Keyword-overlap search, no embeddings — placeholder behind the MemoryStore
    interface so a real vector store can be swapped in later without touching callers."""

    def __init__(self) -> None:
        self._items: list[tuple[str, dict[str, Any]]] = []

    async def add(self, text: str, *, metadata: dict[str, Any] | None = None) -> None:
        self._items.append((text, metadata or {}))

    async def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        query_words = set(query.lower().split())
        scored: list[MemoryHit] = []
        for text, metadata in self._items:
            overlap = len(query_words & set(text.lower().split()))
            if overlap:
                scored.append(MemoryHit(text=text, score=overlap / max(len(query_words), 1), metadata=metadata))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]
