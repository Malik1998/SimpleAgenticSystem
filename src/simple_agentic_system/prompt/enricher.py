from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class PromptContext:
    """What an enricher needs to decide what to add. Intentionally minimal — extend
    `extra` (or add fields) as a real project needs, e.g. user_id, retrieved memories."""

    user_input: str
    extra: dict[str, Any] = field(default_factory=dict)


class PromptEnricher(Protocol):
    async def enrich(self, system_prompt: str, context: PromptContext) -> str: ...


class PromptPipeline:
    """Chain of PromptEnrichers applied in order to build the final system prompt."""

    def __init__(self, enrichers: list[PromptEnricher] | None = None):
        self._enrichers = list(enrichers or [])

    def add(self, enricher: PromptEnricher) -> None:
        self._enrichers.append(enricher)

    async def build(self, context: PromptContext) -> str:
        system_prompt = ""
        for enricher in self._enrichers:
            system_prompt = await enricher.enrich(system_prompt, context)
        return system_prompt
