from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..llm.base import ToolSchema
from .enricher import PromptContext


def _append(system_prompt: str, block: str) -> str:
    return f"{system_prompt}\n\n{block}".strip() if system_prompt else block


@dataclass
class StaticTextEnricher:
    text: str

    async def enrich(self, system_prompt: str, context: PromptContext) -> str:
        return _append(system_prompt, self.text)


@dataclass
class ToolDescriptionsEnricher:
    """Reads tool specs on every call (via a callable, e.g. ToolRegistry.list_specs)
    rather than a fixed list, so it stays accurate as the registry changes."""

    list_specs: Callable[[], list[ToolSchema]]

    async def enrich(self, system_prompt: str, context: PromptContext) -> str:
        specs = self.list_specs()
        if not specs:
            return system_prompt
        lines = "\n".join(f"- {spec.name}: {spec.description}" for spec in specs)
        return _append(system_prompt, f"Available tools:\n{lines}")


@dataclass
class StateSnapshotEnricher:
    describe_state: Callable[[], dict[str, str]]

    async def enrich(self, system_prompt: str, context: PromptContext) -> str:
        snapshot = self.describe_state()
        if not snapshot:
            return system_prompt
        lines = "\n".join(f"- {key}: {value}" for key, value in snapshot.items())
        return _append(system_prompt, f"Current state:\n{lines}")
