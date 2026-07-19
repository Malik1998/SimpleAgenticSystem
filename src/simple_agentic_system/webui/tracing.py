from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordedEvent:
    kind: str
    name: str
    attributes: dict[str, Any]
    output: Any = None
    error: str | None = None


class _RecordingSpan:
    def __init__(self, event: RecordedEvent) -> None:
        self._event = event

    def set_output(self, output: Any) -> None:
        self._event.output = output

    def set_error(self, error: str) -> None:
        self._event.error = error


@dataclass
class RecordingTracer:
    """Collects tool-call spans into a flat list so the web UI can render a
    ChatGPT-style "used tool X" trail alongside the final answer."""

    events: list[RecordedEvent] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, *, kind: str = "span", **attributes: Any) -> Iterator[_RecordingSpan]:
        event = RecordedEvent(kind=kind, name=name, attributes=attributes)
        self.events.append(event)
        yield _RecordingSpan(event)
