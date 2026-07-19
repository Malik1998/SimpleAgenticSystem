from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

_logger = logging.getLogger("simple_agentic_system.trace")


class Span(Protocol):
    def set_output(self, output: Any) -> None: ...
    def set_error(self, error: str) -> None: ...


class Tracer(Protocol):
    def span(self, name: str, *, kind: str = "span", **attributes: Any) -> AbstractContextManager[Span]:
        """kind is one of "agent", "llm", "tool", "span" — backends (e.g. Langfuse) map
        it to their own observation types. Plain sync context manager on purpose: every
        call site here is inside an `async def`, and creating a span never needs to
        await anything (exporting happens on a background thread in real backends)."""
        ...


class NullSpan:
    def set_output(self, output: Any) -> None:
        pass

    def set_error(self, error: str) -> None:
        pass


class NullTracer:
    """Default Tracer: zero overhead, zero dependencies."""

    @contextmanager
    def span(self, name: str, *, kind: str = "span", **attributes: Any) -> Iterator[Span]:
        yield NullSpan()


class _LoggingSpan:
    def __init__(self) -> None:
        self.output: Any = None
        self.error: str | None = None

    def set_output(self, output: Any) -> None:
        self.output = output

    def set_error(self, error: str) -> None:
        self.error = error


class LoggingTracer:
    """Zero-dependency default backend: logs span start/end/duration via stdlib
    logging. Good enough to see "what and how" is happening without wiring Langfuse."""

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or _logger

    @contextmanager
    def span(self, name: str, *, kind: str = "span", **attributes: Any) -> Iterator[Span]:
        record = _LoggingSpan()
        start = time.monotonic()
        self._logger.info("-> %s %s %s", kind, name, attributes or "")
        try:
            yield record
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            self._logger.exception("<- %s %s failed after %.1fms", kind, name, duration_ms)
            raise
        duration_ms = (time.monotonic() - start) * 1000
        if record.error:
            self._logger.warning("<- %s %s error after %.1fms: %s", kind, name, duration_ms, record.error)
        else:
            self._logger.info("<- %s %s done after %.1fms output=%r", kind, name, duration_ms, record.output)
