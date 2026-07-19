from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .tracer import Span

_KIND_TO_LANGFUSE_TYPE = {
    "llm": "generation",
    "tool": "tool",
    "agent": "agent",
    "span": "span",
}


class _LangfuseSpan:
    def __init__(self, native_span: Any):
        self._native = native_span

    def set_output(self, output: Any) -> None:
        self._native.update(output=output)

    def set_error(self, error: str) -> None:
        self._native.update(level="ERROR", status_message=error)


class LangfuseTracer:
    """Tracer backed by Langfuse (https://langfuse.com).

    Optional dependency: `pip install simple-agentic-system[langfuse]`, then set
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST in the environment (or
    pass an already-configured `langfuse.Langfuse` client explicitly).
    """

    def __init__(self, client: Any | None = None):
        if client is None:
            try:
                from langfuse import Langfuse
            except ImportError as exc:
                raise ImportError(
                    "LangfuseTracer requires the 'langfuse' package: "
                    "pip install simple-agentic-system[langfuse]"
                ) from exc
            client = Langfuse()
        self._client = client

    @contextmanager
    def span(self, name: str, *, kind: str = "span", **attributes: Any) -> Iterator[Span]:
        as_type = _KIND_TO_LANGFUSE_TYPE.get(kind, "span")
        with self._client.start_as_current_observation(
            name=name, as_type=as_type, input=attributes or None
        ) as native_span:
            yield _LangfuseSpan(native_span)
