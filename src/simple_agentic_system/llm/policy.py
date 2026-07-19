from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ..observability.tracer import NullTracer, Tracer
from .base import LLMMessage, LLMProvider, LLMResponse, ToolSchema
from .errors import FatalError, LLMError, TransientError

logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0


@dataclass
class ProviderEntry:
    provider: LLMProvider
    priority: int = 0
    """Lower runs first."""


class AllProvidersFailedError(LLMError):
    def __init__(self, errors: dict[str, Exception]):
        self.errors = errors
        super().__init__(f"All providers failed: {errors}")


class LLMRouter:
    """Tries providers in priority order; retries transient errors per-provider before falling
    through to the next one. A FatalError skips straight to the next provider."""

    def __init__(
        self,
        providers: list[ProviderEntry],
        retry_policy: RetryPolicy | None = None,
        tracer: Tracer | None = None,
    ):
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self._entries = sorted(providers, key=lambda entry: entry.priority)
        self._retry_policy = retry_policy or RetryPolicy()
        self._tracer = tracer or NullTracer()

    async def complete(
        self, messages: list[LLMMessage], tools: list[ToolSchema] | None = None, **kwargs: Any
    ) -> LLMResponse:
        tools = tools or []
        errors: dict[str, Exception] = {}
        for entry in self._entries:
            try:
                return await self._complete_with_retries(entry.provider, messages, tools, **kwargs)
            except LLMError as exc:
                errors[entry.provider.name] = exc
                logger.warning("provider %s failed (%s), trying next", entry.provider.name, exc)
        raise AllProvidersFailedError(errors)

    async def _complete_with_retries(
        self, provider: LLMProvider, messages: list[LLMMessage], tools: list[ToolSchema], **kwargs: Any
    ) -> LLMResponse:
        backoff = self._retry_policy.initial_backoff_seconds
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            with self._tracer.span(
                f"llm:{provider.name}", kind="llm", attempt=attempt, model=kwargs.get("model")
            ) as span:
                try:
                    response = await provider.complete(messages, tools, **kwargs)
                except FatalError as exc:
                    span.set_error(str(exc))
                    raise
                except TransientError as exc:
                    span.set_error(str(exc))
                    if attempt == self._retry_policy.max_attempts:
                        raise
                    await asyncio.sleep(backoff)
                    backoff = min(
                        backoff * self._retry_policy.backoff_multiplier, self._retry_policy.max_backoff_seconds
                    )
                    continue
                span.set_output(response.content)
                return response
        raise AssertionError("unreachable")
