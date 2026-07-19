import pytest

from simple_agentic_system.llm import (
    AllProvidersFailedError,
    LLMMessage,
    LLMResponse,
    LLMRouter,
    ProviderEntry,
    RetryPolicy,
)
from simple_agentic_system.llm.errors import FatalError, TransientError


class CountingProvider:
    """Replays a scripted list of outcomes: an Exception instance to raise, or a
    string to succeed with."""

    def __init__(self, name, behavior):
        self.name = name
        self._behavior = behavior
        self.calls = 0

    async def complete(self, messages, tools, **kwargs):
        outcome = self._behavior[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(content=outcome, provider=self.name)


async def test_retries_transient_then_succeeds():
    provider = CountingProvider("p1", [TransientError("boom"), "ok"])
    router = LLMRouter([ProviderEntry(provider)], retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0))
    response = await router.complete([LLMMessage(role="user", content="hi")])
    assert response.content == "ok"
    assert provider.calls == 2


async def test_fatal_error_skips_to_next_provider():
    p1 = CountingProvider("p1", [FatalError("bad key")])
    p2 = CountingProvider("p2", ["ok from p2"])
    router = LLMRouter([ProviderEntry(p1, priority=0), ProviderEntry(p2, priority=1)])
    response = await router.complete([LLMMessage(role="user", content="hi")])
    assert response.content == "ok from p2"
    assert p1.calls == 1
    assert p2.calls == 1


async def test_all_providers_failed_raises():
    p1 = CountingProvider("p1", [FatalError("bad")])
    router = LLMRouter([ProviderEntry(p1)])
    with pytest.raises(AllProvidersFailedError):
        await router.complete([LLMMessage(role="user", content="hi")])


async def test_exhausted_retries_falls_through_to_next_provider():
    p1 = CountingProvider("p1", [TransientError("t1"), TransientError("t2")])
    p2 = CountingProvider("p2", ["ok"])
    router = LLMRouter(
        [ProviderEntry(p1), ProviderEntry(p2)],
        retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0),
    )
    response = await router.complete([LLMMessage(role="user", content="hi")])
    assert response.content == "ok"
    assert p1.calls == 2
