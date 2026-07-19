"""End-to-end demo: FakeLLMProvider + builtin tools, no API keys or Docker required.

Run: uv run python examples/minimal_agent.py
"""

from __future__ import annotations

import asyncio
import logging

from simple_agentic_system.agent import Agent
from simple_agentic_system.context import ContextManager
from simple_agentic_system.llm import LLMResponse, LLMRouter, ProviderEntry, ToolCall
from simple_agentic_system.llm.providers import FakeLLMProvider
from simple_agentic_system.observability import LoggingTracer
from simple_agentic_system.prompt import PromptPipeline, StaticTextEnricher, ToolDescriptionsEnricher
from simple_agentic_system.tools import ToolRegistry
from simple_agentic_system.tools.builtin import calculator


def build_scripted_llm() -> FakeLLMProvider:
    """First turn: call the calculator tool. Second turn: answer using its result."""
    call_count = 0

    def script(messages, tools):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="calculator", arguments={"a": 21, "b": 2, "op": "mul"})],
            )
        tool_message = next(m for m in reversed(messages) if m.role == "tool")
        return LLMResponse(content=f"21 * 2 = {tool_message.content}")

    return FakeLLMProvider(script)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tracer = LoggingTracer()

    registry = ToolRegistry(tracer=tracer)
    registry.register(calculator)

    pipeline = PromptPipeline(
        [StaticTextEnricher("You are a helpful assistant."), ToolDescriptionsEnricher(registry.list_specs)]
    )
    router = LLMRouter([ProviderEntry(build_scripted_llm())], tracer=tracer)
    context = ContextManager()
    agent = Agent(router, registry, pipeline, context, tracer=tracer)

    answer = await agent.run("What is 21 * 2?")
    print("\nFinal answer:", answer)


if __name__ == "__main__":
    asyncio.run(main())
