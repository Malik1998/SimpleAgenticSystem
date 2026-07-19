"""Demo of the self-repair primitive: RetryingTool + AgentAsTool.

A "flaky" divide tool fails on b=0; a fixer Agent (backed by a FakeLLMProvider)
inspects the error and proposes corrected arguments, and RetryingTool retries with
them. No real LLM or Docker involved — this validates the composition, not any
particular inner tool. In a real setup `inner` would typically be
sandbox.PythonExecTool instead of this toy DivideTool.

Run: uv run python examples/self_repair_demo.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from simple_agentic_system.agent import Agent, AgentAsTool
from simple_agentic_system.context import ContextManager
from simple_agentic_system.llm import LLMResponse, LLMRouter, ProviderEntry
from simple_agentic_system.llm.providers import FakeLLMProvider
from simple_agentic_system.observability import LoggingTracer
from simple_agentic_system.prompt import PromptPipeline
from simple_agentic_system.tools import RepairPolicy, RetryingTool, ToolRegistry, ToolResult


class DivideTool:
    """Divides a / b. Fails if b == 0 — the bug the fixer needs to spot."""

    name = "divide"
    description = "Divide a by b"
    parameters = {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}

    async def run(self, **kwargs: Any) -> ToolResult:
        a, b = kwargs["a"], kwargs["b"]
        if b == 0:
            return ToolResult(is_error=True, error="ZeroDivisionError: b must not be 0")
        return ToolResult(output=a / b)


def build_fixer_llm() -> FakeLLMProvider:
    """Deterministic 'fixer': always swaps a bad b=0 for b=1."""

    def script(messages, tools):
        return LLMResponse(content=json.dumps({"a": 10, "b": 1}))

    return FakeLLMProvider(script)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tracer = LoggingTracer()

    fixer_router = LLMRouter([ProviderEntry(build_fixer_llm())], tracer=tracer)
    fixer_agent = Agent(fixer_router, ToolRegistry(tracer=tracer), PromptPipeline(), ContextManager(), tracer=tracer)
    fixer_tool = AgentAsTool(
        fixer_agent,
        name="fix_divide_args",
        description="Given failed divide() arguments and the error, propose corrected arguments.",
        parse_output=json.loads,
    )

    retrying_divide = RetryingTool(
        inner=DivideTool(), fixer=fixer_tool, policy=RepairPolicy(max_attempts=2), name="divide"
    )

    result = await retrying_divide.run(a=10, b=0)
    print("\nFinal result:", result)


if __name__ == "__main__":
    asyncio.run(main())
