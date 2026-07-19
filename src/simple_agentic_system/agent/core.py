from __future__ import annotations

from dataclasses import dataclass

from ..context.manager import ContextManager
from ..llm.base import LLMMessage
from ..llm.policy import LLMRouter
from ..observability.tracer import NullTracer, Tracer
from ..prompt.enricher import PromptContext, PromptPipeline
from ..tools.registry import ToolRegistry


@dataclass
class AgentConfig:
    max_iterations: int = 8


class Agent:
    """Ties LLM + Tools + Prompt + Context together into one tool-calling loop."""

    def __init__(
        self,
        llm: LLMRouter,
        tools: ToolRegistry,
        prompt: PromptPipeline,
        context: ContextManager,
        config: AgentConfig | None = None,
        tracer: Tracer | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.prompt = prompt
        self.context = context
        self.config = config or AgentConfig()
        self._tracer = tracer or NullTracer()

    async def run(self, user_input: str) -> str:
        with self._tracer.span("agent:run", kind="agent", input=user_input) as run_span:
            result = await self._run(user_input)
            run_span.set_output(result)
            return result

    async def _run(self, user_input: str) -> str:
        await self.context.add_message(LLMMessage(role="user", content=user_input))
        system_prompt = await self.prompt.build(PromptContext(user_input=user_input))

        for step in range(self.config.max_iterations):
            with self._tracer.span(f"agent:step:{step}", kind="span"):
                messages = [LLMMessage(role="system", content=system_prompt), *await self.context.get_messages()]
                response = await self.llm.complete(messages, self.tools.list_specs())

                await self.context.add_message(
                    LLMMessage(role="assistant", content=response.content, tool_calls=response.tool_calls)
                )

                if not response.tool_calls:
                    await self.context.save_session()
                    return response.content

                for call in response.tool_calls:
                    result = await self.tools.call(call.name, call.arguments)
                    await self.context.add_message(
                        LLMMessage(role="tool", content=result.as_text(), tool_call_id=call.id)
                    )

        await self.context.save_session()
        return "Max iterations reached without a final answer."
