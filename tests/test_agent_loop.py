from simple_agentic_system.agent import Agent
from simple_agentic_system.context import ContextManager
from simple_agentic_system.llm import LLMResponse, LLMRouter, ProviderEntry, ToolCall
from simple_agentic_system.llm.providers import FakeLLMProvider
from simple_agentic_system.prompt import PromptPipeline
from simple_agentic_system.tools import ToolRegistry
from simple_agentic_system.tools.builtin import calculator


def _build_agent(script):
    registry = ToolRegistry()
    registry.register(calculator)
    router = LLMRouter([ProviderEntry(FakeLLMProvider(script))])
    return Agent(router, registry, PromptPipeline(), ContextManager())


async def test_agent_calls_tool_then_answers():
    call_count = 0

    def script(messages, tools):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content="", tool_calls=[ToolCall(id="1", name="calculator", arguments={"a": 2, "b": 2, "op": "add"})]
            )
        tool_message = next(m for m in reversed(messages) if m.role == "tool")
        return LLMResponse(content=f"answer={tool_message.content}")

    agent = _build_agent(script)
    result = await agent.run("2+2?")
    assert result == "answer=4"

    messages = await agent.context.get_messages()
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]


async def test_agent_answers_directly_without_tools():
    def script(messages, tools):
        return LLMResponse(content="no tools needed")

    agent = _build_agent(script)
    result = await agent.run("hello")
    assert result == "no tools needed"


async def test_agent_stops_at_max_iterations():
    def script(messages, tools):
        return LLMResponse(
            content="", tool_calls=[ToolCall(id="x", name="calculator", arguments={"a": 1, "b": 1, "op": "add"})]
        )

    from simple_agentic_system.agent import AgentConfig

    registry = ToolRegistry()
    registry.register(calculator)
    router = LLMRouter([ProviderEntry(FakeLLMProvider(script))])
    agent = Agent(router, registry, PromptPipeline(), ContextManager(), config=AgentConfig(max_iterations=2))

    result = await agent.run("loop forever")
    assert result == "Max iterations reached without a final answer."
