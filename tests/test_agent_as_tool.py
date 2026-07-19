import json

from simple_agentic_system.agent import Agent, AgentAsTool
from simple_agentic_system.context import ContextManager
from simple_agentic_system.llm import LLMResponse, LLMRouter, ProviderEntry
from simple_agentic_system.llm.providers import FakeLLMProvider
from simple_agentic_system.prompt import PromptPipeline
from simple_agentic_system.tools import ToolRegistry


def _build_agent(reply: str) -> Agent:
    router = LLMRouter([ProviderEntry(FakeLLMProvider(lambda messages, tools: LLMResponse(content=reply)))])
    return Agent(router, ToolRegistry(), PromptPipeline(), ContextManager())


async def test_agent_as_tool_returns_plain_text_by_default():
    tool = AgentAsTool(_build_agent("42"), name="answer", description="answers things")
    result = await tool.run(task="what is the answer?")
    assert not result.is_error
    assert result.output == "42"


async def test_agent_as_tool_parses_structured_output():
    tool = AgentAsTool(
        _build_agent(json.dumps({"x": 7})),
        name="fixer",
        description="returns corrected args",
        parse_output=json.loads,
    )
    result = await tool.run(original_arguments={"x": 1}, error="bad")
    assert not result.is_error
    assert result.output == {"x": 7}


async def test_agent_as_tool_wraps_bad_parse_as_error():
    tool = AgentAsTool(_build_agent("not json"), name="fixer", description="", parse_output=json.loads)
    result = await tool.run(task="fix it")
    assert result.is_error
