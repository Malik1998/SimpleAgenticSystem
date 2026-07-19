import pytest

from simple_agentic_system.tools import ToolRegistry
from simple_agentic_system.tools.builtin import calculator, echo
from simple_agentic_system.tools.registry import ToolNotFoundError


async def test_register_and_call():
    registry = ToolRegistry()
    registry.register(calculator)
    result = await registry.call("calculator", {"a": 2, "b": 3, "op": "add"})
    assert not result.is_error
    assert result.output == 5


async def test_call_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        await registry.call("nope", {})


async def test_tool_exception_becomes_error_result():
    registry = ToolRegistry()
    registry.register(calculator)
    result = await registry.call("calculator", {"a": 1, "b": 2, "op": "unknown"})
    assert result.is_error
    assert "unknown op" in result.error


def test_list_specs():
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(echo)
    names = {spec.name for spec in registry.list_specs()}
    assert names == {"calculator", "echo"}
