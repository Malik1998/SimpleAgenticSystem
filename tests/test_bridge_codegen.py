import sys
import types

from simple_agentic_system.llm.base import ToolSchema
from simple_agentic_system.sandbox.bridge.codegen import render_tool_module


def _exec_generated(source: str, fake_call_tool) -> dict:
    fake_runtime = types.ModuleType("runtime")
    fake_runtime.call_tool = fake_call_tool
    sys.modules["runtime"] = fake_runtime
    try:
        namespace: dict = {}
        exec(compile(source, "<generated>", "exec"), namespace)
        return namespace
    finally:
        del sys.modules["runtime"]


def test_generated_module_compiles_and_dispatches_required_args():
    specs = [
        ToolSchema(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        )
    ]
    source = render_tool_module(specs)
    calls = []
    namespace = _exec_generated(source, lambda name, **kwargs: calls.append((name, kwargs)) or sum(kwargs.values()))

    assert namespace["add"](a=1, b=2) == 3
    assert calls == [("add", {"a": 1, "b": 2})]


def test_generated_module_handles_optional_args():
    specs = [
        ToolSchema(
            name="calculator",
            description="calc",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {"type": "string"},
                },
                "required": ["a", "b"],
            },
        )
    ]
    source = render_tool_module(specs)
    namespace = _exec_generated(source, lambda name, **kwargs: kwargs)

    # optional arg omitted -> defaults to None, still a valid call
    assert namespace["calculator"](a=1, b=2) == {"a": 1, "b": 2, "op": None}
    assert namespace["calculator"](a=1, b=2, op="mul") == {"a": 1, "b": 2, "op": "mul"}


def test_render_tool_module_empty_specs():
    source = render_tool_module([])
    compile(source, "<generated>", "exec")  # must not raise
