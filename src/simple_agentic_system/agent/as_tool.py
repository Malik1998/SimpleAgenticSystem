from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..tools.base import ToolResult
from .core import Agent

_DEFAULT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"task": {"type": "string"}},
    "required": ["task"],
}


class AgentAsTool:
    """Wraps any Agent as a Tool, so it can be called by another Agent's tool-calling
    loop, or used as the `fixer` of a RetryingTool (self-repair, see tools/retrying.py).

    By default the wrapped agent's final text becomes ToolResult.output. Pass
    `parse_output` (e.g. json.loads) when the caller needs structured output — this is
    required for the RetryingTool fixer contract, which expects a dict of corrected
    arguments back.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        parse_output: Callable[[str], Any] | None = None,
    ):
        self._agent = agent
        self.name = name
        self.description = description
        self.parameters = parameters or _DEFAULT_PARAMETERS
        self._parse_output = parse_output

    async def run(self, **kwargs: Any) -> ToolResult:
        task = kwargs.get("task") or _render_task(kwargs)
        try:
            output: Any = await self._agent.run(task)
            if self._parse_output is not None:
                output = self._parse_output(output)
        except Exception as exc:  # noqa: BLE001 - normalize into ToolResult like other tools
            return ToolResult(is_error=True, error=str(exc))
        return ToolResult(output=output)


def _render_task(kwargs: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in kwargs.items())
