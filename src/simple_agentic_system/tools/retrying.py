from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from .base import Tool, ToolResult


@dataclass
class RepairPolicy:
    max_attempts: int = 2
    mode: Literal["sequential", "parallel"] = "sequential"
    parallel_fanout: int = 1
    """Only used when mode == "parallel": how many fix candidates to try per round."""


class RetryingTool:
    """Generic self-repair wrapper around ANY Tool — not specific to code execution.

    On ToolResult.is_error, calls `fixer` with the original arguments and the error,
    expecting back a ToolResult whose `.output` is a dict of corrected arguments;
    `inner` is retried with those. `fixer` is typically an AgentAsTool wrapping a
    small specialized agent, but any Tool honoring that contract works.

    mode="sequential": one fix attempt at a time, each seeing the previous error.
    mode="parallel": each round fans `parallel_fanout` independent fix attempts out
    concurrently (asyncio.gather) and keeps the first one whose retry succeeds —
    useful when the fixer is a stochastic LLM and best-of-N improves the odds.
    """

    def __init__(self, inner: Tool, fixer: Tool, policy: RepairPolicy | None = None, *, name: str | None = None):
        self.inner = inner
        self.fixer = fixer
        self.policy = policy or RepairPolicy()
        self.name = name or inner.name
        self.description = inner.description
        self.parameters = inner.parameters

    async def run(self, **kwargs: Any) -> ToolResult:
        result = await self.inner.run(**kwargs)
        if not result.is_error:
            return result
        if self.policy.mode == "parallel":
            return await self._run_parallel(kwargs, result)
        return await self._run_sequential(kwargs, result)

    async def _run_sequential(self, kwargs: dict[str, Any], result: ToolResult) -> ToolResult:
        current_args = kwargs
        for _attempt in range(self.policy.max_attempts):
            fixed_args = await self._ask_fixer(current_args, result)
            if fixed_args is None:
                break
            current_args = fixed_args
            result = await self.inner.run(**current_args)
            if not result.is_error:
                return result
        return result

    async def _run_parallel(self, kwargs: dict[str, Any], result: ToolResult) -> ToolResult:
        for _round in range(self.policy.max_attempts):
            candidates = await asyncio.gather(
                *(self._ask_fixer(kwargs, result) for _ in range(self.policy.parallel_fanout))
            )
            candidate_args_list = [args for args in candidates if args is not None]
            if not candidate_args_list:
                break
            candidate_results = await asyncio.gather(*(self.inner.run(**args) for args in candidate_args_list))
            for candidate_args, candidate_result in zip(candidate_args_list, candidate_results, strict=True):
                if not candidate_result.is_error:
                    return candidate_result
            kwargs, result = candidate_args_list[0], candidate_results[0]
        return result

    async def _ask_fixer(self, args: dict[str, Any], failed_result: ToolResult) -> dict[str, Any] | None:
        fixer_result = await self.fixer.run(original_arguments=args, error=failed_result.error or "")
        if fixer_result.is_error or not isinstance(fixer_result.output, dict):
            return None
        return fixer_result.output
