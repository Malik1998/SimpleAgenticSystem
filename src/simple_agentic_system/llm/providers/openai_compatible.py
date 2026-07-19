from __future__ import annotations

import json
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError as OpenAIRateLimitError,
)

from ..base import LLMMessage, LLMResponse, ToolCall, ToolSchema
from ..errors import FatalError, RateLimitError, TransientError


class OpenAICompatibleProvider:
    """LLMProvider backed by any OpenAI-compatible chat.completions endpoint.

    Shared implementation for OpenAI itself and any OpenAI-compatible gateway
    (OpenRouter, local vLLM, ...) — only name/api_key/model/base_url differ.
    """

    def __init__(self, *, name: str, api_key: str, model: str, base_url: str | None = None):
        self.name = name
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[LLMMessage], tools: list[ToolSchema], **kwargs: Any) -> LLMResponse:
        model = kwargs.pop("model", self._model)
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[_to_openai_message(m) for m in messages],
                tools=[_to_openai_tool(t) for t in tools] or None,
                **kwargs,
            )
        except OpenAIRateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise TransientError(str(exc)) from exc
        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise FatalError(str(exc)) from exc

        choice = response.choices[0]
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (choice.message.tool_calls or [])
        ]
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            raw=response,
            provider=self.name,
            model=model,
        )


def _to_openai_message(message: LLMMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in message.tool_calls
        ]
    return out


def _to_openai_tool(tool: ToolSchema) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }
