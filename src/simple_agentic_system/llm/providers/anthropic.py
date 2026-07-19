from __future__ import annotations

from typing import Any

import anthropic

from ..base import LLMMessage, LLMResponse, ToolCall, ToolSchema
from ..errors import FatalError, RateLimitError, TransientError


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str = "claude-sonnet-4-5"):
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, messages: list[LLMMessage], tools: list[ToolSchema], **kwargs: Any) -> LLMResponse:
        system, converted = _split_system(messages)
        model = kwargs.pop("model", self._model)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": converted,
            **kwargs,
        }
        if system:
            create_kwargs["system"] = system
        if tools:
            create_kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]

        try:
            response = await self._client.messages.create(**create_kwargs)
        except anthropic.RateLimitError as exc:
            raise RateLimitError(str(exc)) from exc
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            raise TransientError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise FatalError(str(exc)) from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        tool_calls = [
            ToolCall(id=block.id, name=block.name, arguments=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        return LLMResponse(content=text, tool_calls=tool_calls, raw=response, provider=self.name, model=model)


def _split_system(messages: list[LLMMessage]) -> tuple[str, list[dict[str, Any]]]:
    system_parts = [m.content for m in messages if m.role == "system"]
    converted = [_to_anthropic_message(m) for m in messages if m.role != "system"]
    return "\n".join(system_parts), converted


def _to_anthropic_message(message: LLMMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content}],
        }
    if message.tool_calls:
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        content.extend(
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in message.tool_calls
        )
        return {"role": message.role, "content": content}
    return {"role": message.role, "content": message.content}


def _to_anthropic_tool(tool: ToolSchema) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
