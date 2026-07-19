from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from ..agent import Agent, AgentConfig
from ..config import Settings, load_settings
from ..context import ContextManager
from ..llm import LLMRouter, ProviderEntry
from ..llm.policy import AllProvidersFailedError
from ..llm.providers.anthropic import AnthropicProvider
from ..llm.providers.fake import FakeLLMProvider
from ..llm.providers.openai_compatible import OpenAICompatibleProvider
from ..llm.providers.openrouter import OPENROUTER_BASE_URL
from ..prompt import PromptPipeline, StaticTextEnricher, ToolDescriptionsEnricher
from ..tools import ToolRegistry
from ..tools.builtin import calculator, echo
from .tracing import RecordedEvent, RecordingTracer

STATIC_DIR = Path(__file__).parent / "static"

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# In-memory, per-session conversation state. Fine for a single-process local dev UI;
# swap for SqliteSessionStore-backed lookup if this ever needs to survive a restart.
_sessions: dict[str, ContextManager] = {}

_TOOL_CATALOG = {"calculator": calculator, "echo": echo}


class MissingApiKeyError(ValueError):
    pass


def _make_offline_provider() -> FakeLLMProvider:
    """Zero-setup default: no API key required, so the UI is usable out of the box."""

    def script(messages, tools):  # type: ignore[no-untyped-def]
        from ..llm.base import LLMResponse

        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        text = (last_user.content if last_user else "").strip()
        reply = (
            f"[offline demo model — no API key configured]\n\nYou said: {text}"
            if text
            else "[offline demo model] Say something!"
        )
        return LLMResponse(content=reply, provider="fake", model="fake-echo")

    return FakeLLMProvider(script)


def _build_provider(settings: Settings, provider_id: str, model: str | None) -> Any:
    if provider_id == "anthropic":
        if not settings.anthropic_api_key:
            raise MissingApiKeyError("ANTHROPIC_API_KEY is not set (add it to .env)")
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=model or settings.anthropic_model)
    if provider_id == "openai":
        if not settings.openai_api_key:
            raise MissingApiKeyError("OPENAI_API_KEY is not set (add it to .env)")
        return OpenAICompatibleProvider(
            name="openai", api_key=settings.openai_api_key, model=model or settings.openai_model
        )
    if provider_id == "openrouter":
        if not settings.openrouter_api_key:
            raise MissingApiKeyError("OPENROUTER_API_KEY is not set (add it to .env)")
        return OpenAICompatibleProvider(
            name="openrouter",
            api_key=settings.openrouter_api_key,
            model=model or settings.openrouter_model,
            base_url=OPENROUTER_BASE_URL,
        )
    return _make_offline_provider()


def _build_tool_registry(context: ContextManager, enabled: list[str], tracer: RecordingTracer) -> ToolRegistry:
    registry = ToolRegistry(tracer=tracer)
    for name in enabled:
        if name in _TOOL_CATALOG:
            registry.register(_TOOL_CATALOG[name])
    if "search_memory" in enabled:
        registry.register(context.as_memory_tool())
    return registry


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _serialize_events(events: list[RecordedEvent]) -> list[dict[str, Any]]:
    return [
        {
            "name": event.name,
            "attributes": _jsonable(event.attributes),
            "output": _jsonable(event.output),
            "error": event.error,
        }
        for event in events
        if event.kind == "tool"
    ]


async def _handle_index(request: Request) -> Response:
    return FileResponse(STATIC_DIR / "index.html")


async def _handle_defaults(request: Request) -> JSONResponse:
    settings = load_settings()
    providers = [
        {"id": "fake", "label": "Offline demo (no API key)", "model": "fake-echo", "available": True},
        {
            "id": "anthropic",
            "label": "Anthropic",
            "model": settings.anthropic_model,
            "available": bool(settings.anthropic_api_key),
        },
        {
            "id": "openai",
            "label": "OpenAI",
            "model": settings.openai_model,
            "available": bool(settings.openai_api_key),
        },
        {
            "id": "openrouter",
            "label": "OpenRouter",
            "model": settings.openrouter_model,
            "available": bool(settings.openrouter_api_key),
        },
    ]
    tools = [
        {"id": "calculator", "description": calculator.description},
        {"id": "echo", "description": echo.description},
        {"id": "search_memory", "description": "Search past conversation memory for relevant snippets."},
    ]
    return JSONResponse(
        {
            "providers": providers,
            "tools": tools,
            "defaults": {
                "provider": "fake",
                "temperature": 0.7,
                "max_tokens": 1024,
                "max_iterations": 8,
                "tools": [],
            },
        }
    )


async def _handle_chat(request: Request) -> JSONResponse:
    payload = await request.json()
    session_id = str(payload.get("session_id") or "").strip()
    message = str(payload.get("message") or "").strip()
    settings_in = payload.get("settings") or {}

    if not session_id:
        return JSONResponse({"error": "session_id is required"}, status_code=400)
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    context = _sessions.setdefault(session_id, ContextManager(session_id=session_id))

    provider_id = str(settings_in.get("provider") or "fake")
    model = settings_in.get("model") or None
    enabled_tools = [str(t) for t in (settings_in.get("tools") or [])]
    try:
        temperature = float(settings_in.get("temperature", 0.7))
        max_tokens = int(settings_in.get("max_tokens", 1024))
        max_iterations = int(settings_in.get("max_iterations", 8))
    except (TypeError, ValueError):
        return JSONResponse({"error": "temperature/max_tokens/max_iterations must be numbers"}, status_code=400)

    try:
        provider = _build_provider(load_settings(), provider_id, model)
    except MissingApiKeyError as exc:
        return JSONResponse({"reply": None, "steps": [], "error": str(exc)})

    tracer = RecordingTracer()
    registry = _build_tool_registry(context, enabled_tools, tracer)
    pipeline = PromptPipeline(
        [StaticTextEnricher(DEFAULT_SYSTEM_PROMPT), ToolDescriptionsEnricher(registry.list_specs)]
    )
    router = LLMRouter([ProviderEntry(provider)], tracer=tracer)
    agent = Agent(
        router,
        registry,
        pipeline,
        context,
        config=AgentConfig(
            max_iterations=max_iterations,
            llm_kwargs={"temperature": temperature, "max_tokens": max_tokens},
        ),
        tracer=tracer,
    )

    try:
        reply = await agent.run(message)
    except AllProvidersFailedError as exc:
        return JSONResponse({"reply": None, "steps": _serialize_events(tracer.events), "error": str(exc)})

    return JSONResponse({"reply": reply, "steps": _serialize_events(tracer.events), "error": None})


async def _handle_reset(request: Request) -> JSONResponse:
    payload = await request.json()
    session_id = str(payload.get("session_id") or "").strip()
    _sessions.pop(session_id, None)
    return JSONResponse({"ok": True})


def create_app() -> Starlette:
    routes = [
        Route("/", _handle_index, methods=["GET"]),
        Route("/api/defaults", _handle_defaults, methods=["GET"]),
        Route("/api/chat", _handle_chat, methods=["POST"]),
        Route("/api/reset", _handle_reset, methods=["POST"]),
    ]
    return Starlette(routes=routes)
