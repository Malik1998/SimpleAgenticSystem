from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter is OpenAI-compatible, so this is just OpenAICompatibleProvider pointed
    at a different base_url — one key, access to many providers/models."""

    def __init__(self, *, api_key: str, model: str = "anthropic/claude-sonnet-4.5"):
        super().__init__(name="openrouter", api_key=api_key, model=model, base_url=OPENROUTER_BASE_URL)
