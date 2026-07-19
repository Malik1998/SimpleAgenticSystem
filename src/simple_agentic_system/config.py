from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1"

    openrouter_api_key: str | None = None
    openrouter_model: str = "anthropic/claude-sonnet-4.5"

    sandbox_docker_image: str = "simple-agentic-system-sandbox:latest"
    sandbox_max_concurrent_containers: int = 4
    sandbox_idle_ttl_seconds: int = 600


def load_settings() -> Settings:
    return Settings()
