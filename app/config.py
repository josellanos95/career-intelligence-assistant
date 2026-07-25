"""Centralized application configuration, loaded from environment variables.

Kept as a single Pydantic settings object rather than scattered os.getenv()
calls so every module depends on one typed, validated source of truth.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Matches the documented default in .env.example: "openai" so the app
    # works out of the box with a single API key (OpenAI is already required
    # for embeddings regardless of chat provider).
    llm_provider: Literal["claude", "openai"] = "openai"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    vector_store_dir: str = "./data/vector_store"

    log_level: str = "INFO"
    max_context_chunks: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
