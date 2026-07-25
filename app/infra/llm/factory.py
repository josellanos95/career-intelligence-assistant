from __future__ import annotations

from app.config import Settings
from app.infra.llm.base import LLMProvider
from app.infra.llm.claude_provider import ClaudeProvider
from app.infra.llm.openai_provider import OpenAIProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "claude":
        return ClaudeProvider(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    if settings.llm_provider == "openai":
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_chat_model)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
