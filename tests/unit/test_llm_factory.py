import pytest

from app.config import Settings
from app.infra.llm.claude_provider import ClaudeProvider
from app.infra.llm.factory import get_llm_provider
from app.infra.llm.openai_provider import OpenAIProvider


def test_factory_returns_claude_provider_by_default():
    settings = Settings(llm_provider="claude", anthropic_api_key="fake")
    assert isinstance(get_llm_provider(settings), ClaudeProvider)


def test_factory_returns_openai_provider_when_configured():
    settings = Settings(llm_provider="openai", openai_api_key="fake")
    assert isinstance(get_llm_provider(settings), OpenAIProvider)


def test_factory_raises_for_unknown_provider():
    settings = Settings(llm_provider="claude")
    settings.llm_provider = "unknown"  # bypass Literal validation to test the guard clause
    with pytest.raises(ValueError):
        get_llm_provider(settings)
