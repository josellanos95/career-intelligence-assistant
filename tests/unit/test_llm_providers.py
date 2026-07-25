from unittest.mock import MagicMock, patch

from app.domain.models import ChatMessage
from app.infra.llm.claude_provider import ClaudeProvider
from app.infra.llm.openai_provider import OpenAIProvider


def test_claude_provider_parses_response_and_forwards_arguments():
    fake_client = MagicMock()
    fake_block = MagicMock(type="text", text="Hello there")
    fake_response = MagicMock(content=[fake_block])
    fake_response.usage.input_tokens = 10
    fake_response.usage.output_tokens = 5
    fake_client.messages.create.return_value = fake_response

    with patch("app.infra.llm.claude_provider.Anthropic", return_value=fake_client):
        provider = ClaudeProvider(api_key="fake-key", model="claude-test")
        result = provider.complete("system prompt", [ChatMessage(role="user", content="hi")])

    assert result.text == "Hello there"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "claude-test"
    assert result.latency_ms >= 0
    fake_client.messages.create.assert_called_once_with(
        model="claude-test",
        max_tokens=1024,
        temperature=0.2,
        system="system prompt",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_claude_provider_ignores_non_text_blocks():
    fake_client = MagicMock()
    text_block = MagicMock(type="text", text="answer")
    tool_block = MagicMock(type="tool_use")
    fake_response = MagicMock(content=[tool_block, text_block])
    fake_response.usage.input_tokens = 1
    fake_response.usage.output_tokens = 1
    fake_client.messages.create.return_value = fake_response

    with patch("app.infra.llm.claude_provider.Anthropic", return_value=fake_client):
        provider = ClaudeProvider(api_key="fake-key", model="claude-test")
        result = provider.complete("sys", [])

    assert result.text == "answer"


def test_openai_provider_parses_response_and_prepends_system_message():
    fake_client = MagicMock()
    fake_message = MagicMock(content="Hello from GPT")
    fake_choice = MagicMock(message=fake_message)
    fake_response = MagicMock(choices=[fake_choice])
    fake_response.usage.prompt_tokens = 20
    fake_response.usage.completion_tokens = 8
    fake_client.chat.completions.create.return_value = fake_response

    with patch("app.infra.llm.openai_provider.OpenAI", return_value=fake_client):
        provider = OpenAIProvider(api_key="fake-key", model="gpt-test")
        result = provider.complete("system prompt", [ChatMessage(role="user", content="hi")])

    assert result.text == "Hello from GPT"
    assert result.input_tokens == 20
    assert result.output_tokens == 8
    fake_client.chat.completions.create.assert_called_once_with(
        model="gpt-test",
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
        ],
    )
