from __future__ import annotations

import time

from openai import OpenAI

from app.domain.models import ChatMessage
from app.infra.llm.base import LLMResponse


class OpenAIProvider:
    # See ClaudeProvider for why temperature defaults low but not to zero.
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024, temperature: float = 0.2) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system_prompt: str, messages: list[ChatMessage]) -> LLMResponse:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "system", "content": system_prompt}]
            + [{"role": m.role, "content": m.content} for m in messages],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            model=self._model,
        )
