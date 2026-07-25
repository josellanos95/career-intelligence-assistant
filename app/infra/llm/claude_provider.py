from __future__ import annotations

import time

from anthropic import Anthropic

from app.domain.models import ChatMessage
from app.infra.llm.base import LLMResponse


class ClaudeProvider:
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024) -> None:
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, messages: list[ChatMessage]) -> LLMResponse:
        start = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model=self._model,
        )
