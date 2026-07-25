from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain.models import ChatMessage


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class LLMProvider(Protocol):
    def complete(self, system_prompt: str, messages: list[ChatMessage]) -> LLMResponse:
        """Run one chat completion given a system prompt and conversation history."""
        ...
