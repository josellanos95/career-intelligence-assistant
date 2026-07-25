from __future__ import annotations

from app.domain.models import ChatMessage
from app.domain.prompts import AssistantMode, build_system_prompt, build_user_turn
from app.infra.llm.base import LLMProvider, LLMResponse
from app.services.retrieval_service import RetrievalService


class ChatService:
    """Orchestrates retrieval -> prompt assembly -> LLM call for one turn.

    Retrieval always runs against the latest question (no query rewriting
    using prior turns) -- a known limitation for follow-ups like "what about
    the other one?", documented in the README rather than solved here.
    """

    def __init__(self, retrieval: RetrievalService, llm: LLMProvider, top_k: int = 8) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._top_k = top_k

    def ask(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        mode: AssistantMode = AssistantMode.GENERAL,
        doc_id: str | None = None,
    ) -> LLMResponse:
        chunks = self._retrieval.search(question, top_k=self._top_k, doc_id=doc_id)
        system_prompt = build_system_prompt(mode)
        user_turn = build_user_turn(question, chunks)
        messages = [*(history or []), ChatMessage(role="user", content=user_turn)]
        return self._llm.complete(system_prompt, messages)
