from __future__ import annotations

from app.domain.models import ChatMessage, DocumentType
from app.domain.prompts import AssistantMode, build_system_prompt, build_user_turn, default_question_for
from app.infra.llm.base import LLMProvider, LLMResponse
from app.services.retrieval_service import RetrievalService


class ChatService:
    """Orchestrates retrieval -> prompt assembly -> LLM call for one turn.

    Retrieval always runs against the latest question (no query rewriting
    using prior turns) -- a known limitation for follow-ups like "what about
    the other one?", documented in the README rather than solved here.

    Resume and job-description chunks are retrieved in two separate calls
    rather than one filtered call. A single call scoped by `doc_id` (e.g.
    "only Job #2") would exclude the resume entirely, since the resume has a
    different doc_id -- found live, the model filled the gap by inventing
    plausible-looking resume citations instead of refusing. Every skill-gap
    or interview-prep question is inherently resume-vs-job, so the resume
    must always be eligible regardless of which job is being scoped to.

    The resume gets its own generous, fixed budget instead of splitting
    `top_k` evenly with the job: a resume is one short document (a handful
    of chunks), so a 50/50 split under-fetches it -- found live, an even
    split missed the Skills section on a skill-gap question and the model
    cited "(Resume - Skills)" anyway for a claim it was actually inferring
    from the Experience section. Retrieving generously from the resume
    costs little (it's small) and removes that failure mode.

    Job-description retrieval goes through
    RetrievalService.search_job_descriptions rather than a plain search, for
    the same reason: when comparing against every uploaded job at once
    (doc_id=None), a single pooled top-k search can end up returning chunks
    from only one of several uploaded jobs -- see that method's docstring.

    A blank question is only an error in General Q&A mode. Skill-gap and
    interview-prep are meaningful requests on their own once a mode (and
    usually a job) is picked, so a blank box falls back to
    domain.prompts.default_question_for(mode) instead of forcing the user to
    type something the UI already implied.
    """

    _RESUME_TOP_K = 10

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
        question = question.strip() or default_question_for(mode) or ""
        if not question:
            raise ValueError("A question is required for General Q&A.")

        resume_chunks = self._retrieval.search(question, top_k=self._RESUME_TOP_K, doc_type=DocumentType.RESUME)
        job_chunks = self._retrieval.search_job_descriptions(question, top_k=self._top_k, doc_id=doc_id)
        system_prompt = build_system_prompt(mode)
        user_turn = build_user_turn(question, resume_chunks + job_chunks)
        messages = [*(history or []), ChatMessage(role="user", content=user_turn)]
        return self._llm.complete(system_prompt, messages)
