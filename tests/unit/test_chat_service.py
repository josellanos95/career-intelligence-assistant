import pytest

from app.domain.models import ChatMessage, Chunk, DocumentType
from app.domain.prompts import AssistantMode
from app.infra.llm.base import LLMResponse
from app.infra.vectorstore.numpy_store import NumpyVectorStore
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0] for _ in texts]


class FakeLLM:
    def __init__(self):
        self.last_system_prompt = None
        self.last_messages = None

    def complete(self, system_prompt, messages):
        self.last_system_prompt = system_prompt
        self.last_messages = messages
        return LLMResponse(text="fake answer", input_tokens=1, output_tokens=1, latency_ms=0.0, model="fake")


def _make_service(tmp_path, llm=None):
    store = NumpyVectorStore(str(tmp_path))
    retrieval = RetrievalService(embedder=FakeEmbedder(), store=store)
    return ChatService(retrieval=retrieval, llm=llm or FakeLLM()), retrieval


def test_ask_includes_retrieved_context_in_user_message(tmp_path):
    fake_llm = FakeLLM()
    service, retrieval = _make_service(tmp_path, fake_llm)
    retrieval.index_chunks(
        [Chunk(id="c1", doc_id="resume", doc_type=DocumentType.RESUME, doc_title="Resume", section="Skills", text="Python expert")]
    )

    response = service.ask("What are my skills?")

    assert response.text == "fake answer"
    assert "Python expert" in fake_llm.last_messages[-1].content


def test_ask_uses_mode_specific_system_prompt(tmp_path):
    fake_llm = FakeLLM()
    service, _ = _make_service(tmp_path, fake_llm)

    service.ask("Am I a fit?", mode=AssistantMode.SKILL_GAP)

    assert "missing" in fake_llm.last_system_prompt.lower()


def test_ask_forwards_conversation_history_before_the_new_turn(tmp_path):
    fake_llm = FakeLLM()
    service, _ = _make_service(tmp_path, fake_llm)
    history = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")]

    service.ask("follow up question", history=history)

    assert fake_llm.last_messages[0] == history[0]
    assert fake_llm.last_messages[1] == history[1]
    assert "follow up question" in fake_llm.last_messages[2].content


def test_ask_scopes_retrieval_to_a_single_document(tmp_path):
    fake_llm = FakeLLM()
    service, retrieval = _make_service(tmp_path, fake_llm)
    retrieval.index_chunks(
        [
            Chunk(id="j1", doc_id="job-1", doc_type=DocumentType.JOB_DESCRIPTION, doc_title="Job 1", section="Requirements", text="Needs Python"),
            Chunk(id="j2", doc_id="job-2", doc_type=DocumentType.JOB_DESCRIPTION, doc_title="Job 2", section="Requirements", text="Needs Java"),
        ]
    )

    service.ask("What does this job need?", doc_id="job-2")

    assert "Needs Java" in fake_llm.last_messages[-1].content
    assert "Needs Python" not in fake_llm.last_messages[-1].content


def test_ask_always_includes_resume_even_when_scoped_to_one_job(tmp_path):
    """Regression test: scoping to a single job must not exclude the resume.

    Found live: a doc_id filter applied to the whole retrieval call excluded
    the resume (different doc_id), and the model quietly filled the gap by
    fabricating plausible-looking resume citations instead of refusing.
    Resume and job-description chunks must be fetched independently.
    """
    fake_llm = FakeLLM()
    service, retrieval = _make_service(tmp_path, fake_llm)
    retrieval.index_chunks(
        [
            Chunk(id="r1", doc_id="resume", doc_type=DocumentType.RESUME, doc_title="Resume", section="Skills", text="Python expert"),
            Chunk(id="j1", doc_id="job-1", doc_type=DocumentType.JOB_DESCRIPTION, doc_title="Job 1", section="Requirements", text="Needs Python"),
            Chunk(id="j2", doc_id="job-2", doc_type=DocumentType.JOB_DESCRIPTION, doc_title="Job 2", section="Requirements", text="Needs Java"),
        ]
    )

    service.ask("Am I a fit for this job?", mode=AssistantMode.SKILL_GAP, doc_id="job-2")

    content = fake_llm.last_messages[-1].content
    assert "Python expert" in content
    assert "Needs Java" in content
    assert "Needs Python" not in content


def test_ask_falls_back_to_a_default_question_for_skill_gap_when_blank(tmp_path):
    fake_llm = FakeLLM()
    service, _ = _make_service(tmp_path, fake_llm)

    service.ask("   ", mode=AssistantMode.SKILL_GAP)

    assert "missing" in fake_llm.last_messages[-1].content.lower()


def test_ask_falls_back_to_a_default_question_for_interview_prep_when_blank(tmp_path):
    fake_llm = FakeLLM()
    service, _ = _make_service(tmp_path, fake_llm)

    service.ask("", mode=AssistantMode.INTERVIEW_PREP)

    assert "interview" in fake_llm.last_messages[-1].content.lower()


def test_ask_raises_for_a_blank_question_in_general_mode(tmp_path):
    fake_llm = FakeLLM()
    service, _ = _make_service(tmp_path, fake_llm)

    with pytest.raises(ValueError):
        service.ask("", mode=AssistantMode.GENERAL)
