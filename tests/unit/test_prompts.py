from app.domain.models import Chunk, DocumentType, ScoredChunk
from app.domain.prompts import AssistantMode, build_system_prompt, build_user_turn, format_context


def make_scored_chunk(doc_type, title, section, text, score=0.9):
    chunk = Chunk(id="1", doc_id="d1", doc_type=doc_type, doc_title=title, section=section, text=text)
    return ScoredChunk(chunk=chunk, score=score)


def test_system_prompt_includes_grounding_guardrail_for_every_mode():
    for mode in AssistantMode:
        prompt = build_system_prompt(mode).lower()
        assert "only use facts present in the context" in prompt


def test_system_prompt_includes_pii_guardrail():
    prompt = build_system_prompt(AssistantMode.GENERAL).lower()
    assert "contact details" in prompt


def test_skill_gap_prompt_requests_structured_sections():
    prompt = build_system_prompt(AssistantMode.SKILL_GAP).lower()
    assert "missing" in prompt
    assert "matched skills" in prompt


def test_interview_prep_prompt_asks_for_questions():
    prompt = build_system_prompt(AssistantMode.INTERVIEW_PREP).lower()
    assert "interview questions" in prompt


def test_format_context_labels_resume_and_job_chunks_differently():
    chunks = [
        make_scored_chunk(DocumentType.RESUME, "My Resume", "Skills", "Python, FastAPI"),
        make_scored_chunk(DocumentType.JOB_DESCRIPTION, "Forward Deployed Engineer", "What You Bring", "3+ years..."),
    ]

    context = format_context(chunks)

    assert "[Resume - Skills]" in context
    assert "[Job: Forward Deployed Engineer - What You Bring]" in context


def test_format_context_returns_empty_string_for_no_chunks():
    assert format_context([]) == ""


def test_build_user_turn_handles_empty_context():
    result = build_user_turn("What skills am I missing?", [])

    assert "no relevant context" in result.lower()
    assert "What skills am I missing?" in result


def test_build_user_turn_includes_formatted_context():
    chunks = [make_scored_chunk(DocumentType.RESUME, "My Resume", "Skills", "Python, FastAPI")]

    result = build_user_turn("What are my skills?", chunks)

    assert "[Resume - Skills]" in result
    assert "Python, FastAPI" in result
