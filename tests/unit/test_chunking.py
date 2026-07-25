from app.domain.chunking import SectionChunker
from app.domain.models import DocumentType, ParsedDocument

SAMPLE_RESUME = """John Doe
AI Engineer

Summary
Experienced engineer building agentic systems.

Skills
Python, TypeScript, FastAPI, LangGraph

Experience
Acme Corp - Senior Engineer (2022-2024)
Built RAG pipelines and agent orchestration.
"""


def test_splits_into_known_sections():
    doc = ParsedDocument(doc_id="d1", doc_type=DocumentType.RESUME, title="resume.txt", raw_text=SAMPLE_RESUME)
    chunks = SectionChunker().chunk(doc)
    sections = {c.section for c in chunks}
    assert {"Summary", "Skills", "Experience"} <= sections


def test_text_without_headers_falls_back_to_general_section():
    doc = ParsedDocument(
        doc_id="d2",
        doc_type=DocumentType.JOB_DESCRIPTION,
        title="jd.txt",
        raw_text="Just some plain text with no headers at all.",
    )
    chunks = SectionChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].section == "general"


def test_long_section_is_split_with_overlap():
    long_body = "word " * 500
    text = f"Experience\n{long_body}"
    doc = ParsedDocument(doc_id="d3", doc_type=DocumentType.RESUME, title="resume.txt", raw_text=text)
    chunker = SectionChunker(max_chars=200, overlap=50)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(c.section == "Experience" for c in chunks)


def test_chunk_ids_are_unique():
    doc = ParsedDocument(doc_id="d4", doc_type=DocumentType.RESUME, title="resume.txt", raw_text=SAMPLE_RESUME)
    chunks = SectionChunker().chunk(doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_about_company_header_is_detected_generically():
    text = "About Acme Corp\nAcme Corp builds things.\n\nYour Mission\nDo great work."
    doc = ParsedDocument(doc_id="d6", doc_type=DocumentType.JOB_DESCRIPTION, title="jd.txt", raw_text=text)
    chunks = SectionChunker().chunk(doc)
    sections = {c.section for c in chunks}
    assert "About Acme Corp" in sections
    assert "Your Mission" in sections


def test_no_chunk_exceeds_max_chars_by_much():
    long_body = "word " * 1000
    text = f"Skills\n{long_body}"
    doc = ParsedDocument(doc_id="d5", doc_type=DocumentType.RESUME, title="resume.txt", raw_text=text)
    chunker = SectionChunker(max_chars=300, overlap=40)
    chunks = chunker.chunk(doc)
    assert all(len(c.text) <= 300 + 10 for c in chunks)
