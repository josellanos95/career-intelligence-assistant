from app.domain.models import DocumentType
from app.services.ingestion_service import IngestionService


def test_ingest_txt_produces_chunks_with_correct_metadata():
    service = IngestionService()
    content = b"Summary\nBuilds AI systems.\n\nSkills\nPython, FastAPI"

    chunks = service.ingest("resume.txt", content, DocumentType.RESUME, title="My Resume")

    assert len(chunks) >= 1
    assert all(c.doc_type == DocumentType.RESUME for c in chunks)
    assert all(c.doc_title == "My Resume" for c in chunks)


def test_ingest_assigns_same_doc_id_to_all_chunks_of_one_document():
    service = IngestionService()
    content = b"Summary\nSection one.\n\nSkills\nSection two."

    chunks = service.ingest("jd.txt", content, DocumentType.JOB_DESCRIPTION)

    doc_ids = {c.doc_id for c in chunks}
    assert len(doc_ids) == 1


def test_ingest_two_documents_get_different_doc_ids():
    service = IngestionService()
    chunks_a = service.ingest("a.txt", b"Summary\nFirst doc.", DocumentType.RESUME)
    chunks_b = service.ingest("b.txt", b"Summary\nSecond doc.", DocumentType.RESUME)

    assert chunks_a[0].doc_id != chunks_b[0].doc_id
