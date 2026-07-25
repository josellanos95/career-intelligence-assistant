from app.domain.models import Chunk, DocumentType
from app.infra.vectorstore.numpy_store import NumpyVectorStore
from app.services.retrieval_service import RetrievalService


class KeywordEmbedder:
    """Deterministic fake embedder for tests: a 3-dim one-hot-ish vector over
    a tiny fixed vocabulary. No network calls, no API key needed -- mirrors
    the "mock the LLM and network" testing philosophy used for the rest of
    this project's harnesses.
    """

    _VOCAB = ["python", "java", "rag"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if kw in text.lower() else 0.0 for kw in self._VOCAB] for text in texts]


def make_chunk(id: str, doc_id: str, doc_type: DocumentType, text: str) -> Chunk:
    return Chunk(id=id, doc_id=doc_id, doc_type=doc_type, doc_title="t", section="s", text=text)


def test_search_ranks_more_relevant_chunk_first(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    service = RetrievalService(embedder=KeywordEmbedder(), store=store)
    service.index_chunks(
        [
            make_chunk("a", "resume", DocumentType.RESUME, "Experienced Python developer with RAG pipelines."),
            make_chunk("b", "resume", DocumentType.RESUME, "Java backend developer."),
        ]
    )

    results = service.search("python rag experience", top_k=2)

    assert results[0].chunk.id == "a"


def test_search_can_scope_to_a_single_document(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    service = RetrievalService(embedder=KeywordEmbedder(), store=store)
    service.index_chunks(
        [
            make_chunk("a", "job-1", DocumentType.JOB_DESCRIPTION, "Requires Python and RAG."),
            make_chunk("b", "job-2", DocumentType.JOB_DESCRIPTION, "Requires Python and RAG."),
        ]
    )

    results = service.search("python rag", top_k=10, doc_id="job-2")

    assert len(results) == 1
    assert results[0].chunk.doc_id == "job-2"


def test_index_chunks_with_empty_list_does_not_raise(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    service = RetrievalService(embedder=KeywordEmbedder(), store=store)

    service.index_chunks([])

    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


def test_indexing_a_new_resume_replaces_the_previous_one(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    service = RetrievalService(embedder=KeywordEmbedder(), store=store)
    service.index_chunks([make_chunk("r1", "resume-old", DocumentType.RESUME, "Old Python resume.")])

    service.index_chunks([make_chunk("r2", "resume-new", DocumentType.RESUME, "New Java resume.")])

    results = service.search("python java", top_k=10, doc_type=DocumentType.RESUME)
    assert len(results) == 1
    assert results[0].chunk.doc_id == "resume-new"


def test_indexing_a_new_resume_does_not_remove_job_descriptions(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    service = RetrievalService(embedder=KeywordEmbedder(), store=store)
    service.index_chunks([make_chunk("j1", "job-1", DocumentType.JOB_DESCRIPTION, "Requires Python.")])
    service.index_chunks([make_chunk("r1", "resume-a", DocumentType.RESUME, "Python resume.")])

    service.index_chunks([make_chunk("r2", "resume-b", DocumentType.RESUME, "Java resume.")])

    results = service.search("python", top_k=10, doc_type=DocumentType.JOB_DESCRIPTION)
    assert len(results) == 1
    assert results[0].chunk.doc_id == "job-1"


def test_search_job_descriptions_scoped_to_one_job_ignores_the_others(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    service = RetrievalService(embedder=KeywordEmbedder(), store=store)
    service.index_chunks(
        [
            make_chunk("a", "job-a", DocumentType.JOB_DESCRIPTION, "Python Java RAG role."),
            make_chunk("b", "job-b", DocumentType.JOB_DESCRIPTION, "Python only role."),
        ]
    )

    results = service.search_job_descriptions("python java rag", top_k=10, doc_id="job-b")

    assert len(results) == 1
    assert results[0].chunk.doc_id == "job-b"


def test_search_job_descriptions_across_all_jobs_includes_every_job(tmp_path):
    """Regression test: a plain pooled top-k search across every uploaded job
    can silently drop jobs whose chunks embed less closely to the question
    than another job's -- found live, asking a generic skill-gap question
    with three jobs uploaded returned gaps for only one of them.
    search_job_descriptions(doc_id=None) must search each job separately so
    every uploaded job is guaranteed to contribute.
    """
    store = NumpyVectorStore(str(tmp_path))
    service = RetrievalService(embedder=KeywordEmbedder(), store=store)
    service.index_chunks(
        [
            make_chunk("a", "job-a", DocumentType.JOB_DESCRIPTION, "Python Java RAG role."),
            make_chunk("b", "job-b", DocumentType.JOB_DESCRIPTION, "Python only role."),
            make_chunk("c", "job-c", DocumentType.JOB_DESCRIPTION, "RAG only role."),
        ]
    )

    # Demonstrates the failure mode directly: a single pooled top-1 search
    # returns only job-a (a perfect match on all three keywords) and drops
    # job-b and job-c entirely.
    pooled = service.search("python java rag", top_k=1, doc_type=DocumentType.JOB_DESCRIPTION)
    assert {r.chunk.doc_id for r in pooled} == {"job-a"}

    results = service.search_job_descriptions("python java rag", top_k=1, doc_id=None)

    assert {r.chunk.doc_id for r in results} == {"job-a", "job-b", "job-c"}


def test_search_job_descriptions_with_a_single_job_uploaded(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    service = RetrievalService(embedder=KeywordEmbedder(), store=store)
    service.index_chunks([make_chunk("a", "job-a", DocumentType.JOB_DESCRIPTION, "Python role.")])

    results = service.search_job_descriptions("python", top_k=5, doc_id=None)

    assert len(results) == 1
    assert results[0].chunk.doc_id == "job-a"
