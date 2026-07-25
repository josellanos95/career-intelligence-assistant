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
