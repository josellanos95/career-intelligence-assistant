from app.domain.models import Chunk, DocumentType
from app.infra.vectorstore.numpy_store import NumpyVectorStore


def make_chunk(id: str, doc_id: str, doc_type: DocumentType, text: str = "text") -> Chunk:
    return Chunk(id=id, doc_id=doc_id, doc_type=doc_type, doc_title="title", section="general", text=text)


def test_search_returns_most_similar_first(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    chunks = [make_chunk("a", "d1", DocumentType.RESUME), make_chunk("b", "d1", DocumentType.RESUME)]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 0.0], top_k=2)

    assert results[0].chunk.id == "a"
    assert results[0].score > results[1].score


def test_search_filters_by_doc_id():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = NumpyVectorStore(tmp)
        store.add([make_chunk("a", "d1", DocumentType.RESUME)], [[1.0, 0.0]])
        store.add([make_chunk("b", "d2", DocumentType.JOB_DESCRIPTION)], [[1.0, 0.0]])

        results = store.search([1.0, 0.0], top_k=10, doc_id="d2")

        assert len(results) == 1
        assert results[0].chunk.doc_id == "d2"


def test_search_filters_by_doc_type(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    store.add([make_chunk("a", "d1", DocumentType.RESUME)], [[1.0, 0.0]])
    store.add([make_chunk("b", "d2", DocumentType.JOB_DESCRIPTION)], [[1.0, 0.0]])

    results = store.search([1.0, 0.0], top_k=10, doc_type=DocumentType.JOB_DESCRIPTION)

    assert len(results) == 1
    assert results[0].chunk.doc_type == DocumentType.JOB_DESCRIPTION


def test_persists_and_reloads_from_disk(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    store.add([make_chunk("a", "d1", DocumentType.RESUME, text="hello")], [[1.0, 0.0]])

    reloaded = NumpyVectorStore(str(tmp_path))
    results = reloaded.search([1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk.text == "hello"


def test_clear_removes_all_data(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    store.add([make_chunk("a", "d1", DocumentType.RESUME)], [[1.0, 0.0]])

    store.clear()

    assert store.search([1.0, 0.0], top_k=10) == []


def test_search_on_empty_store_returns_empty_list(tmp_path):
    store = NumpyVectorStore(str(tmp_path))
    assert store.search([1.0, 0.0], top_k=5) == []
