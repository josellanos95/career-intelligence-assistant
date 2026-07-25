import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_embedder, get_llm, get_vector_store
from app.infra.llm.base import LLMResponse
from app.infra.vectorstore.numpy_store import NumpyVectorStore
from app.main import app


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeLLM:
    def complete(self, system_prompt, messages):
        return LLMResponse(text="fake answer", input_tokens=1, output_tokens=1, latency_ms=1.0, model="fake")


@pytest.fixture
def client(tmp_path):
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    app.dependency_overrides[get_vector_store] = lambda: NumpyVectorStore(str(tmp_path))
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_upload_resume_and_list_documents(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("resume.txt", b"Skills\nPython, FastAPI", "text/plain")},
        data={"doc_type": "resume"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["num_chunks"] >= 1
    assert body["doc_type"] == "resume"

    listing = client.get("/documents")
    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_upload_unsupported_file_type_returns_400(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("resume.xyz", b"whatever", "application/octet-stream")},
        data={"doc_type": "resume"},
    )
    assert response.status_code == 400


def test_upload_empty_file_returns_400(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
        data={"doc_type": "resume"},
    )
    assert response.status_code == 400


def test_chat_returns_answer_grounded_in_uploaded_document(client):
    client.post(
        "/documents/upload",
        files={"file": ("resume.txt", b"Skills\nPython, FastAPI", "text/plain")},
        data={"doc_type": "resume"},
    )

    response = client.post("/chat", json={"question": "What are my skills?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer"
    assert body["model"] == "fake"


def test_delete_document_removes_it_from_listing(client):
    upload = client.post(
        "/documents/upload",
        files={"file": ("resume.txt", b"Skills\nPython", "text/plain")},
        data={"doc_type": "resume"},
    )
    doc_id = upload.json()["doc_id"]

    delete_response = client.delete(f"/documents/{doc_id}")
    assert delete_response.status_code == 200

    listing = client.get("/documents")
    assert listing.json() == []
