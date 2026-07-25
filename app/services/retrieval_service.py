from __future__ import annotations

from app.domain.models import Chunk, DocumentType, ScoredChunk
from app.infra.embeddings.base import EmbeddingProvider
from app.infra.vectorstore.base import VectorStore


class RetrievalService:
    """Ties an embedding provider to a vector store: index chunks, then
    retrieve the ones most relevant to a query, optionally scoped to one
    document (e.g. "how does my experience align with Job #2?").
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self._embedder.embed([c.text for c in chunks])
        self._store.add(chunks, embeddings)

    def search(
        self,
        query: str,
        top_k: int = 8,
        doc_id: str | None = None,
        doc_type: DocumentType | None = None,
    ) -> list[ScoredChunk]:
        query_embedding = self._embedder.embed([query])[0]
        return self._store.search(query_embedding, top_k=top_k, doc_id=doc_id, doc_type=doc_type)
