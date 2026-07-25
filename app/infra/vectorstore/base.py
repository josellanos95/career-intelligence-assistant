from __future__ import annotations

from typing import Protocol

from app.domain.models import Chunk, DocumentSummary, DocumentType, ScoredChunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        doc_id: str | None = None,
        doc_type: DocumentType | None = None,
    ) -> list[ScoredChunk]: ...

    def list_documents(self) -> list[DocumentSummary]: ...

    def delete_document(self, doc_id: str) -> None: ...

    def clear(self) -> None: ...
