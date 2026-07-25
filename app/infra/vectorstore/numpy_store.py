"""A minimal vector store: numpy array + cosine similarity, no ANN index.

Why not Chroma/Pinecone/pgvector: at this project's scale (a resume plus a
handful of job descriptions -- low hundreds of chunks at most), an
approximate-nearest-neighbor index buys nothing. A brute-force dot product
over a normalized (n, d) numpy array is O(n*d), which for n in the low
thousands is sub-millisecond -- and it drops a native-compiled dependency
(hnswlib/faiss) that is fragile to install across platforms (see the
project's git history: chromadb was tried first and dropped for exactly
this reason on Windows). The first thing to swap when the corpus grows
past what fits in memory, or when multiple concurrent writers are needed,
is this class -- behind the same `VectorStore` interface.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.domain.models import Chunk, DocumentSummary, DocumentType, ScoredChunk


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "id": chunk.id,
        "doc_id": chunk.doc_id,
        "doc_type": chunk.doc_type.value,
        "doc_title": chunk.doc_title,
        "section": chunk.section,
        "text": chunk.text,
    }


def _chunk_from_dict(data: dict) -> Chunk:
    return Chunk(
        id=data["id"],
        doc_id=data["doc_id"],
        doc_type=DocumentType(data["doc_type"]),
        doc_title=data["doc_title"],
        section=data["section"],
        text=data["text"],
    )


class NumpyVectorStore:
    def __init__(self, persist_dir: str) -> None:
        self._dir = Path(persist_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self._dir / "vectors.npy"
        self._metadata_path = self._dir / "metadata.json"
        self._embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        if self._vectors_path.exists() and self._metadata_path.exists():
            self._embeddings = np.load(self._vectors_path)
            with open(self._metadata_path, encoding="utf-8") as f:
                self._chunks = [_chunk_from_dict(d) for d in json.load(f)]

    def _persist(self) -> None:
        np.save(self._vectors_path, self._embeddings)
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump([_chunk_to_dict(c) for c in self._chunks], f, ensure_ascii=False)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        vectors = self._l2_normalize(np.array(embeddings, dtype=np.float32))
        self._embeddings = (
            vectors if self._embeddings.size == 0 else np.vstack([self._embeddings, vectors])
        )
        self._chunks.extend(chunks)
        self._persist()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        doc_id: str | None = None,
        doc_type: DocumentType | None = None,
    ) -> list[ScoredChunk]:
        if not self._chunks:
            return []

        query = self._l2_normalize(np.array([query_embedding], dtype=np.float32))[0]
        scores = self._embeddings @ query  # both sides normalized -> cosine similarity

        results = [
            ScoredChunk(chunk=chunk, score=float(score))
            for chunk, score in zip(self._chunks, scores, strict=True)
            if (doc_id is None or chunk.doc_id == doc_id)
            and (doc_type is None or chunk.doc_type == doc_type)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def list_documents(self) -> list[DocumentSummary]:
        counts: dict[str, DocumentSummary] = {}
        for chunk in self._chunks:
            existing = counts.get(chunk.doc_id)
            if existing is None:
                counts[chunk.doc_id] = DocumentSummary(
                    doc_id=chunk.doc_id, doc_title=chunk.doc_title, doc_type=chunk.doc_type, chunk_count=1
                )
            else:
                counts[chunk.doc_id] = DocumentSummary(
                    doc_id=existing.doc_id,
                    doc_title=existing.doc_title,
                    doc_type=existing.doc_type,
                    chunk_count=existing.chunk_count + 1,
                )
        return list(counts.values())

    def delete_document(self, doc_id: str) -> None:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        self._chunks = [self._chunks[i] for i in keep]
        self._embeddings = self._embeddings[keep] if self._embeddings.size else self._embeddings
        self._persist()

    def clear(self) -> None:
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._chunks = []
        for path in (self._vectors_path, self._metadata_path):
            path.unlink(missing_ok=True)
