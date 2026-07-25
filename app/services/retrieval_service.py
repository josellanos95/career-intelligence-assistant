from __future__ import annotations

from app.domain.models import Chunk, DocumentType, ScoredChunk
from app.infra.embeddings.base import EmbeddingProvider
from app.infra.observability.logging import get_logger
from app.infra.vectorstore.base import VectorStore

logger = get_logger(__name__)


class RetrievalService:
    """Ties an embedding provider to a vector store: index chunks, then
    retrieve the ones most relevant to a query, optionally scoped to one
    document (e.g. "how does my experience align with Job #2?").
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index one newly-ingested document's chunks.

        Only one resume may be active at a time: uploading a new resume
        replaces the previous one instead of accumulating both. Retrieval has
        no "which resume" scoping the way job descriptions do (no `doc_id`
        focus for resumes), and the resume citation label is just "Resume"
        with no title -- so two resumes in the store at once would silently
        blend both people's/versions' chunks into every answer with no way
        for the model, or the user, to tell them apart. Job descriptions are
        unaffected: comparing against multiple jobs at once is the intended
        use case, via the "Focus on job" selector.
        """
        if not chunks:
            return
        if chunks[0].doc_type == DocumentType.RESUME:
            self._replace_existing_resume()
        embeddings = self._embedder.embed([c.text for c in chunks])
        self._store.add(chunks, embeddings)

    def _replace_existing_resume(self) -> None:
        for doc in self._store.list_documents():
            if doc.doc_type == DocumentType.RESUME:
                self._store.delete_document(doc.doc_id)
                logger.info("resume_replaced", replaced_doc_id=doc.doc_id)

    def search(
        self,
        query: str,
        top_k: int = 8,
        doc_id: str | None = None,
        doc_type: DocumentType | None = None,
    ) -> list[ScoredChunk]:
        query_embedding = self._embedder.embed([query])[0]
        return self._store.search(query_embedding, top_k=top_k, doc_id=doc_id, doc_type=doc_type)

    def search_job_descriptions(self, query: str, top_k: int, doc_id: str | None = None) -> list[ScoredChunk]:
        """Retrieve job-description chunks for one chat question.

        Scoped to one job (`doc_id` given), this is a plain top-k search. But
        scoped to every uploaded job at once (`doc_id=None`, the UI's "All
        documents" mode), a single top-k search pooling every job's chunks
        together can end up dominated by whichever job happens to embed
        closest to the question -- found live, a generic "what am I missing"
        question with three jobs uploaded returned gaps for only one of them,
        with nothing to signal that to the user beyond reading each citation
        closely. Searching each job separately and concatenating instead
        guarantees every uploaded job contributes up to `top_k` chunks -- the
        same "a generous, fixed budget beats an even split" call already made
        for the resume budget in ChatService, applied here now that job
        descriptions can also outnumber each other in a shared pool.
        """
        query_embedding = self._embedder.embed([query])[0]

        if doc_id is not None:
            return self._store.search(
                query_embedding, top_k=top_k, doc_id=doc_id, doc_type=DocumentType.JOB_DESCRIPTION
            )

        job_doc_ids = sorted(
            {doc.doc_id for doc in self._store.list_documents() if doc.doc_type == DocumentType.JOB_DESCRIPTION}
        )
        if len(job_doc_ids) <= 1:
            return self._store.search(query_embedding, top_k=top_k, doc_type=DocumentType.JOB_DESCRIPTION)

        results: list[ScoredChunk] = []
        for job_id in job_doc_ids:
            results.extend(
                self._store.search(
                    query_embedding, top_k=top_k, doc_id=job_id, doc_type=DocumentType.JOB_DESCRIPTION
                )
            )
        return results
