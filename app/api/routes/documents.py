from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_ingestion_service, get_retrieval_service, get_vector_store
from app.domain.models import DocumentType
from app.infra.parsers.factory import UnsupportedFileType
from app.infra.vectorstore.numpy_store import NumpyVectorStore
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: DocumentType = Form(...),
    title: str | None = Form(None),
    ingestion: IngestionService = Depends(get_ingestion_service),
    retrieval: RetrievalService = Depends(get_retrieval_service),
) -> dict:
    content = await file.read()
    try:
        chunks = ingestion.ingest(file.filename or "upload", content, doc_type, title=title)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in the uploaded file.")

    retrieval.index_chunks(chunks)
    return {
        "doc_id": chunks[0].doc_id,
        "title": chunks[0].doc_title,
        "doc_type": doc_type.value,
        "num_chunks": len(chunks),
    }


@router.get("")
def list_documents(store: NumpyVectorStore = Depends(get_vector_store)) -> list[dict]:
    return [
        {
            "doc_id": doc.doc_id,
            "title": doc.doc_title,
            "doc_type": doc.doc_type.value,
            "num_chunks": doc.chunk_count,
        }
        for doc in store.list_documents()
    ]


@router.delete("/{doc_id}")
def delete_document(doc_id: str, store: NumpyVectorStore = Depends(get_vector_store)) -> dict:
    store.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
