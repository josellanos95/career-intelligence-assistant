"""Server-rendered UI: FastAPI + Jinja2 + HTMX, no separate frontend build.

Kept deliberately independent from the JSON API in documents.py/chat.py --
both layers call the same service layer (IngestionService, RetrievalService,
ChatService), but this one returns HTML fragments shaped for htmx swaps
instead of JSON, since the two have different response needs.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.deps import get_chat_service, get_ingestion_service, get_retrieval_service, get_vector_store
from app.domain.models import DocumentType
from app.domain.prompts import AssistantMode
from app.infra.parsers.factory import UnsupportedFileType
from app.infra.vectorstore.numpy_store import NumpyVectorStore
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService

router = APIRouter(tags=["ui"])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _documents_context(store: NumpyVectorStore) -> dict:
    documents = [
        {"doc_id": d.doc_id, "doc_title": d.doc_title, "doc_type": d.doc_type.value, "num_chunks": d.chunk_count}
        for d in store.list_documents()
    ]
    return {"documents": documents}


@router.get("/", response_class=HTMLResponse)
def index(request: Request, store: NumpyVectorStore = Depends(get_vector_store)) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, **_documents_context(store)})


@router.post("/ui/upload", response_class=HTMLResponse)
async def ui_upload(
    request: Request,
    file: UploadFile = File(...),
    doc_type: DocumentType = Form(...),
    title: str = Form(""),
    ingestion: IngestionService = Depends(get_ingestion_service),
    retrieval: RetrievalService = Depends(get_retrieval_service),
    store: NumpyVectorStore = Depends(get_vector_store),
) -> HTMLResponse:
    content = await file.read()
    error = None
    try:
        chunks = ingestion.ingest(file.filename or "upload", content, doc_type, title=title or None)
        if not chunks:
            error = "No extractable text found in that file."
        else:
            retrieval.index_chunks(chunks)
    except UnsupportedFileType as exc:
        error = str(exc)

    context = {"request": request, "error": error, **_documents_context(store)}
    return templates.TemplateResponse("_documents_update.html", context)


@router.delete("/ui/documents/{doc_id}", response_class=HTMLResponse)
def ui_delete_document(
    request: Request, doc_id: str, store: NumpyVectorStore = Depends(get_vector_store)
) -> HTMLResponse:
    store.delete_document(doc_id)
    return templates.TemplateResponse("_documents_update.html", {"request": request, **_documents_context(store)})


@router.post("/ui/chat", response_class=HTMLResponse)
def ui_chat(
    request: Request,
    question: str = Form(...),
    mode: AssistantMode = Form(AssistantMode.GENERAL),
    doc_id: str = Form(""),
    chat_service: ChatService = Depends(get_chat_service),
) -> HTMLResponse:
    response = chat_service.ask(question, mode=mode, doc_id=doc_id or None)
    return templates.TemplateResponse(
        "_chat_turn.html", {"request": request, "question": question, "answer": response.text}
    )
