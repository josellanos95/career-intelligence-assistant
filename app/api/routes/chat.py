from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_chat_service
from app.domain.models import ChatMessage
from app.domain.prompts import AssistantMode
from app.infra.observability.logging import get_logger
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # Optional: skill-gap and interview-prep have a sensible default question
    # when left blank (see app.domain.prompts.default_question_for);
    # General Q&A does not and ChatService.ask() raises for it.
    question: str = ""
    mode: AssistantMode = AssistantMode.GENERAL
    doc_id: str | None = None
    # Chat history lives on the client and is resent each turn -- the server
    # holds no per-session state, in keeping with 12-factor's "stateless
    # process" principle. The only durable state is the vector store on disk.
    history: list[ChatTurn] = []


class ChatResponseModel(BaseModel):
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


@router.post("", response_model=ChatResponseModel)
def chat(request: ChatRequest, chat_service: ChatService = Depends(get_chat_service)) -> ChatResponseModel:
    history = [ChatMessage(role=turn.role, content=turn.content) for turn in request.history]
    try:
        response = chat_service.ask(request.question, history=history, mode=request.mode, doc_id=request.doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    # Log length, not the raw question/answer text -- resumes and job
    # questions can carry personal data, and the guardrails already avoid
    # echoing contact details back to the user (app/domain/prompts.py); logs
    # shouldn't be a side channel that undoes that.
    logger.info(
        "chat_completed",
        mode=request.mode.value,
        doc_id=request.doc_id,
        question_length=len(request.question),
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=round(response.latency_ms, 2),
    )
    return ChatResponseModel(
        answer=response.text,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
    )
