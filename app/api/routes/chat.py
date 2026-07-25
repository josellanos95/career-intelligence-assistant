from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_chat_service
from app.domain.models import ChatMessage
from app.domain.prompts import AssistantMode
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
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
    response = chat_service.ask(request.question, history=history, mode=request.mode, doc_id=request.doc_id)
    return ChatResponseModel(
        answer=response.text,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
    )
