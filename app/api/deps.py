"""Dependency providers.

Expensive, stateful resources (vector store, embedder, LLM client) are
process-wide singletons via lru_cache. Cheap orchestration objects
(RetrievalService, ChatService) are built per-request through `Depends(...)`
rather than called directly, so tests can swap `get_embedder`/`get_llm`
with fakes via `app.dependency_overrides` and have the override propagate
through the whole chain -- calling the singleton functions directly here
instead would bypass that.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.config import Settings, get_settings
from app.infra.embeddings.base import EmbeddingProvider
from app.infra.embeddings.openai_embedder import OpenAIEmbedder
from app.infra.llm.base import LLMProvider
from app.infra.llm.factory import get_llm_provider
from app.infra.vectorstore.numpy_store import NumpyVectorStore
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_vector_store() -> NumpyVectorStore:
    return NumpyVectorStore(get_settings().vector_store_dir)


@lru_cache
def get_embedder() -> EmbeddingProvider:
    settings = get_settings()
    return OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.openai_embedding_model)


@lru_cache
def get_llm() -> LLMProvider:
    return get_llm_provider(get_settings())


def get_retrieval_service(
    embedder: EmbeddingProvider = Depends(get_embedder),
    store: NumpyVectorStore = Depends(get_vector_store),
) -> RetrievalService:
    return RetrievalService(embedder=embedder, store=store)


def get_chat_service(
    retrieval: RetrievalService = Depends(get_retrieval_service),
    llm: LLMProvider = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> ChatService:
    return ChatService(retrieval=retrieval, llm=llm, top_k=settings.max_context_chunks)


def get_ingestion_service() -> IngestionService:
    return IngestionService()
