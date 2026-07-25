"""Core domain entities.

Kept free of any framework or infrastructure import (no FastAPI, no OpenAI
client, no file I/O) so the domain layer stays testable and swappable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DocumentType(str, Enum):
    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"


@dataclass(frozen=True)
class ParsedDocument:
    """Raw text extracted from an uploaded file, before chunking."""

    doc_id: str
    doc_type: DocumentType
    title: str
    raw_text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with enough metadata to filter and cite it."""

    id: str
    doc_id: str
    doc_type: DocumentType
    doc_title: str
    section: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
