from __future__ import annotations

import uuid

from app.domain.chunking import SectionChunker
from app.domain.models import Chunk, DocumentType, ParsedDocument
from app.domain.text_normalize import normalize_text
from app.infra.parsers.factory import get_parser_for


class IngestionService:
    """Turns an uploaded file into retrievable chunks: parse -> chunk."""

    def __init__(self, chunker: SectionChunker | None = None) -> None:
        self._chunker = chunker or SectionChunker()

    def ingest(
        self,
        filename: str,
        content: bytes,
        doc_type: DocumentType,
        title: str | None = None,
    ) -> list[Chunk]:
        parser = get_parser_for(filename)
        raw_text = normalize_text(parser.parse(content))
        document = ParsedDocument(
            doc_id=str(uuid.uuid4()),
            doc_type=doc_type,
            title=title or filename,
            raw_text=raw_text,
        )
        return self._chunker.chunk(document)
