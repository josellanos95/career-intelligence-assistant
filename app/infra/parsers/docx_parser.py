from __future__ import annotations

import io

from docx import Document as DocxDocument


class DocxParser:
    def parse(self, content: bytes) -> str:
        document = DocxDocument(io.BytesIO(content))
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs)
