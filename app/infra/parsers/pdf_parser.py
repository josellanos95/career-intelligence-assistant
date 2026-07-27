from __future__ import annotations

import pymupdf


class PdfParser:
    """Extracts text from PDFs using pymupdf.

    pypdf was tried first but collapses real line breaks into single-space
    runs and mangles kerning on design-heavy PDFs (text placed by absolute
    position rather than text flow) -- section headers came out with stray
    spaces inserted mid-word, with no way to tell a header from body text.
    pymupdf preserves the original line structure, which the section
    chunker depends on.
    """

    def parse(self, content: bytes) -> str:
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
