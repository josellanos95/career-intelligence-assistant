from __future__ import annotations

from app.infra.parsers.base import DocumentParser
from app.infra.parsers.docx_parser import DocxParser
from app.infra.parsers.pdf_parser import PdfParser
from app.infra.parsers.txt_parser import TxtParser

_PARSERS_BY_EXTENSION: dict[str, DocumentParser] = {
    ".pdf": PdfParser(),
    ".docx": DocxParser(),
    ".txt": TxtParser(),
}


class UnsupportedFileType(ValueError):
    pass


def get_parser_for(filename: str) -> DocumentParser:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        return _PARSERS_BY_EXTENSION[suffix]
    except KeyError:
        supported = ", ".join(sorted(_PARSERS_BY_EXTENSION))
        raise UnsupportedFileType(
            f"Unsupported file type '{suffix}'. Supported: {supported}"
        ) from None
