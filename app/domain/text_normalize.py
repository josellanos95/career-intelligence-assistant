"""Normalizes typographic artifacts that PDFs/DOCX commonly introduce
(ligatures, smart quotes, non-breaking spaces) so header matching, chunking,
and the text sent to the LLM all see plain, predictable characters.
"""
from __future__ import annotations

_REPLACEMENTS = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    " ": " ",
}


def normalize_text(text: str) -> str:
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text
