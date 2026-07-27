"""Section-aware chunking for resumes and job descriptions.

Why not fixed-size chunking: resumes and job descriptions are already
organized into semantically meaningful sections (Experience, Skills,
Requirements...). Splitting blindly every N characters routinely cuts a
bullet point in half or merges "Education" with "Certifications", which
hurts retrieval precision for questions like "what skills are listed?".
Instead we first split on section headers, then apply a fixed-size
fallback *within* a section only if that section is too long for a
single chunk (e.g. a long "Experience" block spanning several jobs).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.domain.models import Chunk, ParsedDocument

# Bilingual (EN/ES) since resumes and job descriptions in this project can be
# in either language (see README: multilingual retrieval is a deliberate
# demo of the embedding model's cross-lingual robustness, not an accident).
_KNOWN_HEADERS = {
    # English - resume
    "summary", "professional summary", "profile", "objective", "about",
    "experience", "professional experience", "work experience",
    "employment history", "career history",
    "education", "skills", "technical skills", "core competencies",
    "key strengths", "certifications", "projects", "featured projects",
    "languages",
    # English - job description
    "about the role", "about the company", "the role",
    "responsibilities", "what you'll do", "requirements", "what you bring",
    "qualifications", "what we offer", "bonus skills", "bonus skills / experience",
    "nice to have", "preferred qualifications", "ready to apply",
    # Spanish (accents stripped by _normalize before comparison)
    "resumen profesional", "resumen", "perfil", "objetivo",
    "fortalezas clave", "habilidades tecnicas", "habilidades",
    "experiencia profesional", "experiencia laboral", "educacion",
    "certificaciones", "proyectos destacados", "proyectos", "idiomas",
}

_HEADER_LINE = re.compile(r"^\s{0,3}(?P<title>[^\d@|]{1,70})\s*$")
_MAX_HEADER_WORDS = 8

_DEFAULT_SECTION = "general"


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _normalize(title: str) -> str:
    return _strip_accents(title.strip().lower()).rstrip(":?")


def _is_section_header(line: str) -> str | None:
    match = _HEADER_LINE.match(line)
    if not match:
        return None
    title = match.group("title").strip()
    if not title or title.endswith(".") or len(title.split()) > _MAX_HEADER_WORDS:
        return None
    normalized = _normalize(title)
    if normalized in _KNOWN_HEADERS:
        return title.rstrip(":")
    # "About <Company Name>" varies per job posting and can't be enumerated
    # exactly, but as a short, period-less, prefix-matched line it's a safe bet.
    if normalized.startswith("about ") and len(title.split()) <= 5:
        return title.rstrip(":")
    return None


def _split_into_sections(raw_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = _DEFAULT_SECTION
    current_lines: list[str] = []

    for line in raw_text.splitlines():
        header = _is_section_header(line)
        if header is not None:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = header
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # avoid cutting mid-word: back off to the last whitespace in range
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


@dataclass
class SectionChunker:
    max_chars: int = 1000
    overlap: int = 150

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        sections = _split_into_sections(document.raw_text)
        chunks: list[Chunk] = []
        for section_title, section_text in sections:
            pieces = _split_long_text(section_text, self.max_chars, self.overlap)
            for i, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        id=f"{document.doc_id}::{section_title}::{i}",
                        doc_id=document.doc_id,
                        doc_type=document.doc_type,
                        doc_title=document.title,
                        section=section_title,
                        text=piece,
                    )
                )
        return chunks
