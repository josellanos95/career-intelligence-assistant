"""Parser interface. Every concrete parser turns a file's bytes into raw text;
chunking and everything downstream only ever deals in plain text.
"""
from __future__ import annotations

from typing import Protocol


class DocumentParser(Protocol):
    def parse(self, content: bytes) -> str:
        """Extract raw text from a file's bytes."""
        ...
