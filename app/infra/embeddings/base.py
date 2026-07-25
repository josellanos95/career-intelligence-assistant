from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returned order matches input order."""
        ...
