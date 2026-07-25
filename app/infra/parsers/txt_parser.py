from __future__ import annotations


class TxtParser:
    def parse(self, content: bytes) -> str:
        return content.decode("utf-8", errors="replace")
