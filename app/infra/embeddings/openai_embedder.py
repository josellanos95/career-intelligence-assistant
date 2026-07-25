from __future__ import annotations

from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        # OpenAI does not guarantee response order matches input order across
        # all client versions, but does expose an explicit index -- sort on it
        # rather than trust list order.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
