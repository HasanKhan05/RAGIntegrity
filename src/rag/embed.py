"""Local Sentence Transformer embedding adapter."""

from __future__ import annotations

import os

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Load the configured local model only when embeddings are requested."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()
