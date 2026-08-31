"""Generate concise grounded answers with the configured Gemini model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import types

from src.rag.config import Settings
from src.rag.models import GeneratedAnswer, RetrievedChunk, TokenUsage


class GenerationError(RuntimeError):
    """Raised when Gemini cannot return a usable grounded answer."""


def build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{chunk.filename}, p. {chunk.page_number}]\n{chunk.text}"
        for chunk in chunks
    )
    return (
        "Answer only from the brochure context below. Keep the answer concise and "
        "cite sources as [filename, p. N]. If the context is insufficient, say so.\n\n"
        f"Question: {question.strip()}\n\nContext:\n{context}"
    )


class GeminiGenerator:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def generate(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> GeneratedAnswer:
        self.settings.require_generation()
        client = self._client or genai.Client(api_key=self.settings.llm_api_key)
        try:
            response = client.models.generate_content(
                model=self.settings.llm_model,
                contents=build_prompt(question, chunks),
                config=types.GenerateContentConfig(
                    temperature=self.settings.llm_temperature,
                    max_output_tokens=self.settings.max_output_tokens,
                ),
            )
            answer_text = str(response.text or "").strip()
            if not answer_text:
                raise GenerationError("Gemini generation failed")
            usage = getattr(response, "usage_metadata", None)
            token_usage = None
            if usage is not None:
                token_usage = TokenUsage(
                    input_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=getattr(usage, "candidates_token_count", None),
                    total_tokens=getattr(usage, "total_token_count", None),
                )
            return GeneratedAnswer(text=answer_text, token_usage=token_usage)
        except GenerationError:
            raise
        except Exception as error:
            raise GenerationError("Gemini generation failed") from error
