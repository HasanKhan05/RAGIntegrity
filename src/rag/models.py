"""Shared immutable records passed between Phase 1 components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    document_id: str
    filename: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk(TextChunk):
    rank: int
    relevance_score: float | None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    token_usage: TokenUsage | None
