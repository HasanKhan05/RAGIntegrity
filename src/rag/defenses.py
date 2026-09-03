"""Attack-blind retrieval defenses for the local RAG testbed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from src.rag.models import RetrievedChunk


_INSTRUCTION_PATTERNS = (
    "when answering",
    "always state",
    "respond with",
    "ignore previous",
    "prioritize this",
)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class DefenseMode(str, Enum):
    NONE = "none"
    SOURCE_TRUST = "source_trust"
    INSTRUCTION_FILTER = "instruction_filter"
    SIMILARITY_FILTER = "similarity_filter"
    COMBINED = "combined"


@dataclass(frozen=True)
class DefenseStageDecision:
    stage: str
    included: bool
    reason: str | None = None


@dataclass(frozen=True)
class DefenseTraceEntry:
    original_rank: int
    filename: str
    page_number: int
    chunk_id: str
    included: bool
    stage_decisions: tuple[DefenseStageDecision, ...]
    final_rank: int | None


@dataclass(frozen=True)
class DefenseResult:
    chunks: tuple[RetrievedChunk, ...]
    trace: tuple[DefenseTraceEntry, ...]


def load_trusted_filenames(path: Path) -> frozenset[str]:
    """Read the immutable clean-source inventory from an index manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(
        str(document["filename"])
        for document in payload.get("documents", [])
        if "filename" in document
    )


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains_instruction(text: str) -> bool:
    normalized = _normalized_text(text)
    return any(phrase in normalized for phrase in _INSTRUCTION_PATTERNS)


class DefenseCoordinator:
    def __init__(
        self,
        *,
        trusted_filenames: Iterable[str],
        similarity_threshold: float = 0.92,
        embedder: Embedder | None = None,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self._trusted_filenames = frozenset(trusted_filenames)
        self._similarity_threshold = similarity_threshold
        self._embedder = embedder

    def apply(
        self, chunks: tuple[RetrievedChunk, ...], mode: DefenseMode
    ) -> DefenseResult:
        decisions: list[list[DefenseStageDecision]] = [[] for _ in chunks]
        retained = list(range(len(chunks)))

        for stage in self._stages(mode):
            if stage == "instruction_filter":
                retained = self._apply_instruction_filter(chunks, retained, decisions)
            elif stage == "similarity_filter":
                retained = self._apply_similarity_filter(chunks, retained, decisions)
            else:
                retained = self._apply_source_trust(chunks, retained, decisions)

        final_ranks = {index: rank for rank, index in enumerate(retained, start=1)}
        result_chunks = tuple(
            RetrievedChunk(
                document_id=chunks[index].document_id,
                filename=chunks[index].filename,
                page_number=chunks[index].page_number,
                chunk_id=chunks[index].chunk_id,
                text=chunks[index].text,
                rank=final_ranks[index],
                relevance_score=chunks[index].relevance_score,
            )
            for index in retained
        )
        trace = tuple(
            DefenseTraceEntry(
                original_rank=chunk.rank,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                included=index in final_ranks,
                stage_decisions=tuple(decisions[index]),
                final_rank=final_ranks.get(index),
            )
            for index, chunk in enumerate(chunks)
        )
        return DefenseResult(chunks=result_chunks, trace=trace)

    @staticmethod
    def _stages(mode: DefenseMode) -> tuple[str, ...]:
        if mode is DefenseMode.NONE:
            return ()
        if mode is DefenseMode.SOURCE_TRUST:
            return ("source_trust",)
        if mode is DefenseMode.INSTRUCTION_FILTER:
            return ("instruction_filter",)
        if mode is DefenseMode.SIMILARITY_FILTER:
            return ("similarity_filter",)
        if mode is DefenseMode.COMBINED:
            return ("instruction_filter", "similarity_filter", "source_trust")
        raise ValueError(f"unsupported defense mode: {mode}")

    @staticmethod
    def _apply_instruction_filter(
        chunks: tuple[RetrievedChunk, ...],
        retained: list[int],
        decisions: list[list[DefenseStageDecision]],
    ) -> list[int]:
        kept: list[int] = []
        for index in retained:
            if _contains_instruction(chunks[index].text):
                decisions[index].append(
                    DefenseStageDecision("instruction_filter", False, "instruction_content")
                )
            else:
                decisions[index].append(DefenseStageDecision("instruction_filter", True))
                kept.append(index)
        return kept

    def _apply_similarity_filter(
        self,
        chunks: tuple[RetrievedChunk, ...],
        retained: list[int],
        decisions: list[list[DefenseStageDecision]],
    ) -> list[int]:
        if len(retained) < 2:
            for index in retained:
                decisions[index].append(DefenseStageDecision("similarity_filter", True))
            return retained
        if self._embedder is None:
            raise ValueError("an embedder is required for similarity filtering")

        vectors = np.asarray(
            self._embedder.encode([chunks[index].text for index in retained]), dtype=float
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
        similarities = normalized @ normalized.T
        removed: set[int] = set()
        for left_position, left_index in enumerate(retained):
            for right_position in range(left_position + 1, len(retained)):
                right_index = retained[right_position]
                if left_index in removed or right_index in removed:
                    continue
                if similarities[left_position, right_position] >= self._similarity_threshold:
                    _, loser = self._resolve_duplicate(chunks, left_index, right_index)
                    removed.add(loser)
        kept: list[int] = []
        for index in retained:
            if index in removed:
                decisions[index].append(
                    DefenseStageDecision("similarity_filter", False, "near_duplicate")
                )
            else:
                decisions[index].append(DefenseStageDecision("similarity_filter", True))
                kept.append(index)
        return kept

    def _resolve_duplicate(
        self, chunks: tuple[RetrievedChunk, ...], first: int, second: int
    ) -> tuple[int, int]:
        first_key = (
            chunks[first].filename not in self._trusted_filenames,
            chunks[first].rank,
        )
        second_key = (
            chunks[second].filename not in self._trusted_filenames,
            chunks[second].rank,
        )
        return (first, second) if first_key <= second_key else (second, first)

    def _apply_source_trust(
        self,
        chunks: tuple[RetrievedChunk, ...],
        retained: list[int],
        decisions: list[list[DefenseStageDecision]],
    ) -> list[int]:
        kept: list[int] = []
        for index in retained:
            if chunks[index].filename in self._trusted_filenames:
                decisions[index].append(DefenseStageDecision("source_trust", True))
                kept.append(index)
            else:
                decisions[index].append(
                    DefenseStageDecision("source_trust", False, "untrusted_source")
                )
        return kept
