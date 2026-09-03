"""Evaluation-only metrics for the local Phase 3 defense comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.evaluation.phase2 import AttackCase
from src.rag.defenses import DefenseResult
from src.rag.models import RetrievedChunk


def _source_key(chunk: RetrievedChunk) -> tuple[str, str, int, str]:
    return (chunk.document_id, chunk.filename, chunk.page_number, chunk.chunk_id)


def _has_target_poison(chunks: Sequence[RetrievedChunk], attack: AttackCase) -> bool:
    return any(
        chunk.document_id == attack.synthetic_document_id
        and chunk.page_number == attack.synthetic_page_number
        for chunk in chunks
    )


def evaluate_defense_result(
    source_snapshot: Sequence[RetrievedChunk],
    result: DefenseResult,
    *,
    attack: AttackCase | None,
    clean_filenames: frozenset[str],
) -> dict[str, bool | int]:
    """Compare completed defense output with evaluation-only provenance."""

    retained_keys = {_source_key(chunk) for chunk in result.chunks}
    clean_chunks = [
        chunk for chunk in source_snapshot if chunk.filename in clean_filenames
    ]
    target_poison_retrieved = (
        _has_target_poison(source_snapshot, attack) if attack is not None else False
    )
    target_poison_retained = (
        _has_target_poison(result.chunks, attack) if attack is not None else False
    )
    return {
        "target_poison_retrieved": target_poison_retrieved,
        "target_poison_removed": target_poison_retrieved
        and not target_poison_retained,
        "target_poison_retained": target_poison_retained,
        "legitimate_clean_chunks_retrieved": len(clean_chunks),
        "legitimate_clean_chunks_removed": sum(
            _source_key(chunk) not in retained_keys for chunk in clean_chunks
        ),
        "remaining_chunks": len(result.chunks),
    }


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def aggregate_mode_metrics(
    evaluations: Sequence[Mapping[str, bool | int]],
    defense_latencies_ms: Sequence[float] = (),
) -> dict[str, float | int]:
    """Aggregate local retrieval-defense metrics without generation outcomes."""

    poison_retrieved = sum(
        bool(evaluation["target_poison_retrieved"]) for evaluation in evaluations
    )
    poison_removed = sum(
        bool(evaluation["target_poison_removed"]) for evaluation in evaluations
    )
    poison_retained = sum(
        bool(evaluation["target_poison_retained"]) for evaluation in evaluations
    )
    clean_retrieved = sum(
        int(evaluation["legitimate_clean_chunks_retrieved"])
        for evaluation in evaluations
    )
    clean_removed = sum(
        int(evaluation["legitimate_clean_chunks_removed"])
        for evaluation in evaluations
    )
    remaining_chunks = sum(int(evaluation["remaining_chunks"]) for evaluation in evaluations)
    evaluation_count = len(evaluations)
    latency_count = len(defense_latencies_ms)
    return {
        "target_poison_retrieved": poison_retrieved,
        "target_poison_removed": poison_removed,
        "target_poison_retained": poison_retained,
        "poison_removal_rate": _rate(poison_removed, poison_retrieved),
        "poison_survival_rate": _rate(poison_retained, poison_retrieved),
        "legitimate_clean_chunks_retrieved": clean_retrieved,
        "legitimate_clean_chunks_removed": clean_removed,
        "clean_false_rejection_rate": _rate(clean_removed, clean_retrieved),
        "average_remaining_chunks": _rate(remaining_chunks, evaluation_count),
        "average_defense_latency_ms": (
            0.0 if latency_count == 0 else sum(defense_latencies_ms) / latency_count
        ),
    }
