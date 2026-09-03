from __future__ import annotations

from src.evaluation.phase2 import AttackCase
from src.evaluation.phase3 import aggregate_mode_metrics, evaluate_defense_result
from src.rag.defenses import DefenseResult, DefenseTraceEntry
from src.rag.models import RetrievedChunk


def _chunk(
    document_id: str,
    filename: str,
    page_number: int,
    rank: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        chunk_id=f"{document_id}-p{page_number}-c{rank}",
        text="Brochure text.",
        rank=rank,
        relevance_score=0.9,
    )


def _attack() -> AttackCase:
    return AttackCase(
        attack_id="attack_001",
        synthetic_document_id="update-doc",
        synthetic_page_number=1,
        synthetic_filename="update.pdf",
        attack_type="false_specification",
        target_model="RAV4",
        target_topic="fuel tank capacity",
        clean_fact="55 litres",
        false_claim="72 litres",
        clean_source_filename="rav4.pdf",
        clean_source_page=40,
        target_test_question="What is the capacity?",
        false_value="72",
        false_unit_aliases=("l", "litres"),
    )


def _result(chunks: tuple[RetrievedChunk, ...], included: tuple[bool, ...]) -> DefenseResult:
    return DefenseResult(
        chunks=chunks,
        trace=tuple(
            DefenseTraceEntry(
                original_rank=index,
                filename=f"source-{index}.pdf",
                page_number=index,
                chunk_id=f"source-{index}-chunk",
                included=value,
                stage_decisions=(),
                final_rank=index if value else None,
            )
            for index, value in enumerate(included, start=1)
        ),
    )


def test_evaluation_requires_exact_target_document_and_page_after_defense() -> None:
    snapshot = (
        _chunk("update-doc", "update.pdf", 2, 1),
        _chunk("update-doc", "update.pdf", 1, 2),
    )
    result = _result((snapshot[0],), (True, False))

    metrics = evaluate_defense_result(
        snapshot,
        result,
        attack=_attack(),
        clean_filenames=frozenset({"rav4.pdf"}),
    )

    assert metrics["target_poison_retrieved"] is True
    assert metrics["target_poison_removed"] is True
    assert metrics["target_poison_retained"] is False


def test_evaluation_counts_only_removed_clean_inventory_chunks_as_false_rejections() -> None:
    snapshot = (
        _chunk("clean-doc", "rav4.pdf", 40, 1),
        _chunk("update-doc", "update.pdf", 1, 2),
        _chunk("other-doc", "other.pdf", 1, 3),
    )
    result = _result((), (False, False, False))

    metrics = evaluate_defense_result(
        snapshot,
        result,
        attack=_attack(),
        clean_filenames=frozenset({"rav4.pdf"}),
    )

    assert metrics["legitimate_clean_chunks_retrieved"] == 1
    assert metrics["legitimate_clean_chunks_removed"] == 1


def test_aggregate_false_rejection_numerator_excludes_removed_synthetic_chunks() -> None:
    metrics = aggregate_mode_metrics(
        (
            {
                "target_poison_retrieved": True,
                "target_poison_removed": True,
                "target_poison_retained": False,
                "legitimate_clean_chunks_retrieved": 1,
                "legitimate_clean_chunks_removed": 1,
                "remaining_chunks": 0,
            },
            {
                "target_poison_retrieved": True,
                "target_poison_removed": True,
                "target_poison_retained": False,
                "legitimate_clean_chunks_retrieved": 0,
                "legitimate_clean_chunks_removed": 0,
                "remaining_chunks": 0,
            },
        )
    )

    assert metrics["legitimate_clean_chunks_removed"] == 1
    assert metrics["clean_false_rejection_rate"] == 1.0


def test_aggregate_metrics_use_zero_for_empty_denominators() -> None:
    metrics = aggregate_mode_metrics(())

    assert metrics == {
        "target_poison_retrieved": 0,
        "target_poison_removed": 0,
        "target_poison_retained": 0,
        "poison_removal_rate": 0.0,
        "poison_survival_rate": 0.0,
        "legitimate_clean_chunks_retrieved": 0,
        "legitimate_clean_chunks_removed": 0,
        "clean_false_rejection_rate": 0.0,
        "average_remaining_chunks": 0.0,
        "average_defense_latency_ms": 0.0,
    }
