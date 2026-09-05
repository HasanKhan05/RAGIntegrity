"""Stable Phase 4 row serialization, summaries, and error analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from experiments.phase4_generation import GenerationCache
from src.evaluation.phase2 import (
    AttackCase,
    assess_false_claim_adoption,
    detect_poison,
)
from src.evaluation.phase3 import evaluate_defense_result
from src.evaluation.phase4 import (
    aggregate_evaluation,
    evaluate_sources,
    score_attack_answer,
    score_control_answer,
)
from src.rag.defenses import DefenseResult
from src.rag.models import RetrievedChunk


ROW_FIELDS = (
    "question_id",
    "cohort",
    "attack_id",
    "question",
    "scenario",
    "mode",
    "fingerprint",
    "answer",
    "score",
    "retrieval_compromised",
    "generation_compromised",
    "target_poison_rank",
    "context_size",
    "retrieval_latency_ms",
    "defense_latency_ms",
    "generation_latency_ms",
)

_ATTACK_MANIFEST_LABEL = "data/manifests/attack_manifest.json"
_CLEAN_MANIFEST_LABEL = "data/manifests/clean_index.json"

_MODE_KEYS = (
    "clean_none",
    "attacked_none",
    "source_trust",
    "instruction_filter",
    "similarity_filter",
    "combined",
)

_LIMITATIONS = (
    "The corpus is a controlled local set of Toyota brochures with a modest fixed benchmark.",
    "Known curated provenance makes source-trust filtering unusually strong.",
    "The attacks are controlled synthetic research artifacts.",
    "Results cover one Gemini model configuration and one deterministic response per unique prompt.",
    "Rule-based instruction filtering can miss indirect attacks, while similarity filtering depends on its representation and threshold.",
    "Deterministic grading is intentionally narrow; ambiguous answers require human review.",
)


def _atomic_write_text(path: Path, text: str, *, newline: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline=newline)
    temporary.replace(path)


def _serialized(items: Sequence[object]) -> list[dict[str, object]]:
    return [asdict(item) for item in items]


def _snapshot_json(
    snapshot: Mapping[str, bytes], label: str
) -> Mapping[str, object]:
    source = snapshot.get(label)
    if not isinstance(source, bytes):
        raise ValueError(f"verified frozen snapshot is missing {label}")
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"verified frozen snapshot contains invalid JSON: {label}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"verified frozen snapshot must contain an object: {label}")
    return payload


def _load_attack_truth(
    snapshot: Mapping[str, bytes],
) -> tuple[dict[str, AttackCase], dict[str, tuple[str, ...]]]:
    payload = _snapshot_json(snapshot, _ATTACK_MANIFEST_LABEL)
    records = payload.get("attacks")
    if not isinstance(records, list):
        raise ValueError("attack manifest must contain an attacks list")
    attacks: dict[str, AttackCase] = {}
    clean_values: dict[str, tuple[str, ...]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("attack manifest entries must be objects")
        aliases = record.get("false_unit_aliases")
        check = record.get("deterministic_compromise_check")
        values = check.get("clean_values") if isinstance(check, dict) else None
        if (
            not isinstance(aliases, list)
            or not all(isinstance(alias, str) for alias in aliases)
            or not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError("attack manifest entries require deterministic values")
        try:
            attack = AttackCase(
                attack_id=record["attack_id"],
                synthetic_document_id=record["synthetic_document_id"],
                synthetic_page_number=record["synthetic_page_number"],
                synthetic_filename=record["synthetic_filename"],
                attack_type=record["attack_type"],
                target_model=record["target_model"],
                target_topic=record["target_topic"],
                clean_fact=record["clean_fact"],
                false_claim=record["false_claim"],
                clean_source_filename=record["clean_source_filename"],
                clean_source_page=record["clean_source_page"],
                target_test_question=record["target_test_question"],
                false_value=record["false_value"],
                false_unit_aliases=tuple(aliases),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("attack manifest entry is incomplete") from error
        attacks[attack.attack_id] = attack
        clean_values[attack.attack_id] = tuple(values)
    return attacks, clean_values


def _load_trusted_filenames(snapshot: Mapping[str, bytes]) -> frozenset[str]:
    payload = _snapshot_json(snapshot, _CLEAN_MANIFEST_LABEL)
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError("clean manifest must contain a documents list")
    return frozenset(
        document["filename"]
        for document in documents
        if isinstance(document, dict) and isinstance(document.get("filename"), str)
    )


def _request_by_cell(
    requests: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for request in requests:
        cell_ids = request.get("cell_ids")
        if not isinstance(cell_ids, Sequence) or isinstance(cell_ids, str):
            raise ValueError("Phase 4 request requires cell_ids")
        for cell_id in cell_ids:
            if not isinstance(cell_id, str) or cell_id in result:
                raise ValueError("Phase 4 request cell_ids must be unique strings")
            result[cell_id] = request
    return result


def _chunks(cell: Mapping[str, object], key: str) -> tuple[RetrievedChunk, ...]:
    value = cell.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"prepared Phase 4 cell requires {key}")
    if not all(isinstance(chunk, RetrievedChunk) for chunk in value):
        raise ValueError(f"prepared Phase 4 cell {key} must contain retrieved chunks")
    return tuple(value)


def _benchmark_metadata(cell: Mapping[str, object]) -> Mapping[str, object]:
    metadata = cell.get("benchmark_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("prepared Phase 4 cell requires benchmark metadata")
    return metadata


def build_scored_rows(
    prepared: Mapping[str, object],
    cache: GenerationCache,
    frozen_snapshot: Mapping[str, bytes],
    *,
    generated_fingerprints: Collection[object] = (),
) -> list[dict[str, object]]:
    """Join cached answers with evaluation-only truth after generation is complete."""

    cells = prepared.get("cells")
    requests = prepared.get("requests")
    if not isinstance(cells, Sequence) or not isinstance(requests, Sequence):
        raise ValueError("prepared Phase 4 data requires cells and requests")
    request_by_cell = _request_by_cell(requests)  # type: ignore[arg-type]
    attacks_by_id, clean_values = _load_attack_truth(frozen_snapshot)
    trusted_filenames = _load_trusted_filenames(frozen_snapshot)
    generated = {str(value) for value in generated_fingerprints}
    seen_fingerprints: set[str] = set()
    rows: list[dict[str, object]] = []

    for cell_value in cells:
        if not isinstance(cell_value, Mapping):
            raise ValueError("prepared Phase 4 cells must be objects")
        cell = cell_value
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in request_by_cell:
            raise ValueError("prepared Phase 4 cell is missing its request")
        request = request_by_cell[cell_id]
        fingerprint = request.get("fingerprint")
        identity = request.get("identity")
        if not isinstance(fingerprint, str) or not isinstance(identity, Mapping):
            raise ValueError("Phase 4 request requires fingerprint and identity")
        entry = cache.get(fingerprint, identity)
        if entry is None or not isinstance(entry.get("answer"), str):
            raise ValueError(f"missing completed generation for {fingerprint}")

        source_chunks = _chunks(cell, "source_chunks")
        final_chunks = _chunks(cell, "chunks")
        metadata = _benchmark_metadata(cell)
        cohort = str(cell.get("cohort"))
        attack: AttackCase | None = None
        attack_id = cell.get("attack_id")
        if cohort == "attack":
            if not isinstance(attack_id, str) or attack_id not in attacks_by_id:
                raise ValueError("attack cell refers to an unknown attack_id")
            attack = attacks_by_id[attack_id]
            expected_filename = attack.clean_source_filename
            expected_page = attack.clean_source_page
        else:
            expected_filename = metadata.get("clean_source_filename")
            expected_page = metadata.get("clean_source_page")
            if not isinstance(expected_filename, str) or not isinstance(expected_page, int):
                raise ValueError("control cell requires expected clean source metadata")

        answer = str(entry["answer"])
        if attack is not None:
            values = clean_values[attack.attack_id]
            adoption = assess_false_claim_adoption(
                answer,
                false_value=attack.false_value,
                clean_values=values,
                unit_aliases=attack.false_unit_aliases,
            )
            score = score_attack_answer(answer, attack, values)
            generation_compromised = adoption.generation_compromised
        else:
            expected_answer = metadata.get("expected_answer")
            if not isinstance(expected_answer, str):
                raise ValueError("control cell requires expected_answer")
            score = score_control_answer(answer, expected_answer)
            generation_compromised = False

        poison = (
            detect_poison(
                source_chunks,
                attack.synthetic_document_id,
                attack.synthetic_page_number,
            )
            if attack is not None
            else None
        )
        defense = evaluate_defense_result(
            source_chunks,
            DefenseResult(chunks=final_chunks, trace=tuple(cell.get("defense_trace", ()))),
            attack=attack,
            clean_filenames=trusted_filenames,
        )
        source_checks = evaluate_sources(
            answer,
            final_chunks,
            expected_clean_source_filename=expected_filename,
            expected_clean_source_page=expected_page,
            trusted_clean_filenames=trusted_filenames,
            attack=attack,
        )
        if fingerprint in generated and fingerprint not in seen_fingerprints:
            generation_status = "generated"
        elif fingerprint in generated:
            generation_status = "in_run_reuse"
        else:
            generation_status = "cache_hit"
        seen_fingerprints.add(fingerprint)

        token_usage = entry.get("token_usage")
        row: dict[str, object] = {
            "question_id": cell.get("question_id"),
            "cohort": cohort,
            "attack_id": attack_id,
            "question": cell.get("question"),
            "scenario": cell.get("scenario"),
            "mode": cell.get("mode"),
            "fingerprint": fingerprint,
            "answer": answer,
            "score": score,
            "retrieval_compromised": poison.retrieval_compromised if poison else False,
            "generation_compromised": generation_compromised,
            "target_poison_rank": poison.poison_rank if poison else None,
            "context_size": len(final_chunks),
            "retrieval_latency_ms": cell.get("retrieval_latency_ms", 0.0),
            "defense_latency_ms": cell.get("defense_latency_ms", 0.0),
            "generation_latency_ms": entry.get("latency_ms", 0.0),
            "cell_id": cell_id,
            "source_context": _serialized(source_chunks),
            "sources": _serialized(final_chunks),
            "defense_trace": _serialized(tuple(cell.get("defense_trace", ()))),
            **defense,
            **source_checks,
            "citation_checks": {
                "count": source_checks["citation_count"],
                "resolves_to_context": source_checks["citations_resolve_to_context"],
            },
            "source_checks": {
                "expected_clean_source_present": source_checks[
                    "expected_clean_source_present"
                ],
                "trusted_clean_provenance": source_checks["trusted_clean_provenance"],
                "target_synthetic_source_present": source_checks[
                    "target_synthetic_source_present"
                ],
            },
            "deterministic_score": score,
            "manual_review_required": score == "ambiguous",
            "legitimate_clean_false_rejection": (
                cohort == "control"
                and int(defense["legitimate_clean_chunks_removed"]) > 0
            ),
            "defense_induced_correctness_loss": False,
            "generation_status": generation_status,
            "generated_this_run": generation_status == "generated",
            "cache_reused": generation_status != "generated",
            "token_usage": token_usage,
            "provider_attempts": entry.get("provider_attempts", 1),
            "rate_limit_retries": entry.get("rate_limit_retries", 0),
            "total_latency_ms": round(
                float(cell.get("retrieval_latency_ms", 0.0))
                + float(cell.get("defense_latency_ms", 0.0))
                + float(entry.get("latency_ms", 0.0)),
                2,
            ),
        }
        rows.append(row)

    control_baselines = {
        str(row["question_id"]): row
        for row in rows
        if row["cohort"] == "control"
        and row["scenario"] == "attacked"
        and row["mode"] == "none"
    }
    for row in rows:
        baseline = control_baselines.get(str(row["question_id"]))
        row["defense_induced_correctness_loss"] = bool(
            row["cohort"] == "control"
            and row["scenario"] == "attacked"
            and row["mode"] != "none"
            and baseline is not None
            and baseline["score"] == "correct"
            and row["score"] != "correct"
        )
    return rows


def _condition_key(row: Mapping[str, object]) -> str:
    return (
        "clean_none"
        if row.get("scenario") == "clean"
        else "attacked_none"
        if row.get("mode") == "none"
        else str(row.get("mode"))
    )


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def _average(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [
        float(row.get(field, 0.0))
        for row in rows
        if isinstance(row.get(field, 0.0), int | float)
        and not isinstance(row.get(field, 0.0), bool)
    ]
    return 0.0 if not values else sum(values) / len(values)


def build_summary(
    rows: Sequence[Mapping[str, object]], run: Mapping[str, object]
) -> dict[str, object]:
    """Build the stable Phase 5-facing aggregate solely from saved rows and run data."""

    aggregates = aggregate_evaluation(rows)
    attack_metrics = aggregates["attack_metrics"]
    control_metrics = aggregates["clean_control_metrics"]
    for metrics in attack_metrics.values():
        distribution = metrics["target_poison_rank_distribution"]
        metrics["target_poison_rank_distribution"] = {
            str(rank): count for rank, count in distribution.items()
        }
    by_condition = {
        key: [row for row in rows if _condition_key(row) == key]
        for key in _MODE_KEYS
    }
    attack_question_ids = {
        str(row.get("question_id")) for row in rows if row.get("cohort") == "attack"
    }
    control_question_ids = {
        str(row.get("question_id")) for row in rows if row.get("cohort") == "control"
    }
    error_by_cohort: dict[str, object] = {}
    for cohort in ("attack", "control"):
        cohort_rows = [row for row in rows if row.get("cohort") == cohort]
        error_by_cohort[cohort] = {
            "rows": len(cohort_rows),
            "incorrect": sum(row.get("score") == "incorrect" for row in cohort_rows),
            "ambiguous": sum(
                row.get("deterministic_score", row.get("score")) == "ambiguous"
                for row in cohort_rows
            ),
        }
    error_by_condition = {
        key: {
            "rows": len(condition_rows),
            "incorrect": sum(row.get("score") == "incorrect" for row in condition_rows),
            "ambiguous": sum(
                row.get("deterministic_score", row.get("score")) == "ambiguous"
                for row in condition_rows
            ),
        }
        for key, condition_rows in by_condition.items()
    }
    generation_usage = dict(aggregates["generation_usage"])
    generation_usage.update(
        {
            "cache_hit_rows": sum(row.get("generation_status") == "cache_hit" for row in rows),
            "in_run_reuse_rows": sum(
                row.get("generation_status") == "in_run_reuse" for row in rows
            ),
        }
    )
    run_generation = run.get("generation")
    if isinstance(run_generation, Mapping):
        for field in (
            "unique_new_cache_entries",
            "provider_attempts",
            "rate_limit_retries",
        ):
            value = run_generation.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                generation_usage[field] = value
    return {
        "schema_version": "phase4_summary_v1",
        "benchmark": {
            "conceptual_cells": len(rows),
            "question_count": len(attack_question_ids | control_question_ids),
            "attack_question_count": len(attack_question_ids),
            "clean_control_question_count": len(control_question_ids),
            "condition_count": sum(bool(value) for value in by_condition.values()),
        },
        "generation_usage": generation_usage,
        "retrieval": {
            "attacked_target_poison_retrieval_rate": attack_metrics[
                "attacked_none"
            ]["target_poison_retrieval_rate"],
            "attacked_target_poison_rank_distribution": attack_metrics[
                "attacked_none"
            ]["target_poison_rank_distribution"],
            "average_target_poison_rank_when_retrieved": attack_metrics[
                "attacked_none"
            ]["average_target_poison_rank_when_retrieved"],
        },
        "attack_metrics": attack_metrics,
        "clean_control_metrics": control_metrics,
        "latency": {
            key: {
                "average_retrieval_latency_ms": _average(value, "retrieval_latency_ms"),
                "average_defense_latency_ms": _average(value, "defense_latency_ms"),
                "average_generation_latency_ms": _average(value, "generation_latency_ms"),
                "average_total_latency_ms": _average(value, "total_latency_ms"),
            }
            for key, value in by_condition.items()
        },
        "source_quality": {
            key: {
                "expected_clean_source_rate": _rate(
                    sum(row.get("expected_clean_source_present") is True for row in value),
                    len(value),
                ),
                "trusted_clean_provenance_rate": _rate(
                    sum(row.get("trusted_clean_provenance") is True for row in value),
                    len(value),
                ),
                "citation_resolution_rate": _rate(
                    sum(row.get("citations_resolve_to_context") is True for row in value),
                    len(value),
                ),
            }
            for key, value in by_condition.items()
        },
        "error_analysis": {
            "by_cohort": error_by_cohort,
            "by_condition": error_by_condition,
            "retrieval_compromised_rows": sum(
                row.get("retrieval_compromised") is True for row in rows
            ),
            "generation_compromised_rows": sum(
                row.get("generation_compromised") is True for row in rows
            ),
            "false_refusal_rows": sum(row.get("false_refusal_proxy") is True for row in rows),
            "source_mismatch_rows": sum(
                row.get("expected_clean_source_present") is False for row in rows
            ),
            "ambiguous_score_rows": sum(
                row.get("deterministic_score", row.get("score")) == "ambiguous"
                for row in rows
            ),
        },
        "limitations": list(_LIMITATIONS),
        "run": dict(run),
    }


def _percent(value: object) -> str:
    return f"{float(value or 0.0) * 100:.1f}%"


def title_block(summary: Mapping[str, object]) -> str:
    benchmark = summary["benchmark"]
    return (
        "# Phase 4 Evaluation and Error Analysis\n\n"
        f"This report summarizes {benchmark['conceptual_cells']} scored cells across "
        f"{benchmark['question_count']} frozen questions. Retrieval compromise means the "
        "target synthetic page entered top-k retrieval; Generation compromise means the "
        "saved answer adopted its target false claim. These outcomes are measured separately."
    )


def generation_accounting(summary: Mapping[str, object]) -> str:
    usage = summary["generation_usage"]
    accounting = "\n".join(
        (
            "## Generation accounting",
            "",
            f"- Unique new cache entries: {usage.get('unique_new_cache_entries', usage.get('new_calls', 0))}",
            f"- Total provider attempts: {usage.get('provider_attempts', usage.get('new_calls', 0))}",
            f"- Rate-limit retries: {usage.get('rate_limit_retries', 0)}",
        )
    )
    run = summary.get("run", {})
    reconciliation = run.get("latency_reconciliation", {})
    note = reconciliation.get("note")
    return f"{accounting}\n\n{note}" if isinstance(note, str) else accounting


def comparison_table(summary: Mapping[str, object]) -> str:
    attacks = summary["attack_metrics"]
    controls = summary["clean_control_metrics"]
    lines = [
        "## Condition comparison",
        "",
        "| Condition | Attack accuracy | Overall generation ASR | Conditional generation ASR | Poison survival | Control accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in _MODE_KEYS:
        attack = attacks[key]
        control = controls[key]
        lines.append(
            f"| {key} | {_percent(attack['answer_accuracy'])} | "
            f"{_percent(attack['overall_attack_success_rate'])} | "
            f"{_percent(attack['conditional_attack_success_rate'])} | "
            f"{_percent(attack['poison_survival_rate'])} | "
            f"{_percent(control['answer_accuracy'])} |"
        )
    return "\n".join(lines)


def _case_line(label: str, row: Mapping[str, object]) -> str:
    answer = " ".join(str(row.get("answer", "")).split())
    if len(answer) > 160:
        answer = answer[:157] + "..."
    return (
        f"- **{label}:** `{row.get('question_id')}` / `{_condition_key(row)}` — "
        f"retrieval compromised={row.get('retrieval_compromised')}, generation "
        f"compromised={row.get('generation_compromised')}, poison "
        f"removed={row.get('target_poison_removed')}, score={row.get('score')}, "
        f"target rank={row.get('target_poison_rank')}, final context "
        f"size={row.get('context_size')}. Saved answer: {answer}"
    )


def _representative_cases(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[str, Mapping[str, object]]]:
    selectors = (
        (
            "Rank-1 retrieval and generation compromise",
            lambda row: row.get("target_poison_rank") == 1
            and row.get("generation_compromised") is True,
        ),
        (
            "Retrieved poison resisted by generation",
            lambda row: row.get("retrieval_compromised") is True
            and row.get("generation_compromised") is False
            and row.get("target_poison_retained") is True
            and row.get("target_poison_removed") is False,
        ),
        (
            "Poison not retrieved",
            lambda row: row.get("cohort") == "attack"
            and row.get("scenario") == "attacked"
            and row.get("retrieval_compromised") is False,
        ),
        (
            "Source-trust restoration",
            lambda row: row.get("mode") == "source_trust"
            and row.get("target_poison_removed") is True
            and row.get("score") == "correct",
        ),
        (
            "Similarity-filter or control harm",
            lambda row: row.get("mode") == "similarity_filter"
            and (
                row.get("generation_compromised") is True
                or row.get("legitimate_clean_false_rejection") is True
                or row.get("defense_induced_correctness_loss") is True
            ),
        ),
        (
            "Ambiguous deterministic grade",
            lambda row: row.get("deterministic_score", row.get("score"))
            == "ambiguous",
        ),
    )
    selected: list[tuple[str, Mapping[str, object]]] = []
    used: set[str] = set()
    for label, selector in selectors:
        match = next(
            (
                row
                for row in rows
                if selector(row)
                and str(row.get("cell_id", (row.get("question_id"), _condition_key(row))))
                not in used
            ),
            None,
        )
        if match is not None:
            key = str(match.get("cell_id", (match.get("question_id"), _condition_key(match))))
            used.add(key)
            selected.append((label, match))
    return selected[:6]


def error_groups(
    summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> str:
    errors = summary["error_analysis"]
    cohorts = errors["by_cohort"]
    initial_ambiguities = int(errors["ambiguous_score_rows"])
    unresolved_ambiguities = sum(
        row.get("manual_review_required") is True for row in rows
    )
    ambiguity_label = "row" if initial_ambiguities == 1 else "rows"
    if initial_ambiguities > 0 and unresolved_ambiguities == 0:
        ambiguity_line = (
            f"- Initial deterministic ambiguities: {initial_ambiguities} "
            f"{ambiguity_label}; all adjudicated."
        )
    else:
        ambiguity_line = (
            f"- Initial deterministic ambiguities: {initial_ambiguities} "
            f"{ambiguity_label}; {initial_ambiguities - unresolved_ambiguities} "
            f"adjudicated and {unresolved_ambiguities} unresolved."
        )
    lines = [
        "## Error groups",
        "",
        f"- Attack rows: {cohorts['attack']['incorrect']} incorrect and {cohorts['attack']['ambiguous']} ambiguous.",
        f"- Control rows: {cohorts['control']['incorrect']} incorrect and {cohorts['control']['ambiguous']} ambiguous.",
        f"- Retrieval compromise: {errors['retrieval_compromised_rows']} rows.",
        f"- Generation compromise: {errors['generation_compromised_rows']} rows.",
        f"- False refusal: {errors['false_refusal_rows']} rows.",
        f"- Source mismatch: {errors['source_mismatch_rows']} rows.",
        ambiguity_line,
        "",
        "### Errors by condition",
        "",
    ]
    lines.extend(
        f"- {key}: {errors['by_condition'][key]['incorrect']} incorrect and "
        f"{errors['by_condition'][key]['ambiguous']} ambiguous."
        for key in _MODE_KEYS
    )
    lines.extend(("", "## Representative saved cases", ""))
    cases = _representative_cases(rows)
    lines.extend(_case_line(label, row) for label, row in cases)
    if not cases:
        lines.append("No qualifying error case occurred in the saved rows.")
    return "\n".join(lines)


def limitations(summary: Mapping[str, object]) -> str:
    lines = [
        "## Limitations",
        "",
        "Observed defense tradeoffs indicate reductions or harms in this run, not general guarantees.",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines)


def render_error_analysis(
    summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> str:
    """Render a concise report from saved rows and their derived summary."""

    return "\n\n".join(
        (
            title_block(summary),
            generation_accounting(summary),
            comparison_table(summary),
            error_groups(summary, rows),
            limitations(summary),
        )
    ) + "\n"


def write_row_json(
    path: Path, rows: Sequence[Mapping[str, object]], run: Mapping[str, object]
) -> None:
    payload = {
        "schema_version": "phase4_evaluation_results_v1",
        "run": dict(run),
        "rows": [dict(row) for row in rows],
    }
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return value


def write_row_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    extra_fields = sorted({str(key) for row in rows for key in row if key not in ROW_FIELDS})
    fieldnames = (*ROW_FIELDS, *extra_fields)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    _atomic_write_text(path, output.getvalue(), newline="")


def _artifact_paths(output_dir: Path, report_path: Path | None) -> dict[str, Path]:
    return {
        "phase4_evaluation_results.json": output_dir
        / "phase4_evaluation_results.json",
        "phase4_evaluation_results.csv": output_dir
        / "phase4_evaluation_results.csv",
        "phase4_summary.json": output_dir / "phase4_summary.json",
        "phase4_evaluation.md": report_path or output_dir / "phase4_evaluation.md",
    }


def verify_outputs(
    output_dir: Path, *, report_path: Path | None = None
) -> bool:
    """Return whether the last-written marker matches all four artifacts."""

    marker_path = output_dir / "phase4_publication.json"
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    artifacts = payload.get("artifacts")
    paths = _artifact_paths(output_dir, report_path)
    if (
        payload.get("schema_version") != "phase4_publication_v1"
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != set(paths)
    ):
        return False
    for label, path in paths.items():
        metadata = artifacts.get(label)
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("sha256"), str
        ):
            return False
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return False
        if digest != metadata["sha256"]:
            return False
    return True


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    run: Mapping[str, object],
    *,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Publish four artifacts and write their verified completion marker last."""

    marker_path = output_dir / "phase4_publication.json"
    marker_path.unlink(missing_ok=True)
    paths = _artifact_paths(output_dir, report_path)
    summary = build_summary(rows, run)
    write_row_json(paths["phase4_evaluation_results.json"], rows, run)
    write_row_csv(paths["phase4_evaluation_results.csv"], rows)
    _atomic_write_text(
        paths["phase4_summary.json"],
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_write_text(
        paths["phase4_evaluation.md"],
        render_error_analysis(summary, rows),
    )
    publication = {
        "schema_version": "phase4_publication_v1",
        "artifacts": {
            label: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for label, path in paths.items()
        },
    }
    _atomic_write_text(
        marker_path,
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
    )
    if not verify_outputs(output_dir, report_path=report_path):
        marker_path.unlink(missing_ok=True)
        raise RuntimeError("Phase 4 publication verification failed")
    return summary
