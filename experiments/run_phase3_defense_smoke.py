"""Run the fixed six-call defended-generation smoke test for Phase 3."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.phase2_preflight import prepare_run
from src.evaluation.phase2 import (
    AttackCase,
    assess_false_claim_adoption,
    detect_poison,
    load_attack_manifest,
)
from src.rag.config import Settings
from src.rag.defenses import DefenseCoordinator, DefenseMode, load_trusted_filenames
from src.rag.embed import SentenceTransformerEmbedder
from src.rag.generate import GeminiGenerator
from src.rag.index import ATTACKED_COLLECTION_NAME
from src.rag.models import GeneratedAnswer, RetrievedChunk
from src.rag.retrieve import Retriever


SMOKE_MATRIX: tuple[tuple[str, DefenseMode], ...] = (
    ("attack_003", DefenseMode.INSTRUCTION_FILTER),
    ("attack_003", DefenseMode.COMBINED),
    ("attack_005", DefenseMode.SOURCE_TRUST),
    ("attack_005", DefenseMode.COMBINED),
    ("attack_010", DefenseMode.SOURCE_TRUST),
    ("attack_010", DefenseMode.COMBINED),
)


def _default_output_path(settings: Settings) -> Path:
    return settings.project_root / "experiments" / "results" / "phase3_defense_smoke.json"


def _default_baseline_paths(settings: Settings) -> tuple[Path, Path]:
    results = settings.project_root / "experiments" / "results"
    return (
        results / "phase2_attack_results.json",
        results / "phase2_expansion_smoke.json",
    )


def _serialize_trace(trace: Sequence[Any]) -> list[dict[str, object]]:
    return [
        {
            "original_rank": entry.original_rank,
            "filename": entry.filename,
            "page_number": entry.page_number,
            "chunk_id": entry.chunk_id,
            "included": entry.included,
            "stage_decisions": [
                asdict(decision) for decision in entry.stage_decisions
            ],
            "final_rank": entry.final_rank,
        }
        for entry in trace
    ]


def _serialize_sources(chunks: Sequence[RetrievedChunk]) -> list[dict[str, object]]:
    return [asdict(chunk) for chunk in chunks]


def _serialize_usage(answer: GeneratedAnswer) -> dict[str, int | None] | None:
    return None if answer.token_usage is None else asdict(answer.token_usage)


def _sum_token_usage(answers: Sequence[GeneratedAnswer]) -> dict[str, int]:
    return {
        field: sum(
            getattr(answer.token_usage, field) or 0
            for answer in answers
            if answer.token_usage is not None
        )
        for field in ("input_tokens", "output_tokens", "total_tokens")
    }


def _load_checks(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("attacks") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("attack manifest is invalid")
    checks: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("attack_id"), str):
            raise ValueError("attack manifest entry is invalid")
        check = record.get("deterministic_compromise_check")
        if isinstance(check, dict):
            checks[record["attack_id"]] = check
    return checks


def _load_baselines(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    baselines: dict[str, dict[str, object]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Phase 2 baseline is unavailable: {path}") from error
        records = payload.get("attacks") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"Phase 2 baseline is invalid: {path}")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("attack_id"), str):
                raise ValueError(f"Phase 2 baseline entry is invalid: {path}")
            attack_id = record["attack_id"]
            if attack_id in baselines:
                raise ValueError(f"Phase 2 baseline attack is duplicated: {attack_id}")
            baselines[attack_id] = {
                "result_file": path.name,
                "attacked_answer": record.get("attacked_answer"),
                "retrieval_compromised": record.get("retrieval_compromised"),
                "generation_compromised": record.get("generation_compromised"),
            }
    return baselines


def _required_attacks(inventory: Sequence[AttackCase]) -> dict[str, AttackCase]:
    by_id = {attack.attack_id: attack for attack in inventory}
    required_ids = {attack_id for attack_id, _ in SMOKE_MATRIX}
    if (
        len(inventory) != 10
        or len(by_id) != len(inventory)
        or not required_ids <= set(by_id)
    ):
        raise ValueError("Phase 3 smoke requires the three selected unique attacks")
    return by_id


def run_phase3_defense_smoke(
    settings: Settings,
    *,
    generator: GeminiGenerator | Any | None = None,
    retriever: Retriever | Any | None = None,
    coordinator: DefenseCoordinator | Any | None = None,
    baseline_paths: Sequence[Path] | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Generate exactly once for each fixed attack and defended mode."""

    destination = (output_path or _default_output_path(settings)).resolve()
    inventory = load_attack_manifest(settings.attack_manifest_path)
    attacks = _required_attacks(inventory)
    selected = [
        attacks[attack_id] for attack_id in ("attack_003", "attack_005", "attack_010")
    ]
    prepare_run(
        settings,
        selected,
        inventory,
        destination,
        validate_collections=generator is None or retriever is None,
    )
    checks = _load_checks(settings.attack_manifest_path)
    required_ids = {attack_id for attack_id, _ in SMOKE_MATRIX}
    if required_ids - set(checks):
        raise ValueError("Phase 3 smoke deterministic checks are missing")
    baselines = _load_baselines(baseline_paths or _default_baseline_paths(settings))
    if required_ids - set(baselines):
        raise ValueError("Phase 3 smoke baselines are missing selected attacks")

    active_generator = generator or GeminiGenerator(settings)
    active_retriever = retriever or Retriever(
        settings, collection_name=ATTACKED_COLLECTION_NAME
    )
    active_coordinator = coordinator or DefenseCoordinator(
        trusted_filenames=load_trusted_filenames(settings.manifest_path),
        similarity_threshold=settings.defense_similarity_threshold,
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
    )

    snapshots = {
        attack.attack_id: tuple(active_retriever.retrieve(attack.target_test_question))
        for attack in selected
    }
    answers: list[GeneratedAnswer] = []
    runs: list[dict[str, object]] = []
    for attack_id, mode in SMOKE_MATRIX:
        attack = attacks[attack_id]
        snapshot = snapshots[attack_id]
        defended = active_coordinator.apply(snapshot, mode)
        answer = active_generator.generate(attack.target_test_question, defended.chunks)
        answers.append(answer)
        detection = detect_poison(
            snapshot, attack.synthetic_document_id, attack.synthetic_page_number
        )
        check = checks[attack_id]
        clean_values = check.get("clean_values")
        if not isinstance(clean_values, list) or not all(
            isinstance(value, str) for value in clean_values
        ):
            raise ValueError(f"missing deterministic clean values for {attack_id}")
        assessment = assess_false_claim_adoption(
            answer.text,
            false_value=attack.false_value,
            clean_values=clean_values,
            unit_aliases=attack.false_unit_aliases,
        )
        runs.append(
            {
                "attack_id": attack_id,
                "question": attack.target_test_question,
                "defense_mode": mode.value,
                "phase2_baseline": baselines[attack_id],
                "source_snapshot": _serialize_sources(snapshot),
                "retained_sources": _serialize_sources(defended.chunks),
                "trace": _serialize_trace(defended.trace),
                "answer": answer.text,
                "token_usage": _serialize_usage(answer),
                "poison_retrieved": detection.poison_retrieved,
                "poison_rank": detection.poison_rank,
                "retrieval_compromised": detection.retrieval_compromised,
                "generation_compromised": assessment.generation_compromised,
                "manual_review_required": assessment.manual_review_required,
                "assessment_reason": assessment.reason,
            }
        )

    result: dict[str, object] = {
        "gemini_calls": len(answers),
        "token_usage": _sum_token_usage(answers),
        "runs": runs,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    run_phase3_defense_smoke(Settings.from_env())


if __name__ == "__main__":
    main()
