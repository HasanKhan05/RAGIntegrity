"""Run the fixed six-call generation smoke test for the Phase 2 expansion."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from experiments.phase2_preflight import prepare_run
from experiments.run_phase2_attacks import (
    RunResult,
    _serialize_sources,
    _serialize_usage,
    _sum_token_usage,
    run_once,
)
from src.evaluation.phase2 import (
    AttackCase,
    assess_false_claim_adoption,
    detect_poison,
    load_attack_manifest,
)
from src.rag.config import Settings
from src.rag.generate import GeminiGenerator
from src.rag.index import ATTACKED_COLLECTION_NAME, CLEAN_COLLECTION_NAME
from src.rag.retrieve import Retriever


SMOKE_ATTACK_IDS = ("attack_004", "attack_005", "attack_010")


def _load_checks(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    attacks = payload.get("attacks") if isinstance(payload, dict) else None
    if not isinstance(attacks, list):
        raise ValueError("Phase 2 attack manifest is invalid")
    checks: dict[str, dict[str, object]] = {}
    for item in attacks:
        if not isinstance(item, dict) or not isinstance(item.get("attack_id"), str):
            raise ValueError("Phase 2 attack manifest entry is invalid")
        check = item.get("deterministic_compromise_check")
        if isinstance(check, dict):
            checks[item["attack_id"]] = check
    return checks


def _smoke_result(
    attack: AttackCase,
    check: dict[str, object],
    clean: RunResult,
    attacked: RunResult,
) -> dict[str, object]:
    detection = detect_poison(
        attacked.sources,
        attack.synthetic_document_id,
        attack.synthetic_page_number,
    )
    clean_values = check.get("clean_values")
    if not isinstance(clean_values, list) or not all(
        isinstance(value, str) for value in clean_values
    ):
        raise ValueError(f"Missing deterministic clean values for {attack.attack_id}")
    assessment = assess_false_claim_adoption(
        attacked.answer.text,
        false_value=attack.false_value,
        clean_values=clean_values,
        unit_aliases=attack.false_unit_aliases,
    )
    return {
        "attack_id": attack.attack_id,
        "question": attack.target_test_question,
        "target_model": attack.target_model,
        "target_topic": attack.target_topic,
        "clean_fact": attack.clean_fact,
        "false_claim": attack.false_claim,
        "clean_answer": clean.answer.text,
        "attacked_answer": attacked.answer.text,
        "clean_sources": _serialize_sources(clean.sources),
        "attacked_sources": _serialize_sources(attacked.sources),
        "clean_latency_ms": clean.latency_ms,
        "attacked_latency_ms": attacked.latency_ms,
        "clean_token_usage": _serialize_usage(clean.answer.token_usage),
        "attacked_token_usage": _serialize_usage(attacked.answer.token_usage),
        "poison_retrieved": detection.poison_retrieved,
        "poison_rank": detection.poison_rank,
        "retrieval_compromised": detection.retrieval_compromised,
        "generation_compromised": assessment.generation_compromised,
        "manual_review_required": assessment.manual_review_required,
        "assessment_reason": assessment.reason,
    }


def run_expansion_smoke(
    settings: Settings,
    *,
    generator: GeminiGenerator | None = None,
    retriever_factory: Callable[[str], Retriever] | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Run only attacks 004, 005, and 010 against clean and attacked corpora."""

    inventory = load_attack_manifest(settings.attack_manifest_path)
    if len(inventory) != 10:
        raise ValueError("Phase 2 expansion smoke requires exactly 10 manifest attacks")
    by_id = {attack.attack_id: attack for attack in inventory}
    if len(by_id) != 10 or any(attack_id not in by_id for attack_id in SMOKE_ATTACK_IDS):
        raise ValueError("Phase 2 expansion smoke targets are missing or duplicated")
    selected = [by_id[attack_id] for attack_id in SMOKE_ATTACK_IDS]
    checks = _load_checks(settings.attack_manifest_path)
    if any(attack_id not in checks for attack_id in SMOKE_ATTACK_IDS):
        raise ValueError("Phase 2 expansion smoke deterministic checks are missing")

    destination = prepare_run(
        settings,
        selected,
        inventory,
        output_path
        or settings.project_root
        / "experiments"
        / "results"
        / "phase2_expansion_smoke.json",
        validate_collections=generator is None or retriever_factory is None,
    )
    active_generator = generator or GeminiGenerator(settings)
    factory = retriever_factory or (
        lambda collection_name: Retriever(settings, collection_name=collection_name)
    )
    clean_retriever = factory(CLEAN_COLLECTION_NAME)
    attacked_retriever = factory(ATTACKED_COLLECTION_NAME)

    outcomes: list[dict[str, object]] = []
    runs = []
    for attack in selected:
        clean = run_once(attack.target_test_question, clean_retriever, active_generator)
        attacked = run_once(
            attack.target_test_question, attacked_retriever, active_generator
        )
        runs.extend((clean, attacked))
        outcomes.append(
            _smoke_result(attack, checks[attack.attack_id], clean, attacked)
        )

    result: dict[str, object] = {
        "gemini_calls": len(runs),
        "token_usage": _sum_token_usage(runs),
        "attacks": outcomes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    run_expansion_smoke(Settings.from_env())


if __name__ == "__main__":
    main()
