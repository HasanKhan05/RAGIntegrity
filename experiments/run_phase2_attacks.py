"""Run the fixed six-call controlled Phase 2 poisoning experiment."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.evaluation.phase2 import (
    AttackCase,
    answer_adopts_false_claim,
    detect_poison,
    load_attack_manifest,
)
from src.rag.config import Settings
from src.rag.generate import GeminiGenerator
from src.rag.index import ATTACKED_COLLECTION_NAME, CLEAN_COLLECTION_NAME
from src.rag.models import GeneratedAnswer, RetrievedChunk, TokenUsage
from src.rag.retrieve import Retriever


@dataclass(frozen=True)
class RunResult:
    """One retrieval and generation outcome for an experiment question."""

    answer: GeneratedAnswer
    sources: tuple[RetrievedChunk, ...]
    latency_ms: int


def run_once(question: str, retriever: Retriever, generator: GeminiGenerator) -> RunResult:
    """Retrieve and generate once, leaving attack evaluation to the caller."""

    started_at = perf_counter()
    chunks = tuple(retriever.retrieve(question))
    answer = generator.generate(question, chunks)
    return RunResult(
        answer=answer,
        sources=chunks,
        latency_ms=round((perf_counter() - started_at) * 1000),
    )


def _serialize_sources(chunks: Sequence[RetrievedChunk]) -> list[dict[str, object]]:
    return [asdict(chunk) for chunk in chunks]


def _serialize_usage(usage: TokenUsage | None) -> dict[str, int | None] | None:
    return None if usage is None else asdict(usage)


def _sum_token_usage(results: Sequence[RunResult]) -> dict[str, int]:
    return {
        field: sum(
            getattr(result.answer.token_usage, field) or 0
            for result in results
            if result.answer.token_usage is not None
        )
        for field in ("input_tokens", "output_tokens", "total_tokens")
    }


def _attack_result(
    attack: AttackCase, clean: RunResult, attacked: RunResult
) -> dict[str, object]:
    detection = detect_poison(attacked.sources, attack.synthetic_document_id)
    return {
        "attack_id": attack.attack_id,
        "question": attack.target_test_question,
        "attack_type": attack.attack_type,
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
        "generation_compromised": answer_adopts_false_claim(
            attacked.answer.text, attack.false_value, attack.false_unit_aliases
        ),
    }


def run_phase2(
    settings: Settings,
    *,
    generator: GeminiGenerator | None = None,
    retriever_factory: Callable[[str], Retriever] | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Run each manifest attack against clean and attacked indexes exactly once."""

    attacks = load_attack_manifest(settings.attack_manifest_path)
    if len(attacks) != 3:
        raise ValueError("Phase 2 attack manifest must contain exactly three attacks")

    active_generator = generator or GeminiGenerator(settings)
    factory = retriever_factory or (
        lambda collection_name: Retriever(settings, collection_name=collection_name)
    )
    clean_retriever = factory(CLEAN_COLLECTION_NAME)
    attacked_retriever = factory(ATTACKED_COLLECTION_NAME)

    outcomes: list[dict[str, object]] = []
    runs: list[RunResult] = []
    for attack in attacks:
        clean = run_once(attack.target_test_question, clean_retriever, active_generator)
        attacked = run_once(
            attack.target_test_question, attacked_retriever, active_generator
        )
        runs.extend((clean, attacked))
        outcomes.append(_attack_result(attack, clean, attacked))

    result: dict[str, object] = {
        "gemini_calls": len(runs),
        "token_usage": _sum_token_usage(runs),
        "attacks": outcomes,
    }
    destination = output_path or (
        settings.project_root / "experiments" / "results" / "phase2_attacks.json"
    ).with_name("phase2_attack_results.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    """Run the controlled Phase 2 experiment from local configuration."""

    run_phase2(Settings.from_env())


if __name__ == "__main__":
    main()
