"""Run the zero-Gemini local comparison of Phase 3 retrieval defenses."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.evaluation.phase2 import AttackCase, load_attack_manifest
from src.evaluation.phase3 import aggregate_mode_metrics, evaluate_defense_result
from src.rag.config import Settings
from src.rag.defenses import DefenseCoordinator, DefenseMode, load_trusted_filenames
from src.rag.embed import SentenceTransformerEmbedder
from src.rag.index import ATTACKED_COLLECTION_NAME
from src.rag.models import RetrievedChunk
from src.rag.retrieve import Retriever


@dataclass(frozen=True)
class BenchmarkQuestion:
    question_id: str
    question: str
    question_type: str
    attack_id: str | None


def _load_questions(
    path: Path, *, question_type: str, expected_count: int
) -> tuple[BenchmarkQuestion, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {question_type} questions: {path}") from error
    records = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"expected {expected_count} {question_type} questions")

    questions: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{question_type} question entries must be objects")
        question_id = record.get("question_id")
        question = record.get("question")
        if not isinstance(question_id, str) or not isinstance(question, str):
            raise ValueError(f"{question_type} questions require question_id and question")
        if question_id in seen_ids or not question.strip():
            raise ValueError(f"invalid {question_type} question: {question_id}")
        seen_ids.add(question_id)
        attack_id = record.get("attack_id") if question_type == "attack" else None
        if question_type == "attack" and not isinstance(attack_id, str):
            raise ValueError("attack questions require attack_id")
        questions.append(
            BenchmarkQuestion(
                question_id=question_id,
                question=question,
                question_type=question_type,
                attack_id=attack_id,
            )
        )
    return tuple(questions)


def _serialize_sources(chunks: tuple[RetrievedChunk, ...]) -> list[dict[str, object]]:
    return [asdict(chunk) for chunk in chunks]


def _serialize_trace(trace: tuple[Any, ...]) -> list[dict[str, object]]:
    return [
        {
            "original_rank": entry.original_rank,
            "filename": entry.filename,
            "page_number": entry.page_number,
            "chunk_id": entry.chunk_id,
            "included": entry.included,
            "stage_decisions": [asdict(decision) for decision in entry.stage_decisions],
            "final_rank": entry.final_rank,
        }
        for entry in trace
    ]


def _default_output_path(settings: Settings) -> Path:
    return settings.project_root / "experiments" / "results" / "phase3_defense_analysis.json"


def run_phase3_defense_analysis(
    settings: Settings,
    *,
    retriever: Retriever | Any | None = None,
    coordinator: DefenseCoordinator | Any | None = None,
    output_path: Path | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    """Retrieve each fixed benchmark question once and evaluate all defense modes."""

    destination = output_path or _default_output_path(settings)
    if destination.exists():
        raise FileExistsError(f"analysis result already exists: {destination}")

    attacks = load_attack_manifest(settings.attack_manifest_path)
    if len(attacks) != 10:
        raise ValueError("expected 10 evaluation-only attack cases")
    attacks_by_id = {attack.attack_id: attack for attack in attacks}
    if len(attacks_by_id) != len(attacks):
        raise ValueError("attack cases must have unique attack_id values")

    attack_questions = _load_questions(
        settings.attack_questions_path, question_type="attack", expected_count=30
    )
    control_questions = _load_questions(
        settings.clean_control_questions_path,
        question_type="clean_control",
        expected_count=18,
    )
    if any(question.attack_id not in attacks_by_id for question in attack_questions):
        raise ValueError("attack question refers to an unknown attack_id")

    clean_filenames = load_trusted_filenames(settings.manifest_path)
    active_retriever = retriever or Retriever(
        settings, collection_name=ATTACKED_COLLECTION_NAME
    )
    active_coordinator = coordinator or DefenseCoordinator(
        trusted_filenames=clean_filenames,
        similarity_threshold=settings.defense_similarity_threshold,
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
    )

    questions = (*attack_questions, *control_questions)
    outcomes: list[dict[str, object]] = []
    evaluations_by_mode: dict[str, list[dict[str, bool | int]]] = {
        mode.value: [] for mode in DefenseMode
    }
    latencies_by_mode: dict[str, list[float]] = {
        mode.value: [] for mode in DefenseMode
    }
    for benchmark_question in questions:
        retrieval_started = clock()
        snapshot = tuple(active_retriever.retrieve(benchmark_question.question))
        retrieval_latency_ms = round((clock() - retrieval_started) * 1000, 2)

        defense_results: dict[DefenseMode, tuple[Any, float]] = {}
        for mode in DefenseMode:
            defense_started = clock()
            result = active_coordinator.apply(snapshot, mode)
            defense_latency_ms = round((clock() - defense_started) * 1000, 2)
            defense_results[mode] = (result, defense_latency_ms)

        attack = (
            attacks_by_id[benchmark_question.attack_id]
            if benchmark_question.attack_id is not None
            else None
        )
        defenses: dict[str, object] = {}
        for mode in DefenseMode:
            result, defense_latency_ms = defense_results[mode]
            evaluation = evaluate_defense_result(
                snapshot,
                result,
                attack=attack,
                clean_filenames=clean_filenames,
            )
            evaluations_by_mode[mode.value].append(evaluation)
            latencies_by_mode[mode.value].append(defense_latency_ms)
            defenses[mode.value] = {
                "sources": _serialize_sources(result.chunks),
                "trace": _serialize_trace(result.trace),
                "defense_latency_ms": defense_latency_ms,
                "evaluation": evaluation,
            }
        outcomes.append(
            {
                "question_id": benchmark_question.question_id,
                "question": benchmark_question.question,
                "question_type": benchmark_question.question_type,
                "attack_id": benchmark_question.attack_id,
                "retrieval_latency_ms": retrieval_latency_ms,
                "source_snapshot": _serialize_sources(snapshot),
                "defenses": defenses,
            }
        )

    result: dict[str, object] = {
        "gemini_calls": 0,
        "question_count": len(questions),
        "attack_question_count": len(attack_questions),
        "clean_control_question_count": len(control_questions),
        "defense_modes": [mode.value for mode in DefenseMode],
        "questions": outcomes,
        "summary": {
            mode.value: aggregate_mode_metrics(
                evaluations_by_mode[mode.value], latencies_by_mode[mode.value]
            )
            for mode in DefenseMode
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    """Run the local-only defense analysis from project configuration."""

    run_phase3_defense_analysis(Settings.from_env())


if __name__ == "__main__":
    main()
