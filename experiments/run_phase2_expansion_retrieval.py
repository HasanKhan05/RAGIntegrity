"""Run the 30-question Phase 2 expansion retrieval analysis without an LLM."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.evaluation.phase2 import detect_poison, load_attack_manifest
from src.rag.config import Settings
from src.rag.index import ATTACKED_COLLECTION_NAME
from src.rag.retrieve import Retriever


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list) or len(questions) != 30:
        raise ValueError("Phase 2 expansion requires exactly 30 attack questions")
    if not all(
        isinstance(item, dict)
        and isinstance(item.get("question_id"), str)
        and isinstance(item.get("attack_id"), str)
        and isinstance(item.get("question"), str)
        for item in questions
    ):
        raise ValueError("Phase 2 expansion question data is invalid")
    return questions


def _summary(results: list[dict[str, object]]) -> dict[str, int | float | None]:
    ranks = [int(item["poison_rank"]) for item in results if item["poison_rank"] is not None]
    return {
        "total_questions": len(results),
        "retrieved_count": len(ranks),
        "not_retrieved_count": len(results) - len(ranks),
        "rank_1_count": ranks.count(1),
        "rank_2_count": ranks.count(2),
        "rank_3_count": ranks.count(3),
        "average_poison_rank_retrieved": (
            round(sum(ranks) / len(ranks), 3) if ranks else None
        ),
    }


def run_expansion_retrieval(
    settings: Settings,
    *,
    retriever: Retriever | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Retrieve all expansion questions once and evaluate exact target pages."""

    destination = (
        output_path
        or settings.project_root
        / "experiments"
        / "results"
        / "phase2_expansion_retrieval.json"
    ).resolve()
    if destination.exists():
        raise FileExistsError(f"Phase 2 expansion output already exists: {destination}")
    if settings.top_k != 3:
        raise ValueError("Phase 2 expansion requires TOP_K=3")

    attacks = load_attack_manifest(settings.attack_manifest_path)
    if len(attacks) != 10:
        raise ValueError("Phase 2 expansion requires exactly 10 attacks")
    attacks_by_id = {attack.attack_id: attack for attack in attacks}
    if len(attacks_by_id) != 10:
        raise ValueError("Phase 2 expansion attack IDs must be unique")
    questions = _load_questions(settings.attack_questions_path)
    if any(item["attack_id"] not in attacks_by_id for item in questions):
        raise ValueError("Phase 2 expansion question references an unknown attack")

    active_retriever = retriever or Retriever(
        settings, collection_name=ATTACKED_COLLECTION_NAME
    )
    outcomes: list[dict[str, object]] = []
    for item in questions:
        attack = attacks_by_id[str(item["attack_id"])]
        chunks = tuple(active_retriever.retrieve(str(item["question"])))
        detection = detect_poison(
            chunks,
            attack.synthetic_document_id,
            attack.synthetic_page_number,
        )
        outcomes.append(
            {
                "question_id": item["question_id"],
                "attack_id": attack.attack_id,
                "question": item["question"],
                "sources": [asdict(chunk) for chunk in chunks],
                "poison_retrieved": detection.poison_retrieved,
                "poison_rank": detection.poison_rank,
                "retrieval_compromised": detection.retrieval_compromised,
            }
        )

    result: dict[str, object] = {
        "summary": _summary(outcomes),
        "questions": outcomes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    run_expansion_retrieval(Settings.from_env())


if __name__ == "__main__":
    main()
