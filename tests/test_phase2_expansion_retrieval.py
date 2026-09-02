from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.run_phase2_expansion_retrieval import run_expansion_retrieval
from src.rag.config import Settings
from src.rag.models import RetrievedChunk


def _settings(tmp_path: Path) -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=gemini\nLLM_MODEL=gemini-3.5-flash-lite\n",
        encoding="utf-8",
    )
    settings = Settings.from_env(env_file)
    attacks = []
    questions = []
    for number in range(1, 11):
        document_id = "shared" if number in (4, 8) else f"doc-{number}"
        page_number = 1 if number != 8 else 3
        attacks.append(
            {
                "attack_id": f"attack_{number:03d}",
                "synthetic_document_id": document_id,
                "synthetic_page_number": page_number,
                "synthetic_filename": "shared.pdf" if document_id == "shared" else f"doc-{number}.pdf",
                "attack_type": "false_specification",
                "target_model": f"Model {number}",
                "target_topic": "topic",
                "clean_fact": "Clean value 1.",
                "false_claim": "False value 2.",
                "clean_source_filename": "clean.pdf",
                "clean_source_page": 1,
                "target_test_question": f"Question {number}-1",
                "false_value": "2",
                "false_unit_aliases": ["units"],
            }
        )
        questions.extend(
            {
                "question_id": f"attack_{number:03d}_q{variant}",
                "attack_id": f"attack_{number:03d}",
                "question": f"Question {number}-{variant}",
                "target_model": f"Model {number}",
                "target_topic": "topic",
            }
            for variant in range(1, 4)
        )
    settings.attack_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.attack_manifest_path.write_text(
        json.dumps({"attacks": attacks}), encoding="utf-8"
    )
    settings.attack_questions_path.parent.mkdir(parents=True, exist_ok=True)
    settings.attack_questions_path.write_text(
        json.dumps({"questions": questions}), encoding="utf-8"
    )
    return settings


def _chunk(document_id: str, page: int, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=page,
        chunk_id=f"{document_id}-p{page}-c0",
        text="Ordinary retrieved text.",
        rank=rank,
        relevance_score=1.0 - rank / 10,
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        self.calls.append(question)
        if question == "Question 1-1":
            return [_chunk("doc-1", 1, 1)]
        if question == "Question 2-1":
            return [_chunk("clean", 1, 1), _chunk("doc-2", 1, 2)]
        if question == "Question 3-1":
            return [
                _chunk("clean", 1, 1),
                _chunk("other", 1, 2),
                _chunk("doc-3", 1, 3),
            ]
        if question == "Question 4-1":
            return [_chunk("shared", 3, 1)]  # RAV4 page, not Aygo page.
        return [_chunk("clean", 1, 1)]


def test_retrieval_runner_is_page_aware_and_aggregates_ranks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    retriever = FakeRetriever()

    result = run_expansion_retrieval(
        settings,
        retriever=retriever,
        output_path=tmp_path / "retrieval.json",
    )

    assert len(retriever.calls) == 30
    assert len(result["questions"]) == 30
    by_question = {item["question"]: item for item in result["questions"]}
    assert [
        by_question[f"Question {number}-1"]["poison_rank"]
        for number in range(1, 5)
    ] == [1, 2, 3, None]
    assert by_question["Question 4-1"]["poison_retrieved"] is False
    assert result["summary"] == {
        "total_questions": 30,
        "retrieved_count": 3,
        "not_retrieved_count": 27,
        "rank_1_count": 1,
        "rank_2_count": 1,
        "rank_3_count": 1,
        "average_poison_rank_retrieved": 2.0,
    }
    assert set(by_question["Question 1-1"]["sources"][0]) == {
        "document_id",
        "filename",
        "page_number",
        "chunk_id",
        "text",
        "rank",
        "relevance_score",
    }
    assert json.loads((tmp_path / "retrieval.json").read_text(encoding="utf-8")) == result


def test_retrieval_runner_refuses_to_overwrite(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output = tmp_path / "retrieval.json"
    output.write_text("existing", encoding="utf-8")
    retriever = FakeRetriever()

    with pytest.raises(FileExistsError, match="already exists"):
        run_expansion_retrieval(settings, retriever=retriever, output_path=output)

    assert retriever.calls == []
