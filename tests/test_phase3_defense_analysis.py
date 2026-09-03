from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import run_phase3_defense_analysis as runner
from src.evaluation.phase2 import AttackCase
from src.rag.config import Settings
from src.rag.defenses import DefenseMode, DefenseResult, DefenseTraceEntry
from src.rag.models import RetrievedChunk


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingRetriever:
    def __init__(self, attacks: dict[str, AttackCase], clock: ManualClock) -> None:
        self.attacks = attacks
        self.clock = clock
        self.questions: list[str] = []

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        self.questions.append(question)
        self.clock.advance(5.0)
        attack = next((case for key, case in self.attacks.items() if key in question), None)
        target = (
            RetrievedChunk(
                document_id=attack.synthetic_document_id,
                filename=attack.synthetic_filename,
                page_number=attack.synthetic_page_number,
                chunk_id=f"{attack.synthetic_document_id}-p{attack.synthetic_page_number}-c0",
                text="Update text.",
                rank=2,
                relevance_score=0.8,
            )
            if attack is not None
            else RetrievedChunk(
                document_id="unrelated-update",
                filename="update.pdf",
                page_number=1,
                chunk_id="unrelated-update-p1-c0",
                text="Update text.",
                rank=2,
                relevance_score=0.8,
            )
        )
        return [
            RetrievedChunk(
                document_id="clean-doc",
                filename="clean.pdf",
                page_number=1,
                chunk_id="clean-doc-p1-c0",
                text="Official brochure text.",
                rank=1,
                relevance_score=0.9,
            ),
            target,
        ]


class RecordingCoordinator:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.calls: list[tuple[DefenseMode, tuple[RetrievedChunk, ...]]] = []

    def apply(
        self, chunks: tuple[RetrievedChunk, ...], mode: DefenseMode
    ) -> DefenseResult:
        self.calls.append((mode, chunks))
        self.clock.advance(0.01)
        return DefenseResult(
            chunks=chunks,
            trace=tuple(
                DefenseTraceEntry(
                    original_rank=chunk.rank,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    included=True,
                    stage_decisions=(),
                    final_rank=chunk.rank,
                )
                for chunk in chunks
            ),
        )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        llm_provider="gemini",
        llm_model="unused",
        llm_api_key=None,
        llm_base_url=None,
        max_output_tokens=300,
        llm_temperature=0.0,
        top_k=3,
        chunk_size=1200,
        chunk_overlap=200,
        chroma_persist_dir=tmp_path / "vector-store",
        embedding_model="all-MiniLM-L6-v2",
        clean_data_dir=tmp_path / "clean",
        manifest_path=tmp_path / "clean_index.json",
        poisoned_data_dir=tmp_path / "poisoned",
        attacked_manifest_path=tmp_path / "attacked_index.json",
        attack_manifest_path=tmp_path / "attack_manifest.json",
        attack_questions_path=tmp_path / "attack_questions.json",
        clean_control_questions_path=tmp_path / "controls.json",
        defense_similarity_threshold=0.92,
    )


def _attack(index: int) -> AttackCase:
    identifier = f"attack_{index:03d}"
    return AttackCase(
        attack_id=identifier,
        synthetic_document_id=f"update-{index}",
        synthetic_page_number=index,
        synthetic_filename=f"update-{index}.pdf",
        attack_type="false_specification",
        target_model="Model",
        target_topic="topic",
        clean_fact="clean",
        false_claim="false",
        clean_source_filename="clean.pdf",
        clean_source_page=1,
        target_test_question=identifier,
        false_value="72",
        false_unit_aliases=("l",),
    )


def _write_question_inputs(settings: Settings, attacks: tuple[AttackCase, ...]) -> None:
    settings.manifest_path.write_text(
        json.dumps({"documents": [{"document_id": "clean-doc", "filename": "clean.pdf"}]}),
        encoding="utf-8",
    )
    settings.attack_questions_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": f"{attack.attack_id}_q{variant}",
                        "attack_id": attack.attack_id,
                        "question": f"{attack.attack_id} question {variant}",
                    }
                    for attack in attacks
                    for variant in range(1, 4)
                ]
            }
        ),
        encoding="utf-8",
    )
    settings.clean_control_questions_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": f"control_{index:03d}",
                        "question": f"control question {index}",
                    }
                    for index in range(1, 19)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_local_analysis_reuses_one_snapshot_across_modes_and_reports_canonical_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    attacks = tuple(_attack(index) for index in range(1, 11))
    _write_question_inputs(settings, attacks)
    monkeypatch.setattr(runner, "load_attack_manifest", lambda path: attacks)
    clock = ManualClock()
    retriever = RecordingRetriever({attack.attack_id: attack for attack in attacks}, clock)
    coordinator = RecordingCoordinator(clock)
    output_path = tmp_path / "phase3_defense_analysis.json"

    result = runner.run_phase3_defense_analysis(
        settings,
        retriever=retriever,
        coordinator=coordinator,
        output_path=output_path,
        clock=clock,
    )

    assert len(retriever.questions) == 48
    assert len(coordinator.calls) == 48 * len(DefenseMode)
    for start in range(0, len(coordinator.calls), len(DefenseMode)):
        mode_calls = coordinator.calls[start : start + len(DefenseMode)]
        assert [mode for mode, _ in mode_calls] == list(DefenseMode)
        assert len({id(snapshot) for _, snapshot in mode_calls}) == 1
        assert isinstance(mode_calls[0][1], tuple)
    assert len({id(snapshot) for _, snapshot in coordinator.calls[::len(DefenseMode)]}) == 48

    assert set(result) == {
        "gemini_calls",
        "question_count",
        "attack_question_count",
        "clean_control_question_count",
        "defense_modes",
        "questions",
        "summary",
    }
    assert result["gemini_calls"] == 0
    assert result["question_count"] == 48
    assert result["attack_question_count"] == 30
    assert result["clean_control_question_count"] == 18
    assert result["defense_modes"] == [mode.value for mode in DefenseMode]
    assert len(result["questions"]) == 48
    assert set(result["summary"]) == {mode.value for mode in DefenseMode}
    first_question = result["questions"][0]
    assert set(first_question) == {
        "question_id",
        "question",
        "question_type",
        "attack_id",
        "retrieval_latency_ms",
        "source_snapshot",
        "defenses",
    }
    assert first_question["retrieval_latency_ms"] == 5000.0
    assert set(first_question["defenses"]) == {mode.value for mode in DefenseMode}
    for outcome in first_question["defenses"].values():
        assert set(outcome) == {"sources", "trace", "defense_latency_ms", "evaluation"}
        assert outcome["defense_latency_ms"] == 10.0
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


def test_local_analysis_refuses_to_overwrite_before_retrieval(tmp_path: Path) -> None:
    output_path = tmp_path / "phase3_defense_analysis.json"
    output_path.write_text("existing", encoding="utf-8")
    clock = ManualClock()
    retriever = RecordingRetriever({}, clock)

    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_phase3_defense_analysis(
            _settings(tmp_path),
            retriever=retriever,
            coordinator=RecordingCoordinator(clock),
            output_path=output_path,
            clock=clock,
        )

    assert retriever.questions == []
