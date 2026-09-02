from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.run_phase2_expansion_smoke as smoke
from experiments.run_phase2_expansion_smoke import run_expansion_smoke
from src.attacks.benchmark import write_expansion_data
from src.rag.config import Settings
from src.rag.models import GeneratedAnswer, RetrievedChunk, TokenUsage


def _settings(tmp_path: Path) -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=gemini",
                "LLM_API_KEY=test-key",
                "LLM_MODEL=gemini-3.5-flash-lite",
                "TOP_K=3",
                "LLM_TEMPERATURE=0",
            ]
        ),
        encoding="utf-8",
    )
    settings = Settings.from_env(env_file)
    write_expansion_data(settings)
    return settings


def _chunk(document_id: str, page: int, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=page,
        chunk_id=f"{document_id}-p{page}-c0",
        text="Ordinary source text.",
        rank=rank,
        relevance_score=0.9,
    )


class RecordingGenerator:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        self.calls.append(question)
        return GeneratedAnswer(self.answers.pop(0), TokenUsage(1, 1, 2))


class RetrieverFactory:
    def __init__(self, document_ids: dict[str, str] | None = None) -> None:
        self.document_ids = document_ids or {}

    def __call__(self, collection_name: str) -> object:
        document_ids = self.document_ids

        def retrieve(instance: object, question: str) -> list[RetrievedChunk]:
            if collection_name == "clean_brochures":
                return [_chunk("clean", 1)]
            if "Aygo X" in question:
                return [_chunk(document_ids.get("attack_004", "a4"), 3)]
            if "Yaris" in question:
                return [_chunk(document_ids.get("attack_005", "a5"), 1, 2)]
            return [_chunk(document_ids.get("attack_010", "a10"), 1, 3)]

        return type("FakeRetriever", (), {"retrieve": retrieve})()


def test_smoke_runner_caps_calls_and_keeps_compromise_outcomes_separate(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    manifest = json.loads(settings.attack_manifest_path.read_text(encoding="utf-8"))
    ids = {item["attack_id"]: item["synthetic_document_id"] for item in manifest["attacks"]}
    # Keep test chunks coupled to the generated document IDs.
    factory = RetrieverFactory(ids)
    generator = RecordingGenerator(
        [
            "231 litres.",
            "The update says 285 litres, but the official brochure says 231 litres.",
            "130 DIN hp.",
            "The Yaris produces 145 DIN hp.",
            "3,000 kg.",
            "The Land Cruiser towing capacity is 3,000 kg.",
        ]
    )

    result = run_expansion_smoke(
        settings,
        generator=generator,
        retriever_factory=factory,
        output_path=tmp_path / "smoke.json",
    )

    assert len(generator.calls) == 6
    assert result["gemini_calls"] == 6
    assert result["token_usage"] == {
        "input_tokens": 6,
        "output_tokens": 6,
        "total_tokens": 12,
    }
    assert [item["attack_id"] for item in result["attacks"]] == [
        "attack_004",
        "attack_005",
        "attack_010",
    ]
    aygo, yaris, towing = result["attacks"]
    assert aygo["poison_retrieved"] is False  # page 3 cannot satisfy page 1
    assert aygo["generation_compromised"] is None
    assert aygo["manual_review_required"] is True
    assert yaris["poison_rank"] == 2
    assert yaris["generation_compromised"] is True
    assert towing["poison_rank"] == 3
    assert towing["generation_compromised"] is False
    assert set(yaris["attacked_sources"][0]) == {
        "document_id",
        "filename",
        "page_number",
        "chunk_id",
        "text",
        "rank",
        "relevance_score",
    }


def test_smoke_runner_refuses_existing_output_before_generation(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output = tmp_path / "smoke.json"
    output.write_text("existing", encoding="utf-8")
    generator = RecordingGenerator(["unused"] * 6)

    with pytest.raises(FileExistsError, match="already exists"):
        run_expansion_smoke(
            settings,
            generator=generator,
            retriever_factory=RetrieverFactory(),
            output_path=output,
        )

    assert generator.calls == []


def test_smoke_runner_passes_full_inventory_to_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    observed: list[tuple[list[str], int]] = []

    def fake_prepare(
        actual_settings: Settings,
        selected: object,
        inventory: object,
        output_path: Path | None,
        *,
        validate_collections: bool,
    ) -> Path:
        observed.append(([item.attack_id for item in selected], len(inventory)))
        return (tmp_path / "smoke.json").resolve()

    monkeypatch.setattr(smoke, "prepare_run", fake_prepare)
    run_expansion_smoke(
        settings,
        generator=RecordingGenerator(["Answer"] * 6),
        retriever_factory=RetrieverFactory(),
        output_path=tmp_path / "smoke.json",
    )

    assert observed == [(["attack_004", "attack_005", "attack_010"], 10)]
