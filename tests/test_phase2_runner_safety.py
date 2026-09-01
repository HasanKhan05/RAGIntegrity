import json
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_phase2_attacks as runner
import experiments.phase2_preflight as preflight
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
                "CHROMA_PERSIST_DIR=vector-store",
            ]
        ),
        encoding="utf-8",
    )
    settings = Settings.from_env(env_file)
    settings.attack_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.attack_manifest_path.write_text(
        json.dumps(
            {
                "attacks": [
                    {
                        "attack_id": f"attack_{number}",
                        "synthetic_document_id": f"synthetic-{number}",
                        "synthetic_filename": f"update-{number}.pdf",
                        "attack_type": "false_specification",
                        "target_model": "Test model",
                        "target_topic": "test topic",
                        "clean_fact": "The value is 1 litre.",
                        "false_claim": "The value is 71 litres.",
                        "clean_source_filename": "brochure.pdf",
                        "clean_source_page": 1,
                        "target_test_question": "What is the value?",
                        "false_value": "71",
                        "false_unit_aliases": ["litre"],
                    }
                    for number in range(1, 4)
                ]
            }
        ),
        encoding="utf-8",
    )
    return settings


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        self.calls.append(question)
        return GeneratedAnswer("Answer", TokenUsage(1, 1, 2))


class FakeRetrieverFactory:
    def __call__(self, collection_name: str) -> object:
        return type(
            "FakeRetriever",
            (),
            {"retrieve": lambda self, question: []},
        )()


def _configure_production_inventory(settings: Settings) -> set[tuple[str, str]]:
    manifest = json.loads(settings.attack_manifest_path.read_text(encoding="utf-8"))
    attacks = manifest["attacks"]
    settings.poisoned_data_dir.mkdir(parents=True, exist_ok=True)
    for attack in attacks:
        (settings.poisoned_data_dir / attack["synthetic_filename"]).write_bytes(b"")
    clean_pair = ("clean", "brochure.pdf")
    expected_poison = {
        (attack["synthetic_document_id"], attack["synthetic_filename"])
        for attack in attacks
    }
    settings.manifest_path.write_text(
        json.dumps({"documents": [{"document_id": clean_pair[0], "filename": clean_pair[1]}]}),
        encoding="utf-8",
    )
    settings.attacked_manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"document_id": clean_pair[0], "filename": clean_pair[1]},
                    *[
                        {"document_id": document_id, "filename": filename}
                        for document_id, filename in sorted(expected_poison)
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    return expected_poison


def test_existing_output_stops_before_any_generation(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    output_path.write_text("existing", encoding="utf-8")
    generator = RecordingGenerator()

    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_phase2(
            _settings(tmp_path),
            generator=generator,
            retriever_factory=FakeRetrieverFactory(),
            output_path=output_path,
        )

    assert generator.calls == []


@pytest.mark.parametrize("change", [{"llm_temperature": 0.1}, {"top_k": 2}])
def test_unsafe_settings_stop_before_any_generation(
    tmp_path: Path, change: dict[str, object]
) -> None:
    generator = RecordingGenerator()

    with pytest.raises(ValueError):
        runner.run_phase2(
            replace(_settings(tmp_path), **change),
            generator=generator,
            retriever_factory=FakeRetrieverFactory(),
            output_path=tmp_path / "results.json",
        )

    assert generator.calls == []


def test_missing_generation_config_stops_before_any_generation(tmp_path: Path) -> None:
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        runner.run_phase2(
            replace(_settings(tmp_path), llm_api_key=None),
            generator=generator,
            retriever_factory=FakeRetrieverFactory(),
            output_path=tmp_path / "results.json",
        )

    assert generator.calls == []


def test_production_inventory_mismatch_stops_before_any_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="poisoned PDF inventory"):
        runner.run_phase2(
            _settings(tmp_path), generator=generator, output_path=tmp_path / "results.json"
        )

    assert generator.calls == []


def test_production_missing_collection_stops_before_any_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    manifest = json.loads(settings.attack_manifest_path.read_text(encoding="utf-8"))
    attacks = manifest["attacks"]
    settings.poisoned_data_dir.mkdir(parents=True, exist_ok=True)
    for attack in attacks:
        (settings.poisoned_data_dir / attack["synthetic_filename"]).write_bytes(b"")
    settings.manifest_path.write_text(
        json.dumps({"documents": [{"document_id": "clean", "filename": "brochure.pdf"}]}),
        encoding="utf-8",
    )
    settings.attacked_manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"document_id": "clean", "filename": "brochure.pdf"},
                    *[
                        {
                            "document_id": attack["synthetic_document_id"],
                            "filename": attack["synthetic_filename"],
                        }
                        for attack in attacks
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )

    def missing_collection(
        actual_settings: Settings, collection_name: str
    ) -> set[tuple[str, str]]:
        if collection_name == "attacked_brochures":
            raise ValueError("required collection is unavailable: attacked_brochures")
        return set()

    monkeypatch.setattr(preflight, "_collection_pairs", missing_collection)
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="required collection is unavailable"):
        runner.run_phase2(
            settings, generator=generator, output_path=tmp_path / "results.json"
        )

    assert generator.calls == []


def test_extra_poisoned_pdf_stops_before_any_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    expected_poison = _configure_production_inventory(settings)
    (settings.poisoned_data_dir / "unlisted-update.pdf").write_bytes(b"")
    monkeypatch.setattr(
        preflight,
        "_collection_pairs",
        lambda actual_settings, collection_name: (
            {("clean", "brochure.pdf")} if collection_name == "clean_brochures" else expected_poison
        ),
    )
    monkeypatch.setattr(
        runner,
        "Retriever",
        lambda actual_settings, collection_name: type(
            "FakeRetriever", (), {"retrieve": lambda self, question: []}
        )(),
    )
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="inventory"):
        runner.run_phase2(
            settings, generator=generator, output_path=tmp_path / "results.json"
        )

    assert generator.calls == []


def test_extra_attacked_manifest_document_stops_before_any_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    expected_poison = _configure_production_inventory(settings)
    attacked = json.loads(settings.attacked_manifest_path.read_text(encoding="utf-8"))
    attacked["documents"].append(
        {"document_id": "unlisted", "filename": "unlisted-update.pdf"}
    )
    settings.attacked_manifest_path.write_text(json.dumps(attacked), encoding="utf-8")
    monkeypatch.setattr(
        preflight,
        "_collection_pairs",
        lambda actual_settings, collection_name: (
            {("clean", "brochure.pdf")} if collection_name == "clean_brochures" else expected_poison
        ),
    )
    monkeypatch.setattr(
        runner,
        "Retriever",
        lambda actual_settings, collection_name: type(
            "FakeRetriever", (), {"retrieve": lambda self, question: []}
        )(),
    )
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="inventory"):
        runner.run_phase2(
            settings, generator=generator, output_path=tmp_path / "results.json"
        )

    assert generator.calls == []
