import json
from dataclasses import replace
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

import experiments.run_phase2_attacks as runner
import experiments.phase2_preflight as preflight
from src.evaluation.phase2 import AttackCase
from src.rag.config import Settings
from src.rag.models import GeneratedAnswer, RetrievedChunk, TokenUsage
from src.rag.retrieve import Retriever


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
                        "attack_id": f"attack_{number:03d}",
                        "synthetic_document_id": f"synthetic-{number}",
                        "synthetic_page_number": 1,
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


def _attack_case(
    attack_id: str,
    document_id: str,
    filename: str,
    page_number: int,
) -> AttackCase:
    return AttackCase(
        attack_id=attack_id,
        synthetic_document_id=document_id,
        synthetic_page_number=page_number,
        synthetic_filename=filename,
        attack_type="false_specification",
        target_model="Model",
        target_topic="Topic",
        clean_fact="Clean",
        false_claim="False",
        clean_source_filename="clean.pdf",
        clean_source_page=1,
        target_test_question="Question?",
        false_value="71",
        false_unit_aliases=("litres",),
    )


def test_attack_inventory_allows_shared_documents_with_unique_pages() -> None:
    attacks = [
        _attack_case("attack_004", "cargo", "cargo.pdf", 1),
        _attack_case("attack_006", "cargo", "cargo.pdf", 2),
        _attack_case("attack_005", "power", "power.pdf", 1),
        _attack_case("attack_007", "power", "power.pdf", 2),
    ]

    assert preflight._attack_inventory(attacks) == {
        ("cargo", "cargo.pdf"),
        ("power", "power.pdf"),
    }


@pytest.mark.parametrize(
    "attacks, message",
    [
        (
            [
                _attack_case("a", "same-id", "one.pdf", 1),
                _attack_case("b", "same-id", "two.pdf", 2),
            ],
            "conflicting filenames",
        ),
        (
            [
                _attack_case("a", "one-id", "same.pdf", 1),
                _attack_case("b", "two-id", "same.pdf", 2),
            ],
            "conflicting document IDs",
        ),
        (
            [
                _attack_case("a", "same-id", "same.pdf", 1),
                _attack_case("b", "same-id", "same.pdf", 1),
            ],
            "document-plus-page target is not unique",
        ),
    ],
)
def test_attack_inventory_rejects_conflicts(
    attacks: list[AttackCase], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        preflight._attack_inventory(attacks)


def test_preflight_and_retriever_share_chroma_client_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.create_collection("attacked_brochures")
    collection.add(
        ids=["chunk-1"],
        documents=["Ordinary text."],
        metadatas=[
            {
                "document_id": "doc-1",
                "filename": "doc-1.pdf",
                "page_number": 1,
                "chunk_id": "chunk-1",
            }
        ],
        embeddings=[[0.0, 1.0]],
    )

    assert preflight._collection_pairs(settings, "attacked_brochures") == {
        ("doc-1", "doc-1.pdf")
    }
    chunks = Retriever(
        settings,
        collection_name="attacked_brochures",
        embedder=type("FakeEmbedder", (), {"encode": lambda self, texts: [[0.0, 1.0]]})(),
    ).retrieve("question")
    assert chunks[0].document_id == "doc-1"


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
        if collection_name == "clean_brochures":
            return {("clean", "brochure.pdf")}
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


def test_missing_clean_collection_document_stops_before_any_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    expected_poison = _configure_production_inventory(settings)
    clean_manifest = {
        "documents": [
            {"document_id": "clean", "filename": "brochure.pdf"},
            {"document_id": "missing", "filename": "missing.pdf"},
        ]
    }
    settings.manifest_path.write_text(json.dumps(clean_manifest), encoding="utf-8")
    settings.attacked_manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    *clean_manifest["documents"],
                    *[
                        {"document_id": document_id, "filename": filename}
                        for document_id, filename in sorted(expected_poison)
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preflight,
        "_collection_pairs",
        lambda actual_settings, collection_name: (
            {("clean", "brochure.pdf")}
            if collection_name == "clean_brochures"
            else {("clean", "brochure.pdf"), ("missing", "missing.pdf")} | expected_poison
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

    with pytest.raises(ValueError, match="clean collection inventory"):
        runner.run_phase2(
            settings, generator=generator, output_path=tmp_path / "results.json"
        )

    assert generator.calls == []


def test_extra_clean_collection_document_stops_before_any_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    expected_poison = _configure_production_inventory(settings)
    clean_pairs = {("clean", "brochure.pdf"), ("extra", "extra.pdf")}
    attacked_pairs = {("clean", "brochure.pdf")} | expected_poison
    monkeypatch.setattr(
        preflight,
        "_collection_pairs",
        lambda actual_settings, collection_name: (
            clean_pairs if collection_name == "clean_brochures" else attacked_pairs
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

    with pytest.raises(ValueError, match="clean collection inventory"):
        runner.run_phase2(
            settings, generator=generator, output_path=tmp_path / "results.json"
        )

    assert generator.calls == []


@pytest.mark.parametrize(
    "entry",
    [
        {"document_id": "missing-filename"},
        {"filename": "missing-id"},
        "not an object",
    ],
)
def test_manifest_pairs_rejects_malformed_document_entries(
    tmp_path: Path, entry: object
) -> None:
    manifest_path = tmp_path / "index.json"
    manifest_path.write_text(json.dumps({"documents": [entry]}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid document entry"):
        preflight._manifest_pairs(manifest_path)
