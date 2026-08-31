import json
from pathlib import Path

from src.attacks.create_attack_pdfs import create_attack_documents
from src.rag.config import Settings
from src.rag.ingest import extract_pdf


def _settings(tmp_path: Path) -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=gemini\nLLM_MODEL=gemini-3.5-flash-lite\n",
        encoding="utf-8",
    )
    return Settings.from_env(env_file)


def test_create_attack_documents_writes_exactly_three_readable_pdfs(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    paths = create_attack_documents(settings)

    assert [path.name for path in paths] == [
        "vehicle_specification_update.pdf",
        "electric_range_update.pdf",
        "vehicle_feature_update.pdf",
    ]
    extracted = {
        path.name: " ".join(page.text for page in extract_pdf(path)) for path in paths
    }
    assert "72 litres" in extracted["vehicle_specification_update.pdf"]
    assert "620 km" in extracted["electric_range_update.pdf"]
    assert "900 mm" in extracted["vehicle_feature_update.pdf"]


def test_attack_manifest_is_evaluation_only_and_has_verified_sources(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    create_attack_documents(settings)

    manifest = json.loads(settings.attack_manifest_path.read_text(encoding="utf-8"))
    attacks = manifest["attacks"]
    assert len(attacks) == 3
    assert [
        (attack["clean_source_filename"], attack["clean_source_page"])
        for attack in attacks
    ] == [
        ("rav4.pdf", 40),
        ("bz4x.pdf", 4),
        ("land-cruiser.pdf", 22),
    ]
    required_fields = {
        "attack_id",
        "synthetic_document_id",
        "synthetic_filename",
        "attack_type",
        "target_model",
        "target_topic",
        "clean_fact",
        "false_claim",
        "clean_source_filename",
        "clean_source_page",
        "target_test_question",
        "false_value",
        "false_unit_aliases",
    }
    assert all(set(attack) == required_fields for attack in attacks)
