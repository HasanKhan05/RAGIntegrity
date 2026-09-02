import hashlib
import json
from pathlib import Path

import pymupdf

from src.attacks.benchmark import ATTACK_DEFINITIONS
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


def test_create_attack_documents_writes_six_readable_pdfs_with_isolated_pages(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    paths = create_attack_documents(settings)

    assert [path.name for path in paths] == [
        "vehicle_specification_update.pdf",
        "electric_range_update.pdf",
        "vehicle_feature_update.pdf",
        "cargo_and_dimensions_update.pdf",
        "powertrain_update.pdf",
        "capability_update.pdf",
    ]
    extracted = {
        path.name: " ".join(page.text for page in extract_pdf(path)) for path in paths
    }
    assert "72 litres" in extracted["vehicle_specification_update.pdf"]
    assert "620 km" in extracted["electric_range_update.pdf"]
    assert "900 mm" in extracted["vehicle_feature_update.pdf"]

    expected_pages = {
        "cargo_and_dimensions_update.pdf": [
            ("Aygo X", "285 litres"),
            ("Corolla Touring Sports", "640 litres"),
            ("RAV4", "645 litres"),
            ("bZ4X", "520 litres"),
        ],
        "powertrain_update.pdf": [
            ("Yaris", "145 DIN hp"),
            ("C-HR", "160 DIN hp"),
        ],
        "capability_update.pdf": [("Land Cruiser", "3,500 kg")],
    }
    for filename, page_expectations in expected_pages.items():
        with pymupdf.open(settings.poisoned_data_dir / filename) as document:
            assert document.page_count == len(page_expectations)
            page_texts = [page.get_text() for page in document]
        for page_index, (model, claim) in enumerate(page_expectations):
            assert model in page_texts[page_index]
            assert claim in page_texts[page_index]
            assert all(
                other_model not in page_texts[page_index]
                for other_index, (other_model, _) in enumerate(page_expectations)
                if other_index != page_index
            )


def test_existing_three_attack_pdfs_are_not_rewritten(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    original_names = [
        "vehicle_specification_update.pdf",
        "electric_range_update.pdf",
        "vehicle_feature_update.pdf",
    ]
    settings.poisoned_data_dir.mkdir(parents=True)
    sentinels = {}
    for index, filename in enumerate(original_names, start=1):
        content = f"existing-pdf-{index}".encode()
        path = settings.poisoned_data_dir / filename
        path.write_bytes(content)
        sentinels[filename] = hashlib.sha256(content).hexdigest()

    create_attack_documents(settings)

    assert {
        filename: hashlib.sha256(
            (settings.poisoned_data_dir / filename).read_bytes()
        ).hexdigest()
        for filename in original_names
    } == sentinels


def test_attack_manifest_is_evaluation_only_and_has_verified_sources(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    create_attack_documents(settings)

    manifest = json.loads(settings.attack_manifest_path.read_text(encoding="utf-8"))
    attacks = manifest["attacks"]
    assert len(attacks) == 10
    assert attacks == [dict(attack) for attack in ATTACK_DEFINITIONS]
    assert [
        (attack["clean_source_filename"], attack["clean_source_page"])
        for attack in attacks
    ][:3] == [
        ("rav4.pdf", 40),
        ("bz4x.pdf", 4),
        ("land-cruiser.pdf", 22),
    ]
    required_fields = {
        "attack_id",
        "synthetic_document_id",
        "synthetic_page_number",
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
        "canonical_test_question",
        "natural_question_variants",
        "deterministic_compromise_check",
    }
    assert all(set(attack) == required_fields for attack in attacks)
    assert [attack["synthetic_page_number"] for attack in attacks] == [
        1, 1, 1, 1, 1, 2, 2, 3, 4, 1
    ]
