"""Safety checks for the paid controlled Phase 2 experiment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from src.evaluation.phase2 import AttackCase
from src.rag.config import Settings
from src.rag.index import ATTACKED_COLLECTION_NAME, CLEAN_COLLECTION_NAME


def prepare_run(
    settings: Settings,
    run_attacks: Sequence[AttackCase],
    inventory_attacks: Sequence[AttackCase],
    output_path: Path | None,
    *,
    validate_collections: bool,
) -> Path:
    """Validate a safe, fixed experiment before any retriever or model is built."""

    destination = output_path or (
        settings.project_root / "experiments" / "results" / "phase2_attack_results.json"
    )
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Phase 2 results output already exists: {destination}")
    settings.require_generation()
    if settings.llm_temperature != 0:
        raise ValueError("Phase 2 requires LLM_TEMPERATURE=0")
    if settings.top_k != 3:
        raise ValueError("Phase 2 requires TOP_K=3")
    _attack_inventory(inventory_attacks)
    if validate_collections:
        _validate_collections(settings, inventory_attacks)
    return destination


def _attack_inventory(
    attacks: Sequence[AttackCase],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    document_names: dict[str, str] = {}
    filename_ids: dict[str, str] = {}
    target_pages: set[tuple[str, int]] = set()
    for attack in attacks:
        pair = (attack.synthetic_document_id, attack.synthetic_filename)
        previous_name = document_names.setdefault(*pair)
        if previous_name != attack.synthetic_filename:
            raise ValueError("synthetic document ID maps to conflicting filenames")
        previous_id = filename_ids.setdefault(attack.synthetic_filename, attack.synthetic_document_id)
        if previous_id != attack.synthetic_document_id:
            raise ValueError("synthetic filename maps to conflicting document IDs")
        target = (attack.synthetic_document_id, attack.synthetic_page_number)
        if target in target_pages:
            raise ValueError("attack manifest document-plus-page target is not unique")
        target_pages.add(target)
        pairs.add(pair)
    return pairs


def _manifest_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        raise ValueError(f"index manifest is unavailable: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list):
        raise ValueError(f"index manifest is invalid: {path}")
    pairs: set[tuple[str, str]] = set()
    for document in documents:
        if (
            not isinstance(document, dict)
            or not isinstance(document.get("document_id"), str)
            or not isinstance(document.get("filename"), str)
            or not document["document_id"]
            or not document["filename"]
        ):
            raise ValueError(f"index manifest contains invalid document entry: {path}")
        pairs.add((document["document_id"], document["filename"]))
    return pairs


def _collection_pairs(settings: Settings, collection_name: str) -> set[tuple[str, str]]:
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(collection_name)
    except NotFoundError as error:
        raise ValueError(f"required collection is unavailable: {collection_name}") from error
    metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
    return {
        (str(metadata["document_id"]), str(metadata["filename"]))
        for metadata in metadatas
        if isinstance(metadata, dict)
        and isinstance(metadata.get("document_id"), str)
        and isinstance(metadata.get("filename"), str)
    }


def _validate_collections(settings: Settings, attacks: Sequence[AttackCase]) -> None:
    expected = _attack_inventory(attacks)
    expected_ids = {document_id for document_id, _ in expected}
    expected_names = {filename for _, filename in expected}
    poisoned_names = {path.name for path in settings.poisoned_data_dir.glob("*.pdf")}
    if poisoned_names != expected_names:
        raise ValueError("generated poisoned PDF inventory does not match attack manifest")
    clean_manifest = _manifest_pairs(settings.manifest_path)
    clean_pairs = _collection_pairs(settings, CLEAN_COLLECTION_NAME)
    if clean_pairs != clean_manifest:
        raise ValueError("clean collection inventory does not match clean index manifest")
    if any(
        document_id in expected_ids or filename in expected_names
        for document_id, filename in clean_pairs
    ):
        raise ValueError("clean collection contains synthetic document IDs")
    attacked_manifest = _manifest_pairs(settings.attacked_manifest_path)
    attacked_pairs = _collection_pairs(settings, ATTACKED_COLLECTION_NAME)
    expected_attacked = clean_manifest | expected
    if attacked_manifest != expected_attacked:
        raise ValueError("attacked collection inventory does not match attack manifest")
    if attacked_pairs != expected_attacked:
        raise ValueError("attacked collection inventory does not match attack manifest")
