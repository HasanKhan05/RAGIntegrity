from pathlib import Path

import chromadb
import pymupdf
import pytest

from src.rag.config import Settings
from src.rag.index import (
    ATTACKED_COLLECTION_NAME,
    CLEAN_COLLECTION_NAME,
    NoCleanPdfsError,
    index_attacked_corpus,
    index_clean_corpus,
)


class DeterministicEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)]


def _settings(tmp_path: Path) -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=gemini",
                "LLM_MODEL=gemini-3.5-flash-lite",
                "CHROMA_PERSIST_DIR=vector-store",
            ]
        ),
        encoding="utf-8",
    )
    return Settings.from_env(env_file)


def _write_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_attacked_index_isolated_and_contains_clean_and_synthetic_pdfs(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.clean_data_dir.mkdir(parents=True)
    settings.poisoned_data_dir.mkdir(parents=True)
    _write_pdf(settings.clean_data_dir / "corolla.pdf", "Corolla clean fact one.")
    _write_pdf(settings.clean_data_dir / "rav4.pdf", "RAV4 clean fact two.")
    _write_pdf(
        settings.poisoned_data_dir / "service-update.pdf",
        "Synthetic service update fact.",
    )
    embedder = DeterministicEmbedder()

    clean_result = index_clean_corpus(settings, embedder=embedder)
    attacked_result = index_attacked_corpus(settings, embedder=embedder)

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    clean = client.get_collection(CLEAN_COLLECTION_NAME)
    attacked = client.get_collection(ATTACKED_COLLECTION_NAME)
    stored = attacked.get(include=["metadatas"])

    assert clean_result.document_count == 2
    assert attacked_result.document_count == 3
    assert clean.count() == 2
    assert attacked.count() == 3
    assert all(
        set(metadata) == {"document_id", "filename", "page_number", "chunk_id"}
        for metadata in stored["metadatas"]
    )


def test_attacked_index_rejects_duplicate_clean_and_synthetic_filenames(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.clean_data_dir.mkdir(parents=True)
    settings.poisoned_data_dir.mkdir(parents=True)
    _write_pdf(settings.clean_data_dir / "foo.pdf", "Clean brochure fact.")
    _write_pdf(settings.poisoned_data_dir / "foo.pdf", "Synthetic brochure fact.")

    with pytest.raises(ValueError, match="unique filenames"):
        index_attacked_corpus(settings, embedder=DeterministicEmbedder())


def test_attacked_index_requires_clean_pdfs_before_synthetic_pdfs(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.poisoned_data_dir.mkdir(parents=True)
    _write_pdf(settings.poisoned_data_dir / "service-update.pdf", "Synthetic fact.")

    with pytest.raises(NoCleanPdfsError, match="official brochure PDFs"):
        index_attacked_corpus(settings, embedder=DeterministicEmbedder())
