import json
from pathlib import Path

import pymupdf
import pytest

from src.rag.config import Settings
from src.rag.index import NoCleanPdfsError, index_clean_corpus


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


class RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, float(index + 1), 0.5] for index, _ in enumerate(texts)]


def _write_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_index_rejects_missing_official_pdfs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.clean_data_dir.mkdir(parents=True)

    with pytest.raises(NoCleanPdfsError, match="official brochure PDFs"):
        index_clean_corpus(settings, embedder=RecordingEmbedder())


def test_index_persists_safe_manifest_and_reuses_unchanged_corpus(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.clean_data_dir.mkdir(parents=True)
    _write_pdf(
        settings.clean_data_dir / "rav4.pdf",
        "Toyota RAV4 luggage capacity is 580 litres.",
    )
    embedder = RecordingEmbedder()

    first = index_clean_corpus(settings, embedder=embedder)
    second = index_clean_corpus(settings, embedder=embedder)

    assert first.reused is False
    assert first.document_count == 1
    assert first.page_count == 1
    assert first.chunk_count == 1
    assert second.reused is True
    assert len(embedder.calls) == 1

    manifest_text = settings.manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["documents"] == [
        {
            "document_id": first.document_ids[0],
            "filename": "rav4.pdf",
            "page_count": 1,
        }
    ]
    assert "secret" not in manifest_text
    assert "poison" not in manifest_text.lower()
