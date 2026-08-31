from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from src.rag.config import Settings
from src.rag.index import COLLECTION_NAME
from src.rag.retrieve import IndexUnavailableError, Retriever


def _settings(tmp_path: Path) -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=gemini",
                "LLM_MODEL=gemini-3.5-flash-lite",
                "CHROMA_PERSIST_DIR=vector-store",
                "TOP_K=2",
            ]
        ),
        encoding="utf-8",
    )
    return Settings.from_env(env_file)


class QueryEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def _create_collection(settings: Settings) -> None:
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=["doc-a-p1-c0", "doc-b-p2-c0"],
        documents=["RAV4 luggage capacity", "Yaris wheelbase"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadatas=[
            {
                "document_id": "doc-a",
                "filename": "rav4.pdf",
                "page_number": 1,
                "chunk_id": "doc-a-p1-c0",
            },
            {
                "document_id": "doc-b",
                "filename": "yaris.pdf",
                "page_number": 2,
                "chunk_id": "doc-b-p2-c0",
            },
        ],
    )


def test_retrieve_returns_ranked_source_metadata(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _create_collection(settings)
    retriever = Retriever(settings, embedder=QueryEmbedder())

    results = retriever.retrieve("What is the luggage capacity?")

    assert [result.rank for result in results] == [1, 2]
    assert results[0].document_id == "doc-a"
    assert results[0].filename == "rav4.pdf"
    assert results[0].page_number == 1
    assert results[0].chunk_id == "doc-a-p1-c0"
    assert results[0].text == "RAV4 luggage capacity"
    assert results[0].relevance_score == pytest.approx(1.0)
    assert results[1].relevance_score == pytest.approx(0.0)


def test_retrieve_rejects_blank_question(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(ValueError, match="question"):
        Retriever(settings, embedder=QueryEmbedder()).retrieve("   ")


def test_retrieve_reports_missing_index(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(IndexUnavailableError, match="index"):
        Retriever(settings, embedder=QueryEmbedder()).retrieve("RAV4 capacity")
