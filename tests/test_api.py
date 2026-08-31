from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.rag.config import Settings
from src.rag.models import GeneratedAnswer, RetrievedChunk, TokenUsage


def _settings(tmp_path: Path, api_key: str = "test-key") -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=gemini",
                f"LLM_API_KEY={api_key}",
                "LLM_MODEL=gemini-3.5-flash-lite",
                "CHROMA_PERSIST_DIR=vector-store",
            ]
        ),
        encoding="utf-8",
    )
    return Settings.from_env(env_file)


class FakeRetriever:
    def __init__(self, filename: str = "rav4.pdf") -> None:
        self.filename = filename

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                document_id="doc-a",
                filename=self.filename,
                page_number=3,
                chunk_id="doc-a-p3-c0",
                text="The luggage capacity is 580 litres.",
                rank=1,
                relevance_score=0.91,
            )
        ]


class FakeGenerator:
    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            text="The luggage capacity is 580 litres. [rav4.pdf, p. 3]",
            token_usage=TokenUsage(24, 12, 36),
        )


def test_health_and_documents_are_safe_before_indexing(tmp_path: Path) -> None:
    client = TestClient(create_app(settings=_settings(tmp_path)))

    assert client.get("/health").json() == {
        "status": "ok",
        "index_available": False,
    }
    assert client.get("/documents").json() == []


def test_ask_rejects_whitespace_question(tmp_path: Path) -> None:
    client = TestClient(create_app(settings=_settings(tmp_path)))

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_reports_missing_index_without_loading_model(tmp_path: Path) -> None:
    client = TestClient(create_app(settings=_settings(tmp_path)))

    response = client.post("/ask", json={"question": "RAV4 capacity?"})

    assert response.status_code == 503
    assert "index" in response.json()["detail"].lower()


def test_ask_returns_answer_ranked_sources_latency_and_usage(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )
    response = TestClient(app).post(
        "/ask", json={"question": "What is the luggage capacity?"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == (
        "The luggage capacity is 580 litres. [rav4.pdf, p. 3]"
    )
    assert payload["sources"] == [
        {
            "rank": 1,
            "document_id": "doc-a",
            "filename": "rav4.pdf",
            "page_number": 3,
            "chunk_id": "doc-a-p3-c0",
            "relevance_score": 0.91,
            "text": "The luggage capacity is 580 litres.",
        }
    ]
    assert payload["latency_ms"] >= 0
    assert payload["token_usage"] == {
        "input_tokens": 24,
        "output_tokens": 12,
        "total_tokens": 36,
    }


def test_ask_maps_missing_generation_configuration_to_503(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path, api_key=""),
        retriever=FakeRetriever(),
    )

    response = TestClient(app).post("/ask", json={"question": "RAV4 capacity?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Gemini generation is not configured"


def test_ask_defaults_to_clean_and_can_select_attacked(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        retriever=FakeRetriever(filename="clean.pdf"),
        attacked_retriever=FakeRetriever(filename="update.pdf"),
        generator=FakeGenerator(),
    )
    client = TestClient(app)

    clean_response = client.post("/ask", json={"question": "capacity"})
    attacked_response = client.post(
        "/ask", json={"question": "capacity", "corpus_mode": "attacked"}
    )

    assert clean_response.json()["sources"][0]["filename"] == "clean.pdf"
    assert attacked_response.json()["sources"][0]["filename"] == "update.pdf"
