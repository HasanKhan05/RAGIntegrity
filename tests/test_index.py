from dataclasses import replace
from pathlib import Path

from src.rag.config import Settings
from src.rag.index import build_corpus_fingerprint


def _settings(tmp_path: Path) -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=gemini",
                "LLM_MODEL=gemini-3.5-flash-lite",
                "EMBEDDING_MODEL=all-MiniLM-L6-v2",
                "CHROMA_PERSIST_DIR=vector-store",
                "CHUNK_SIZE=1200",
                "CHUNK_OVERLAP=200",
            ]
        ),
        encoding="utf-8",
    )
    return Settings.from_env(env_file)


def test_corpus_fingerprint_is_stable_and_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(b"first brochure")
    second.write_bytes(b"second brochure")
    settings = _settings(tmp_path)

    forward = build_corpus_fingerprint([first, second], settings)
    reverse = build_corpus_fingerprint([second, first], settings)

    assert forward == reverse
    assert [source["filename"] for source in forward["sources"]] == [
        "a.pdf",
        "b.pdf",
    ]


def test_corpus_fingerprint_changes_with_content_or_index_settings(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "brochure.pdf"
    pdf_path.write_bytes(b"version one")
    settings = _settings(tmp_path)
    original = build_corpus_fingerprint([pdf_path], settings)

    pdf_path.write_bytes(b"version two")
    changed_content = build_corpus_fingerprint([pdf_path], settings)
    changed_model = build_corpus_fingerprint(
        [pdf_path], replace(settings, embedding_model="different-model")
    )
    changed_chunking = build_corpus_fingerprint(
        [pdf_path], replace(settings, chunk_size=1300)
    )

    assert changed_content != original
    assert changed_model != original
    assert changed_chunking != original
