from pathlib import Path

import pymupdf

from src.rag.config import Settings
from src.rag.index import index_clean_corpus


class RecordingEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_index_embeds_filename_with_chunk_text_for_model_specific_queries(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=gemini\nCHROMA_PERSIST_DIR=vector-store\n",
        encoding="utf-8",
    )
    settings = Settings.from_env(env_file)
    settings.clean_data_dir.mkdir(parents=True)
    pdf_path = settings.clean_data_dir / "rav4.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Fuel tank capacity is 55 litres.")
    document.save(pdf_path)
    document.close()
    embedder = RecordingEmbedder()

    index_clean_corpus(settings, embedder=embedder)

    assert embedder.texts == ["rav4.pdf\nFuel tank capacity is 55 litres."]
