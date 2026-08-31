import os
import sys
from types import ModuleType

from src.rag.embed import SentenceTransformerEmbedder


class FakeVectors:
    def tolist(self) -> list[list[float]]:
        return [[1.0, 0.0]]


class FakeModel:
    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> FakeVectors:
        assert texts == ["question"]
        assert normalize_embeddings is True
        assert show_progress_bar is False
        return FakeVectors()


def test_embedder_disables_progress_warnings_and_telemetry_before_model_import(
    monkeypatch,
) -> None:
    for name in (
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        monkeypatch.delenv(name, raising=False)
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = lambda model_name: FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    vectors = SentenceTransformerEmbedder("all-MiniLM-L6-v2").encode(["question"])

    assert vectors == [[1.0, 0.0]]
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
