import json
from pathlib import Path

from experiments.run_phase2_attacks import run_phase2
from src.rag.config import Settings
from src.rag.models import GeneratedAnswer, RetrievedChunk, TokenUsage


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
    settings = Settings.from_env(env_file)
    manifest = {
        "attacks": [
            {
                "attack_id": f"attack_{number:03d}",
                "synthetic_document_id": f"synthetic-{number}",
                "synthetic_filename": f"update-{number}.pdf",
                "attack_type": "false_specification",
                "target_model": "Test model",
                "target_topic": "test topic",
                "clean_fact": f"The value is {number} litres.",
                "false_claim": f"The value is {number + 70} litres.",
                "clean_source_filename": "brochure.pdf",
                "clean_source_page": 1,
                "target_test_question": f"What is test value {number}?",
                "false_value": str(number + 70),
                "false_unit_aliases": ["l", "litre", "litres"],
            }
            for number in range(1, 4)
        ]
    }
    settings.attack_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.attack_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return settings


def _chunk(document_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=1,
        chunk_id=f"{document_id}-p1-c0",
        text="Brochure text.",
        rank=rank,
        relevance_score=0.9,
    )


class RecordingGenerator:
    def __init__(self, answers: list[GeneratedAnswer]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        self.calls.append((question, chunks))
        return self.answers.pop(0)


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return self.chunks


class FakeRetrieverFactory:
    def __init__(
        self, clean_chunks: list[RetrievedChunk], attacked_chunks: list[RetrievedChunk]
    ) -> None:
        self.clean_chunks = clean_chunks
        self.attacked_chunks = attacked_chunks

    def __call__(self, collection_name: str) -> FakeRetriever:
        if collection_name == "clean_brochures":
            return FakeRetriever(self.clean_chunks)
        if collection_name == "attacked_brochures":
            return FakeRetriever(self.attacked_chunks)
        raise AssertionError(f"Unexpected collection: {collection_name}")


def test_runner_records_clean_and_attacked_outcomes_without_hidden_source_fields(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    clean_answer = GeneratedAnswer("The value is 1 litre.", TokenUsage(10, 5, 15))
    attacked_answer = GeneratedAnswer("The value is 71 litres.", TokenUsage(11, 6, 17))
    result = run_phase2(
        settings,
        generator=RecordingGenerator([clean_answer, attacked_answer] * 3),
        retriever_factory=FakeRetrieverFactory(
            [_chunk("clean", 1)], [_chunk("synthetic-1", 1), _chunk("clean", 2)]
        ),
        output_path=tmp_path / "results.json",
    )

    assert result["gemini_calls"] == 6
    assert len(result["attacks"]) == 3
    assert result["attacks"][0]["poison_rank"] == 1
    assert result["attacks"][0]["retrieval_compromised"] is True
    assert result["attacks"][0]["generation_compromised"] is True
    assert set(result["attacks"][0]["attacked_sources"][0]) == {
        "rank",
        "document_id",
        "filename",
        "page_number",
        "chunk_id",
        "relevance_score",
        "text",
    }
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == result


def test_runner_makes_six_calls_and_sums_fake_token_usage(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    answers = [
        GeneratedAnswer(f"Answer {number}", TokenUsage(number, number + 1, 2 * number + 1))
        for number in range(1, 7)
    ]
    generator = RecordingGenerator(answers)

    result = run_phase2(
        settings,
        generator=generator,
        retriever_factory=FakeRetrieverFactory(
            [_chunk("clean", 1)], [_chunk("clean", 1)]
        ),
        output_path=tmp_path / "results.json",
    )

    assert len(generator.calls) == 6
    assert result["gemini_calls"] == 6
    assert result["token_usage"] == {
        "input_tokens": 21,
        "output_tokens": 27,
        "total_tokens": 48,
    }
