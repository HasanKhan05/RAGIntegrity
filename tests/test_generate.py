from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag.config import ConfigurationError, Settings
from src.rag.generate import GenerationError, GeminiGenerator, build_prompt
from src.rag.models import RetrievedChunk


def _settings(tmp_path: Path, api_key: str = "test-key") -> Settings:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=gemini",
                f"LLM_API_KEY={api_key}",
                "LLM_MODEL=gemini-3.5-flash-lite",
                "MAX_OUTPUT_TOKENS=300",
                "LLM_TEMPERATURE=0",
            ]
        ),
        encoding="utf-8",
    )
    return Settings.from_env(env_file)


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            document_id="doc-a",
            filename="rav4.pdf",
            page_number=3,
            chunk_id="doc-a-p3-c0",
            text="The luggage capacity is 580 litres.",
            rank=1,
            relevance_score=0.91,
        )
    ]


class RecordingModels:
    def __init__(self, response: object | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_build_prompt_grounds_answer_and_labels_sources() -> None:
    prompt = build_prompt("What is the luggage capacity?", _chunks())

    assert "What is the luggage capacity?" in prompt
    assert "The luggage capacity is 580 litres." in prompt
    assert "[rav4.pdf, p. 3]" in prompt
    assert "insufficient" in prompt.lower()


def test_generate_maps_answer_usage_and_low_token_configuration(tmp_path: Path) -> None:
    response = SimpleNamespace(
        text="The luggage capacity is 580 litres. [rav4.pdf, p. 3]",
        usage_metadata=SimpleNamespace(
            prompt_token_count=24,
            candidates_token_count=12,
            total_token_count=36,
        ),
    )
    models = RecordingModels(response=response)
    client = SimpleNamespace(models=models)
    settings = _settings(tmp_path)

    result = GeminiGenerator(settings, client=client).generate(
        "What is the luggage capacity?", _chunks()
    )

    assert result.text == "The luggage capacity is 580 litres. [rav4.pdf, p. 3]"
    assert result.token_usage is not None
    assert result.token_usage.input_tokens == 24
    assert result.token_usage.output_tokens == 12
    assert result.token_usage.total_tokens == 36
    assert len(models.calls) == 1
    assert models.calls[0]["model"] == "gemini-3.5-flash-lite"
    config = models.calls[0]["config"]
    assert config.temperature == 0
    assert config.max_output_tokens == 300
    assert "test-key" not in str(models.calls[0]["contents"])


def test_generate_requires_configuration_before_provider_call(tmp_path: Path) -> None:
    models = RecordingModels()
    settings = _settings(tmp_path, api_key="")

    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        GeminiGenerator(settings, client=SimpleNamespace(models=models)).generate(
            "Question", _chunks()
        )

    assert models.calls == []


def test_generate_hides_provider_details_and_secret(tmp_path: Path) -> None:
    models = RecordingModels(error=RuntimeError("provider rejected test-key"))
    settings = _settings(tmp_path)

    with pytest.raises(GenerationError) as error:
        GeminiGenerator(settings, client=SimpleNamespace(models=models)).generate(
            "Question", _chunks()
        )

    assert str(error.value) == "Gemini generation failed"
    assert "test-key" not in str(error.value)
