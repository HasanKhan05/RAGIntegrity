from pathlib import Path

import pytest

from src.rag.config import ConfigurationError, Settings


def _write_env(project_root: Path) -> Path:
    env_file = project_root / ".env"
    env_file.write_text(
        "LLM_PROVIDER=gemini\nLLM_MODEL=gemini-3.5-flash-lite\n",
        encoding="utf-8",
    )
    return env_file


def test_phase2_paths_are_rooted_beside_clean_data(tmp_path: Path) -> None:
    settings = Settings.from_env(_write_env(tmp_path))

    assert settings.poisoned_data_dir == (tmp_path / "data" / "poisoned").resolve()
    assert settings.attacked_manifest_path == (
        tmp_path / "data" / "manifests" / "attacked_index.json"
    ).resolve()
    assert settings.attack_manifest_path == (
        tmp_path / "data" / "manifests" / "attack_manifest.json"
    ).resolve()


def test_settings_load_safe_defaults(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=gemini\nLLM_MODEL=gemini-3.5-flash-lite\n",
        encoding="utf-8",
    )

    settings = Settings.from_env(env_file)

    assert settings.top_k == 3
    assert settings.max_output_tokens == 300
    assert settings.llm_temperature == 0
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200


def test_generation_requires_api_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=gemini\nLLM_MODEL=gemini-3.5-flash-lite\n",
        encoding="utf-8",
    )
    settings = Settings.from_env(env_file)

    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        settings.require_generation()


def test_only_gemini_provider_is_supported(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=other\nLLM_API_KEY=secret\nLLM_MODEL=model\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="gemini"):
        Settings.from_env(env_file)
