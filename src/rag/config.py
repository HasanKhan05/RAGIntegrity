"""Small, explicit environment configuration for the clean RAG baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """Raised when local configuration is missing or unsafe."""


def _read_int(values: Mapping[str, str | None], name: str, default: int) -> int:
    raw = os.environ.get(name, values.get(name))
    try:
        return default if raw in (None, "") else int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error


def _read_float(
    values: Mapping[str, str | None], name: str, default: float
) -> float:
    raw = os.environ.get(name, values.get(name))
    try:
        return default if raw in (None, "") else float(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error


@dataclass(frozen=True)
class Settings:
    """Configuration values used by Phase 1 components."""

    project_root: Path
    llm_provider: str
    llm_model: str
    llm_api_key: str | None = field(repr=False)
    llm_base_url: str | None
    max_output_tokens: int
    llm_temperature: float
    top_k: int
    chunk_size: int
    chunk_overlap: int
    chroma_persist_dir: Path
    embedding_model: str
    clean_data_dir: Path
    manifest_path: Path
    poisoned_data_dir: Path
    attacked_manifest_path: Path
    attack_manifest_path: Path

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        resolved_env = (env_file or (Path.cwd() / ".env")).resolve()
        project_root = resolved_env.parent
        values = dotenv_values(resolved_env)

        def read(name: str, default: str = "") -> str:
            value = os.environ.get(name, values.get(name, default))
            return "" if value is None else value.strip()

        provider = read("LLM_PROVIDER", "gemini").lower()
        if provider != "gemini":
            raise ConfigurationError("LLM_PROVIDER must be gemini in Phase 1")

        max_output_tokens = _read_int(values, "MAX_OUTPUT_TOKENS", 300)
        top_k = _read_int(values, "TOP_K", 3)
        chunk_size = _read_int(values, "CHUNK_SIZE", 1200)
        chunk_overlap = _read_int(values, "CHUNK_OVERLAP", 200)
        temperature = _read_float(values, "LLM_TEMPERATURE", 0.0)

        if max_output_tokens <= 0:
            raise ConfigurationError("MAX_OUTPUT_TOKENS must be positive")
        if top_k <= 0:
            raise ConfigurationError("TOP_K must be positive")
        if chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP must be between 0 and CHUNK_SIZE")
        if not 0 <= temperature <= 2:
            raise ConfigurationError("LLM_TEMPERATURE must be between 0 and 2")

        persist_value = Path(read("CHROMA_PERSIST_DIR", "data/vector_store"))
        persist_dir = persist_value if persist_value.is_absolute() else project_root / persist_value

        key = read("LLM_API_KEY") or None
        if key in {"PASTE_YOUR_KEY_HERE", "YOUR_API_KEY"}:
            key = None

        return cls(
            project_root=project_root,
            llm_provider=provider,
            llm_model=read("LLM_MODEL", "gemini-3.5-flash-lite"),
            llm_api_key=key,
            llm_base_url=read("LLM_BASE_URL") or None,
            max_output_tokens=max_output_tokens,
            llm_temperature=temperature,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chroma_persist_dir=persist_dir.resolve(),
            embedding_model=read("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            clean_data_dir=(project_root / "data" / "clean").resolve(),
            manifest_path=(project_root / "data" / "manifests" / "clean_index.json").resolve(),
            poisoned_data_dir=(project_root / "data" / "poisoned").resolve(),
            attacked_manifest_path=(
                project_root / "data" / "manifests" / "attacked_index.json"
            ).resolve(),
            attack_manifest_path=(
                project_root / "data" / "manifests" / "attack_manifest.json"
            ).resolve(),
        )

    def require_generation(self) -> None:
        if not self.llm_api_key:
            raise ConfigurationError("LLM_API_KEY is required for Gemini generation")
        if not self.llm_model:
            raise ConfigurationError("LLM_MODEL is required for Gemini generation")
