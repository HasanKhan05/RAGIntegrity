"""Run the frozen Phase 4 evaluation behind a provider-free dry-run gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep, time
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from experiments.phase4_generation import (
    GenerationAttemptAudit,
    GenerationBudget,
    GenerationCache,
    build_dry_run_accounting,
    deduplicate_requests,
)
from experiments.phase4_reporting import build_scored_rows, write_outputs
from src.rag.config import Settings
from src.rag.defenses import (
    DefenseCoordinator,
    DefenseMode,
    DefenseResult,
    load_trusted_filenames,
)
from src.rag.embed import SentenceTransformerEmbedder
from src.rag.generate import GenerationError, GeminiGenerator
from src.rag.index import ATTACKED_COLLECTION_NAME, CLEAN_COLLECTION_NAME
from src.rag.models import RetrievedChunk
from src.rag.retrieve import Retriever


ATTACKED_MODES = (
    DefenseMode.NONE,
    DefenseMode.SOURCE_TRUST,
    DefenseMode.INSTRUCTION_FILTER,
    DefenseMode.SIMILARITY_FILTER,
    DefenseMode.COMBINED,
)
SCENARIOS = (
    ("clean", DefenseMode.NONE),
    ("attacked", DefenseMode.NONE),
    ("attacked", DefenseMode.SOURCE_TRUST),
    ("attacked", DefenseMode.INSTRUCTION_FILTER),
    ("attacked", DefenseMode.SIMILARITY_FILTER),
    ("attacked", DefenseMode.COMBINED),
)
_FROZEN_CONDITIONS = frozenset(
    {
        ("clean", "none"),
        ("attacked", "none"),
        ("attacked", "source_trust"),
        ("attacked", "instruction_filter"),
        ("attacked", "similarity_filter"),
        ("attacked", "combined"),
    }
)
HARD_MAX_NEW_CALLS = 120
HARD_MAX_ESTIMATED_INPUT_TOKENS = 150_000
APPROVED_UNIQUE_RESULT_CAP = 103
MIN_ATTEMPT_INTERVAL_SECONDS = 5.0
MAX_RATE_LIMIT_RETRIES = 5
MAX_RETRY_BACKOFF_SECONDS = 60.0

FROZEN_FILE_HASHES = {
    "data/manifests/clean_index.json": (
        "4e7a56f6b7fb75b5349f26bc4db90eb1318a7f1cca247346a6938d3131ce3774"
    ),
    "data/manifests/attacked_index.json": (
        "1545519b36425512dca705b4db35c1a18a9b1c7307c9eb2ec8ec81bd2160cf2d"
    ),
    "data/manifests/attack_manifest.json": (
        "7d2c71ff8e526be382f1ba43c51e5dc5ec6b0d998821583d6e8b76c6cf6037e5"
    ),
    "data/evaluation/attack_questions.json": (
        "60d9f51c2b7139a1bfcb69b16d544414ff7dc268c4860525f1a8337687da5118"
    ),
    "data/evaluation/clean_control_questions.json": (
        "2c4c522dfd48133795c0743155213ec5631e690ff82d8cbf860b9e2b3486cc1f"
    ),
}
FROZEN_COLLECTION_COUNTS = {
    CLEAN_COLLECTION_NAME: 307,
    ATTACKED_COLLECTION_NAME: 317,
}
_HIDDEN_LABEL_KEYS = {
    "attack_type",
    "intended_false_claim",
    "is_poison",
    "is_poisoned",
    "is_synthetic",
    "poison",
    "poison_label",
    "synthetic",
    "target_fact",
}
_RAG_VISIBLE_METADATA_KEYS = {
    "document_id",
    "filename",
    "page_number",
    "chunk_id",
}
_PRIOR_RESULT_PATHS = (
    "experiments/results/phase2_attack_results.json",
    "experiments/results/phase2_expansion_retrieval.json",
    "experiments/results/phase2_expansion_smoke.json",
    "experiments/results/phase3_defense_analysis.json",
    "experiments/results/phase3_defense_smoke.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_paths(settings: Settings) -> dict[str, Path]:
    return {
        "data/manifests/clean_index.json": settings.manifest_path,
        "data/manifests/attacked_index.json": settings.attacked_manifest_path,
        "data/manifests/attack_manifest.json": settings.attack_manifest_path,
        "data/evaluation/attack_questions.json": settings.attack_questions_path,
        "data/evaluation/clean_control_questions.json": (
            settings.clean_control_questions_path
        ),
    }


def _validate_frozen_configuration(settings: Settings) -> dict[str, object]:
    actual = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_output_tokens": settings.max_output_tokens,
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "defense_similarity_threshold": settings.defense_similarity_threshold,
        "clean_collection": CLEAN_COLLECTION_NAME,
        "attacked_collection": ATTACKED_COLLECTION_NAME,
    }
    expected = {
        "llm_provider": "gemini",
        "llm_model": "gemini-3.5-flash-lite",
        "temperature": 0.0,
        "max_output_tokens": 300,
        "top_k": 3,
        "chunk_size": 1200,
        "chunk_overlap": 200,
        "embedding_model": "all-MiniLM-L6-v2",
        "defense_similarity_threshold": 0.92,
        "clean_collection": "clean_brochures",
        "attacked_collection": "attacked_brochures",
    }
    if actual != expected:
        differences = [
            f"{key}={actual[key]!r} (expected {value!r})"
            for key, value in expected.items()
            if actual[key] != value
        ]
        raise ValueError("Phase 4 configuration is not frozen: " + ", ".join(differences))
    return actual


def load_frozen_snapshot(settings: Settings) -> dict[str, bytes]:
    """Read and hash each frozen file once for post-generation scoring."""

    snapshot: dict[str, bytes] = {}
    for label, path in _frozen_paths(settings).items():
        try:
            source = path.read_bytes()
        except OSError as error:
            raise ValueError(f"missing frozen Phase 4 input: {label}") from error
        if hashlib.sha256(source).hexdigest() != FROZEN_FILE_HASHES[label]:
            raise ValueError(f"frozen Phase 4 input hash mismatch: {label}")
        snapshot[label] = source
    return snapshot


def _validate_frozen_hashes(settings: Settings) -> dict[str, str]:
    actual: dict[str, str] = {}
    for label, path in _frozen_paths(settings).items():
        if not path.is_file():
            raise ValueError(f"missing frozen Phase 4 input: {label}")
        actual[label] = _sha256_file(path)
        if actual[label] != FROZEN_FILE_HASHES[label]:
            raise ValueError(f"frozen Phase 4 input hash mismatch: {label}")
    return actual


def _read_records(
    path: Path, label: str, expected_count: int
) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read frozen {label}: {path}") from error
    records = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"expected exactly {expected_count} {label}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{label} entries must be objects")
    return tuple(dict(record) for record in records)


def _load_benchmark(settings: Settings) -> tuple[dict[str, object], ...]:
    attack_questions = _read_records(
        settings.attack_questions_path, "attack questions", 30
    )
    controls = _read_records(
        settings.clean_control_questions_path, "clean control questions", 18
    )
    try:
        attack_payload = json.loads(
            settings.attack_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("unable to read frozen attack manifest") from error
    attacks = attack_payload.get("attacks") if isinstance(attack_payload, dict) else None
    if not isinstance(attacks, list) or len(attacks) != 10:
        raise ValueError("expected exactly 10 attack manifest entries")
    attack_ids = [
        attack.get("attack_id") for attack in attacks if isinstance(attack, dict)
    ]
    if len(attack_ids) != 10 or any(not isinstance(value, str) for value in attack_ids):
        raise ValueError("attack manifest entries require attack_id")
    if len(set(attack_ids)) != 10:
        raise ValueError("attack manifest attack_id values must be unique")

    questions: list[dict[str, object]] = []
    for cohort, records in (("attack", attack_questions), ("control", controls)):
        for record in records:
            question_id = record.get("question_id")
            question = record.get("question")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(f"{cohort} questions require non-empty question_id")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{cohort} questions require non-empty question")
            if cohort == "attack" and record.get("attack_id") not in attack_ids:
                raise ValueError("attack question refers to an unknown attack_id")
            questions.append({**record, "cohort": cohort})
    question_ids = [str(question["question_id"]) for question in questions]
    question_texts = [
        str(question["question"]).strip().casefold() for question in questions
    ]
    if len(set(question_ids)) != 48:
        raise ValueError("benchmark question_id values must be unique")
    if len(set(question_texts)) != 48:
        raise ValueError("benchmark question texts must be unique")
    return tuple(questions)


def _default_collection_inspector(
    settings: Settings, name: str
) -> dict[str, object]:
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name)
    result = collection.get(include=["metadatas"])
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadatas": result.get("metadatas") or [],
    }


def _validate_collection(
    settings: Settings,
    name: str,
    inspector: Callable[[Settings, str], Mapping[str, object]],
) -> dict[str, object]:
    snapshot = dict(inspector(settings, name))
    if snapshot.get("name") != name:
        raise ValueError(f"expected collection {name!r}")
    if snapshot.get("count") != FROZEN_COLLECTION_COUNTS[name]:
        raise ValueError(
            f"collection {name!r} must contain {FROZEN_COLLECTION_COUNTS[name]} chunks"
        )
    metadatas = snapshot.get("metadatas")
    if not isinstance(metadatas, Sequence) or isinstance(metadatas, str):
        raise ValueError(f"collection {name!r} metadata could not be inspected")
    if len(metadatas) != snapshot["count"]:
        raise ValueError(
            f"collection {name!r} inspected metadata count must equal chunk count"
        )
    for metadata in metadatas:
        if not isinstance(metadata, Mapping):
            raise ValueError(f"collection {name!r} contains invalid chunk metadata")
        metadata_keys = set(metadata)
        normalized_keys = {str(key).casefold() for key in metadata_keys}
        leaked = normalized_keys & _HIDDEN_LABEL_KEYS
        if leaked:
            raise ValueError(
                f"collection {name!r} exposes hidden poison labels: {sorted(leaked)}"
            )
        if metadata_keys != _RAG_VISIBLE_METADATA_KEYS:
            raise ValueError(
                f"collection {name!r} metadata must contain exactly the RAG-visible keys"
            )
    return {"name": name, "count": snapshot["count"]}


def _validate_retrieval(chunks: tuple[RetrievedChunk, ...], label: str) -> None:
    if len(chunks) != 3:
        raise ValueError(f"{label} retrieval must return exactly top-3 chunks")
    for chunk in chunks:
        leaked = {key.casefold() for key in vars(chunk)} & _HIDDEN_LABEL_KEYS
        if leaked:
            raise ValueError(f"{label} retrieval exposes hidden poison labels")


def _elapsed_ms(clock: Callable[[], float], started: float) -> float:
    return round((clock() - started) * 1000, 2)


class _AttemptPacer:
    """Keep all initial and retry attempts at or below the approved 12 RPM."""

    def __init__(
        self,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
        *,
        last_attempt: Mapping[str, Any] | None = None,
        wall_clock: Callable[[], float] = time,
    ) -> None:
        self._clock = clock
        self._sleeper = sleeper
        self._last_attempt_at: float | None = None
        if last_attempt is not None:
            # Pending and legacy records cannot prove when checkpoint I/O ended.
            # Give them a full interval from resume; completed new records contain
            # the actual outbound start, which is safe to translate to this clock.
            elapsed = 0.0
            if last_attempt.get("finished_at") is not None:
                started = datetime.fromisoformat(last_attempt["started_at"]).timestamp()
                elapsed = max(0.0, wall_clock() - started)
            self._last_attempt_at = clock() - elapsed

    def wait_for_attempt(self, retry_after: float | None = None) -> None:
        now = self._clock()
        spacing = (
            0.0
            if self._last_attempt_at is None
            else max(
                0.0,
                MIN_ATTEMPT_INTERVAL_SECONDS - (now - self._last_attempt_at),
            )
        )
        delay = max(spacing, retry_after or 0.0)
        if delay > 0:
            self._sleeper(delay)

    def mark_attempt_started(self) -> None:
        """Anchor spacing after the durable pre-call checkpoint completes."""

        self._last_attempt_at = self._clock()


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        for linked in (current.__context__, current.__cause__):
            if linked is not None:
                pending.append(linked)
    return tuple(chain)


def _is_rate_limit_error(error: GenerationError) -> bool:
    for item in _exception_chain(error):
        for attribute in ("code", "status_code", "status"):
            value = getattr(item, attribute, None)
            if value == 429 or str(value).upper() == "RESOURCE_EXHAUSTED":
                return True
        message = str(item).upper()
        if "RESOURCE_EXHAUSTED" in message or "429" in message.split():
            return True
    return False


def _numeric_retry_delay(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float | str):
        try:
            parsed = float(value)
        except (ValueError, OverflowError):
            return None
        if not math.isfinite(parsed) or parsed < 0:
            return None
        return min(parsed, MAX_RETRY_BACKOFF_SECONDS)
    seconds = getattr(value, "seconds", None)
    return _numeric_retry_delay(seconds)


def _retry_after_seconds(error: GenerationError) -> float | None:
    delays: list[float] = []
    retry_keys = {
        "retryafter",
        "retryafterseconds",
        "retrydelay",
        "retrydelayseconds",
    }
    for item in _exception_chain(error):
        for attribute in (
            "retry_after",
            "retry_after_seconds",
            "retry_delay",
            "retry_delay_seconds",
        ):
            delay = _numeric_retry_delay(getattr(item, attribute, None))
            if delay is not None:
                delays.append(delay)
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", None)
        if isinstance(headers, Mapping):
            for key, value in headers.items():
                if str(key).casefold() == "retry-after":
                    delay = _numeric_retry_delay(value)
                    if delay is not None:
                        delays.append(delay)
        for metadata in (
            getattr(item, "metadata", None),
            getattr(item, "details", None),
            getattr(item, "response_json", None),
        ):
            pending: list[object] = [metadata]
            seen: set[int] = set()
            while pending:
                value = pending.pop()
                if value is None or id(value) in seen:
                    continue
                seen.add(id(value))
                if isinstance(value, Mapping):
                    for key, child in value.items():
                        normalized = (
                            str(key).casefold().replace("-", "").replace("_", "")
                        )
                        if normalized in retry_keys:
                            delay = _numeric_retry_delay(child)
                            if delay is not None:
                                delays.append(delay)
                        if isinstance(child, Mapping | list | tuple):
                            pending.append(child)
                elif isinstance(value, list | tuple):
                    pending.extend(value)
    return max(delays) if delays else None


def _fallback_retry_delay(retry_number: int) -> float:
    return min(
        MIN_ATTEMPT_INTERVAL_SECONDS * (2 ** (retry_number - 1)),
        MAX_RETRY_BACKOFF_SECONDS,
    )


def _resumed_retry_delay(
    record: Mapping[str, Any] | None,
    attempts: int,
    wall_clock: Callable[[], float],
) -> float | None:
    if record is None or record["status"] != "rate_limit":
        return None
    delay = _numeric_retry_delay(record.get("retry_after_seconds"))
    if delay is None:
        delay = _fallback_retry_delay(attempts)
    # Older audits have no reliable end timestamp or saved provider retry delay.
    # Keep their cumulative ceiling, and conservatively wait a full fallback stage.
    finished_at = record.get("finished_at")
    elapsed = (max(0.0, wall_clock() - datetime.fromisoformat(finished_at).timestamp())
               if finished_at is not None else 0.0)
    return max(0.0, delay - elapsed)


def _serialize_chunks(chunks: Sequence[RetrievedChunk]) -> list[dict[str, object]]:
    return [asdict(chunk) for chunk in chunks]


def _serialize_trace(trace: Sequence[Any]) -> list[dict[str, object]]:
    return [
        {
            **asdict(entry),
            "stage_decisions": [asdict(decision) for decision in entry.stage_decisions],
        }
        for entry in trace
    ]


def _prior_result_hashes(project_root: Path) -> dict[str, str]:
    return {
        label: _sha256_file(project_root / label)
        for label in _PRIOR_RESULT_PATHS
        if (project_root / label).is_file()
    }


def prepare_phase4_cells(
    settings: Settings,
    *,
    clean_retriever: Retriever | Any | None = None,
    attacked_retriever: Retriever | Any | None = None,
    coordinator: DefenseCoordinator | Any | None = None,
    similarity_embedder: SentenceTransformerEmbedder | Any | None = None,
    collection_inspector: Callable[[Settings, str], Mapping[str, object]] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    """Validate and prepare all 288 exact local generation inputs."""

    frozen_configuration = _validate_frozen_configuration(settings)
    frozen_hashes = _validate_frozen_hashes(settings)
    questions = _load_benchmark(settings)
    inspect = collection_inspector or _default_collection_inspector
    collections = {
        name: _validate_collection(settings, name, inspect)
        for name in (CLEAN_COLLECTION_NAME, ATTACKED_COLLECTION_NAME)
    }

    shared_embedder = similarity_embedder or SentenceTransformerEmbedder(
        settings.embedding_model
    )
    active_clean_retriever = clean_retriever or Retriever(
        settings, collection_name=CLEAN_COLLECTION_NAME, embedder=shared_embedder
    )
    active_attacked_retriever = attacked_retriever or Retriever(
        settings, collection_name=ATTACKED_COLLECTION_NAME, embedder=shared_embedder
    )
    active_coordinator = coordinator or DefenseCoordinator(
        trusted_filenames=load_trusted_filenames(settings.manifest_path),
        similarity_threshold=settings.defense_similarity_threshold,
        embedder=shared_embedder,
    )

    warmup_started = clock()
    shared_embedder.encode(["Phase 4 similarity defense warm-up"])
    warmup_latency_ms = _elapsed_ms(clock, warmup_started)

    retrievals: list[dict[str, object]] = []
    for question in questions:
        question_text = str(question["question"])
        clean_started = clock()
        clean_chunks = tuple(active_clean_retriever.retrieve(question_text))
        clean_latency_ms = _elapsed_ms(clock, clean_started)
        attacked_started = clock()
        attacked_chunks = tuple(active_attacked_retriever.retrieve(question_text))
        attacked_latency_ms = _elapsed_ms(clock, attacked_started)
        _validate_retrieval(clean_chunks, "clean")
        _validate_retrieval(attacked_chunks, "attacked")
        retrievals.append(
            {
                "question": question,
                "clean_chunks": clean_chunks,
                "clean_latency_ms": clean_latency_ms,
                "attacked_chunks": attacked_chunks,
                "attacked_latency_ms": attacked_latency_ms,
            }
        )

    cells: list[dict[str, object]] = []
    for retrieval in retrievals:
        question = retrieval["question"]
        if not isinstance(question, dict):
            raise ValueError("invalid prepared benchmark question")
        question_id = str(question["question_id"])
        question_text = str(question["question"])
        benchmark_metadata = {
            key: value
            for key, value in question.items()
            if key not in {"question_id", "question", "cohort"}
        }
        clean_chunks = retrieval["clean_chunks"]
        attacked_chunks = retrieval["attacked_chunks"]
        if not isinstance(clean_chunks, tuple) or not isinstance(attacked_chunks, tuple):
            raise ValueError("retrieval snapshots must be tuples")
        cells.append(
            {
                "cell_id": f"{question_id}:clean:none",
                "question_id": question_id,
                "question": question_text,
                "cohort": question["cohort"],
                "scenario": "clean",
                "mode": DefenseMode.NONE.value,
                "attack_id": question.get("attack_id"),
                "benchmark_metadata": benchmark_metadata,
                "source_chunks": clean_chunks,
                "chunks": clean_chunks,
                "retrieval_latency_ms": retrieval["clean_latency_ms"],
                "defense_latency_ms": 0.0,
                "defense_trace": (),
                "settings": settings,
            }
        )
        for mode in ATTACKED_MODES:
            defense_started = clock()
            result = active_coordinator.apply(attacked_chunks, mode)
            defense_latency_ms = _elapsed_ms(clock, defense_started)
            if not isinstance(result, DefenseResult):
                raise ValueError("defense coordinator must return DefenseResult")
            cells.append(
                {
                    "cell_id": f"{question_id}:attacked:{mode.value}",
                    "question_id": question_id,
                    "question": question_text,
                    "cohort": question["cohort"],
                    "scenario": "attacked",
                    "mode": mode.value,
                    "attack_id": question.get("attack_id"),
                    "benchmark_metadata": benchmark_metadata,
                    "source_chunks": attacked_chunks,
                    "chunks": result.chunks,
                    "retrieval_latency_ms": retrieval["attacked_latency_ms"],
                    "defense_latency_ms": defense_latency_ms,
                    "defense_trace": result.trace,
                    "settings": settings,
                }
            )

    observed_conditions = {
        (str(cell["scenario"]), str(cell["mode"])) for cell in cells
    }
    if len(cells) != 288:
        raise ValueError("frozen Phase 4 matrix must contain exactly 288 cells")
    if observed_conditions != _FROZEN_CONDITIONS:
        raise ValueError("frozen Phase 4 matrix must contain exactly six conditions")

    requests = deduplicate_requests(cells)
    fingerprint_by_cell = {
        cell_id: request["fingerprint"]
        for request in requests
        for cell_id in request["cell_ids"]
    }
    calls_without_cache = len(requests)
    tokens_without_cache = sum(
        int(request["estimated_input_tokens"]) for request in requests
    )
    plan_cells = [
        {
            key: value
            for key, value in cell.items()
            if key not in {"settings", "source_chunks", "chunks", "defense_trace"}
        }
        | {
            "source_context": _serialize_chunks(cell["source_chunks"]),
            "final_context": _serialize_chunks(cell["chunks"]),
            "defense_trace": _serialize_trace(cell["defense_trace"]),
            "fingerprint": fingerprint_by_cell[str(cell["cell_id"])],
        }
        for cell in cells
    ]
    plan = {
        "schema_version": "phase4_evaluation_plan_v1",
        "conceptual_cells": len(cells),
        "scenario_order": [
            {"scenario": scenario, "mode": mode.value}
            for scenario, mode in SCENARIOS
        ],
        "frozen_configuration": frozen_configuration,
        "frozen_file_hashes": frozen_hashes,
        "collections": collections,
        "benchmark": {
            "question_count": len(questions),
            "attack_question_count": 30,
            "clean_control_question_count": 18,
        },
        "budget": {
            "hard_max_new_calls": HARD_MAX_NEW_CALLS,
            "approved_unique_result_cap": APPROVED_UNIQUE_RESULT_CAP,
            "hard_max_estimated_input_tokens": HARD_MAX_ESTIMATED_INPUT_TOKENS,
            "calls_without_cache": calls_without_cache,
            "estimated_input_tokens_without_cache": tokens_without_cache,
            "within_budget_without_cache": (
                calls_without_cache <= HARD_MAX_NEW_CALLS
                and calls_without_cache <= APPROVED_UNIQUE_RESULT_CAP
                and tokens_without_cache <= HARD_MAX_ESTIMATED_INPUT_TOKENS
            ),
        },
        "warmup_latency_ms": warmup_latency_ms,
        "prior_result_hashes_before": _prior_result_hashes(settings.project_root),
        "cells": plan_cells,
        "requests": [
            {key: value for key, value in request.items() if key != "chunks"}
            for request in requests
        ],
    }
    return {"cells": tuple(cells), "requests": requests, "plan": plan}


def build_dry_run(
    prepared: Mapping[str, object], cache: GenerationCache
) -> dict[str, object]:
    """Return official cache/budget accounting without constructing a generator."""

    cells = prepared.get("cells")
    requests = prepared.get("requests")
    if not isinstance(cells, Sequence) or not isinstance(requests, Sequence):
        raise ValueError("prepared Phase 4 data requires cells and requests")
    accounting = build_dry_run_accounting(cells, requests, cache)
    multiplicities = {
        str(request["fingerprint"]): len(request["cell_ids"])
        for request in requests
    }
    call_cap_passes = accounting["new_calls"] <= HARD_MAX_NEW_CALLS
    unique_result_cap_passes = (
        accounting["new_calls"] <= APPROVED_UNIQUE_RESULT_CAP
    )
    token_cap_passes = (
        accounting["estimated_new_input_tokens"]
        <= HARD_MAX_ESTIMATED_INPUT_TOKENS
    )
    plan = prepared.get("plan")
    plan_mapping = plan if isinstance(plan, Mapping) else {}
    return {
        "schema_version": "phase4_dry_run_v1",
        **accounting,
        "hard_max_new_calls": HARD_MAX_NEW_CALLS,
        "approved_unique_result_cap": APPROVED_UNIQUE_RESULT_CAP,
        "hard_max_estimated_input_tokens": HARD_MAX_ESTIMATED_INPUT_TOKENS,
        "call_cap_passes": call_cap_passes,
        "unique_result_cap_passes": unique_result_cap_passes,
        "input_token_cap_passes": token_cap_passes,
        "within_budget": (
            call_cap_passes and unique_result_cap_passes and token_cap_passes
        ),
        "gemini_calls_made": 0,
        "frozen_configuration": plan_mapping.get("frozen_configuration", {}),
        "frozen_file_hashes": plan_mapping.get("frozen_file_hashes", {}),
        "fingerprint_multiplicities": multiplicities,
    }


def write_json(path: Path, payload: object) -> None:
    """Write a Phase 4 artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_manual_reviews(
    path: Path,
) -> dict[tuple[str, str, str], dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("unable to read Phase 4 manual reviews") from error
    records = payload.get("reviews") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != "phase4_manual_reviews_v1"
        or not isinstance(records, list)
    ):
        raise ValueError("invalid Phase 4 manual review file")
    reviews: dict[tuple[str, str, str], dict[str, object]] = {}
    required_fields = {
        "cell_id",
        "fingerprint",
        "answer_sha256",
        "resolution",
        "generation_compromised",
        "notes",
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != required_fields:
            raise ValueError("manual reviews may contain only row resolutions and notes")
        cell_id = record["cell_id"]
        fingerprint = record["fingerprint"]
        answer_sha256 = record["answer_sha256"]
        resolution = record["resolution"]
        generation_compromised = record["generation_compromised"]
        notes = record["notes"]
        if (
            not isinstance(cell_id, str)
            or not isinstance(fingerprint, str)
            or not isinstance(answer_sha256, str)
            or len(answer_sha256) != 64
            or any(character not in "0123456789abcdef" for character in answer_sha256)
            or resolution not in {None, "correct", "incorrect"}
            or (
                generation_compromised is not None
                and not isinstance(generation_compromised, bool)
            )
            or not isinstance(notes, str)
        ):
            raise ValueError("invalid Phase 4 manual review entry")
        key = (cell_id, fingerprint, answer_sha256)
        if key in reviews:
            raise ValueError("duplicate Phase 4 manual review entry")
        reviews[key] = dict(record)
    return reviews


def _validate_manual_review_targets(
    existing: Mapping[tuple[str, str, str], Mapping[str, object]],
    prepared: Mapping[str, object],
    cache: GenerationCache,
    settings: Settings,
) -> None:
    requests = prepared.get("requests")
    if not isinstance(requests, Sequence):
        raise ValueError("prepared Phase 4 data requires requests")
    targets: dict[tuple[str, str], tuple[Mapping[str, object], Mapping[str, object]]] = {}
    for request_value in requests:
        if not isinstance(request_value, Mapping):
            raise ValueError("prepared Phase 4 requests must be objects")
        fingerprint = request_value.get("fingerprint")
        identity = request_value.get("identity")
        cell_ids = request_value.get("cell_ids")
        if (
            not isinstance(fingerprint, str)
            or not isinstance(identity, Mapping)
            or not isinstance(cell_ids, Sequence)
            or isinstance(cell_ids, str)
        ):
            raise ValueError("prepared Phase 4 request is incomplete")
        for cell_id in cell_ids:
            if not isinstance(cell_id, str):
                raise ValueError("prepared Phase 4 request cell_ids must be strings")
            targets[(cell_id, fingerprint)] = (request_value, identity)

    for cell_id, fingerprint, answer_sha256 in existing:
        target = targets.get((cell_id, fingerprint))
        if target is None:
            raise ValueError("manual review target is not in the current evaluation")
        _, identity = target
        entry = cache.get(fingerprint, identity)
        if entry is None or not isinstance(entry.get("answer"), str):
            raise ValueError("manual review target has no current cached answer")
        current_hash = hashlib.sha256(entry["answer"].encode("utf-8")).hexdigest()
        if current_hash != answer_sha256:
            raise ValueError("manual review answer does not match the current cache entry")

    if existing:
        reviewed_cell_ids = {cell_id for cell_id, _, _ in existing}
        cells = prepared.get("cells")
        if not isinstance(cells, Sequence):
            raise ValueError("prepared Phase 4 data requires cells")
        reviewed_cells = tuple(
            cell
            for cell in cells
            if isinstance(cell, Mapping) and cell.get("cell_id") in reviewed_cell_ids
        )
        reviewed_requests = tuple(
            request
            for request in requests
            if isinstance(request, Mapping)
            and isinstance(request.get("cell_ids"), Sequence)
            and any(cell_id in reviewed_cell_ids for cell_id in request["cell_ids"])
        )
        reviewed_rows = build_scored_rows(
            {"cells": reviewed_cells, "requests": reviewed_requests},
            cache,
            load_frozen_snapshot(settings),
        )
        ambiguous_rows = {
            (
                str(row["cell_id"]),
                str(row["fingerprint"]),
                hashlib.sha256(str(row["answer"]).encode("utf-8")).hexdigest(),
            ): row
            for row in reviewed_rows
            if row.get("deterministic_score") == "ambiguous"
        }
        if set(existing) != set(ambiguous_rows):
            raise ValueError("manual reviews may target only current ambiguous rows")
        for key, review in existing.items():
            if (
                review.get("generation_compromised") is not None
                and ambiguous_rows[key].get("generation_compromised") is not None
            ):
                raise ValueError(
                    "generation_compromised may only adjudicate a null deterministic value"
                )


def _apply_manual_reviews(
    rows: Sequence[dict[str, object]],
    existing: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    review_rows: list[dict[str, object]] = []
    matched: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("deterministic_score") != "ambiguous":
            continue
        answer_sha256 = hashlib.sha256(str(row["answer"]).encode("utf-8")).hexdigest()
        key = (str(row["cell_id"]), str(row["fingerprint"]), answer_sha256)
        matched.add(key)
        previous = existing.get(key, {})
        resolution = previous.get("resolution")
        generation_compromised = previous.get("generation_compromised")
        notes = previous.get("notes", "")
        if resolution in {"correct", "incorrect"}:
            row["score"] = resolution
            row["manual_review_required"] = False
            row["manual_review_resolution"] = resolution
        if generation_compromised is not None:
            if row.get("generation_compromised") is not None:
                raise ValueError(
                    "generation_compromised may only adjudicate a null deterministic value"
                )
            row["generation_compromised"] = generation_compromised
        row["manual_review_notes"] = notes
        review_rows.append(
            {
                "cell_id": key[0],
                "fingerprint": key[1],
                "answer_sha256": key[2],
                "resolution": resolution,
                "generation_compromised": generation_compromised,
                "notes": notes,
            }
        )
    if not set(existing).issubset(matched):
        raise ValueError(
            "manual review answer or row identity does not match current ambiguous rows"
        )
    return review_rows


def _cumulative_attempt_totals(
    requests: Sequence[Mapping[str, object]],
    audit: GenerationAttemptAudit,
    cache: GenerationCache,
) -> dict[str, int]:
    unique = {str(request["fingerprint"]): request for request in requests}
    totals = audit.totals(set(unique))
    # Older completed entries predate the separate audit. Use their saved counts
    # only when the fingerprint has no audit history, never alongside that history.
    for fingerprint, request in unique.items():
        if audit.last_status(fingerprint) is not None:
            continue
        entry = cache.get(fingerprint, request["identity"])
        if entry is not None:
            totals["provider_attempts"] += int(entry.get("provider_attempts", 1))
            totals["rate_limit_retries"] += int(entry.get("rate_limit_retries", 0))
    return totals


def execute_phase4(
    settings: Settings,
    prepared: Mapping[str, object],
    cache: GenerationCache,
    dry_run: Mapping[str, object],
    *,
    generator_factory: Callable[[Settings], Any] | None = None,
    clock: Callable[[], float] = perf_counter,
    sleeper: Callable[[float], None] = sleep,
    wall_clock: Callable[[], float] = time,
) -> dict[str, object]:
    """Resume cache misses, score every cell, and publish final artifacts."""

    _validate_frozen_configuration(settings)
    current_dry_run = build_dry_run(prepared, cache)
    if dict(dry_run) != current_dry_run or dry_run.get("within_budget") is not True:
        raise ValueError("--execute requires a passing, current official dry-run")
    results_dir = settings.project_root / "experiments" / "results"
    manual_review_path = results_dir / "phase4_manual_reviews.json"
    existing_reviews = _read_manual_reviews(manual_review_path)
    _validate_manual_review_targets(existing_reviews, prepared, cache, settings)
    requests = prepared.get("requests")
    if not isinstance(requests, Sequence):
        raise ValueError("prepared Phase 4 data requires requests")
    budget = GenerationBudget(
        max_new_calls=APPROVED_UNIQUE_RESULT_CAP,
        max_estimated_input_tokens=HARD_MAX_ESTIMATED_INPUT_TOKENS,
    )
    audit = GenerationAttemptAudit(results_dir / "phase4_generation_attempts.json",
                                   wall_clock=wall_clock)
    pacer = _AttemptPacer(clock, sleeper, last_attempt=audit.last_attempt(),
                          wall_clock=wall_clock)
    generated_fingerprints: set[str] = set()
    generator: Any | None = None
    for request_value in requests:
        if not isinstance(request_value, Mapping):
            raise ValueError("prepared Phase 4 requests must be objects")
        request = request_value
        fingerprint = request.get("fingerprint")
        identity = request.get("identity")
        estimated_tokens = request.get("estimated_input_tokens")
        question = request.get("question")
        chunks = request.get("chunks")
        if (
            not isinstance(fingerprint, str)
            or not isinstance(identity, Mapping)
            or not isinstance(estimated_tokens, int)
            or not isinstance(question, str)
            or not isinstance(chunks, Sequence)
            or isinstance(chunks, str)
        ):
            raise ValueError("prepared Phase 4 request is incomplete")
        if cache.get(fingerprint, identity) is not None:
            continue
        if identity.get("model") != settings.llm_model:
            raise ValueError("Phase 4 request model does not match frozen settings")
        request_attempts = audit.totals({fingerprint})["provider_attempts"]
        if request_attempts >= 1 + MAX_RATE_LIMIT_RETRIES:
            raise GenerationError("Phase 4 attempt limit reached; explicit audit reset required")
        retry_after = _resumed_retry_delay(audit.last_attempt(fingerprint),
                                           request_attempts, wall_clock)
        budget.require_next_call(estimated_tokens)
        if generator is None:
            factory = generator_factory or GeminiGenerator
            generator = factory(settings)
        while True:
            pacer.wait_for_attempt(retry_after)
            is_retry = audit.last_status(fingerprint) == "rate_limit"
            attempt = audit.begin(fingerprint, rate_limit_retry=is_retry)
            pacer.mark_attempt_started()
            budget.record_provider_attempt()
            request_attempts += 1
            if is_retry:
                budget.record_rate_limit_retry()
            attempt_started_epoch = wall_clock()
            attempt_started = clock()
            try:
                answer = generator.generate(question, chunks)
            except BaseException as error:
                attempt_latency_ms = _elapsed_ms(clock, attempt_started)
                attempt_finished_epoch = wall_clock()
                rate_limited = isinstance(error, GenerationError) and _is_rate_limit_error(error)
                # Store only a fixed category, never provider messages or payloads.
                error_type = (
                    "GenerationError" if isinstance(error, GenerationError)
                    else "KeyboardInterrupt" if isinstance(error, KeyboardInterrupt)
                    else "SystemExit" if isinstance(error, SystemExit)
                    else "Exception" if isinstance(error, Exception)
                    else "BaseException"
                )
                retry_after = _retry_after_seconds(error) if rate_limited else None
                if rate_limited and retry_after is None:
                    retry_after = _fallback_retry_delay(request_attempts)
                audit.finish(attempt, status="rate_limit" if rate_limited else "failure",
                             error_type=error_type, latency_ms=attempt_latency_ms,
                             started_at=attempt_started_epoch, finished_at=attempt_finished_epoch,
                             retry_after_seconds=retry_after)
                if (
                    not rate_limited
                    or request_attempts >= 1 + MAX_RATE_LIMIT_RETRIES
                ):
                    raise
                continue
            attempt_latency_ms = _elapsed_ms(clock, attempt_started)
            attempt_finished_epoch = wall_clock()
            cache.store(
                fingerprint,
                identity,
                answer,
                attempt_latency_ms,
                **audit.totals({fingerprint}),
            )
            audit.finish(attempt, status="success", error_type=None,
                         latency_ms=attempt_latency_ms,
                         started_at=attempt_started_epoch, finished_at=attempt_finished_epoch)
            break
        generated_fingerprints.add(fingerprint)

    request_fingerprints = {
        str(request["fingerprint"])
        for request in requests
        if isinstance(request, Mapping) and isinstance(request.get("fingerprint"), str)
    }
    latency_reconciliation: dict[str, object] | None = None
    if request_fingerprints and all(
        audit.last_status(fingerprint) == "success"
        for fingerprint in request_fingerprints
    ):
        updated_entries = cache.reconcile_successful_latencies(audit)
        latency_reconciliation = {
            "source": "successful_attempt_audit",
            "updated_cache_entries": updated_entries,
            "note": (
                "Cached generation latency was validated against successful-attempt "
                "audit latency_ms. Historical evidence excludes pacing and pre-call "
                "audit work but includes the brief success-cache checkpoint before "
                "the old audit finish; new timings stop when provider invocation returns."
            ),
        }

    plan = prepared.get("plan")
    plan_mapping = plan if isinstance(plan, Mapping) else {}
    before_hashes = plan_mapping.get("prior_result_hashes_before", {})
    after_hashes = _prior_result_hashes(settings.project_root)
    if before_hashes != after_hashes:
        raise RuntimeError("a prior Phase 2 or Phase 3 result changed during Phase 4")
    run = {
        "schema_version": "phase4_run_v1",
        "completed_at": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "frozen_configuration": plan_mapping.get("frozen_configuration", {}),
        "frozen_file_hashes": plan_mapping.get("frozen_file_hashes", {}),
        "prior_result_hashes_before": before_hashes,
        "prior_result_hashes_after": after_hashes,
        "dry_run": dict(dry_run),
        "generation": {
            **budget.as_dict(),
            **_cumulative_attempt_totals(requests, audit, cache),
        },
    }
    if latency_reconciliation is not None:
        run["latency_reconciliation"] = latency_reconciliation
    frozen_snapshot = load_frozen_snapshot(settings)
    rows = build_scored_rows(
        prepared,
        cache,
        frozen_snapshot,
        generated_fingerprints=generated_fingerprints,
    )
    reviews = _apply_manual_reviews(rows, existing_reviews)
    summary = write_outputs(
        results_dir,
        rows,
        run,
        report_path=settings.project_root / "reports" / "phase4_evaluation.md",
    )
    if reviews:
        write_json(
            manual_review_path,
            {"schema_version": "phase4_manual_reviews_v1", "reviews": reviews},
        )
    elif manual_review_path.exists():
        manual_review_path.unlink()
    return {"run": run, "rows": rows, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Phase 4 evaluation")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.from_env()

    prepared = prepare_phase4_cells(settings)
    results_dir = settings.project_root / "experiments" / "results"
    cache = GenerationCache(results_dir / "phase4_generation_cache.json")
    dry_run = build_dry_run(prepared, cache)
    write_json(results_dir / "phase4_evaluation_plan.json", prepared["plan"])
    write_json(results_dir / "phase4_dry_run.json", dry_run)
    if not dry_run["within_budget"]:
        return 1
    if args.execute:
        execute_phase4(settings, prepared, cache, dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
