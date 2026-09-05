"""Exact, local-only generation request caching for the Phase 4 benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import time
from typing import Any

from src.rag.config import Settings
from src.rag.generate import build_prompt
from src.rag.models import GeneratedAnswer, RetrievedChunk


IDENTITY_SCHEMA_VERSION = "phase4_generation_v1"
PROMPT_TEMPLATE_VERSION = "build_prompt_v1"
CACHE_SCHEMA_VERSION = "phase4_generation_cache_v1"


class BudgetExceeded(RuntimeError):
    """Raised before a proposed provider call would exceed a hard Phase 4 cap."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_request_identity(
    question: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings,
) -> dict[str, object]:
    """Return the complete canonical payload for one exact Gemini input."""

    normalized_question = question.strip()
    ordered_chunks = [
        {
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "text_sha256": _sha256_text(chunk.text),
        }
        for chunk in chunks
    ]
    prompt = build_prompt(normalized_question, chunks)
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "question": normalized_question,
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_output_tokens": settings.max_output_tokens,
        "prompt_sha256": _sha256_text(prompt),
        "chunks": ordered_chunks,
    }


def fingerprint_request(identity: Mapping[str, object]) -> str:
    """Hash a canonical request payload without any scenario-only labels."""

    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_text(payload)


def estimate_input_tokens(prompt: str) -> int:
    """Use the approved conservative local estimate without provider tokenization."""

    return max(1, (len(prompt) + 3) // 4)


def _atomic_write_json(path: Path, payload: object) -> None:
    """Persist a cache update atomically so completed calls survive interruption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class GenerationCache:
    """A validated content-addressed cache of successful generation responses."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    @classmethod
    def empty(cls) -> GenerationCache:
        """Create an in-memory-empty cache for dry-run calculations."""

        return cls(None)

    def _entries(self) -> dict[str, object]:
        if self.path is None or not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return {}
        entries = payload.get("entries")
        return entries if isinstance(entries, dict) else {}

    def get(
        self, fingerprint: str, identity: Mapping[str, object]
    ) -> dict[str, object] | None:
        """Return a response only when its fingerprint and full identity both match."""

        entry = self._entries().get(fingerprint)
        if not isinstance(entry, dict):
            return None
        if entry.get("fingerprint") != fingerprint or entry.get("identity") != dict(identity):
            return None
        return entry

    def store(
        self,
        fingerprint: str,
        identity: Mapping[str, object],
        answer: GeneratedAnswer,
        latency_ms: float = 0.0,
        *,
        created_at: str | None = None,
        provider_attempts: int = 1,
        rate_limit_retries: int = 0,
    ) -> dict[str, object]:
        """Atomically store one completed answer before another call can begin."""

        if self.path is None:
            raise ValueError("an in-memory empty cache cannot store generation results")
        if fingerprint != fingerprint_request(identity):
            raise ValueError("cache fingerprint must match the canonical identity")
        entries = self._entries()
        entry: dict[str, object] = {
            "fingerprint": fingerprint,
            "identity": dict(identity),
            "answer": answer.text,
            "model": identity.get("model"),
            "token_usage": asdict(answer.token_usage) if answer.token_usage else None,
            "latency_ms": latency_ms,
            "provider_attempts": provider_attempts,
            "rate_limit_retries": rate_limit_retries,
            "created_at": created_at
            or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        entries[fingerprint] = entry
        _atomic_write_json(
            self.path,
            {"schema_version": CACHE_SCHEMA_VERSION, "entries": entries},
        )
        return entry


    def reconcile_successful_latencies(self, audit: GenerationAttemptAudit) -> int:
        """Explicitly repair cached timing from durable successful attempts only.

        This offline operation never regenerates an answer or changes the audit.
        Historical audit durations exclude pacing but include the success cache
        checkpoint that preceded the old audit finish call. Validate the entire
        cache before writing, and leave an already reconciled file byte-identical.
        """

        entries = self._entries()
        if self.path is None or not entries:
            raise ValueError("latency reconciliation requires a completed cache")
        updated = 0
        for fingerprint, entry in entries.items():
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("identity"), dict)
                or entry.get("fingerprint") != fingerprint
                or fingerprint_request(entry["identity"]) != fingerprint
            ):
                raise ValueError("cache identity cannot be matched to a success audit")
            successful = [attempt for attempt in audit.attempts
                          if attempt["fingerprint"] == fingerprint and attempt["status"] == "success"]
            if len(successful) != 1:
                raise ValueError("each cached fingerprint requires one success audit record")
            attempt = successful[0]
            latency = attempt.get("latency_ms")
            try:
                started = datetime.fromisoformat(attempt["started_at"])
                finished = datetime.fromisoformat(attempt["finished_at"])
                elapsed_ms = (finished - started).total_seconds() * 1000
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid successful attempt timing evidence") from error
            if (
                not isinstance(latency, int | float) or isinstance(latency, bool)
                or not math.isfinite(latency) or latency < 0
                or started.tzinfo is None or finished.tzinfo is None
                or elapsed_ms < 0 or abs(elapsed_ms - latency) > 1.0
            ):
                raise ValueError("invalid successful attempt timing evidence")
            if entry.get("latency_ms") != latency:
                entry["latency_ms"] = latency
                updated += 1
        if updated:
            _atomic_write_json(self.path, {"schema_version": CACHE_SCHEMA_VERSION, "entries": entries})
        return updated


class GenerationAttemptAudit:
    """Durable attempt outcomes, independent of the success-only answer cache.

    A pending record is conservative evidence of a possibly sent attempt when a
    process dies before recording its outcome. Never infer an answer from it.
    """

    def __init__(self, path: Path, *, wall_clock: Callable[[], float] = time) -> None:
        self.path = path
        self._wall_clock = wall_clock
        self.attempts: list[dict[str, Any]] = []
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != "phase4_generation_attempts_v1":
                raise ValueError("invalid Phase 4 attempt audit")
            records = payload.get("attempts")
            if not isinstance(records, list) or any(
                not isinstance(record, dict)
                or not isinstance(record.get("fingerprint"), str)
                or record.get("status") not in {"pending", "success", "rate_limit", "failure"}
                or not isinstance(record.get("rate_limit_retry"), bool)
                or record.get("attempt") != index
                for index, record in enumerate(records, 1)
            ):
                raise ValueError("invalid Phase 4 attempt audit records")
            self.attempts = records

    def _save(self) -> None:
        _atomic_write_json(self.path, {
            "schema_version": "phase4_generation_attempts_v1",
            "attempts": self.attempts,
        })

    def last_status(self, fingerprint: str) -> str | None:
        record = self.last_attempt(fingerprint)
        return record["status"] if record is not None else None

    def last_attempt(self, fingerprint: str | None = None) -> dict[str, Any] | None:
        return next((record for record in reversed(self.attempts)
                     if fingerprint is None or record["fingerprint"] == fingerprint), None)

    def begin(self, fingerprint: str, *, rate_limit_retry: bool) -> int:
        attempt = len(self.attempts) + 1
        self.attempts.append({
            "fingerprint": fingerprint,
            "attempt": attempt,
            "status": "pending",
            "error_type": None,
            "rate_limit_retry": rate_limit_retry,
            "started_at": datetime.fromtimestamp(self._wall_clock(), UTC).isoformat(),
            "finished_at": None,
            "retry_after_seconds": None,
            "latency_ms": None,
        })
        self._save()
        return attempt

    def finish(self, attempt: int, *, status: str, error_type: str | None,
               latency_ms: float, started_at: float,
               finished_at: float | None = None,
               retry_after_seconds: float | None = None) -> None:
        self.attempts[attempt - 1].update(
            status=status, error_type=error_type, latency_ms=latency_ms,
            # Replace the pre-checkpoint timestamp with the actual outbound start.
            started_at=datetime.fromtimestamp(started_at, UTC).isoformat(),
            finished_at=datetime.fromtimestamp(
                self._wall_clock() if finished_at is None else finished_at, UTC
            ).isoformat(),
            retry_after_seconds=retry_after_seconds,
        )
        self._save()

    def totals(self, fingerprints: set[str]) -> dict[str, int]:
        records = [record for record in self.attempts
                   if record["fingerprint"] in fingerprints]
        return {
            "provider_attempts": len(records),
            "rate_limit_retries": sum(record["rate_limit_retry"] for record in records),
        }


class GenerationBudget:
    """Hard per-run limits applied immediately before each provider invocation."""

    def __init__(self, *, max_new_calls: int, max_estimated_input_tokens: int) -> None:
        self.max_new_calls = max_new_calls
        self.max_estimated_input_tokens = max_estimated_input_tokens
        self.new_calls = 0
        self.estimated_input_tokens = 0
        self.provider_attempts = 0
        self.rate_limit_retries = 0

    def require_next_call(self, estimated_input_tokens: int) -> None:
        """Reserve one call only when both caps still hold after it."""

        if estimated_input_tokens < 0:
            raise ValueError("estimated input tokens must not be negative")
        if self.new_calls + 1 > self.max_new_calls:
            raise BudgetExceeded("Phase 4 maximum new generation calls would be exceeded")
        if self.estimated_input_tokens + estimated_input_tokens > self.max_estimated_input_tokens:
            raise BudgetExceeded("Phase 4 estimated input token budget would be exceeded")
        self.new_calls += 1
        self.estimated_input_tokens += estimated_input_tokens

    def record_planned_call(self, *, estimated_input_tokens: int) -> None:
        """Compatibility name for reserving a planned provider call."""

        self.require_next_call(estimated_input_tokens)

    def record_provider_attempt(self) -> None:
        """Count every outbound attempt, including an allowed 429 retry."""

        self.provider_attempts += 1

    def record_rate_limit_retry(self) -> None:
        """Count a retry of the same fingerprint after a provider 429."""

        self.rate_limit_retries += 1

    def as_dict(self) -> dict[str, int]:
        """Expose counters for dry-run and execution reporting."""

        return {
            "new_calls": self.new_calls,
            "unique_new_cache_entries": self.new_calls,
            "estimated_new_input_tokens": self.estimated_input_tokens,
            "provider_attempts": self.provider_attempts,
            "rate_limit_retries": self.rate_limit_retries,
        }


def _cell_chunks(cell: Mapping[str, object]) -> Sequence[RetrievedChunk]:
    chunks = cell.get("chunks", cell.get("final_chunks"))
    if not isinstance(chunks, Sequence) or isinstance(chunks, str):
        raise ValueError("Phase 4 generation cell requires ordered chunks")
    return chunks  # type: ignore[return-value]


def build_dry_run_accounting(
    cells: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    cache: GenerationCache,
) -> dict[str, int]:
    """Count exact cache reuse and provider work without creating a generator."""

    cache_hits = 0
    estimated_new_input_tokens = 0
    seen: set[str] = set()
    for request in requests:
        fingerprint = request.get("fingerprint")
        identity = request.get("identity")
        estimated_tokens = request.get("estimated_input_tokens")
        if not isinstance(fingerprint, str) or not isinstance(identity, Mapping):
            raise ValueError("Phase 4 request requires fingerprint and identity")
        if not isinstance(estimated_tokens, int) or estimated_tokens < 0:
            raise ValueError("Phase 4 request requires estimated input tokens")
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        if cache.get(fingerprint, identity) is not None:
            cache_hits += 1
        else:
            estimated_new_input_tokens += estimated_tokens
    return {
        "conceptual_cells": len(cells),
        "unique_fingerprints": len(seen),
        "cache_hits": cache_hits,
        "new_calls": len(seen) - cache_hits,
        "estimated_new_input_tokens": estimated_new_input_tokens,
    }


def deduplicate_requests(
    cells: Sequence[Mapping[str, object]], *, settings: Settings | None = None
) -> tuple[dict[str, object], ...]:
    """Collapse cells only after their retained generation inputs are validated."""

    requests: dict[str, dict[str, object]] = {}
    for cell in cells:
        question = cell.get("question")
        cell_settings = cell.get("settings", settings)
        if not isinstance(question, str) or cell_settings is None:
            raise ValueError("Phase 4 generation cell requires question and settings")
        chunks = _cell_chunks(cell)
        identity = build_request_identity(question, chunks, cell_settings)  # type: ignore[arg-type]
        supplied_identity = cell.get("identity")
        if isinstance(supplied_identity, Mapping) and dict(supplied_identity) != identity:
            raise ValueError("Phase 4 generation cell identity does not match retained inputs")
        fingerprint = fingerprint_request(identity)
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str):
            raise ValueError("Phase 4 generation cell requires a string cell_id")
        request = requests.get(fingerprint)
        if request is None:
            request = {
                "fingerprint": fingerprint,
                "identity": identity,
                "question": question.strip(),
                "chunks": tuple(chunks),
                "estimated_input_tokens": estimate_input_tokens(
                    build_prompt(question.strip(), chunks)
                ),
                "cell_ids": (),
            }
            requests[fingerprint] = request
        request["cell_ids"] = (*request["cell_ids"], cell_id)  # type: ignore[operator]
    return tuple(requests.values())
