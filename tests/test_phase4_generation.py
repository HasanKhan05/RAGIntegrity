from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.phase4_generation import (
    BudgetExceeded,
    GenerationBudget,
    GenerationCache,
    GenerationAttemptAudit,
    build_dry_run_accounting,
    build_request_identity,
    deduplicate_requests,
    estimate_input_tokens,
    fingerprint_request,
)
from src.rag.models import GeneratedAnswer, RetrievedChunk, TokenUsage


def make_chunk(document_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        page_number=rank,
        chunk_id=f"{document_id}-chunk-{rank}",
        text=text,
        rank=rank,
        relevance_score=0.9,
    )


def make_settings(model: str = "m1") -> SimpleNamespace:
    return SimpleNamespace(
        llm_model=model,
        llm_temperature=0.0,
        max_output_tokens=300,
    )


def build_identity(
    *, model: str = "m1", chunks: tuple[RetrievedChunk, ...]
) -> dict[str, object]:
    return build_request_identity(
        "  What is the official range?  ", chunks, make_settings(model)
    )


def make_cell(mode: str) -> dict[str, object]:
    chunks = (make_chunk("bz4x", "The official range is 514 km.", 1),)
    return {
        "cell_id": f"attack_002:{mode}",
        "scenario": "attacked",
        "mode": mode,
        "question": "What is the official range?",
        "chunks": chunks,
        "settings": make_settings(),
    }


def test_fingerprint_changes_when_ordered_context_or_model_changes() -> None:
    chunk_a = make_chunk("a", "first brochure passage", 1)
    chunk_b = make_chunk("b", "second brochure passage", 2)

    first = fingerprint_request(build_identity(model="m1", chunks=(chunk_a, chunk_b)))

    assert first != fingerprint_request(build_identity(model="m1", chunks=(chunk_b, chunk_a)))
    assert first != fingerprint_request(build_identity(model="m2", chunks=(chunk_a, chunk_b)))


def test_identity_uses_exact_prompt_inputs_but_not_conceptual_mode_labels() -> None:
    chunk = make_chunk("bz4x", "The official range is 514 km.", 1)
    identity = build_request_identity(
        "  What is the official range?  ", (chunk,), make_settings()
    )

    assert identity["question"] == "What is the official range?"
    assert identity["chunks"] == [
        {
            "document_id": "bz4x",
            "filename": "bz4x.pdf",
            "page_number": 1,
            "chunk_id": "bz4x-chunk-1",
            "text_sha256": "338a153ba27ea519c841814a4b03727059a8b56be1e60564b8e8bb1f03f1c132",
        }
    ]
    assert set(identity).isdisjoint({"scenario", "mode"})


def test_identical_generation_inputs_deduplicate_across_modes() -> None:
    requests = deduplicate_requests((make_cell("source_trust"), make_cell("combined")))

    assert len(requests) == 1
    assert requests[0]["cell_ids"] == ("attack_002:source_trust", "attack_002:combined")


def test_token_estimate_rounds_up_and_never_returns_zero() -> None:
    assert estimate_input_tokens("") == 1
    assert estimate_input_tokens("abcd") == 1
    assert estimate_input_tokens("abcde") == 2


def test_cache_persists_each_success_and_resumes_only_missing(tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    cache = GenerationCache(cache_path)
    identity = build_identity(chunks=(make_chunk("a", "first brochure passage", 1),))
    answer = GeneratedAnswer(
        text="ok", token_usage=TokenUsage(input_tokens=5, output_tokens=2, total_tokens=7)
    )

    fingerprint = fingerprint_request(identity)
    cache.store(fingerprint, identity, answer, latency_ms=12.5, created_at="2026-09-04T00:00:00Z")

    stored = GenerationCache(cache_path).get(fingerprint, identity)
    assert stored is not None
    assert stored["answer"] == "ok"
    assert stored["model"] == "m1"
    assert stored["token_usage"] == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
    assert stored["latency_ms"] == 12.5
    assert stored["created_at"] == "2026-09-04T00:00:00Z"
    assert not cache_path.with_suffix(".json.tmp").exists()
    assert GenerationCache(cache_path).get("missing", identity) is None


def _cached_success_with_audit(tmp_path):
    cache = GenerationCache(tmp_path / "cache.json")
    current_time = [100.0]
    audit = GenerationAttemptAudit(tmp_path / "attempts.json", wall_clock=lambda: current_time[0])
    for index in range(2):
        identity = build_identity(chunks=(make_chunk(str(index), "brochure passage", 1),))
        fingerprint = fingerprint_request(identity)
        cache.store(fingerprint, identity, GeneratedAnswer(
            f"saved answer {index}", TokenUsage(input_tokens=5, output_tokens=2, total_tokens=7)
        ), latency_ms=5000.0, created_at="2026-09-04T00:00:00Z")
        attempt = audit.begin(fingerprint, rate_limit_retry=False)
        started = current_time[0]
        current_time[0] += 0.25
        audit.finish(attempt, status="success", error_type=None,
                     latency_ms=250.0, started_at=started)
    return cache, audit


def test_cache_latency_reconciliation_preserves_payloads_and_is_idempotent(tmp_path) -> None:
    cache, audit = _cached_success_with_audit(tmp_path)
    before = json.loads(cache.path.read_text())
    audit_bytes = audit.path.read_bytes()

    assert cache.reconcile_successful_latencies(audit) == 2

    after = json.loads(cache.path.read_text())
    assert len(after["entries"]) == 2
    for fingerprint, entry in after["entries"].items():
        assert entry == {**before["entries"][fingerprint], "latency_ms": 250.0}
    reconciled_bytes = cache.path.read_bytes()
    assert cache.reconcile_successful_latencies(audit) == 0
    assert cache.path.read_bytes() == reconciled_bytes
    assert audit.path.read_bytes() == audit_bytes


@pytest.mark.parametrize("bad_evidence", ("missing", "duplicate", "pending", "negative", "inconsistent_time"))
def test_cache_latency_reconciliation_requires_success_evidence_before_any_write(tmp_path, bad_evidence) -> None:
    cache, audit = _cached_success_with_audit(tmp_path)
    if bad_evidence == "missing":
        audit.attempts.pop()
    elif bad_evidence == "duplicate":
        audit.attempts.append(dict(audit.attempts[-1]))
    elif bad_evidence == "pending":
        audit.attempts[-1]["status"] = "pending"
    elif bad_evidence == "negative":
        audit.attempts[-1]["latency_ms"] = -1.0
    else:
        audit.attempts[-1]["finished_at"] = audit.attempts[-1]["started_at"]
    original = cache.path.read_bytes()
    with pytest.raises(ValueError, match="successful.*timing|success.*audit"):
        cache.reconcile_successful_latencies(audit)
    assert cache.path.read_bytes() == original


def test_cache_rejects_entry_with_different_canonical_identity(tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    first = build_identity(chunks=(make_chunk("a", "first brochure passage", 1),))
    changed = build_identity(chunks=(make_chunk("a", "changed brochure passage", 1),))
    fingerprint = fingerprint_request(first)
    GenerationCache(cache_path).store(fingerprint, first, GeneratedAnswer("ok", None))

    assert GenerationCache(cache_path).get(fingerprint, changed) is None


def test_cache_ignores_malformed_or_nonmatching_fingerprint_entries(tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    identity = build_identity(chunks=(make_chunk("a", "first brochure passage", 1),))
    cache_path.write_text(
        json.dumps({"entries": {"abc": {"fingerprint": "wrong", "identity": identity}}}),
        encoding="utf-8",
    )

    assert GenerationCache(cache_path).get("abc", identity) is None


def test_budget_rejects_next_call_before_either_hard_limit_is_exceeded() -> None:
    budget = GenerationBudget(max_new_calls=1, max_estimated_input_tokens=10)
    budget.record_planned_call(estimated_input_tokens=6)

    with pytest.raises(BudgetExceeded):
        budget.record_planned_call(estimated_input_tokens=5)

    assert budget.new_calls == 1
    assert budget.estimated_input_tokens == 6


def test_budget_rejects_call_limit_before_recording_another_call() -> None:
    budget = GenerationBudget(max_new_calls=1, max_estimated_input_tokens=20)
    budget.record_planned_call(estimated_input_tokens=6)

    with pytest.raises(BudgetExceeded):
        budget.require_next_call(estimated_input_tokens=1)

    assert budget.new_calls == 1
    assert budget.estimated_input_tokens == 6


def test_deduplication_rejects_a_stale_identity_for_changed_retained_inputs() -> None:
    cell = make_cell("source_trust")
    cell["identity"] = build_request_identity(
        str(cell["question"]), tuple(cell["chunks"]), cell["settings"]
    )
    cell["chunks"] = (make_chunk("bz4x", "A changed retained passage.", 1),)

    with pytest.raises(ValueError, match="identity"):
        deduplicate_requests((cell,))


def test_dry_run_accounting_counts_cells_unique_inputs_hits_and_misses(tmp_path) -> None:
    first = make_cell("source_trust")
    duplicate = make_cell("combined")
    distinct = make_cell("none")
    distinct["question"] = "What is the official battery capacity?"
    requests = deduplicate_requests((first, duplicate, distinct))
    cache = GenerationCache(tmp_path / "cache.json")
    cached = requests[0]
    cache.store(
        str(cached["fingerprint"]),
        cached["identity"],
        GeneratedAnswer("cached", None),
    )

    assert build_dry_run_accounting((first, duplicate, distinct), requests, cache) == {
        "conceptual_cells": 3,
        "unique_fingerprints": 2,
        "cache_hits": 1,
        "new_calls": 1,
        "estimated_new_input_tokens": requests[1]["estimated_input_tokens"],
    }


def test_cache_rejects_fingerprint_that_does_not_match_its_identity(tmp_path) -> None:
    cache = GenerationCache(tmp_path / "cache.json")
    identity = build_identity(chunks=(make_chunk("a", "first brochure passage", 1),))

    with pytest.raises(ValueError, match="fingerprint"):
        cache.store("not-the-identity-fingerprint", identity, GeneratedAnswer("ok", None))
