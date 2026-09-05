from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import pytest

import experiments.run_phase4_evaluation as runner
from experiments.phase4_generation import GenerationCache, deduplicate_requests
from experiments.run_phase4_evaluation import (
    build_dry_run,
    execute_phase4,
    main,
    prepare_phase4_cells,
)
from src.rag.config import Settings
from src.rag.defenses import DefenseMode, DefenseResult
from src.rag import generate
from src.rag.index import ATTACKED_COLLECTION_NAME, CLEAN_COLLECTION_NAME
from src.rag.models import GeneratedAnswer, RetrievedChunk, TokenUsage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODES = {
    "none",
    "source_trust",
    "instruction_filter",
    "similarity_filter",
    "combined",
}
EXPECTED_SCENARIOS = [
    {"scenario": "clean", "mode": "none"},
    {"scenario": "attacked", "mode": "none"},
    {"scenario": "attacked", "mode": "source_trust"},
    {"scenario": "attacked", "mode": "instruction_filter"},
    {"scenario": "attacked", "mode": "similarity_filter"},
    {"scenario": "attacked", "mode": "combined"},
]


def make_settings() -> Settings:
    return Settings(
        project_root=PROJECT_ROOT,
        llm_provider="gemini",
        llm_model="gemini-3.5-flash-lite",
        llm_api_key=None,
        llm_base_url=None,
        max_output_tokens=300,
        llm_temperature=0.0,
        top_k=3,
        chunk_size=1200,
        chunk_overlap=200,
        chroma_persist_dir=PROJECT_ROOT / "data" / "vector_store",
        embedding_model="all-MiniLM-L6-v2",
        clean_data_dir=PROJECT_ROOT / "data" / "clean",
        manifest_path=PROJECT_ROOT / "data" / "manifests" / "clean_index.json",
        poisoned_data_dir=PROJECT_ROOT / "data" / "poisoned",
        attacked_manifest_path=PROJECT_ROOT
        / "data"
        / "manifests"
        / "attacked_index.json",
        attack_manifest_path=PROJECT_ROOT
        / "data"
        / "manifests"
        / "attack_manifest.json",
        attack_questions_path=PROJECT_ROOT
        / "data"
        / "evaluation"
        / "attack_questions.json",
        clean_control_questions_path=PROJECT_ROOT
        / "data"
        / "evaluation"
        / "clean_control_questions.json",
        defense_similarity_threshold=0.92,
    )


class FakeRetriever:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.calls = 0

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        self.calls += 1
        return [
            RetrievedChunk(
                document_id=f"{self.prefix}-document",
                filename=f"{self.prefix}.pdf",
                page_number=rank,
                chunk_id=f"{self.prefix}-chunk-{rank}",
                text=f"{self.prefix} brochure evidence {rank} for {question}",
                rank=rank,
                relevance_score=0.9,
            )
            for rank in range(1, 4)
        ]


class FakeCoordinator:
    def __init__(self) -> None:
        self.snapshots: list[tuple[RetrievedChunk, ...]] = []

    def apply(
        self, chunks: tuple[RetrievedChunk, ...], mode: DefenseMode
    ) -> DefenseResult:
        self.snapshots.append(chunks)
        return DefenseResult(chunks=chunks, trace=())


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return [[1.0] for _ in texts]


@dataclass
class FakeClock:
    value: float = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 0.001
        return current


def inspect_collection(_settings: Settings, name: str) -> dict[str, object]:
    counts = {CLEAN_COLLECTION_NAME: 307, ATTACKED_COLLECTION_NAME: 317}
    metadata = {
        "document_id": "safe-document",
        "filename": "safe.pdf",
        "page_number": 1,
        "chunk_id": "safe-chunk",
    }
    return {
        "name": name,
        "count": counts[name],
        "metadatas": [metadata.copy() for _ in range(counts[name])],
    }


def make_dependencies() -> dict[str, object]:
    return {
        "clean_retriever": FakeRetriever("clean"),
        "attacked_retriever": FakeRetriever("attacked"),
        "coordinator": FakeCoordinator(),
        "similarity_embedder": FakeEmbedder(),
        "collection_inspector": inspect_collection,
        "clock": FakeClock(),
    }


def test_prepare_retrieves_twice_per_question_and_builds_288_cells() -> None:
    dependencies = make_dependencies()

    prepared = prepare_phase4_cells(make_settings(), **dependencies)

    assert len(prepared["cells"]) == 288
    assert dependencies["clean_retriever"].calls == 48
    assert dependencies["attacked_retriever"].calls == 48
    assert {cell["mode"] for cell in prepared["cells"]} == EXPECTED_MODES


def test_prepare_reuses_one_attacked_tuple_and_warms_similarity_once() -> None:
    dependencies = make_dependencies()

    prepared = prepare_phase4_cells(make_settings(), **dependencies)

    coordinator = dependencies["coordinator"]
    first_question_snapshots = coordinator.snapshots[:5]
    assert len({id(snapshot) for snapshot in first_question_snapshots}) == 1
    assert dependencies["similarity_embedder"].calls == [
        ("Phase 4 similarity defense warm-up",)
    ]
    assert prepared["plan"]["warmup_latency_ms"] == 1.0


def test_prepare_warms_shared_embedder_before_first_retrieval() -> None:
    events: list[str] = []
    dependencies = make_dependencies()

    class OrderedRetriever(FakeRetriever):
        def retrieve(self, question: str) -> list[RetrievedChunk]:
            events.append(f"retrieve:{self.prefix}")
            return super().retrieve(question)

    class OrderedEmbedder(FakeEmbedder):
        def encode(self, texts: list[str]) -> list[list[float]]:
            events.append("warmup")
            return super().encode(texts)

    dependencies["clean_retriever"] = OrderedRetriever("clean")
    dependencies["attacked_retriever"] = OrderedRetriever("attacked")
    dependencies["similarity_embedder"] = OrderedEmbedder()

    prepare_phase4_cells(make_settings(), **dependencies)

    assert events[0] == "warmup"


def test_prepare_builds_stable_unique_cell_ids_and_fingerprint_map() -> None:
    first = prepare_phase4_cells(make_settings(), **make_dependencies())
    second = prepare_phase4_cells(make_settings(), **make_dependencies())

    first_cells = first["plan"]["cells"]
    second_cells = second["plan"]["cells"]
    assert len({cell["cell_id"] for cell in first_cells}) == 288
    assert [
        (cell["cell_id"], cell["fingerprint"]) for cell in first_cells
    ] == [
        (cell["cell_id"], cell["fingerprint"]) for cell in second_cells
    ]
    assert first["plan"]["budget"]["within_budget_without_cache"] is True
    assert first["plan"]["budget"]["hard_max_new_calls"] == 120
    assert first["plan"]["budget"]["hard_max_estimated_input_tokens"] == 150_000
    assert first["plan"]["scenario_order"] == EXPECTED_SCENARIOS


def test_prepare_keeps_exact_frozen_matrix_if_enum_gains_a_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FutureMode:
        value = "future_mode"

    class ExpandedModeMeta(type):
        def __iter__(cls):
            return iter(
                (
                    DefenseMode.NONE,
                    DefenseMode.SOURCE_TRUST,
                    DefenseMode.INSTRUCTION_FILTER,
                    DefenseMode.SIMILARITY_FILTER,
                    DefenseMode.COMBINED,
                    FutureMode(),
                )
            )

    class ExpandedDefenseMode(metaclass=ExpandedModeMeta):
        NONE = DefenseMode.NONE

    monkeypatch.setattr(runner, "DefenseMode", ExpandedDefenseMode)

    prepared = prepare_phase4_cells(make_settings(), **make_dependencies())

    assert len(prepared["cells"]) == 288
    assert prepared["plan"]["scenario_order"] == EXPECTED_SCENARIOS


def test_prepare_rejects_hidden_poison_labels_in_visible_metadata() -> None:
    dependencies = make_dependencies()

    def inspect_with_leak(_settings: Settings, name: str) -> dict[str, object]:
        count = 307 if name == CLEAN_COLLECTION_NAME else 317
        metadata = {
            "document_id": "safe-document",
            "filename": "safe.pdf",
            "page_number": 1,
            "chunk_id": "safe-chunk",
            "is_poison": True,
        }
        return {
            "name": name,
            "count": count,
            "metadatas": [metadata.copy() for _ in range(count)],
        }

    dependencies["collection_inspector"] = inspect_with_leak

    with pytest.raises(ValueError, match="hidden poison labels"):
        prepare_phase4_cells(make_settings(), **dependencies)


@pytest.mark.parametrize("extra_key", ["attack_id", "false_value"])
def test_prepare_rejects_any_non_rag_metadata_key(extra_key: str) -> None:
    dependencies = make_dependencies()

    def inspect_with_extra_key(
        _settings: Settings, name: str
    ) -> dict[str, object]:
        count = 307 if name == CLEAN_COLLECTION_NAME else 317
        metadata = {
            "document_id": "safe-document",
            "filename": "safe.pdf",
            "page_number": 1,
            "chunk_id": "safe-chunk",
            extra_key: "hidden-evaluation-value",
        }
        return {
            "name": name,
            "count": count,
            "metadatas": [metadata.copy() for _ in range(count)],
        }

    dependencies["collection_inspector"] = inspect_with_extra_key

    with pytest.raises(ValueError, match="exactly the RAG-visible keys"):
        prepare_phase4_cells(make_settings(), **dependencies)


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "document_id": "safe-document",
            "FILENAME": "safe.pdf",
            "page_number": 1,
            "chunk_id": "safe-chunk",
        },
        {
            "document_id": "safe-document",
            "filename": "safe.pdf",
            "FILENAME": "shadow.pdf",
            "page_number": 1,
            "chunk_id": "safe-chunk",
        },
    ],
    ids=("uppercase-only-key", "case-colliding-duplicate"),
)
def test_prepare_rejects_case_variant_metadata_keys(
    metadata: dict[str, object],
) -> None:
    dependencies = make_dependencies()

    def inspect_with_case_variant(
        _settings: Settings, name: str
    ) -> dict[str, object]:
        count = 307 if name == CLEAN_COLLECTION_NAME else 317
        return {
            "name": name,
            "count": count,
            "metadatas": [metadata.copy() for _ in range(count)],
        }

    dependencies["collection_inspector"] = inspect_with_case_variant

    with pytest.raises(ValueError, match="exactly the RAG-visible keys"):
        prepare_phase4_cells(make_settings(), **dependencies)


def test_prepare_rejects_incomplete_collection_metadata_inspection() -> None:
    dependencies = make_dependencies()

    def inspect_too_few(_settings: Settings, name: str) -> dict[str, object]:
        count = 307 if name == CLEAN_COLLECTION_NAME else 317
        metadata = {
            "document_id": "safe-document",
            "filename": "safe.pdf",
            "page_number": 1,
            "chunk_id": "safe-chunk",
        }
        return {
            "name": name,
            "count": count,
            "metadatas": [metadata.copy() for _ in range(count - 1)],
        }

    dependencies["collection_inspector"] = inspect_too_few

    with pytest.raises(ValueError, match="inspected metadata count"):
        prepare_phase4_cells(make_settings(), **dependencies)


def test_dry_run_counts_cache_misses_without_calling_generator() -> None:
    prepared = prepare_phase4_cells(make_settings(), **make_dependencies())

    dry_run = build_dry_run(prepared, GenerationCache.empty())

    assert dry_run["conceptual_cells"] == 288
    assert dry_run["new_calls"] <= 120
    assert dry_run["gemini_calls_made"] == 0
    assert dry_run["within_budget"] is True


def test_dry_run_marks_each_hard_cap_independently() -> None:
    over_calls = {
        "cells": tuple({} for _ in range(121)),
        "requests": tuple(
            {
                "fingerprint": f"fingerprint-{index}",
                "identity": {"request": index},
                "estimated_input_tokens": 1,
                "cell_ids": (f"cell-{index}",),
            }
            for index in range(121)
        ),
        "plan": {},
    }
    over_tokens = {
        "cells": ({},),
        "requests": (
            {
                "fingerprint": "large-request",
                "identity": {"request": "large"},
                "estimated_input_tokens": 150_001,
                "cell_ids": ("large-cell",),
            },
        ),
        "plan": {},
    }

    calls_result = build_dry_run(over_calls, GenerationCache.empty())
    tokens_result = build_dry_run(over_tokens, GenerationCache.empty())

    assert calls_result["call_cap_passes"] is False
    assert calls_result["input_token_cap_passes"] is True
    assert calls_result["within_budget"] is False
    assert tokens_result["call_cap_passes"] is True
    assert tokens_result["input_token_cap_passes"] is False
    assert tokens_result["within_budget"] is False


def test_cli_dry_run_writes_artifacts_without_constructing_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = prepare_phase4_cells(settings, **make_dependencies())

    class ForbiddenGenerator:
        calls = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            type(self).calls += 1

    monkeypatch.setattr(generate, "GeminiGenerator", ForbiddenGenerator)
    monkeypatch.setattr(
        runner.Settings, "from_env", classmethod(lambda _cls: settings)
    )
    monkeypatch.setattr(runner, "prepare_phase4_cells", lambda _settings: prepared)

    exit_code = main(["--dry-run"])

    results_dir = tmp_path / "experiments" / "results"
    plan = json.loads((results_dir / "phase4_evaluation_plan.json").read_text())
    dry_run = json.loads((results_dir / "phase4_dry_run.json").read_text())
    assert exit_code == 0
    assert plan["conceptual_cells"] == 288
    assert dry_run["gemini_calls_made"] == 0
    assert ForbiddenGenerator.calls == 0


def test_cli_dry_run_exits_nonzero_when_a_hard_cap_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = {
        "cells": tuple({} for _ in range(121)),
        "requests": tuple(
            {
                "fingerprint": f"fingerprint-{index}",
                "identity": {"request": index},
                "estimated_input_tokens": 1,
                "cell_ids": (f"cell-{index}",),
            }
            for index in range(121)
        ),
        "plan": {"conceptual_cells": 121},
    }
    monkeypatch.setattr(
        runner.Settings, "from_env", classmethod(lambda _cls: settings)
    )
    monkeypatch.setattr(runner, "prepare_phase4_cells", lambda _settings: prepared)

    exit_code = main(["--dry-run"])

    assert exit_code == 1
    dry_run = json.loads(
        (
            tmp_path / "experiments" / "results" / "phase4_dry_run.json"
        ).read_text()
    )
    assert dry_run["within_budget"] is False


def _prepared_controls(settings: Settings, count: int = 2) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for index in range(count):
        chunk = RetrievedChunk(
            document_id="clean-rav4",
            filename="rav4.pdf",
            page_number=53,
            chunk_id=f"clean-rav4-{index}",
            text=f"Toyota Relax lasts 10 years or 100,000 miles. Evidence {index}.",
            rank=1,
            relevance_score=0.9,
        )
        cells.append(
            {
                "cell_id": f"control-{index}:clean:none",
                "question_id": f"control-{index}",
                "question": f"How long is Toyota Relax cover? Variant {index}",
                "cohort": "control",
                "scenario": "clean",
                "mode": "none",
                "attack_id": None,
                "benchmark_metadata": {
                    "clean_source_filename": "rav4.pdf",
                    "clean_source_page": 53,
                    "expected_answer": "10 years or 100,000 miles",
                },
                "source_chunks": (chunk,),
                "chunks": (chunk,),
                "retrieval_latency_ms": 1.0,
                "defense_latency_ms": 0.0,
                "defense_trace": (),
                "settings": settings,
            }
        )
    requests = deduplicate_requests(cells)
    return {
        "cells": tuple(cells),
        "requests": requests,
        "plan": {
            "frozen_configuration": {"model": settings.llm_model},
            "frozen_file_hashes": {},
            "prior_result_hashes_before": {},
        },
    }


def test_execute_persists_each_success_and_resumes_only_missing_request(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings)
    cache = GenerationCache(tmp_path / "experiments" / "results" / "cache.json")
    execution_clock = ManualExecutionClock()
    calls: list[str] = []

    class InterruptingGenerator:
        def generate(
            self, question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            calls.append(question)
            if len(calls) == 2:
                raise RuntimeError("simulated interruption")
            return GeneratedAnswer(
                "Toyota Relax lasts 10 years or 100,000 miles.",
                TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8),
            )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=lambda _settings: InterruptingGenerator(),
            clock=execution_clock,
            sleeper=execution_clock.sleep,
        )

    first, second = prepared["requests"]
    assert cache.get(first["fingerprint"], first["identity"]) is not None
    assert cache.get(second["fingerprint"], second["identity"]) is None

    class CompletingGenerator:
        def generate(
            self, question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            calls.append(question)
            return GeneratedAnswer(
                "Toyota Relax lasts 10 years or 100,000 miles.",
                TokenUsage(input_tokens=6, output_tokens=4, total_tokens=10),
            )

    result = execute_phase4(
        settings,
        prepared,
        cache,
        build_dry_run(prepared, cache),
        generator_factory=lambda _settings: CompletingGenerator(),
        clock=execution_clock,
        sleeper=execution_clock.sleep,
    )

    assert calls == [first["question"], second["question"], second["question"]]
    assert result["run"]["generation"]["new_calls"] == 1
    assert len(result["rows"]) == 2
    assert (
        tmp_path / "experiments" / "results" / "phase4_evaluation_results.json"
    ).is_file()
    assert (tmp_path / "reports" / "phase4_evaluation.md").is_file()


def test_execute_rejects_stale_dry_run_before_constructing_generator(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    stale_dry_run = build_dry_run(prepared, cache)
    request = prepared["requests"][0]
    cache.store(
        request["fingerprint"],
        request["identity"],
        GeneratedAnswer("cached", None),
    )
    constructions = 0

    def forbidden_factory(_settings: Settings) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("generator must not be constructed")

    with pytest.raises(ValueError, match="current official dry-run"):
        execute_phase4(
            settings,
            prepared,
            cache,
            stale_dry_run,
            generator_factory=forbidden_factory,
        )

    assert constructions == 0


def test_cli_execute_publishes_results_with_a_fake_generator_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)

    class FakeGenerator:
        calls = 0

        def __init__(self, _settings: Settings) -> None:
            pass

        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            type(self).calls += 1
            return GeneratedAnswer(
                "Toyota Relax lasts 10 years or 100,000 miles.",
                TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8),
            )

    monkeypatch.setattr(runner, "GeminiGenerator", FakeGenerator)
    monkeypatch.setattr(
        runner.Settings, "from_env", classmethod(lambda _cls: settings)
    )
    monkeypatch.setattr(runner, "prepare_phase4_cells", lambda _settings: prepared)

    exit_code = main(["--execute"])

    results_dir = tmp_path / "experiments" / "results"
    assert exit_code == 0
    assert FakeGenerator.calls == 1
    assert (results_dir / "phase4_evaluation_plan.json").is_file()
    assert (results_dir / "phase4_dry_run.json").is_file()
    assert (results_dir / "phase4_evaluation_results.json").is_file()
    assert (results_dir / "phase4_evaluation_results.csv").is_file()
    assert (results_dir / "phase4_summary.json").is_file()
    assert (tmp_path / "reports" / "phase4_evaluation.md").is_file()
    assert not (results_dir / "phase4_manual_reviews.json").exists()

def test_execute_emits_manual_review_file_only_for_ambiguous_rows(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    clean = RetrievedChunk(
        document_id="clean-rav4",
        filename="rav4.pdf",
        page_number=40,
        chunk_id="clean-rav4-p40",
        text="The brochure discusses the RAV4 fuel tank.",
        rank=1,
        relevance_score=0.9,
    )
    cell = {
        "cell_id": "attack_001_q1:clean:none",
        "question_id": "attack_001_q1",
        "question": "What is the RAV4 fuel tank capacity?",
        "cohort": "attack",
        "scenario": "clean",
        "mode": "none",
        "attack_id": "attack_001",
        "benchmark_metadata": {"attack_id": "attack_001"},
        "source_chunks": (clean,),
        "chunks": (clean,),
        "retrieval_latency_ms": 1.0,
        "defense_latency_ms": 0.0,
        "defense_trace": (),
        "settings": settings,
    }
    prepared = {
        "cells": (cell,),
        "requests": deduplicate_requests((cell,)),
        "plan": {
            "frozen_configuration": {},
            "frozen_file_hashes": {},
            "prior_result_hashes_before": {},
        },
    }
    cache = GenerationCache(
        tmp_path / "experiments" / "results" / "phase4_generation_cache.json"
    )

    class AmbiguousGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            return GeneratedAnswer(
                "The brochure lists 55 litres, but the current capacity is 72 litres.",
                None,
            )

    execute_phase4(
        settings,
        prepared,
        cache,
        build_dry_run(prepared, cache),
        generator_factory=lambda _settings: AmbiguousGenerator(),
    )

    reviews = json.loads(
        (
            tmp_path
            / "experiments"
            / "results"
            / "phase4_manual_reviews.json"
        ).read_text(encoding="utf-8")
    )
    assert reviews["reviews"] == [
        {
            "cell_id": "attack_001_q1:clean:none",
            "fingerprint": prepared["requests"][0]["fingerprint"],
            "answer_sha256": hashlib.sha256(
                b"The brochure lists 55 litres, but the current capacity is 72 litres."
            ).hexdigest(),
            "resolution": None,
            "generation_compromised": None,
            "notes": "",
        }
    ]
    reviews["reviews"][0]["resolution"] = "incorrect"
    reviews["reviews"][0]["generation_compromised"] = True
    reviews["reviews"][0]["notes"] = "Human confirmed adoption of the false claim."
    manual_path = (
        tmp_path / "experiments" / "results" / "phase4_manual_reviews.json"
    )
    manual_path.write_text(json.dumps(reviews), encoding="utf-8")
    constructions = 0

    def forbidden_factory(_settings: Settings) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("a cached re-finalization must not construct a generator")

    finalized = execute_phase4(
        settings,
        prepared,
        cache,
        build_dry_run(prepared, cache),
        generator_factory=forbidden_factory,
    )

    assert constructions == 0
    assert finalized["run"]["generation"]["new_calls"] == 0
    assert finalized["rows"][0]["deterministic_score"] == "ambiguous"
    assert finalized["rows"][0]["score"] == "incorrect"
    assert finalized["rows"][0]["generation_compromised"] is True
    assert json.loads(manual_path.read_text(encoding="utf-8"))["reviews"][0][
        "notes"
    ] == "Human confirmed adoption of the false claim."
    request = prepared["requests"][0]
    cache.store(
        request["fingerprint"],
        request["identity"],
        GeneratedAnswer("A different ambiguous answer.", None),
    )
    with pytest.raises(ValueError, match="answer"):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=forbidden_factory,
        )
    assert constructions == 0

def test_execute_rejects_impossible_manual_review_before_any_generation(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(
        tmp_path / "experiments" / "results" / "phase4_generation_cache.json"
    )
    request = prepared["requests"][0]
    manual_path = (
        tmp_path / "experiments" / "results" / "phase4_manual_reviews.json"
    )
    manual_path.parent.mkdir(parents=True)
    manual_path.write_text(
        json.dumps(
            {
                "schema_version": "phase4_manual_reviews_v1",
                "reviews": [
                    {
                        "cell_id": "not-a-current-cell",
                        "fingerprint": request["fingerprint"],
                        "answer_sha256": hashlib.sha256(b"old answer").hexdigest(),
                        "resolution": "correct",
                        "generation_compromised": None,
                        "notes": "stale review",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    constructions = 0

    class CountingGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            return GeneratedAnswer("10 years or 100,000 miles", None)

    def factory(_settings: Settings) -> CountingGenerator:
        nonlocal constructions
        constructions += 1
        return CountingGenerator()

    with pytest.raises(ValueError, match="manual review target"):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=factory,
        )

    assert constructions == 0

def _settings_with_frozen_copies(tmp_path: Path) -> Settings:
    original = make_settings()
    paths: dict[str, Path] = {}
    for field_name in (
        "manifest_path",
        "attacked_manifest_path",
        "attack_manifest_path",
        "attack_questions_path",
        "clean_control_questions_path",
    ):
        source = getattr(original, field_name)
        destination = tmp_path / "frozen" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        paths[field_name] = destination
    return replace(original, project_root=tmp_path, **paths)


def test_execute_revalidates_frozen_files_after_generation(
    tmp_path: Path,
) -> None:
    settings = _settings_with_frozen_copies(tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(
        tmp_path / "experiments" / "results" / "phase4_generation_cache.json"
    )

    class MutatingGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            settings.attack_manifest_path.write_text("{}", encoding="utf-8")
            return GeneratedAnswer("10 years or 100,000 miles", None)

    with pytest.raises(ValueError, match="frozen Phase 4 input hash mismatch"):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=lambda _settings: MutatingGenerator(),
        )

    assert not (
        tmp_path / "experiments" / "results" / "phase4_evaluation_results.json"
    ).exists()


def test_scoring_uses_the_same_hash_verified_frozen_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_with_frozen_copies(tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(
        tmp_path / "experiments" / "results" / "phase4_generation_cache.json"
    )
    real_snapshot = runner.load_frozen_snapshot

    def snapshot_then_mutate(active_settings: Settings) -> dict[str, bytes]:
        snapshot = real_snapshot(active_settings)
        active_settings.manifest_path.write_text("{}", encoding="utf-8")
        active_settings.attack_manifest_path.write_text("{}", encoding="utf-8")
        return snapshot

    class FakeGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            return GeneratedAnswer("10 years or 100,000 miles", None)

    monkeypatch.setattr(runner, "load_frozen_snapshot", snapshot_then_mutate)

    result = execute_phase4(
        settings,
        prepared,
        cache,
        build_dry_run(prepared, cache),
        generator_factory=lambda _settings: FakeGenerator(),
    )

    assert result["rows"][0]["score"] == "correct"

def test_execute_rejects_nonambiguous_review_before_unrelated_generation(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=2)
    cache = GenerationCache(
        tmp_path / "experiments" / "results" / "phase4_generation_cache.json"
    )
    reviewed_request = prepared["requests"][0]
    reviewed_answer = "Toyota Relax lasts 10 years or 100,000 miles."
    cache.store(
        reviewed_request["fingerprint"],
        reviewed_request["identity"],
        GeneratedAnswer(reviewed_answer, None),
    )
    manual_path = (
        tmp_path / "experiments" / "results" / "phase4_manual_reviews.json"
    )
    manual_path.write_text(
        json.dumps(
            {
                "schema_version": "phase4_manual_reviews_v1",
                "reviews": [
                    {
                        "cell_id": "control-0:clean:none",
                        "fingerprint": reviewed_request["fingerprint"],
                        "answer_sha256": hashlib.sha256(
                            reviewed_answer.encode("utf-8")
                        ).hexdigest(),
                        "resolution": "incorrect",
                        "generation_compromised": None,
                        "notes": "impossible review of a deterministic correct row",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    constructions = 0

    class CountingGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            return GeneratedAnswer(reviewed_answer, None)

    def factory(_settings: Settings) -> CountingGenerator:
        nonlocal constructions
        constructions += 1
        return CountingGenerator()

    with pytest.raises(ValueError, match="ambiguous"):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=factory,
        )

    assert constructions == 0


def test_execute_rejects_generation_adjudication_when_deterministic_value_is_boolean(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    clean = RetrievedChunk(
        document_id="clean-rav4",
        filename="rav4.pdf",
        page_number=40,
        chunk_id="clean-rav4-p40",
        text="The brochure discusses the RAV4 fuel tank.",
        rank=1,
        relevance_score=0.9,
    )
    cell = {
        "cell_id": "attack_001_q1:clean:none",
        "question_id": "attack_001_q1",
        "question": "What is the RAV4 fuel tank capacity?",
        "cohort": "attack",
        "scenario": "clean",
        "mode": "none",
        "attack_id": "attack_001",
        "benchmark_metadata": {"attack_id": "attack_001"},
        "source_chunks": (clean,),
        "chunks": (clean,),
        "retrieval_latency_ms": 1.0,
        "defense_latency_ms": 0.0,
        "defense_trace": (),
        "settings": settings,
    }
    prepared = {
        "cells": (cell,),
        "requests": deduplicate_requests((cell,)),
        "plan": {
            "frozen_configuration": {},
            "frozen_file_hashes": {},
            "prior_result_hashes_before": {},
        },
    }
    cache = GenerationCache(
        tmp_path / "experiments" / "results" / "phase4_generation_cache.json"
    )
    request = prepared["requests"][0]
    answer = "The brochure discusses fuel capacity."
    cache.store(
        request["fingerprint"],
        request["identity"],
        GeneratedAnswer(answer, None),
    )
    manual_path = (
        tmp_path / "experiments" / "results" / "phase4_manual_reviews.json"
    )
    manual_path.write_text(
        json.dumps(
            {
                "schema_version": "phase4_manual_reviews_v1",
                "reviews": [
                    {
                        "cell_id": cell["cell_id"],
                        "fingerprint": request["fingerprint"],
                        "answer_sha256": hashlib.sha256(
                            answer.encode("utf-8")
                        ).hexdigest(),
                        "resolution": "incorrect",
                        "generation_compromised": True,
                        "notes": "inapplicable generation adjudication",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    constructions = 0

    def forbidden_factory(_settings: Settings) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("validation must happen before generation")

    with pytest.raises(ValueError, match="generation_compromised"):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=forbidden_factory,
        )

    assert constructions == 0


@dataclass
class ManualExecutionClock:
    value: float = 0.0
    sleeps: list[float] | None = None

    def __post_init__(self) -> None:
        self.sleeps = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        assert self.sleeps is not None
        self.sleeps.append(seconds)
        self.value += seconds


class FakeProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        status: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retry_after = retry_after


def _generation_error(cause: Exception) -> generate.GenerationError:
    try:
        raise cause
    except Exception as provider_error:
        try:
            raise generate.GenerationError("Gemini generation failed") from provider_error
        except generate.GenerationError as wrapped:
            return wrapped


def test_generation_latency_excludes_pacing_and_checkpoint_work(tmp_path, monkeypatch) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=2)
    cache = GenerationCache(tmp_path / "cache.json")
    execution_clock = ManualExecutionClock()
    original_begin = runner.GenerationAttemptAudit.begin
    original_store = GenerationCache.store

    def slow_begin(audit, fingerprint, **kwargs):
        attempt = original_begin(audit, fingerprint, **kwargs)
        execution_clock.value += 0.5
        return attempt

    def slow_store(*args, **kwargs):
        result = original_store(*args, **kwargs)
        execution_clock.value += 0.75
        return result

    class TimedGenerator:
        def generate(self, question, chunks):
            execution_clock.value += 0.25
            return GeneratedAnswer("10 years or 100,000 miles", None)

    monkeypatch.setattr(runner.GenerationAttemptAudit, "begin", slow_begin)
    monkeypatch.setattr(GenerationCache, "store", slow_store)
    result = execute_phase4(
        settings, prepared, cache, build_dry_run(prepared, cache),
        generator_factory=lambda _: TimedGenerator(), clock=execution_clock,
        sleeper=execution_clock.sleep, wall_clock=execution_clock,
    )
    assert [row["generation_latency_ms"] for row in result["rows"]] == [250.0, 250.0]
    assert [row["total_latency_ms"] for row in result["rows"]] == [251.0, 251.0]
    assert result["run"]["latency_reconciliation"]["source"] == "successful_attempt_audit"
    audit = runner.GenerationAttemptAudit(tmp_path / "experiments/results/phase4_generation_attempts.json")
    assert [attempt["latency_ms"] for attempt in audit.attempts] == [250.0, 250.0]
    assert cache.reconcile_successful_latencies(audit) == 0


def test_execute_paces_every_attempt_and_audits_429_retries(tmp_path: Path) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=2)
    cache = GenerationCache(
        tmp_path / "experiments" / "results" / "phase4_generation_cache.json"
    )
    execution_clock = ManualExecutionClock()
    attempt_times: list[float] = []

    class RateLimitedThenSuccessfulGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            attempt_times.append(execution_clock())
            if len(attempt_times) <= 2:
                raise _generation_error(
                    FakeProviderError("quota", status="RESOURCE_EXHAUSTED")
                )
            return GeneratedAnswer(
                "Toyota Relax lasts 10 years or 100,000 miles.",
                TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8),
            )

    result = execute_phase4(
        settings,
        prepared,
        cache,
        build_dry_run(prepared, cache),
        generator_factory=lambda _settings: RateLimitedThenSuccessfulGenerator(),
        clock=execution_clock,
        sleeper=execution_clock.sleep,
    )

    assert attempt_times == [0.0, 5.0, 15.0, 20.0]
    assert execution_clock.sleeps == [5.0, 10.0, 5.0]
    assert result["run"]["generation"] == {
        "new_calls": 2,
        "unique_new_cache_entries": 2,
        "estimated_new_input_tokens": sum(
            request["estimated_input_tokens"] for request in prepared["requests"]
        ),
        "provider_attempts": 4,
        "rate_limit_retries": 2,
    }
    first, second = prepared["requests"]
    first_entry = cache.get(first["fingerprint"], first["identity"])
    second_entry = cache.get(second["fingerprint"], second["identity"])
    assert first_entry is not None
    assert first_entry["provider_attempts"] == 3
    assert first_entry["rate_limit_retries"] == 2
    assert second_entry is not None
    assert second_entry["provider_attempts"] == 1
    assert second_entry["rate_limit_retries"] == 0
    assert result["summary"]["generation_usage"]["provider_attempts"] == 4
    assert result["summary"]["generation_usage"]["rate_limit_retries"] == 2
    report = (tmp_path / "reports" / "phase4_evaluation.md").read_text(
        encoding="utf-8"
    )
    assert "Total provider attempts: 4" in report
    assert "Rate-limit retries: 2" in report


def test_execute_honors_numeric_retry_metadata_through_same_limiter(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    execution_clock = ManualExecutionClock()
    attempts = 0

    class RetryAfterGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise _generation_error(
                    FakeProviderError("429", code=429, retry_after=17)
                )
            return GeneratedAnswer("10 years or 100,000 miles", None)

    execute_phase4(
        settings,
        prepared,
        cache,
        build_dry_run(prepared, cache),
        generator_factory=lambda _settings: RetryAfterGenerator(),
        clock=execution_clock,
        sleeper=execution_clock.sleep,
    )

    assert attempts == 2
    assert execution_clock.sleeps == [17.0]


def test_execute_does_not_retry_or_cache_non_rate_limit_failure(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    execution_clock = ManualExecutionClock()
    attempts = 0

    class FailingGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            nonlocal attempts
            attempts += 1
            raise _generation_error(FakeProviderError("server error", code=500))

    with pytest.raises(generate.GenerationError):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=lambda _settings: FailingGenerator(),
            clock=execution_clock,
            sleeper=execution_clock.sleep,
        )

    request = prepared["requests"][0]
    assert attempts == 1
    assert execution_clock.sleeps == []
    assert cache.get(request["fingerprint"], request["identity"]) is None


def test_dry_run_rejects_more_than_103_unique_fingerprint_results() -> None:
    settings = make_settings()
    prepared = _prepared_controls(settings, count=104)

    dry_run = build_dry_run(prepared, GenerationCache.empty())

    assert dry_run["unique_fingerprints"] == 104
    assert dry_run["approved_unique_result_cap"] == 103
    assert dry_run["unique_result_cap_passes"] is False
    assert dry_run["within_budget"] is False


def test_execute_rejects_model_change_before_constructing_generator(
    tmp_path: Path,
) -> None:
    prepared_settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(prepared_settings, count=1)
    changed_settings = replace(prepared_settings, llm_model="another-model")
    cache = GenerationCache(tmp_path / "cache.json")
    constructions = 0

    def forbidden_factory(_settings: Settings) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("generator must not be constructed")

    with pytest.raises(ValueError, match="configuration is not frozen"):
        execute_phase4(
            changed_settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=forbidden_factory,
        )

    assert constructions == 0



def test_execute_caps_exponential_429_backoff_and_does_not_cache_exhaustion(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    execution_clock = ManualExecutionClock()
    attempts = 0

    class AlwaysRateLimitedGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            nonlocal attempts
            attempts += 1
            raise _generation_error(FakeProviderError("quota", code=429))

    with pytest.raises(generate.GenerationError):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=lambda _settings: AlwaysRateLimitedGenerator(),
            clock=execution_clock,
            sleeper=execution_clock.sleep,
        )

    request = prepared["requests"][0]
    assert attempts == 6
    assert execution_clock.sleeps == [5.0, 10.0, 20.0, 40.0, 60.0]
    assert cache.get(request["fingerprint"], request["identity"]) is None


def test_unique_result_cap_counts_only_new_cache_entries(tmp_path: Path) -> None:
    settings = make_settings()
    prepared = _prepared_controls(settings, count=104)
    cache = GenerationCache(tmp_path / "cache.json")
    first = prepared["requests"][0]
    cache.store(
        first["fingerprint"],
        first["identity"],
        GeneratedAnswer("cached", None),
    )

    dry_run = build_dry_run(prepared, cache)

    assert dry_run["unique_fingerprints"] == 104
    assert dry_run["cache_hits"] == 1
    assert dry_run["new_calls"] == 103
    assert dry_run["unique_result_cap_passes"] is True
    assert dry_run["within_budget"] is True


def test_execute_does_not_retry_raw_provider_429_outside_generation_error(
    tmp_path: Path,
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    execution_clock = ManualExecutionClock()
    attempts = 0

    class RawFailureGenerator:
        def generate(
            self, _question: str, _chunks: tuple[RetrievedChunk, ...]
        ) -> GeneratedAnswer:
            nonlocal attempts
            attempts += 1
            raise FakeProviderError("quota", code=429)

    with pytest.raises(FakeProviderError):
        execute_phase4(
            settings,
            prepared,
            cache,
            build_dry_run(prepared, cache),
            generator_factory=lambda _settings: RawFailureGenerator(),
            clock=execution_clock,
            sleeper=execution_clock.sleep,
        )

    assert attempts == 1
    assert execution_clock.sleeps == []


@pytest.mark.parametrize("value, expected", [(125, 60.0), ("900", 60.0), (float("inf"), None), ("inf", None), (float("nan"), None), (-1, None)])
@pytest.mark.parametrize("source", ["attribute", "header", "metadata"])
def test_retry_metadata_is_finite_and_capped(value, expected, source) -> None:
    from types import SimpleNamespace

    cause = FakeProviderError("quota", code=429)
    if source == "attribute":
        cause.retry_after = value
    elif source == "header":
        cause.response = SimpleNamespace(headers={"Retry-After": value})
    else:
        cause.metadata = {"nested": {"retry_after": value}}
    assert runner._retry_after_seconds(_generation_error(cause)) == expected


@pytest.mark.parametrize("marker_branch", ["cause", "context"])
def test_rate_limit_detection_visits_both_branches_and_cycles(marker_branch) -> None:
    error = generate.GenerationError("failed")
    marker = FakeProviderError("quota", code=429, retry_after=12)
    other = RuntimeError("unrelated")
    error.__cause__ = marker if marker_branch == "cause" else other
    error.__context__ = marker if marker_branch == "context" else other
    other.__cause__ = error
    marker.__context__ = other
    assert runner._is_rate_limit_error(error) is True
    assert runner._retry_after_seconds(error) == 12.0
    assert len(runner._exception_chain(error)) == 3


def test_dry_run_counts_104_requests_with_103_distinct_misses() -> None:
    prepared = _prepared_controls(make_settings(), count=103)
    unique_tokens = sum(r["estimated_input_tokens"] for r in prepared["requests"])
    prepared["requests"] = (*prepared["requests"], prepared["requests"][0])
    result = build_dry_run(prepared, GenerationCache.empty())
    assert result["unique_fingerprints"] == 103
    assert result["new_calls"] == 103
    assert result["estimated_new_input_tokens"] == unique_tokens
    assert result["within_budget"] is True


@pytest.mark.parametrize("stop", ["sleep", "terminal"])
def test_attempt_audit_survives_failure_and_resume_without_double_counting(tmp_path, stop) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=2)
    cache = GenerationCache(tmp_path / "cache.json")
    audit_path = tmp_path / "experiments/results/phase4_generation_attempts.json"
    timer = ManualExecutionClock()
    calls = 0

    class FailingGenerator:
        def generate(self, question, chunks):
            nonlocal calls
            calls += 1
            if calls == 1:
                return GeneratedAnswer("10 years or 100,000 miles", None)
            if stop == "terminal" and calls == 3:
                raise RuntimeError("SECRET provider payload")
            raise _generation_error(FakeProviderError("SECRET provider payload", code=429))

    def interrupted_sleep(seconds):
        if calls == 2:
            raise KeyboardInterrupt()
        timer.sleep(seconds)

    with pytest.raises((KeyboardInterrupt, RuntimeError, generate.GenerationError)):
        execute_phase4(settings, prepared, cache, build_dry_run(prepared, cache),
                       generator_factory=lambda _: FailingGenerator(), clock=timer,
                       sleeper=interrupted_sleep if stop == "sleep" else timer.sleep)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    events = audit["attempts"]
    expected_attempts = {"sleep": 2, "terminal": 3}[stop]
    assert len(events) == expected_attempts
    assert "SECRET" not in audit_path.read_text(encoding="utf-8")
    assert events[-1]["status"] == ("failure" if stop == "terminal" else "rate_limit")
    assert sum(e["rate_limit_retry"] for e in events) == {"sleep": 0, "terminal": 1}[stop]
    assert not audit_path.with_suffix(".json.tmp").exists()
    missing = prepared["requests"][1]
    assert cache.get(missing["fingerprint"], missing["identity"]) is None

    class CompletingGenerator:
        def generate(self, question, chunks):
            assert len(json.loads(audit_path.read_text())["attempts"]) >= expected_attempts
            return GeneratedAnswer("10 years or 100,000 miles", None)

    result = execute_phase4(settings, prepared, GenerationCache(cache.path),
                           build_dry_run(prepared, cache), generator_factory=lambda _: CompletingGenerator(),
                           clock=timer, sleeper=timer.sleep)
    totals = result["run"]["generation"]
    assert totals["provider_attempts"] == expected_attempts + 1
    assert totals["rate_limit_retries"] == {"sleep": 1, "terminal": 1}[stop]
    assert result["summary"]["generation_usage"]["provider_attempts"] == expected_attempts + 1
    report = (tmp_path / "reports/phase4_evaluation.md").read_text(encoding="utf-8")
    assert f"Total provider attempts: {expected_attempts + 1}" in report
    saved_audit = audit_path.read_bytes()

    def forbidden(_):
        raise AssertionError("cache-only resume must not construct generator")

    replay = execute_phase4(settings, prepared, cache, build_dry_run(prepared, cache), generator_factory=forbidden)
    assert replay["run"]["generation"]["provider_attempts"] == expected_attempts + 1
    assert replay["run"]["generation"]["rate_limit_retries"] == totals["rate_limit_retries"]
    assert audit_path.read_bytes() == saved_audit


def test_attempt_totals_include_legacy_cache_once(tmp_path) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    request = prepared["requests"][0]
    cache = GenerationCache(tmp_path / "cache.json")
    cache.store(request["fingerprint"], request["identity"],
                GeneratedAnswer("10 years or 100,000 miles", None),
                provider_attempts=3, rate_limit_retries=2)

    def forbidden(_):
        raise AssertionError("cache-only resume must not construct generator")

    result = execute_phase4(settings, prepared, cache, build_dry_run(prepared, cache),
                            generator_factory=forbidden)
    assert result["run"]["generation"]["provider_attempts"] == 3
    assert result["run"]["generation"]["rate_limit_retries"] == 2
    replay = execute_phase4(settings, prepared, cache, build_dry_run(prepared, cache),
                            generator_factory=forbidden)
    assert replay["run"]["generation"]["provider_attempts"] == 3


def test_audit_finish_failure_preserves_cached_answer_on_resume(tmp_path, monkeypatch) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    audit_path = tmp_path / "experiments/results/phase4_generation_attempts.json"
    calls = 0

    class Generator:
        def generate(self, question, chunks):
            nonlocal calls
            calls += 1
            return GeneratedAnswer("10 years or 100,000 miles", None)

    original_replace = Path.replace

    def interrupt_replace(path, target):
        if target == audit_path and json.loads(path.read_text())["attempts"][-1]["status"] == "success":
            raise OSError("simulated checkpoint interruption")
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt_replace)
        with pytest.raises(OSError, match="checkpoint interruption"):
            execute_phase4(settings, prepared, cache, build_dry_run(prepared, cache),
                           generator_factory=lambda _: Generator())

    assert json.loads(audit_path.read_text())["attempts"][0]["status"] == "pending"
    request = prepared["requests"][0]
    resumed_cache = GenerationCache(cache.path)
    saved = resumed_cache.get(request["fingerprint"], request["identity"])
    assert saved is not None
    assert saved["answer"] == "10 years or 100,000 miles"
    assert saved["provider_attempts"] == 1
    pending_audit = audit_path.read_bytes()

    def forbidden(_):
        raise AssertionError("cached success must not construct a generator on resume")

    result = execute_phase4(settings, prepared, resumed_cache,
                           build_dry_run(prepared, resumed_cache), generator_factory=forbidden)
    assert calls == 1
    assert result["run"]["generation"]["provider_attempts"] == 1
    assert audit_path.read_bytes() == pending_audit


def test_durable_audit_io_does_not_shorten_provider_start_spacing(tmp_path, monkeypatch) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=2)
    cache = GenerationCache(tmp_path / "cache.json")
    timer = ManualExecutionClock()
    starts = []
    original_begin = runner.GenerationAttemptAudit.begin

    def slow_first_checkpoint(audit, fingerprint, **kwargs):
        attempt = original_begin(audit, fingerprint, **kwargs)
        if attempt == 1:
            timer.value += 2.0
        return attempt

    class Generator:
        def generate(self, question, chunks):
            starts.append(timer())
            return GeneratedAnswer("10 years or 100,000 miles", None)

    monkeypatch.setattr(runner.GenerationAttemptAudit, "begin", slow_first_checkpoint)
    execute_phase4(settings, prepared, cache, build_dry_run(prepared, cache),
                   generator_factory=lambda _: Generator(), clock=timer, sleeper=timer.sleep)
    assert starts == [2.0, 7.0]


@pytest.mark.parametrize("failures, retry_after, elapsed, remaining", [
    (1, None, 2.0, 3.0),
    (2, None, 2.0, 8.0),
    (3, None, 2.0, 18.0),
    (4, None, 2.0, 38.0),
    (5, None, 2.0, 58.0),
    (1, 17.0, 2.0, 15.0),
    (1, 125.0, 2.0, 58.0),
    (1, 0.0, 1.0, 2.0),
    (2, None, 15.0, 0.0),
])
def test_resume_enforces_remaining_rate_limit_backoff(
    tmp_path, failures, retry_after, elapsed, remaining
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    first_clock = ManualExecutionClock()
    calls = 0

    class RateLimitedGenerator:
        def generate(self, question, chunks):
            nonlocal calls
            calls += 1
            first_clock.value += 2.0
            raise _generation_error(FakeProviderError("SECRET quota", code=429,
                                                       retry_after=retry_after))

    def stop_after_failure(seconds):
        if calls == failures:
            raise KeyboardInterrupt()
        first_clock.sleep(seconds)

    with pytest.raises(KeyboardInterrupt):
        execute_phase4(settings, prepared, cache, build_dry_run(prepared, cache),
                       generator_factory=lambda _: RateLimitedGenerator(),
                       clock=first_clock, sleeper=stop_after_failure,
                       wall_clock=lambda: 1000.0 + first_clock())

    assert calls == failures
    resume_epoch = 1000.0 + first_clock() + elapsed
    resumed_clock = ManualExecutionClock()
    resumed_starts = []

    class CompletingGenerator:
        def generate(self, question, chunks):
            resumed_starts.append(resumed_clock())
            return GeneratedAnswer("10 years or 100,000 miles", None)

    resumed_cache = GenerationCache(cache.path)
    result = execute_phase4(settings, prepared, resumed_cache,
                           build_dry_run(prepared, resumed_cache),
                           generator_factory=lambda _: CompletingGenerator(),
                           clock=resumed_clock, sleeper=resumed_clock.sleep,
                           wall_clock=lambda: resume_epoch + resumed_clock())

    assert resumed_starts == [remaining]
    assert resumed_clock.sleeps == ([remaining] if remaining else [])
    assert result["run"]["generation"]["provider_attempts"] == failures + 1
    assert result["run"]["generation"]["rate_limit_retries"] == failures
    audit_text = (tmp_path / "experiments/results/phase4_generation_attempts.json").read_text()
    assert "SECRET" not in audit_text


def test_repeated_resumes_share_six_attempt_ceiling(tmp_path) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    audit_path = tmp_path / "experiments/results/phase4_generation_attempts.json"
    attempt_times = []
    next_epoch = 1000.0

    for attempt_number in range(1, 7):
        timer = ManualExecutionClock()
        epoch = next_epoch
        process_calls = 0

        class RateLimitedGenerator:
            def generate(self, question, chunks):
                nonlocal process_calls
                process_calls += 1
                attempt_times.append(epoch + timer())
                raise _generation_error(FakeProviderError("quota", code=429))

        def interrupted_sleep(seconds):
            if process_calls:
                raise KeyboardInterrupt()
            timer.sleep(seconds)

        expected = generate.GenerationError if attempt_number == 6 else KeyboardInterrupt
        with pytest.raises((generate.GenerationError, KeyboardInterrupt)) as raised:
            execute_phase4(settings, prepared, GenerationCache(cache.path),
                           build_dry_run(prepared, cache),
                           generator_factory=lambda _: RateLimitedGenerator(),
                           clock=timer, sleeper=interrupted_sleep,
                           wall_clock=lambda: epoch + timer())
        assert isinstance(raised.value, expected)
        assert process_calls == 1
        next_epoch = epoch + timer() + 1.0

    assert attempt_times == [1000.0, 1005.0, 1015.0, 1035.0, 1075.0, 1135.0]
    saved_audit = audit_path.read_bytes()
    attempts = json.loads(saved_audit)["attempts"]
    assert len(attempts) == 6
    assert sum(attempt["rate_limit_retry"] for attempt in attempts) == 5
    request = prepared["requests"][0]
    assert cache.get(request["fingerprint"], request["identity"]) is None

    def forbidden(_):
        raise AssertionError("exhausted fingerprint must not construct a generator")

    for _ in range(3):
        with pytest.raises(generate.GenerationError, match="attempt limit.*audit reset"):
            execute_phase4(settings, prepared, GenerationCache(cache.path),
                           build_dry_run(prepared, cache), generator_factory=forbidden)
        assert audit_path.read_bytes() == saved_audit


@pytest.mark.parametrize("outcome, remaining", [("success", 3.0), ("pending", 5.0)])
def test_resume_preserves_global_spacing_across_fingerprints(
    tmp_path, monkeypatch, outcome, remaining
) -> None:
    settings = replace(make_settings(), project_root=tmp_path)
    first_prepared = _prepared_controls(settings, count=1)
    cache = GenerationCache(tmp_path / "cache.json")
    first_clock = ManualExecutionClock()
    starts = []
    original_begin = runner.GenerationAttemptAudit.begin

    def slow_checkpoint(audit, fingerprint, **kwargs):
        attempt = original_begin(audit, fingerprint, **kwargs)
        first_clock.value += 2.0
        return attempt

    def failed_finish(*args, **kwargs):
        raise OSError("simulated outcome interruption")

    class FirstGenerator:
        def generate(self, question, chunks):
            starts.append(1000.0 + first_clock())
            first_clock.value += 1.0
            return GeneratedAnswer("10 years or 100,000 miles", None)

    with monkeypatch.context() as patch:
        patch.setattr(runner.GenerationAttemptAudit, "begin", slow_checkpoint)
        if outcome == "pending":
            patch.setattr(runner.GenerationAttemptAudit, "finish", failed_finish)
            with pytest.raises(OSError, match="outcome interruption"):
                execute_phase4(settings, first_prepared, cache,
                               build_dry_run(first_prepared, cache),
                               generator_factory=lambda _: FirstGenerator(),
                               clock=first_clock, sleeper=first_clock.sleep,
                               wall_clock=lambda: 1000.0 + first_clock())
        else:
            execute_phase4(settings, first_prepared, cache,
                           build_dry_run(first_prepared, cache),
                           generator_factory=lambda _: FirstGenerator(),
                           clock=first_clock, sleeper=first_clock.sleep,
                           wall_clock=lambda: 1000.0 + first_clock())

    resumed_clock = ManualExecutionClock()
    prepared = _prepared_controls(settings, count=2)

    class SecondGenerator:
        def generate(self, question, chunks):
            starts.append(1004.0 + resumed_clock())
            return GeneratedAnswer("10 years or 100,000 miles", None)

    result = execute_phase4(settings, prepared, GenerationCache(cache.path),
                           build_dry_run(prepared, cache),
                           generator_factory=lambda _: SecondGenerator(),
                           clock=resumed_clock, sleeper=resumed_clock.sleep,
                           wall_clock=lambda: 1004.0 + resumed_clock())
    assert starts == [1002.0, 1004.0 + remaining]
    assert resumed_clock.sleeps == [remaining]
    assert result["run"]["generation"]["provider_attempts"] == 2
