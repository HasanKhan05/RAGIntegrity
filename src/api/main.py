"""Phase 1 API endpoints for health, clean documents, and free-form questions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src.rag.config import ConfigurationError, Settings
from src.rag.defenses import (
    DefenseCoordinator,
    DefenseMode,
    DefenseTraceEntry,
    load_trusted_filenames,
)
from src.rag.embed import SentenceTransformerEmbedder
from src.rag.generate import GenerationError, GeminiGenerator
from src.rag.index import ATTACKED_COLLECTION_NAME, CLEAN_COLLECTION_NAME
from src.rag.retrieve import IndexUnavailableError, Retriever


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    corpus_mode: Literal["clean", "attacked"] = "clean"
    defense_mode: DefenseMode = DefenseMode.NONE

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int


class SourceResponse(BaseModel):
    rank: int
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    relevance_score: float | None
    text: str
    is_injected_test_document: bool = False


class TokenUsageResponse(BaseModel):
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class DefenseStageDecisionResponse(BaseModel):
    stage: str
    included: bool
    reason: str | None


class DefenseTraceResponse(BaseModel):
    original_rank: int
    filename: str
    page_number: int
    chunk_id: str
    included: bool
    stage_decisions: list[DefenseStageDecisionResponse]
    final_rank: int | None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    latency_ms: float
    token_usage: TokenUsageResponse | None
    defense_mode: DefenseMode
    defense_latency_ms: float
    defense_trace: list[DefenseTraceResponse]


def _manifest_documents(settings: Settings) -> list[dict[str, Any]]:
    if not settings.manifest_path.exists():
        return []
    try:
        manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    documents = manifest.get("documents", [])
    return documents if isinstance(documents, list) else []


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _synthetic_document_ids(settings: Settings) -> frozenset[str]:
    attacks = _read_json_object(settings.attack_manifest_path).get("attacks", [])
    if not isinstance(attacks, list):
        return frozenset()
    return frozenset(
        str(attack["synthetic_document_id"])
        for attack in attacks
        if isinstance(attack, dict) and attack.get("synthetic_document_id")
    )


def _clean_display_name(filename: str) -> str:
    model_names = {
        "aygo-x": "Aygo X", "bz4x": "bZ4X", "c-hr": "C-HR",
        "corolla": "Corolla", "land-cruiser": "Land Cruiser",
        "rav4": "RAV4", "yaris": "Yaris",
    }
    model = model_names.get(Path(filename).stem.lower(), Path(filename).stem.title())
    return f"Toyota {model} — official brochure"


def _synthetic_display_name(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _document_catalog(settings: Settings) -> dict[str, list[dict[str, Any]]]:
    clean_documents = _manifest_documents(settings)
    clean_ids = {str(item.get("document_id")) for item in clean_documents}
    attacked_documents = _read_json_object(settings.attacked_manifest_path).get("documents", [])
    if not isinstance(attacked_documents, list):
        attacked_documents = []

    def item(document: dict[str, Any], *, synthetic: bool) -> dict[str, Any]:
        filename = str(document.get("filename", ""))
        return {
            "document_id": str(document.get("document_id", "")),
            "filename": filename,
            "page_count": int(document.get("page_count", 0)),
            "display_name": _synthetic_display_name(filename) if synthetic else _clean_display_name(filename),
            "pdf_url": f"/documents/file/{'synthetic' if synthetic else 'clean'}/{filename}",
        }

    synthetic_documents = [
        document for document in attacked_documents
        if isinstance(document, dict) and str(document.get("document_id")) not in clean_ids
    ]
    return {
        "official_clean": [item(document, synthetic=False) for document in clean_documents if isinstance(document, dict)],
        "synthetic_test": [item(document, synthetic=True) for document in synthetic_documents],
    }


def _frontend_results_summary(settings: Settings) -> dict[str, Any]:
    results_dir = settings.project_root / "experiments" / "results"
    summary = _read_json_object(results_dir / "phase4_summary.json")
    cache = _read_json_object(results_dir / "phase4_generation_cache.json")
    attack_metrics = summary.get("attack_metrics", {})
    control_metrics = summary.get("clean_control_metrics", {})
    required_modes = ("clean_none", "attacked_none", "source_trust", "instruction_filter", "similarity_filter", "combined")
    if not all(isinstance(attack_metrics.get(mode), dict) for mode in required_modes):
        raise HTTPException(status_code=503, detail="Saved Phase 4 results are unavailable")
    entries = cache.get("entries", {})
    cache_entries = list(entries.values()) if isinstance(entries, dict) else []
    provider_calls = sum(int(entry.get("provider_attempts", 0)) for entry in cache_entries if isinstance(entry, dict))
    total_tokens = sum(int((entry.get("token_usage") or {}).get("total_tokens") or 0) for entry in cache_entries if isinstance(entry, dict))
    attacked_none = attack_metrics["attacked_none"]
    combined = attack_metrics["combined"]
    return {
        "benchmark": summary.get("benchmark", {}),
        "clean_answer_quality_rate": attack_metrics["clean_none"]["answer_accuracy"],
        "clean_answer_correct": attack_metrics["clean_none"]["correct_answers"],
        "clean_answer_total": attack_metrics["clean_none"]["question_count"],
        "retrieval_attack_success_rate": summary.get("retrieval", {}).get(
            "attacked_target_poison_retrieval_rate"
        ),
        "undefended_attack_success_rate": attacked_none["overall_attack_success_rate"],
        "conditional_attack_success_rate": attacked_none["conditional_attack_success_rate"],
        "selected_defense": "combined",
        "selected_defense_attack_success_rate": combined["overall_attack_success_rate"],
        "average_extra_latency_ms": round(combined["average_defense_latency_ms"] - attacked_none["average_defense_latency_ms"], 2),
        "clean_control_accuracy_rate": control_metrics.get("combined", {}).get("answer_accuracy"),
        "defense_comparison": {mode: attack_metrics[mode]["overall_attack_success_rate"] for mode in ("attacked_none", "source_trust", "instruction_filter", "similarity_filter", "combined")},
        "selected_defense_tradeoffs": {
            "poison_removal_rate": combined["poison_removal_rate"],
            "poison_survival_rate": combined["poison_survival_rate"],
            "clean_false_rejection_rate": combined["clean_false_rejection_rate"],
        },
        "generation_usage": {"provider_calls": provider_calls, "total_tokens": total_tokens},
    }


def _index_available(settings: Settings) -> bool:
    if not settings.manifest_path.exists() or not settings.chroma_persist_dir.exists():
        return False
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        return client.get_collection(CLEAN_COLLECTION_NAME).count() > 0
    except NotFoundError:
        return False


def _public_defense_trace(entry: DefenseTraceEntry) -> DefenseTraceResponse:
    return DefenseTraceResponse(
        original_rank=entry.original_rank,
        filename=entry.filename,
        page_number=entry.page_number,
        chunk_id=entry.chunk_id,
        included=entry.included,
        stage_decisions=[
            DefenseStageDecisionResponse(
                stage=decision.stage,
                included=decision.included,
                reason=decision.reason,
            )
            for decision in entry.stage_decisions
        ],
        final_rank=entry.final_rank,
    )


def create_app(
    *,
    settings: Settings | None = None,
    retriever: Any | None = None,
    attacked_retriever: Any | None = None,
    generator: Any | None = None,
    defense_coordinator: Any | None = None,
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    active_retriever = retriever or Retriever(
        active_settings, collection_name=CLEAN_COLLECTION_NAME
    )
    active_attacked_retriever = attacked_retriever or Retriever(
        active_settings, collection_name=ATTACKED_COLLECTION_NAME
    )
    active_generator = generator or GeminiGenerator(active_settings)
    trusted_filenames = (
        load_trusted_filenames(active_settings.manifest_path)
        if active_settings.manifest_path.exists()
        else frozenset()
    )
    active_defense_coordinator = defense_coordinator or DefenseCoordinator(
        trusted_filenames=trusted_filenames,
        similarity_threshold=active_settings.defense_similarity_threshold,
        embedder=SentenceTransformerEmbedder(active_settings.embedding_model),
    )
    application = FastAPI(title="RAGIntegrity — Evaluating Retrieval Poisoning Attacks and Defenses", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    injected_document_ids = _synthetic_document_ids(active_settings)

    @application.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "index_available": _index_available(active_settings),
        }

    @application.get("/documents", response_model=list[DocumentResponse])
    def documents() -> list[dict[str, Any]]:
        return _manifest_documents(active_settings)

    @application.get("/documents/catalog")
    def document_catalog() -> dict[str, list[dict[str, Any]]]:
        return _document_catalog(active_settings)

    @application.get("/documents/file/{collection}/{filename}")
    def document_file(
        collection: Literal["clean", "synthetic"], filename: str
    ) -> FileResponse:
        catalog = _document_catalog(active_settings)
        key = "official_clean" if collection == "clean" else "synthetic_test"
        allowed_filenames = {item["filename"] for item in catalog[key]}
        if filename not in allowed_filenames or Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Document not found")
        directory = active_settings.clean_data_dir if collection == "clean" else active_settings.poisoned_data_dir
        document_path = directory / filename
        if not document_path.is_file():
            raise HTTPException(status_code=404, detail="Document not found")
        return FileResponse(document_path, media_type="application/pdf", filename=filename)

    @application.get("/results/summary")
    def results_summary() -> dict[str, Any]:
        return _frontend_results_summary(active_settings)

    @application.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        try:
            selected_retriever = (
                active_retriever
                if request.corpus_mode == "clean"
                else active_attacked_retriever
            )
            chunks = selected_retriever.retrieve(request.question)
            defense_started = time.perf_counter()
            defense_result = active_defense_coordinator.apply(
                tuple(chunks), request.defense_mode
            )
            defense_latency_ms = round(
                (time.perf_counter() - defense_started) * 1000, 2
            )
            generated = active_generator.generate(
                request.question, defense_result.chunks
            )
        except IndexUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ConfigurationError as error:
            raise HTTPException(
                status_code=503, detail="Gemini generation is not configured"
            ) from error
        except GenerationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        token_usage = None
        if generated.token_usage is not None:
            token_usage = TokenUsageResponse(
                input_tokens=generated.token_usage.input_tokens,
                output_tokens=generated.token_usage.output_tokens,
                total_tokens=generated.token_usage.total_tokens,
            )
        return AskResponse(
            answer=generated.text,
            sources=[
                SourceResponse(
                    rank=chunk.rank,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    relevance_score=chunk.relevance_score,
                    text=chunk.text,
                    is_injected_test_document=chunk.document_id in injected_document_ids,
                )
                for chunk in defense_result.chunks
            ],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            token_usage=token_usage,
            defense_mode=request.defense_mode,
            defense_latency_ms=defense_latency_ms,
            defense_trace=[
                _public_defense_trace(entry) for entry in defense_result.trace
            ],
        )

    return application


app = create_app()
