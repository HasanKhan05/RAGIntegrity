"""Phase 1 API endpoints for health, clean documents, and free-form questions."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from fastapi import FastAPI, HTTPException
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
    application = FastAPI(title="RAG Poisoning Testbed", version="0.1.0")

    @application.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "index_available": _index_available(active_settings),
        }

    @application.get("/documents", response_model=list[DocumentResponse])
    def documents() -> list[dict[str, Any]]:
        return _manifest_documents(active_settings)

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
