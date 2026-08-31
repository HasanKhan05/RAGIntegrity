"""Rank clean brochure chunks independently from answer generation."""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from src.rag.config import Settings
from src.rag.embed import Embedder, SentenceTransformerEmbedder
from src.rag.index import COLLECTION_NAME
from src.rag.models import RetrievedChunk


class IndexUnavailableError(RuntimeError):
    """Raised when retrieval is attempted before indexing."""


class Retriever:
    def __init__(self, settings: Settings, *, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or SentenceTransformerEmbedder(
            settings.embedding_model
        )

    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be blank")
        result_limit = self.settings.top_k if top_k is None else top_k
        if result_limit <= 0:
            raise ValueError("top_k must be positive")

        client = chromadb.PersistentClient(
            path=str(self.settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except NotFoundError as error:
            raise IndexUnavailableError(
                "Clean index is unavailable; run python -m src.rag.index first"
            ) from error
        if collection.count() == 0:
            raise IndexUnavailableError(
                "Clean index is empty; run python -m src.rag.index first"
            )

        query_embedding = self.embedder.encode([normalized_question])[0]
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(result_limit, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        ):
            results.append(
                RetrievedChunk(
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    page_number=int(metadata["page_number"]),
                    chunk_id=str(chunk_id),
                    text=str(text),
                    rank=rank,
                    relevance_score=1.0 - float(distance),
                )
            )
        return results
