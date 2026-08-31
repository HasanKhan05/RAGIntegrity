from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.rag.config import Settings
from src.rag.index import COLLECTION_NAME
from src.rag.retrieve import Retriever


class QueryEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_retrieve_reranks_exact_query_terms_from_a_larger_candidate_pool(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=gemini\nCHROMA_PERSIST_DIR=vector-store\nTOP_K=1\n",
        encoding="utf-8",
    )
    settings = Settings.from_env(env_file)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.upsert(
        ids=["overview", "specification"],
        documents=["Aygo X city crossover", "Wheelbase (mm) 2430"],
        embeddings=[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]],
        metadatas=[
            {
                "document_id": "aygo",
                "filename": "aygo-x.pdf",
                "page_number": 3,
                "chunk_id": "overview",
            },
            {
                "document_id": "aygo",
                "filename": "aygo-x.pdf",
                "page_number": 23,
                "chunk_id": "specification",
            },
        ],
    )

    results = Retriever(settings, embedder=QueryEmbedder()).retrieve(
        "What is the Aygo X wheelbase?"
    )

    assert [result.chunk_id for result in results] == ["specification"]
    assert results[0].rank == 1
