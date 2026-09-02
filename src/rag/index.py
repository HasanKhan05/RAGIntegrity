"""Build and reuse persistent local Chroma indexes for brochure corpora."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.errors import NotFoundError

from src.rag.chunk import chunk_pages
from src.rag.config import Settings
from src.rag.embed import Embedder, SentenceTransformerEmbedder
from src.rag.ingest import load_pdfs


CLEAN_COLLECTION_NAME = "clean_brochures"
ATTACKED_COLLECTION_NAME = "attacked_brochures"
COLLECTION_NAME = CLEAN_COLLECTION_NAME


class NoCleanPdfsError(RuntimeError):
    """Raised when the clean corpus does not contain official PDFs."""


class NoPoisonedPdfsError(RuntimeError):
    """Raised when the attacked corpus lacks synthetic PDFs."""


class EmptyCleanCorpusError(RuntimeError):
    """Raised when PDFs exist but yield no readable text."""


@dataclass(frozen=True)
class IndexResult:
    reused: bool
    document_count: int
    page_count: int
    chunk_count: int
    document_ids: tuple[str, ...]
    manifest_path: Path


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_corpus_fingerprint(
    pdf_paths: Sequence[Path], settings: Settings
) -> dict[str, Any]:
    sources = [
        {"filename": path.name, "sha256": _file_sha256(path)}
        for path in sorted(pdf_paths, key=lambda item: item.name.lower())
    ]
    return {
        "schema_version": 2,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "sources": sources,
    }


def _discover_pdfs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.lower(),
    )


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _manifest_matches(
    manifest: dict[str, Any] | None, fingerprint: dict[str, Any]
) -> bool:
    if manifest is None:
        return False
    return all(manifest.get(key) == value for key, value in fingerprint.items())


def _persistent_client(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _index_corpus(
    settings: Settings,
    *,
    pdf_paths: Sequence[Path],
    pages: Sequence[Any],
    collection_name: str,
    manifest_path: Path,
    embedder: Embedder | None,
    force: bool,
) -> IndexResult:
    fingerprint = build_corpus_fingerprint(pdf_paths, settings)
    manifest = _read_manifest(manifest_path)
    client = _persistent_client(settings.chroma_persist_dir)
    try:
        existing_collection = client.get_collection(collection_name)
    except NotFoundError:
        existing_collection = None

    if (
        not force
        and _manifest_matches(manifest, fingerprint)
        and existing_collection is not None
        and existing_collection.count() == int(manifest.get("chunk_count", -1))
        and existing_collection.count() > 0
    ):
        documents = manifest.get("documents", []) if manifest else []
        return IndexResult(
            reused=True,
            document_count=len(documents),
            page_count=int(manifest.get("page_count", 0)),
            chunk_count=int(manifest.get("chunk_count", 0)),
            document_ids=tuple(item["document_id"] for item in documents),
            manifest_path=manifest_path,
        )

    if not pages:
        raise EmptyCleanCorpusError("Brochure PDFs contained no readable text")
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise EmptyCleanCorpusError("Brochure PDFs produced no text chunks")

    active_embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    embeddings = active_embedder.encode(
        [f"{chunk.filename}\n{chunk.text}" for chunk in chunks]
    )

    if existing_collection is not None:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ],
    )

    pages_by_document: dict[tuple[str, str], set[int]] = defaultdict(set)
    for page in pages:
        pages_by_document[(page.document_id, page.filename)].add(page.page_number)
    documents = [
        {
            "document_id": document_id,
            "filename": filename,
            "page_count": len(page_numbers),
        }
        for (document_id, filename), page_numbers in sorted(
            pages_by_document.items(), key=lambda item: item[0][1].lower()
        )
    ]
    completed_manifest = {
        **fingerprint,
        "documents": documents,
        "page_count": len(pages),
        "chunk_count": len(chunks),
    }
    _write_manifest(manifest_path, completed_manifest)

    return IndexResult(
        reused=False,
        document_count=len(documents),
        page_count=len(pages),
        chunk_count=len(chunks),
        document_ids=tuple(item["document_id"] for item in documents),
        manifest_path=manifest_path,
    )


def index_clean_corpus(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    force: bool = False,
) -> IndexResult:
    """Build the clean Chroma collection, or reuse it when inputs match."""

    pdf_paths = _discover_pdfs(settings.clean_data_dir)
    if not pdf_paths:
        raise NoCleanPdfsError(
            "Add official brochure PDFs to data/clean before indexing"
        )
    return _index_corpus(
        settings,
        pdf_paths=pdf_paths,
        pages=load_pdfs(settings.clean_data_dir),
        collection_name=CLEAN_COLLECTION_NAME,
        manifest_path=settings.manifest_path,
        embedder=embedder,
        force=force,
    )


def index_attacked_corpus(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    force: bool = False,
) -> IndexResult:
    """Build an isolated collection from clean and synthetic PDF inputs."""

    clean_paths = _discover_pdfs(settings.clean_data_dir)
    if not clean_paths:
        raise NoCleanPdfsError(
            "Add official brochure PDFs to data/clean before indexing"
        )
    poisoned_paths = _discover_pdfs(settings.poisoned_data_dir)
    if not poisoned_paths:
        raise NoPoisonedPdfsError(
            "Add synthetic PDFs to data/poisoned before indexing"
        )
    duplicate_filenames = sorted(
        {path.name.casefold() for path in clean_paths}
        & {path.name.casefold() for path in poisoned_paths}
    )
    if duplicate_filenames:
        raise ValueError(
            "Clean and synthetic PDFs must use unique filenames: "
            + ", ".join(duplicate_filenames)
        )
    return _index_corpus(
        settings,
        pdf_paths=[*clean_paths, *poisoned_paths],
        pages=[
            *load_pdfs(settings.clean_data_dir),
            *load_pdfs(settings.poisoned_data_dir),
        ],
        collection_name=ATTACKED_COLLECTION_NAME,
        manifest_path=settings.attacked_manifest_path,
        embedder=embedder,
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local brochure PDFs")
    parser.add_argument(
        "--force", action="store_true", help="Rebuild the selected index"
    )
    parser.add_argument("--corpus", choices=("clean", "attacked"), default="clean")
    arguments = parser.parse_args()
    try:
        indexer = (
            index_clean_corpus if arguments.corpus == "clean" else index_attacked_corpus
        )
        result = indexer(Settings.from_env(), force=arguments.force)
    except (NoCleanPdfsError, NoPoisonedPdfsError, EmptyCleanCorpusError) as error:
        parser.exit(1, f"Indexing stopped: {error}\n")
    action = "Reused" if result.reused else "Built"
    print(
        f"{action} {arguments.corpus} index: {result.document_count} documents, "
        f"{result.page_count} pages, {result.chunk_count} chunks"
    )


if __name__ == "__main__":
    main()
