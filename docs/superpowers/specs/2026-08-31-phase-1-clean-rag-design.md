# Phase 1 Clean RAG Baseline Design

## Scope

Phase 1 builds a local, understandable clean RAG baseline over official Toyota UK brochure PDFs. It includes document ingestion, fixed-size chunking, local Sentence Transformer embeddings, persistent ChromaDB retrieval, concise Gemini generation, and a minimal FastAPI interface. It does not include synthetic documents, attacks, defenses, aggregate evaluation, or frontend code.

The implementation prioritizes correctness, clarity, and low API usage. It uses no paid embedding service and makes no Gemini call during automated tests.

## Architecture

The system is one Python application with focused modules:

```text
official PDFs in data/clean/
        -> PyMuPDF page extraction
        -> page-bounded text chunks
        -> all-MiniLM-L6-v2 local embeddings
        -> persistent ChromaDB collection
        -> top-k retrieval
        -> concise Gemini prompt
        -> answer, ranked sources, latency, and token usage
```

Indexing is an explicit local command rather than an automatic API startup side effect. The API opens the existing local index and never rebuilds embeddings just because the server restarted.

## Components

### Configuration

`src/rag/config.py` loads the root `.env` and exposes a small immutable settings object. It validates numeric bounds and keeps these defaults:

- `LLM_PROVIDER=gemini`
- `LLM_MODEL=gemini-3.5-flash-lite`
- `MAX_OUTPUT_TOKENS=300`
- `LLM_TEMPERATURE=0`
- `TOP_K=3`
- `CHROMA_PERSIST_DIR=data/vector_store`
- `EMBEDDING_MODEL=all-MiniLM-L6-v2`

The API key is never returned, logged, committed, or included in exception text.

### PDF extraction

`src/rag/ingest.py` reads every `.pdf` file in `data/clean/` with PyMuPDF. Each non-empty page becomes a record containing a deterministic document ID, filename, one-based page number, and normalized text. Empty pages are skipped. Corrupt or unreadable PDFs produce a clear filename-specific error and do not silently create a partial index.

### Chunking

`src/rag/chunk.py` splits each page independently into approximately 1,200-character chunks with 200 characters of overlap. It prefers nearby whitespace boundaries while remaining deterministic. Chunks never span pages, so every citation retains an exact page number. Each chunk carries document ID, filename, page number, and deterministic chunk ID.

### Embeddings and indexing

`src/rag/embed.py` wraps `sentence-transformers` using `all-MiniLM-L6-v2` locally. `src/rag/index.py` orchestrates extraction, chunking, embedding, and ChromaDB persistence.

The indexer records a small JSON manifest under `data/manifests/` containing source filenames, SHA-256 hashes, embedding model, and chunk settings. If those values match the existing manifest and Chroma collection, indexing reports that the corpus is unchanged and reuses the store. If they differ, it rebuilds the clean collection deterministically. No API key or hidden poison label is stored in metadata.

### Retrieval

`src/rag/retrieve.py` embeds the free-form question locally and queries ChromaDB with the configured `top_k`. Results are returned in rank order with document ID, filename, page number, chunk ID, relevance score when available, and the retrieved text. Retrieval has no dependency on Gemini generation.

### Gemini generation

`src/rag/generate.py` uses the official `google-genai` Python SDK and only the configured Gemini model. It sends a short instruction, the question, and only the retrieved chunks. The prompt requires an answer grounded in context, an explicit insufficient-context response when needed, and simple `[filename, p. N]` references.

Generation uses temperature `0` and at most 300 output tokens. It returns provider-reported token counts when available. A missing key, model, or unsupported provider produces a safe configuration error before any network call.

### API

`src/api/main.py` provides only:

- `GET /health`: process health and whether a local index is available.
- `GET /documents`: clean source metadata from the current index manifest.
- `POST /ask`: validates a non-empty free-form question, retrieves top-k chunks, makes one Gemini call, and returns the answer, ranked sources, total latency, and optional token usage.

Missing configuration or an absent index is reported as a clear service-unavailable response. Invalid questions receive a validation response. No authentication, users, middleware stack, or production infrastructure is added.

## Data Flow

1. The owner places official manufacturer brochure PDFs in `data/clean/`.
2. `python -m src.rag.index` computes the corpus fingerprint.
3. If unchanged, the existing local Chroma collection is reused.
4. If changed, text is extracted and chunked, embeddings are generated locally, and the clean collection plus manifest are replaced.
5. `/ask` embeds one question locally and retrieves the top three chunks.
6. One concise Gemini request uses those chunks as context.
7. The response includes the answer, source ranks and metadata, latency, and token usage if Gemini reports it.

## Error Handling

- No PDFs: indexing exits with instructions to add official brochures; it does not download substitutes.
- Unreadable PDF: indexing names the file and fails without claiming success.
- Empty extracted corpus: indexing fails before creating a usable manifest.
- Missing local model or dependency: the command reports the missing requirement; it does not fall back to paid embeddings.
- Missing Chroma index: retrieval and `/ask` explain that indexing must run first.
- Missing Gemini configuration: generation fails safely without exposing the key.
- Gemini network, quota, or provider error: `/ask` returns a concise provider failure and does not retry repeatedly.

## Testing and QA

Development follows red-green-refactor. Automated tests use generated temporary PDFs and deterministic test embeddings, so they require no official brochure and make no paid calls.

Tests cover:

- extraction of text and one-based page metadata;
- skipping empty pages and rejecting unreadable PDFs;
- deterministic chunking, overlap, and metadata preservation;
- index manifest change detection;
- ranked retrieval response structure using a temporary Chroma store;
- `/health`, `/documents`, and `/ask` validation behavior;
- safe missing-index and missing-Gemini-configuration behavior;
- Gemini response and token-usage mapping through an injected fake client.

Manual Phase 1 QA requires Python 3.13, official brochures, the downloaded local embedding model, and the configured Gemini key. It will index the real brochures, inspect retrieval for roughly three to five factual questions, and make only a handful of live Gemini calls. Phase 1 remains `Blocked` until that QA, GitHub connection, commit, and push succeed.

## External Requirements and Completion Boundary

The current workspace still requires:

- Python 3.13 and a project virtual environment;
- official Toyota UK brochure PDFs supplied by the user;
- GitHub repository creation/authentication and an `origin` remote;
- one successful low-volume Gemini smoke test.

Code and offline tests may be completed before every external item is present, but `PHASE_STATUS.md` must say `Blocked` until every Phase 1 completion criterion is met. Work stops after Phase 1 under all circumstances.
