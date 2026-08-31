# Phase 1 Clean RAG Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a simple clean RAG baseline that indexes official Toyota UK brochure PDFs locally, retrieves ranked chunks, and answers free-form questions with a concise Gemini response.

**Architecture:** A small Python package separates configuration, page extraction, page-bounded chunking, local embedding/indexing, retrieval, Gemini generation, and FastAPI transport. Indexing is an explicit command with a corpus fingerprint so unchanged brochures reuse the persistent ChromaDB collection; automated tests use generated PDFs and fakes and never call Gemini.

**Tech Stack:** Python 3.13, FastAPI, Uvicorn, Pydantic, python-dotenv, PyMuPDF, sentence-transformers, ChromaDB, NumPy, pandas, google-genai, httpx, pytest

**Spec:** `docs/superpowers/specs/2026-08-31-phase-1-clean-rag-design.md`

## Global Constraints

- Implement Phase 1 only; do not create poisoning, defense, evaluation, or frontend behavior.
- Use only official manufacturer brochure PDFs supplied by the user as the real clean corpus.
- Keep embeddings and vector storage local; never use a paid embedding API.
- Use `all-MiniLM-L6-v2`, `TOP_K=3`, `LLM_TEMPERATURE=0`, and `MAX_OUTPUT_TOKENS=300` by default.
- Never print, log, return, or commit `LLM_API_KEY`; `.env` must remain ignored.
- Automated tests must make zero network calls and zero Gemini calls.
- Manual QA may make only three to five live Gemini calls.
- Keep modules explicit and small; add no production infrastructure or multi-provider abstraction.
- Mark Phase 1 `Blocked` unless Python 3.13, official PDFs, real indexing/retrieval, Gemini smoke QA, GitHub remote, commit, and push all succeed.

## File Structure

- `requirements.txt`: minimal Phase 1 dependencies.
- `src/rag/models.py`: shared immutable page, chunk, retrieval, generation, and token-usage records.
- `src/rag/config.py`: environment loading and validation without secret exposure.
- `src/rag/ingest.py`: PDF discovery and PyMuPDF page extraction.
- `src/rag/chunk.py`: deterministic page-bounded fixed-size chunking.
- `src/rag/embed.py`: local Sentence Transformer adapter.
- `src/rag/index.py`: fingerprint, manifest, Chroma collection, and indexing CLI.
- `src/rag/retrieve.py`: ranked Chroma retrieval independent of generation.
- `src/rag/generate.py`: Gemini-only prompt construction and response mapping.
- `src/api/main.py`: health, documents, and ask endpoints plus dependency wiring.
- `tests/`: behavior-focused tests and generated temporary fixtures.

---

### Task 1: Runtime Configuration and Phase 1 Scaffold

**Files:**
- Create: `requirements.txt`
- Modify: `.env.example`
- Create: `src/__init__.py`
- Create: `src/rag/__init__.py`
- Create: `src/rag/config.py`
- Create: `src/rag/models.py`
- Create: `tests/test_config.py`
- Create: `data/clean/.gitkeep`
- Create: `data/manifests/.gitkeep`

**Interfaces:**
- Produces: `Settings.from_env(env_file: Path | None = None) -> Settings`
- Produces: `Settings.require_generation() -> None`
- Produces: shared frozen dataclasses `PageText`, `TextChunk`, `RetrievedChunk`, `TokenUsage`, and `GeneratedAnswer`

- [ ] **Step 1: Verify Python 3.13 and create the virtual environment**

Run:

```powershell
python --version
git --version
node --version
gh --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Expected: Python reports `3.13.x`, Git and Node report installed versions, GitHub CLI availability is recorded, and `.venv\Scripts\python.exe` exists. If Python is absent, install the official 64-bit Python 3.13 distribution before continuing.

- [ ] **Step 2: Add minimal dependencies and install them**

Create `requirements.txt` with:

```text
fastapi
uvicorn
pydantic
python-dotenv
pymupdf
sentence-transformers
chromadb
numpy
pandas
google-genai
httpx
pytest
```

Run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: installation exits `0` without adding CUDA-specific packages manually.

- [ ] **Step 3: Write failing configuration tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from src.rag.config import ConfigurationError, Settings


def test_settings_load_safe_defaults(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=gemini\nLLM_MODEL=gemini-3.5-flash-lite\n")

    settings = Settings.from_env(env_file)

    assert settings.top_k == 3
    assert settings.max_output_tokens == 300
    assert settings.llm_temperature == 0
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200


def test_generation_requires_key_without_exposing_secret(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=gemini\nLLM_MODEL=gemini-3.5-flash-lite\n")
    settings = Settings.from_env(env_file)

    with pytest.raises(ConfigurationError, match="LLM_API_KEY") as error:
        settings.require_generation()

    assert "AIza" not in str(error.value)


def test_only_gemini_provider_is_supported(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=other\nLLM_API_KEY=secret\nLLM_MODEL=model\n")

    with pytest.raises(ConfigurationError, match="gemini"):
        Settings.from_env(env_file)
```

- [ ] **Step 4: Run tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_config.py -q
```

Expected: collection fails because `src.rag.config` does not exist.

- [ ] **Step 5: Implement minimal settings and shared models**

Implement `Settings` as a frozen dataclass that reads `dotenv_values`, converts `TOP_K`, `MAX_OUTPUT_TOKENS`, `LLM_TEMPERATURE`, `CHUNK_SIZE`, and `CHUNK_OVERLAP`, validates positive sizes, `chunk_overlap < chunk_size`, and `0 <= temperature <= 2`, accepts only `gemini`, and stores the key only in memory. Add `CHUNK_SIZE=1200` and `CHUNK_OVERLAP=200` to `.env.example`. Define the shared frozen records with these fields:

```python
@dataclass(frozen=True)
class PageText:
    document_id: str
    filename: str
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk(TextChunk):
    rank: int
    relevance_score: float | None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    token_usage: TokenUsage | None
```

- [ ] **Step 6: Run tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q`

Expected: `3 passed`.

- [ ] **Step 7: Commit**

```powershell
git add requirements.txt .env.example src tests/test_config.py data/clean/.gitkeep data/manifests/.gitkeep
git commit -m "chore: add phase 1 runtime configuration"
```

---

### Task 2: PDF Extraction and Page-Bounded Chunking

**Files:**
- Create: `src/rag/ingest.py`
- Create: `src/rag/chunk.py`
- Create: `tests/test_ingest.py`
- Create: `tests/test_chunk.py`

**Interfaces:**
- Consumes: `PageText`, `TextChunk`
- Produces: `extract_pdf(path: Path) -> list[PageText]`
- Produces: `load_clean_pdfs(directory: Path) -> list[PageText]`
- Produces: `chunk_pages(pages: Iterable[PageText], chunk_size: int = 1200, overlap: int = 200) -> list[TextChunk]`

- [ ] **Step 1: Read the test-quality rules before editing tests**

Read `superpowers/test-driven-development/writing-good-tests.md` completely and name the production behavior each test would catch.

- [ ] **Step 2: Write failing PDF extraction tests**

Create a two-page temporary PDF with PyMuPDF in `tests/test_ingest.py`, leaving page two empty. Assert that `extract_pdf` returns one page, uses one-based page `1`, preserves the filename, and derives the same document ID on repeated calls. Add a second test asserting an unreadable `.pdf` raises `PdfExtractionError` containing the filename.

- [ ] **Step 3: Run extraction tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingest.py -q`

Expected: import failure because `src.rag.ingest` is absent.

- [ ] **Step 4: Implement extraction minimally**

Use `fitz.open(path)`, normalize runs of whitespace with `" ".join(text.split())`, skip blank pages, and compute `document_id` as `sha256(path.name.encode("utf-8")).hexdigest()[:16]`. Wrap PyMuPDF open/read failures in `PdfExtractionError(f"Could not read PDF: {path.name}")` without swallowing the original exception chain.

- [ ] **Step 5: Verify extraction GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingest.py -q`

Expected: all extraction tests pass.

- [ ] **Step 6: Write failing chunk tests**

Test that a 2,600-character page produces multiple chunks, adjacent chunks overlap, IDs are deterministic, metadata is preserved, and no chunk contains text from a second page. Test invalid values: `chunk_size <= 0`, `overlap < 0`, and `overlap >= chunk_size` raise `ValueError`.

- [ ] **Step 7: Run chunk tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chunk.py -q`

Expected: import failure because `src.rag.chunk` is absent.

- [ ] **Step 8: Implement deterministic chunking**

Advance by `chunk_size - overlap`, prefer the last whitespace within the final 15% of the target window, strip each chunk, and build IDs as `f"{document_id}-p{page_number}-c{chunk_index}"`. Do not merge pages.

- [ ] **Step 9: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ingest.py tests/test_chunk.py -q`

Expected: all tests pass.

```powershell
git add src/rag/ingest.py src/rag/chunk.py tests/test_ingest.py tests/test_chunk.py
git commit -m "feat: extract and chunk clean brochures"
```

---

### Task 3: Local Embeddings, Corpus Fingerprint, and Persistent Chroma Index

**Files:**
- Create: `src/rag/embed.py`
- Create: `src/rag/index.py`
- Create: `tests/test_index.py`
- Create: `data/vector_store/.gitkeep`

**Interfaces:**
- Consumes: `Settings`, `TextChunk`, `load_clean_pdfs`, `chunk_pages`
- Produces: `SentenceTransformerEmbedder.encode(texts: Sequence[str]) -> list[list[float]]`
- Produces: `build_corpus_fingerprint(pdf_paths: Sequence[Path], settings: Settings) -> dict[str, object]`
- Produces: `index_clean_corpus(settings: Settings, force: bool = False) -> IndexResult`
- Produces: CLI `python -m src.rag.index [--force]`

- [ ] **Step 1: Write failing pure fingerprint tests**

In `tests/test_index.py`, create temporary `.pdf` byte files and assert that the fingerprint is stable for identical filename/content/config, independent of discovery order, and changes when bytes, embedding model, chunk size, or overlap changes.

- [ ] **Step 2: Run fingerprint tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_index.py -q`

Expected: import failure because `src.rag.index` is absent.

- [ ] **Step 3: Implement fingerprint and manifest helpers**

Hash each PDF with SHA-256 using buffered reads. Return a JSON-serializable object sorted by lowercase filename:

```python
{
    "schema_version": 1,
    "embedding_model": settings.embedding_model,
    "chunk_size": settings.chunk_size,
    "chunk_overlap": settings.chunk_overlap,
    "sources": [{"filename": path.name, "sha256": digest}, ...],
}
```

Write JSON atomically through a sibling temporary file and `Path.replace`. The persisted manifest extends the fingerprint with a `documents` list containing each deterministic document ID, filename, and extracted page count so `/documents` never needs to inspect Chroma internals.

- [ ] **Step 4: Verify fingerprint GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_index.py -q`

Expected: fingerprint tests pass.

- [ ] **Step 5: Write failing index orchestration tests**

Inject a fake embedder and fake collection factory. Assert an empty clean directory raises `NoCleanPdfsError`; an unchanged manifest returns `reused=True` without calling `encode`; a changed corpus clears/upserts the clean collection with IDs, documents, embeddings, and only RAG-visible metadata; and the manifest contains no API key or poison field.

- [ ] **Step 6: Run orchestration tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_index.py -q`

Expected: new orchestration assertions fail because indexing is not implemented.

- [ ] **Step 7: Implement local embedder and Chroma indexing**

Lazily import and instantiate `SentenceTransformer(settings.embedding_model)`. Use `model.encode(texts, normalize_embeddings=True, show_progress_bar=False)` and convert the NumPy result with `.tolist()`. Use `chromadb.PersistentClient(path=str(settings.chroma_persist_dir))`, collection name `clean_brochures`, and cosine space metadata. Store filename, document ID, page number, and chunk ID only.

If the manifest equals the new fingerprint and the collection has records, return reuse without embedding. Otherwise delete/recreate only `clean_brochures`, upsert the new chunks, then write the manifest.

- [ ] **Step 8: Verify index GREEN and CLI missing-PDF behavior**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_index.py -q
.venv\Scripts\python.exe -m src.rag.index
```

Expected: tests pass; CLI exits non-zero with a clear request for official PDFs because `data/clean/` is currently empty.

- [ ] **Step 9: Commit**

```powershell
git add src/rag/embed.py src/rag/index.py tests/test_index.py data/vector_store/.gitkeep
git commit -m "feat: index clean brochures in local chroma"
```

---

### Task 4: Ranked Retrieval Independent of Generation

**Files:**
- Create: `src/rag/retrieve.py`
- Create: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `Settings`, `SentenceTransformerEmbedder`, `RetrievedChunk`
- Produces: `Retriever.retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]`

- [ ] **Step 1: Write failing retrieval tests**

Use a fake embedder returning one query vector and a fake collection returning Chroma-shaped nested `ids`, `documents`, `metadatas`, and `distances`. Assert whitespace-only questions raise `ValueError`; results preserve the returned order; ranks are one-based; metadata maps exactly; `top_k` defaults to settings; and cosine distance `0.2` maps to relevance `0.8`.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_retrieve.py -q`

Expected: import failure because `src.rag.retrieve` is absent.

- [ ] **Step 3: Implement minimal retrieval**

Encode only the stripped question, call collection query with `n_results`, request documents/metadatas/distances, zip the first result lists, and construct `RetrievedChunk` records. Return an empty list if Chroma returns no IDs. Raise `IndexUnavailableError` if the clean collection is missing or empty.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_retrieve.py -q`

Expected: all retrieval tests pass.

```powershell
git add src/rag/retrieve.py tests/test_retrieve.py
git commit -m "feat: return ranked clean rag retrieval"
```

---

### Task 5: Concise Gemini Generation and Token Mapping

**Files:**
- Create: `src/rag/generate.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Consumes: `Settings`, `RetrievedChunk`, `GeneratedAnswer`, `TokenUsage`
- Produces: `build_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str`
- Produces: `GeminiGenerator.generate(question: str, chunks: Sequence[RetrievedChunk]) -> GeneratedAnswer`

- [ ] **Step 1: Write failing prompt and response tests**

Test that the prompt contains the question, only retrieved chunk text, and `[filename, p. N]` labels; requests an insufficient-context response; and excludes the API key. Inject a fake Google client whose `models.generate_content` records arguments and returns text plus usage metadata. Assert model, temperature `0`, maximum `300`, answer text, and input/output/total token mapping.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generate.py -q`

Expected: import failure because `src.rag.generate` is absent.

- [ ] **Step 3: Implement Gemini-only generation**

Construct `genai.Client(api_key=settings.llm_api_key)` only after `settings.require_generation()`. Call:

```python
response = client.models.generate_content(
    model=settings.llm_model,
    contents=build_prompt(question, chunks),
    config=types.GenerateContentConfig(
        temperature=settings.llm_temperature,
        max_output_tokens=settings.max_output_tokens,
    ),
)
```

Map `response.usage_metadata.prompt_token_count`, `candidates_token_count`, and `total_token_count` when present. Do not retry automatically. Wrap provider failures in `GenerationError("Gemini generation failed")` without including credentials or full prompts.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generate.py -q`

Expected: all generation tests pass and fake call count is exactly one.

```powershell
git add src/rag/generate.py tests/test_generate.py
git commit -m "feat: generate grounded answers with gemini"
```

---

### Task 6: Minimal FastAPI Endpoints

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/main.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `Settings`, `Retriever`, `GeminiGenerator`, current manifest
- Produces: `create_app(settings: Settings | None = None, retriever: Retriever | None = None, generator: GeminiGenerator | None = None) -> FastAPI`
- Produces: `GET /health`, `GET /documents`, `POST /ask`

- [ ] **Step 1: Write failing API tests**

Using `fastapi.testclient.TestClient`, assert `/health` returns `{"status": "ok", "index_available": false}` without an index; `/documents` returns an empty list when no manifest exists; `/ask` rejects a blank question with `422`; missing index/config maps to `503` with a safe detail; and injected fake retriever/generator returns answer, ordered sources, elapsed milliseconds, and optional token usage.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api.py -q`

Expected: import failure because `src.api.main` is absent.

- [ ] **Step 3: Implement API schemas and dependency wiring**

Use Pydantic request/response models. `AskRequest.question` must have `min_length=1` and whitespace validation. Measure `/ask` with `time.perf_counter()`. Call retrieval once and generation once. Return source fields `rank`, `document_id`, `filename`, `page_number`, `chunk_id`, `relevance_score`, and `text`. Map known configuration/index/generation failures to concise `503` responses.

- [ ] **Step 4: Verify GREEN and full offline suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py -q
.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass with no network calls, warnings, or secret output.

- [ ] **Step 5: Manual API QA without generation**

Run:

```powershell
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Check `/health` and `/docs`; stop the server. Expected: both load locally and no Gemini call occurs.

- [ ] **Step 6: Commit**

```powershell
git add src/api tests/test_api.py
git commit -m "feat: expose clean rag api"
```

---

### Task 7: Real Brochure QA, Documentation, GitHub, and Phase Status

**Files:**
- Modify: `README.md`
- Modify: `SETUP.md` only if actual commands differ
- Modify: `PHASE_STATUS.md`
- Create locally only: official PDFs under `data/clean/`
- Generated and ignored: `data/vector_store/`
- Generated: `data/manifests/clean_index.json`

**Interfaces:**
- Consumes: all Phase 1 commands and API endpoints
- Produces: reproducible owner instructions, verified clean index, low-volume live QA record, accurate phase status, GitHub `origin`

- [ ] **Step 1: Request official brochure PDFs**

Ask the user to place one or more readable official Toyota UK brochure PDFs in `data/clean/`. Verify filenames and PDF headers; do not download or substitute third-party documents.

- [ ] **Step 2: Index the real clean corpus**

Run:

```powershell
.venv\Scripts\python.exe -m src.rag.index
.venv\Scripts\python.exe -m src.rag.index
```

Expected: first run extracts/chunks/embeds locally and persists Chroma; second run reports reuse without rebuilding.

- [ ] **Step 3: Run three to five retrieval checks before generation**

Choose objective questions based on facts visibly present in the supplied brochures. Inspect returned filenames, pages, and ranks. Record issues; do not make Gemini calls until retrieval is relevant.

- [ ] **Step 4: Run no more than three to five live Gemini smoke calls**

For the same questions, call `/ask` once each. Confirm concise grounded answers, source references, ranks, latency, and token usage when reported. Do not run aggregate evaluation.

- [ ] **Step 5: Update practical documentation**

Document exact commands to create/activate `.venv`, install requirements, configure `.env`, place PDFs, run `python -m src.rag.index`, start Uvicorn, and call `/ask`. State that embeddings and Chroma are local and Gemini is called only for final answer generation.

- [ ] **Step 6: Update phase status accurately**

Set Phase 1 to `Complete` only if real brochure indexing/retrieval, live Gemini smoke QA, tests, documentation, commit, GitHub remote, and push succeed. Otherwise set `Blocked`, list the exact missing external requirement, and keep later phases `Not Started`.

- [ ] **Step 7: Create/connect GitHub repository**

Create `rag-poisoning-testbed`, set `origin`, and confirm `main`. If authentication is unavailable, stop and ask the user to authenticate; never invent credentials.

- [ ] **Step 8: Run final verification**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q
git check-ignore -v .env
git status --short
git remote -v
```

Expected: zero test failures; `.env` ignored; no secret tracked; only intended documentation/status changes pending; `origin` points to the repository.

- [ ] **Step 9: Commit and push Phase 1**

```powershell
git add README.md SETUP.md PHASE_STATUS.md data/manifests/clean_index.json
git commit -m "feat: complete phase 1 clean rag baseline"
git push -u origin main
```

Expected: push succeeds. Record the commit hash and remote URL, then stop without starting Phase 2.


