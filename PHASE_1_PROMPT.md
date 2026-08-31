# Codex Prompt — Phase 1 Only

You are implementing **Phase 1 — Setup, Clean Data, and Baseline RAG** for the project in this folder.

Before writing code, read these files in order:

1. `00_START_HERE.md`
2. `AGENTS.md`
3. `PROJECT_PLAN.md`
4. `ARCHITECTURE_AND_THREAT_MODEL.md`
5. `DATA_AND_EVALUATION.md`
6. `FRONTEND_SPEC.md`
7. `SETUP.md`
8. `PHASE_STATUS.md`

## Highest-priority constraints

> **DO NOT USE TOKENS HEAVILY.**

> **DO NOT TAKE THIS PROJECT TO PRODUCTION LEVEL.**

Optimize for:
- correctness
- speed of development
- low API/token usage
- simple understandable code
- a reliable local demo

Do not add unnecessary infrastructure or abstractions.

Do not begin Phase 2.

---

# Phase 1 objective

Create a working **clean RAG baseline** over official car brochure PDFs.

The user must be able to ask a free-form question and receive:
- a concise answer
- retrieved source document names
- retrieval ranks
- latency
- token usage if the configured provider returns it

No poisoning or defenses yet.

---

# Step 1 — Verify environment

Check, without making unnecessary changes:

- Python version
- Git
- whether a virtual environment already exists
- whether Node exists (frontend will be Phase 5, so do not install frontend packages yet)
- GitHub authentication/connection options

Do not install Docker, database servers, CUDA, LangChain, LlamaIndex, or other unnecessary tools.

---

# Step 2 — Create and connect the GitHub repository

The project folder and GitHub repository must be connected during Phase 1.

Preferred repository name:

`rag-poisoning-testbed`

Do the following:

1. Initialize Git in the current project folder if not already initialized.
2. Preserve the starter documents already present.
3. Create a GitHub repository named `rag-poisoning-testbed`.
4. Connect the current local folder to that GitHub repository as `origin`.
5. Use `main` as the default branch.
6. Ensure `.env` is ignored before any commit.

Use the user's authenticated GitHub integration or GitHub CLI if available.

If GitHub repository creation cannot proceed only because authentication is unavailable, stop and ask the user to authenticate. Do not invent credentials and do not create a different hosting setup.

Do not push the actual `.env`.

---

# Step 3 — Minimal project scaffold

Create only the structure needed for Phase 1:

```text
data/
  clean/
  manifests/
  vector_store/

src/
  api/
  rag/

tests/
```

Future-phase folders may remain empty if already present, but do not implement future-phase code.

Create a minimal `requirements.txt`.

Do not add production config files.

---

# Step 4 — API configuration

Use the existing root `.env`.

Expected variables:

```text
LLM_PROVIDER=
LLM_API_KEY=
LLM_MODEL=
LLM_BASE_URL=
MAX_OUTPUT_TOKENS=300
LLM_TEMPERATURE=0
TOP_K=3
```

Rules:
- never print the API key
- never log the API key
- never commit `.env`
- do not create a large multi-provider framework
- support only what is needed for the provider the user configures

If a live LLM generation test becomes necessary and the provider/model/key are not configured, stop and ask the user to fill them in.

Do not ask for the secret value in chat if the user can place it directly into `.env`.

---

# Step 5 — Clean brochure PDFs

Look in:

`data/clean/`

If official car brochure PDFs are not present:

1. Do not download random replacements.
2. Do not use third-party car-spec websites as substitutes.
3. Stop at the point where real ingestion testing requires PDFs.
4. Tell the user exactly which PDFs are needed and ask them to place the official brochures in `data/clean/`.

The user has explicitly said they will provide/download the car brochure PDFs when needed.

If PDFs are present, continue.

---

# Step 6 — Implement PDF ingestion

Use PyMuPDF.

Requirements:
- read each PDF
- extract page text
- preserve filename and page number metadata
- skip empty text safely
- keep implementation readable
- do not use OCR unless clearly necessary

Create a small clean-source manifest if useful.

---

# Step 7 — Chunking

Create a simple chunker.

Do not over-engineer semantic chunking.

Use a sensible fixed-size text chunk strategy with small overlap.

Preserve:
- document id
- filename
- page number
- chunk id

Keep configuration centralized and easy to change.

---

# Step 8 — Local embeddings and ChromaDB

Use:

`sentence-transformers`

Recommended model:

`all-MiniLM-L6-v2`

Use local persistent ChromaDB.

Do not use a paid embedding API.

Persist the vector store under:

`data/vector_store/`

Avoid rebuilding embeddings unnecessarily if the clean corpus has not changed.

---

# Step 9 — Retrieval

Implement top-k retrieval.

Default:

`TOP_K=3`

For each retrieved chunk, return:
- rank
- document id
- filename
- page number
- chunk id
- similarity/relevance score if readily available
- text excerpt/context

Keep retrieval code independent from answer generation.

---

# Step 10 — LLM answer generation

Use only the configured LLM provider.

Token rules:
- short system prompt
- short user prompt
- send only retrieved context
- do not send full PDFs
- temperature `0`
- max output around `300` tokens
- do not repeatedly call the LLM during development for the same test

Prompt behavior:
- answer only from retrieved context
- if context is insufficient, say so
- include simple source references based on retrieved document names/pages

Keep this implementation small and understandable.

---

# Step 11 — Minimal FastAPI API

Create only what Phase 1 needs.

Suggested endpoints:

### `GET /health`
Returns service health.

### `GET /documents`
Returns clean document metadata currently indexed.

### `POST /ask`
Accepts a free-form question and returns:
- answer
- retrieved sources with ranks
- latency
- token usage if available

Do not add authentication, users, admin APIs, databases, or production middleware.

FastAPI `/docs` is enough for Phase 1 testing.

---

# Step 12 — Tests and smoke QA

Create lightweight tests for:
- PDF text extraction
- chunking
- retrieval structure
- API health endpoint
- safe behavior when API config is missing

For live LLM smoke testing:
- use only a few questions
- do not run a large evaluation
- do not generate aggregate attack percentages in Phase 1

If the brochures are available, manually test roughly 3–5 sensible factual questions total, not dozens.

Record any issue clearly.

---

# Step 13 — Documentation

Update:
- `README.md`
- `PHASE_STATUS.md`
- any setup instructions that changed

The README should explain how to:
- activate `.venv`
- install requirements
- place clean PDFs
- configure `.env`
- ingest/index documents
- run FastAPI
- ask a question

Keep README practical and concise.

---

# Step 14 — Phase 1 completion criteria

Phase 1 is complete only if:

- clean official brochure PDFs can be ingested
- text is chunked
- embeddings are generated locally
- ChromaDB retrieval works
- a free-form question can retrieve relevant brochure chunks
- configured LLM can produce a concise answer from retrieved context
- source names/ranks are returned
- basic tests pass
- token usage is kept low
- `.env` is not tracked
- GitHub repository is connected
- changes are committed
- changes are pushed

If any required external item is missing, mark the phase `Blocked` rather than pretending it is complete.

---

# Git completion

When QA passes:

1. Verify `git status`.
2. Verify `.env` is not tracked.
3. Update `PHASE_STATUS.md`.
4. Commit with a clear message such as:

`feat: complete phase 1 clean rag baseline`

5. Push `main` to `origin`.
6. Show the user:
   - what was implemented
   - tests run
   - any API calls made
   - whether PDFs/API configuration were required
   - Git commit hash
   - remote repository URL
7. Stop.

**Do not start Phase 2.**
