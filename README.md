# RAG Poisoning Testbed

**Muhammad Hasan Dad Khan**

A local portfolio/research demo showing how retrieval-augmented generation can be influenced by synthetic documents and how simple defenses can reduce that effect.

```text
Clean RAG → Poisoned RAG → Defended RAG
```

Phase 1 provides the clean baseline only: official Toyota brochure ingestion, fixed-size page-aware chunking, local `all-MiniLM-L6-v2` embeddings, persistent ChromaDB retrieval, grounded Gemini answers, and a minimal FastAPI API. Poisoning and defenses are intentionally not implemented yet.

## Phase 1 quick start

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place readable official brochure PDFs in `data/clean/`. Copy `.env.example` to `.env` and configure the local secret:

```dotenv
LLM_PROVIDER=gemini
LLM_API_KEY=your_key_here
LLM_MODEL=gemini-3.5-flash-lite
LLM_BASE_URL=
MAX_OUTPUT_TOKENS=300
LLM_TEMPERATURE=0
TOP_K=3
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
CHROMA_PERSIST_DIR=data/vector_store
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Never commit `.env`. Build or reuse the clean index, then start the API:

```powershell
python -m src.rag.index
uvicorn src.api.main:app --reload
```

Open `http://localhost:8000/docs`, or ask from PowerShell:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/ask `
  -ContentType 'application/json' `
  -Body '{"question":"What is the RAV4 fuel tank capacity?"}'
```

The response includes the concise answer, retrieved source chunks and ranks, latency, and provider token usage when available. Run the tests with:

```powershell
python -m pytest -q
```

The vector store is local and ignored by Git. Re-running the index command reuses it when the PDFs and index settings have not changed.

## Project references

- `00_START_HERE.md` — execution entry point
- `PROJECT_PLAN.md` — phased plan
- `ARCHITECTURE_AND_THREAT_MODEL.md` — scope and trust boundaries
- `DATA_AND_EVALUATION.md` — corpus and evaluation rules
- `PHASE_STATUS.md` — completion tracker
- [Final Figma](https://www.figma.com/design/m8D51hB7Q9KA8llSRHRBhb/RAG-Poisoning-Testbed-%E2%80%94-Muhammad-Hasan-Dad-Khan?node-id=1-169)

> **DO NOT USE TOKENS HEAVILY.** Keep this project small, understandable, and inexpensive to run.
