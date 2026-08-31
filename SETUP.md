# Local Setup

Keep the machine setup minimal.

## Required

- Python 3.13.x, 64-bit
- Node.js 24 LTS
- Git
- Codex / code editor
- modern browser

## Do not install for this project unless later requested

- Docker Desktop
- WSL just for this project
- CUDA
- PostgreSQL
- MongoDB
- Redis
- Pinecone
- Kubernetes
- LangChain
- LlamaIndex
- Anaconda

---

# Python environment

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Phase 1 should create a minimal `requirements.txt`.

Expected core packages:

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
httpx
pytest
```

Add only the SDK needed for the user's selected LLM provider.

Phase 2 may add:

```text
reportlab
```

---

# Environment variables

A local `.env` is included in the starter pack.

Fill these before live LLM generation is required:

```text
LLM_PROVIDER=
LLM_API_KEY=
LLM_MODEL=
LLM_BASE_URL=
```

Do not commit `.env`.

The project should fail clearly and safely if a live LLM call is attempted without required configuration.

---

# Clean PDFs

Expected location:

```text
data/clean/
```

When Phase 1 reaches real ingestion testing, if this directory does not contain official brochure PDFs, Codex should stop and ask the user to provide them.

Do not substitute unofficial documents without permission.

---

# Local run targets

Once Phase 1 is implemented, likely commands should remain simple, for example:

```powershell
uvicorn src.api.main:app --reload
```

FastAPI docs:

```text
http://localhost:8000/docs
```

Frontend is not implemented until Phase 5.
