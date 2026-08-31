# Local Setup

## Requirements

- Python 3.13.x, 64-bit
- Git
- a modern browser
- Node.js 24 LTS is optional until the Phase 5 frontend

Do not install Docker, WSL, CUDA, database servers, LangChain, LlamaIndex, or other infrastructure for Phase 1.

## Python environment

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, PyMuPDF may require the current Microsoft Visual C++ Redistributable.

## Gemini configuration

Copy `.env.example` to `.env`, then set only the secret locally:

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

`.env` is ignored by Git. The application fails safely if live generation is requested without a usable key.

## Clean corpus and index

Place readable official brochure PDFs in `data/clean/`, then run:

```powershell
python -m src.rag.index
```

The command writes `data/manifests/clean_index.json` and the ignored local Chroma store under `data/vector_store/`. It skips rebuilding when the brochure hashes and index settings are unchanged.

## API and tests

```powershell
uvicorn src.api.main:app --reload
```

- API documentation: `http://localhost:8000/docs`
- Health: `GET http://localhost:8000/health`
- Indexed documents: `GET http://localhost:8000/documents`
- Free-form grounded answer: `POST http://localhost:8000/ask`

Run the offline test suite with:

```powershell
python -m pytest -q
```

Live Gemini calls are not part of the automated tests.
