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

Start the backend from the repository root using the existing virtual environment.
Activation is optional when using its executable directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`, then check health in another terminal:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health'
```

An existing index should return `status: ok` and `index_available: True`.
The optional `--reload` flag is for development; first verify startup without it
so an import failure appears directly in the terminal.

Open a second terminal at the repository root, then start the frontend:

```powershell
Set-Location frontend
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://127.0.0.1:5173/`. Vite forwards `/api` requests to the backend on
port 8000. Keep both terminals running, and press `Ctrl+C` in each to stop.

- API documentation: `http://127.0.0.1:8000/docs`
- Document catalog: `GET http://127.0.0.1:8000/documents/catalog`
- Saved evaluation: `GET http://127.0.0.1:8000/results/summary`
- Free-form grounded answer: `POST http://127.0.0.1:8000/ask`

The existing `.env` needs no changes to serve documents and saved results.
Live Demo answers require its Gemini settings and use provider calls.

The first Demo run loads the local embedding model and can take longer than
subsequent runs. Local retrieval and defense processing are serialized within
the API to avoid concurrent Chroma/model initialization; generation runs outside
that lock. The three Demo requests still use their respective corpus and defense.

PDF extraction loads PyMuPDF only when ingesting PDFs. Serving existing indexes,
PDF files, and saved results does not import its native DLL. If Windows blocks
that DLL, PyMuPDF may report `No module named 'mupdf'` after hiding the original
Application Control error. Resolve that dependency with the machine's security
administrator before rebuilding indexes; do not disable security controls.

Run the offline test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Live Gemini calls are not part of the automated tests.
