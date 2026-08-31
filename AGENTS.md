# AGENTS.md — Project Rules for Codex

## Project

Build a simple local **RAG Poisoning Testbed** using official car brochure PDFs as the clean knowledge base. The demo should show:

**Clean RAG → Poisoned RAG → Defended RAG**

The project is for a personal technical portfolio. It must be easy to explain, easy to run, and scientifically clear enough to demonstrate retrieval compromise versus generation compromise.

---

# HIGHEST PRIORITY RULE

> **DO NOT USE TOKENS HEAVILY.**

Treat API/token efficiency as a first-class project constraint.

### Token-efficiency rules

- Prefer local processing whenever possible.
- Embeddings must be local unless the user explicitly changes this.
- Vector storage must be local.
- Do not repeatedly call the LLM for the same input during development.
- Use only a few live LLM calls for smoke testing.
- Keep prompts short and focused.
- Default to a small/low-cost model chosen by the user/provider.
- Default temperature: `0`.
- Default max answer length: approximately `300` output tokens unless a task clearly requires more.
- Default retrieval `top_k`: `3`.
- Cache or reuse development outputs where this does not invalidate an experiment.
- Do not run a large evaluation automatically.
- Ask before any test likely to make many paid API calls.
- In Phase 1, do not run more than a handful of generation calls just to prove the baseline works.

---

# Scope rule

> **DO NOT TAKE THIS PROJECT TO PRODUCTION LEVEL.**

Do not add any of the following unless the user explicitly requests them:

- Docker
- Kubernetes
- Redis
- PostgreSQL
- cloud vector databases
- user authentication
- roles/permissions
- distributed workers
- message queues
- observability platforms
- complex CI/CD
- microservices
- LangChain
- LlamaIndex
- unnecessary design patterns
- multi-provider abstraction layers
- enterprise security controls
- deployment infrastructure

The main concern is that the project **runs correctly, runs quickly, is easy to understand, and does not waste API tokens**.

---

# Preferred stack

## Backend / RAG
- Python 3.13
- FastAPI
- Uvicorn
- PyMuPDF
- sentence-transformers
- ChromaDB
- NumPy
- pandas
- python-dotenv
- pytest
- ReportLab only when synthetic PDFs are needed in Phase 2

## Embeddings
Use a small local Sentence Transformers model such as:

`all-MiniLM-L6-v2`

Do not use a paid embedding API unless explicitly requested.

## Frontend
Frontend implementation is Phase 5 only:
- React
- Vite
- TypeScript

Do not replace the finalized Figma design with a different UI.

---

# Data rules

- Clean corpus: official manufacturer car brochure PDFs, initially Toyota UK.
- The user will provide/download brochure PDFs when needed.
- Do not silently download random dealer, forum, scraped, or unofficial PDFs as ground truth.
- The exact downloaded clean PDF is the experiment's ground truth for that run.
- Keep original PDFs readable and accessible.
- Demo questions are free-form and must not be restricted to predefined questions.
- A small fixed evaluation set may exist later only for repeatable aggregate metrics; it must not restrict the live demo.

---

# Poisoning rules

The RAG system must **not** be given a hidden `is_poison` label.

Synthetic attack PDFs are created in Phase 2 and should look like ordinary informational documents to the retriever/LLM.

Store poison identity separately in an evaluation-only manifest.

Example separation:

RAG-visible:
- document id
- filename/title
- text
- ordinary source metadata

Evaluation-only:
- document id
- synthetic/poison ground-truth label
- attack type
- target fact
- intended false claim

The frontend may label an injected test document **after retrieval** by matching its document ID against the hidden evaluation manifest.

Never let the retriever, LLM, or defense use the hidden poison ground-truth label.

---

# Experiment rules

Keep separate:

1. **Retrieval compromise**  
   The synthetic document appears in top-k retrieval.

2. **Generation compromise**  
   The synthetic content actually changes the final answer.

Do not claim a successful generation attack just because poison was retrieved.

Do not claim a defense is perfect. Prefer wording such as:

- Attack reduced
- Poison down-ranked
- Synthetic document filtered
- Clean document incorrectly rejected

---

# Coding rules

- Keep functions short and readable.
- Prefer explicit code over clever abstractions.
- Avoid premature refactoring.
- Do not create files that are not needed for the current phase.
- Add type hints where useful, but do not over-engineer.
- Use simple JSON for experiment outputs where possible.
- Keep configuration in environment variables or a small config module.
- Never hardcode actual experiment percentages into the frontend.
- Any `—`, `— %`, `— ms`, `# —`, or `Yes / No` placeholder in Figma must later be populated from real backend/experiment values.

---

# API key rules

- `.env` is local only.
- `.env` must be ignored by Git.
- Never commit, log, print, or expose the key.
- The project should read:
  - `LLM_PROVIDER`
  - `LLM_API_KEY`
  - `LLM_MODEL`
  - optional `LLM_BASE_URL`
- Do not make paid calls if these are not configured.
- Ask the user only when a live generation call becomes necessary.

---

# Git / phase workflow

For every phase:

1. Implement only the current phase.
2. Run targeted tests.
3. Perform a quick manual QA.
4. Update `PHASE_STATUS.md`.
5. Update relevant docs if behavior changed.
6. Commit with a clear message.
7. Push to GitHub if tests pass.
8. Stop and wait for the next phase prompt.

Do not continue to the next phase automatically.
