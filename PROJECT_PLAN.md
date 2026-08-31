# Project Plan — RAG Poisoning Testbed

## Goal

Create a small, understandable RAG-security testbed using official car brochure PDFs.

The final demo should let a visitor:

1. Understand what the project is.
2. Ask a free-form question about the car brochures.
3. See the clean RAG answer and retrieved sources.
4. Add a controlled synthetic document.
5. Ask the same or another question.
6. See whether the synthetic document was retrieved and whether it changed the answer.
7. Apply a simple defense.
8. Compare the result.
9. View saved aggregate experiment metrics.

## Final phase count

Exactly **5 phases**.

---

# Phase 1 — Setup, Clean Data, and Baseline RAG

## Objective

Build a working clean RAG pipeline over official car brochure PDFs.

## Main work

- Create/initialize local project repository.
- Create GitHub repository and connect remote.
- Verify local environment.
- Add required project structure.
- Configure `.env` loading safely.
- Obtain clean brochure PDFs from the user if not already present.
- Extract PDF text with PyMuPDF.
- Chunk text.
- Create local embeddings with Sentence Transformers.
- Store embeddings in persistent local ChromaDB.
- Retrieve top-k relevant chunks.
- Generate an answer through the configured LLM API.
- Return the retrieved sources and ranks.
- Add minimal FastAPI endpoints.
- Add basic tests.
- Run only a few API calls for smoke testing.
- Commit and push after QA.

## Phase 1 deliverable

A working **clean RAG baseline** that can answer free-form questions from the clean brochure corpus.

---

# Phase 2 — Poisoning Attacks

## Objective

Create controlled synthetic PDFs and demonstrate retrieval/generation compromise.

## Main work

- Inspect facts in the actual clean brochures.
- Select a few objective facts suitable for controlled attack tests.
- Create approximately 3–5 readable synthetic PDFs.
- Do not write “poisoned”, “fake”, or equivalent labels inside the PDFs.
- Create a hidden evaluation-only attack manifest.
- Add synthetic documents to the same searchable corpus.
- Implement:
  - false-fact poison
  - retrieval-targeted poison
  - instruction-style poison only if it remains simple and useful
- Record poison rank.
- Record retrieval compromise separately from generation compromise.
- Keep attack experiments bounded and local.
- Commit and push after QA.

## Phase 2 deliverable

A reproducible poisoned-RAG demonstration using synthetic documents tied to real brochure facts.

---

# Phase 3 — Defenses

## Objective

Implement a few understandable defenses and measure their tradeoffs.

## Main defenses

1. Source / provenance trust filter
2. Instruction-content filter
3. Suspicious similarity / duplicate filter
4. Optional simple combined defense

## Requirements

- Defenses may use realistic metadata or text signals.
- Defenses must never use hidden `is_poison` ground truth.
- Measure whether clean documents are wrongly rejected.
- Measure latency overhead.
- Keep implementations simple enough to explain in an interview.
- Commit and push after QA.

## Phase 3 deliverable

A defended RAG pipeline that can be compared against clean and attacked runs.

---

# Phase 4 — Evaluation and Error Analysis

## Objective

Produce real aggregate results from a small reproducible benchmark.

## Main work

- Create a compact factual benchmark from the clean PDFs.
- Keep the live demo free-form.
- Use the fixed benchmark only for repeatable aggregate metrics.
- Prefer objective factual questions with known answers.
- Avoid heavy LLM-as-judge use.
- If an LLM judge is used, use it only where deterministic checks are insufficient.
- Run a modest evaluation set, not a massive benchmark.
- Save outputs to JSON/CSV.

## Metrics

- clean answer correctness
- poison retrieval rate
- poison rank
- retrieval attack success
- generation attack success
- source/citation correctness
- defense attack-success reduction
- clean false rejection rate
- latency
- token usage
- optional synthetic canary leakage proxy, clearly labeled as a proxy

## Error analysis

Keep only a few useful examples:
- poison retrieved but answer unchanged
- answer changed even though poison was not rank #1
- defense removed a clean document
- bad answer without successful poison

## Phase 4 deliverable

Saved real experiment results ready for the Results page.

---

# Phase 5 — Frontend, Integration, Demo, and Documentation

## Objective

Implement the finalized Figma design and connect it to real backend data.

## Final page order

1. About
2. Demo
3. Documents
4. Results

## Main work

- Implement finalized Figma in React/Vite/TypeScript.
- Do not add extra pages/features.
- Connect free-form question input to backend.
- Show clean / attacked / defended runs.
- Show retrieved documents.
- Label injected test documents only in the evaluation UI after retrieval.
- Show real current-run metrics.
- Load aggregate Results page metrics from saved experiment output.
- Allow brochure PDFs to be viewed.
- Final README and project explanation.
- Final QA.
- Commit and push.

## Phase 5 deliverable

A polished local portfolio demo matching the final Figma and backed by real experiment outputs.

---

# Suggested repository structure

```text
rag-poisoning-testbed/
│
├── README.md
├── AGENTS.md
├── PROJECT_PLAN.md
├── ARCHITECTURE_AND_THREAT_MODEL.md
├── DATA_AND_EVALUATION.md
├── FRONTEND_SPEC.md
├── PHASE_STATUS.md
├── SETUP.md
├── .env
├── .env.example
├── .gitignore
│
├── data/
│   ├── clean/
│   ├── poisoned/
│   ├── evaluation/
│   ├── manifests/
│   └── vector_store/
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── rag/
│   │   ├── ingest.py
│   │   ├── chunk.py
│   │   ├── embed.py
│   │   ├── retrieve.py
│   │   └── generate.py
│   ├── attacks/
│   ├── defenses/
│   └── evaluation/
│
├── experiments/
│   ├── run_experiments.py
│   └── results/
│
├── frontend/
│
├── reports/
│   ├── figures/
│   └── results.md
│
└── tests/
```

Create only what is needed in the current phase. Empty future-phase folders may be created for orientation, but do not fill them with speculative code.
