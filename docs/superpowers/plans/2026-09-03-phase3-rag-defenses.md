# Phase 3 RAG Defenses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add attack-blind post-retrieval defenses, compare them locally on the fixed Phase 2 benchmark, and run the approved six-call defended-generation smoke test.

**Architecture:** A small `src.rag.defenses` coordinator accepts one immutable retrieval snapshot plus clean provenance, applies one selected defense pipeline, and returns re-ranked chunks with an attack-blind stage trace and defense-only latency. The API and experiment runners use that coordinator; hidden attack labels enter only evaluation code after every defense result has been produced.

**Tech Stack:** Python 3.13, NumPy, sentence-transformers, FastAPI/Pydantic, pytest, local JSON artifacts, Gemini only for the fixed six-call smoke test.

**Spec:** `docs/superpowers/specs/2026-09-03-phase3-rag-defenses-design.md`

## Phase 3 completion record

- Completed on 2026-09-03 with 133 tests passing.
- Local analysis: 48 questions, one immutable top-3 retrieval snapshot per question reused across all five modes, zero Gemini calls, and 23 exact target-page retrievals.
- At the default `0.92` threshold, target removal was 0/23 (`none`), 23/23 (`source_trust`), 2/23 (`instruction_filter`), 0/23 (`similarity_filter`), and 23/23 (`combined`). Clean false rejection was 0/112 for all modes except `similarity_filter` and `combined`, each 4/112.
- The fixed smoke made exactly six Gemini calls: 4,270 input tokens, 179 output tokens, and 4,449 total tokens reported by the provider. Retrieval compromise remained true in all six same-snapshot runs; generation compromise was false in all six defended answers.
- Defense code and traces are attack-blind. Hidden labels are used only after defense for evaluation. Source trust is explicitly limited to the closed, curated clean filename inventory and is not a general open-upload provenance guarantee.
- Recorded workflow ruling: this worker commits locally but does not push; the controller pushes only after task and whole-branch reviews pass.

## Global Constraints

- Phase 3 only; no frontend, Phase 4 evaluation, production infrastructure, or new attack documents.
- Local embeddings and vector storage remain unchanged.
- Use `TOP_K=3`, `LLM_TEMPERATURE=0`, and no more than six new Gemini calls.
- The attack manifest is evaluation-only and never enters a defense, trace, API prompt, or defense configuration.
- Retrieve each benchmark question once; apply every mode to the same immutable tuple.
- Measure defense processing separately from retrieval.
- Preserve `/ask` behavior through `defense_mode="none"` by default.
- Do not overwrite Phase 2 result files or existing local changes.

---

### Task 1: Attack-blind defense core and configuration

**Files:**
- Create: `src/rag/defenses.py`
- Modify: `src/rag/config.py`
- Modify: `.env.example`
- Create: `tests/test_defenses.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `RetrievedChunk`, `Embedder.encode(texts)`, and clean index manifest `documents[].filename`.
- Produces: `DefenseMode`, `DefenseStageDecision`, `DefenseTraceEntry`, `DefenseResult`, `load_trusted_filenames(path)`, and `DefenseCoordinator.apply(chunks, mode)`.

- [x] **Step 1: Write failing configuration and core behavior tests**

  Cover a default `DEFENSE_SIMILARITY_THRESHOLD=0.92` constrained to `[0, 1]`; immutable `none`; source trust based only on clean filenames; the five narrow normalized instruction phrases; ordinary brochure language retention; cosine threshold boundaries; trusted-over-untrusted and better-rank tie resolution; fixed combined order; contiguous final ranks; and traces containing only ordinary source fields, stage decisions, and allowed reasons.

- [x] **Step 2: Verify RED**

  Run `.venv\Scripts\python.exe -m pytest -q tests/test_config.py tests/test_defenses.py` and confirm failures are caused by missing Phase 3 settings/module behavior.

- [x] **Step 3: Implement minimal defense core**

  Add `similarity_threshold: float` to `Settings`; parse `DEFENSE_SIMILARITY_THRESHOLD` with default `0.92`. In `defenses.py`, load a frozen trusted filename set from the clean manifest, normalize instruction text with whitespace/case folding, normalize NumPy vectors before cosine comparison, resolve conflicts deterministically by trust then original rank, preserve first-removal stage decisions, and re-create retained `RetrievedChunk` values with ranks `1..N`.

- [x] **Step 4: Verify GREEN and refactor**

  Run the same focused tests, then keep the coordinator explicit and attack-vocabulary-free.

### Task 2: Backward-compatible API defense selection

**Files:**
- Modify: `src/api/main.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `DefenseCoordinator.apply(chunks, request.defense_mode)`.
- Produces: optional request enum `defense_mode`; response fields `defense_mode`, `defense_latency_ms`, and `defense_trace`; generator receives only `DefenseResult.chunks`.

- [x] **Step 1: Write failing API tests**

  Assert a request without `defense_mode` uses `none`, preserves the existing source and generator context, and returns attack-blind trace data. Assert each explicit mode is accepted and an untrusted attacked chunk removed by `source_trust` never reaches the generator.

- [x] **Step 2: Verify RED**

  Run `.venv\Scripts\python.exe -m pytest -q tests/test_api.py` and confirm the new response/retained-context assertions fail.

- [x] **Step 3: Implement minimal API integration**

  Build one default coordinator from `Settings`, the clean manifest, and the existing local embedder; permit dependency injection in `create_app`; retrieve once, apply the selected mode, generate from retained chunks, and serialize only public trace fields.

- [x] **Step 4: Verify GREEN**

  Run the API tests and the pre-existing prompt tests to prove no hidden labels enter generation.

### Task 3: Zero-Gemini local defense analysis

**Files:**
- Create: `src/evaluation/phase3.py`
- Create: `experiments/run_phase3_defense_analysis.py`
- Create: `tests/test_phase3_evaluation.py`
- Create: `tests/test_phase3_defense_analysis.py`

**Interfaces:**
- Consumes: 30 attack questions, 18 clean controls, 10 evaluation-only attack cases, one attacked `Retriever`, all five defense modes, and clean filenames.
- Produces: `experiments/results/phase3_defense_analysis.json` with per-question immutable source snapshots, per-mode traces, and aggregate poison removal/survival, clean false-rejection, remaining-context, and defense-latency metrics.

- [x] **Step 1: Write failing evaluation tests**

  Test exact target document-plus-page matching after defenses, legitimate-clean removal counting by clean inventory, exclusion of removed synthetic chunks from the false-rejection numerator, and safe zero denominators.

- [x] **Step 2: Verify RED, implement, and verify GREEN**

  Run `.venv\Scripts\python.exe -m pytest -q tests/test_phase3_evaluation.py`; add attack-label comparisons only in `src.evaluation.phase3`; rerun until green.

- [x] **Step 3: Write failing runner tests**

  Use a recording retriever and coordinator to prove exactly 48 retrieval calls, one immutable tuple per question reused across all five modes, no generator/provider object, retrieval timing outside defense timing, canonical output schema, and refusal to overwrite an existing result.

- [x] **Step 4: Verify RED, implement, and verify GREEN**

  Run `.venv\Scripts\python.exe -m pytest -q tests/test_phase3_defense_analysis.py`; implement the runner without importing `GeminiGenerator`; rerun until green.

### Task 4: Fixed six-call defended-generation smoke test

**Files:**
- Create: `experiments/run_phase3_defense_smoke.py`
- Create: `tests/test_phase3_defense_smoke.py`

**Interfaces:**
- Consumes: Phase 2 no-defense outputs, attacked retrieval for attacks `003`, `005`, and `010`, fixed modes (`instruction_filter`/`combined` for `003`; `source_trust`/`combined` for `005` and `010`), the Phase 2 deterministic evaluator, and `GeminiGenerator`.
- Produces: `experiments/results/phase3_defense_smoke.json` with exactly six new calls, provider token totals, defended traces/sources/answers, compromise assessments, and reused Phase 2 baseline references.

- [x] **Step 1: Write failing smoke-runner tests**

  Assert the fixed attack/mode matrix, exactly three retrievals and six generation calls, temperature/top-k preflight, output refusal before calls, retained-only generator contexts, deterministic assessments, token summation, and `gemini_calls == 6`.

- [x] **Step 2: Verify RED**

  Run `.venv\Scripts\python.exe -m pytest -q tests/test_phase3_defense_smoke.py` and confirm failure is the missing runner.

- [x] **Step 3: Implement minimal capped runner**

  Validate all local inputs before generation, retrieve each selected attacked question once, apply the two fixed modes to its tuple, call Gemini once per defended mode, and write only the new Phase 3 output.

- [x] **Step 4: Verify GREEN**

  Run the focused smoke tests; inspect the fake generator call count and output schema.

### Task 5: Produce artifacts, document actual results, and complete Phase 3

**Files:**
- Create: `experiments/results/phase3_defense_analysis.json`
- Create: `experiments/results/phase3_defense_smoke.json`
- Modify: `README.md`
- Modify: `ARCHITECTURE_AND_THREAT_MODEL.md`
- Modify: `DATA_AND_EVALUATION.md`
- Modify: `PHASE_STATUS.md`
- Modify: `docs/superpowers/plans/2026-09-03-phase3-rag-defenses.md`

**Interfaces:**
- Consumes: verified local indexes, `.env` generation configuration, final Phase 3 runners.
- Produces: reproducible Phase 3 metrics, six-call/token accounting, documented threshold/limitations, and completion evidence.

- [x] **Step 1: Run the zero-call local analysis**

  Run `.venv\Scripts\python.exe -m experiments.run_phase3_defense_analysis`, inspect all 48 outcomes and summaries, and confirm no Gemini code path ran.

- [x] **Step 2: Run the paid smoke once**

  After all preflight checks pass, run `.venv\Scripts\python.exe -m experiments.run_phase3_defense_smoke` exactly once; do not retry in a way that exceeds six new calls. Record provider-reported input/output/total tokens.

- [x] **Step 3: Update documentation with measured values**

  Explain defense modes, `0.92` default threshold, attack-blind boundary, same-snapshot methodology, retrieval versus generation compromise, source-trust threat-model limitation, local metrics, smoke outcomes, and exact Gemini usage. Mark Phase 3 complete after verification and the local commit; the recorded controller workflow defers push until reviews pass.

- [x] **Step 4: Run completion verification**

  Run `.venv\Scripts\python.exe -m pytest -q`; `git diff --check`; verify `.env` is ignored and untracked; inspect both result JSON files; scan tracked Phase 3 code/trace/result fields for `LLM_API_KEY`, real key material, `is_poison`, and attack-manifest leakage into `src.rag.defenses` or API traces.

- [x] **Step 5: Commit locally; controller pushes after review**

  Stage only intended Phase 3 files, commit with `feat: complete phase 3 rag defenses`, verify `git status --short --branch`, and stop without starting Phase 4. Under the recorded workflow ruling, do not push here; the controller pushes only after task and whole-branch reviews pass.
