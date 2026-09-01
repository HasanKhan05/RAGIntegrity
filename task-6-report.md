# Task 6 — Phase 2 Runner Safety Report

Date: 2026-09-02

## Scope

Completed the interrupted safety fix for the controlled six-call Phase 2 runner.

## Changes

- Refuse to run when the destination results file already exists.
- Require a configured generation key/model, LLM_TEMPERATURE=0, and TOP_K=3
  before constructing retrievers or generating answers.
- Validate production runs against the exact synthetic PDF filename inventory,
  clean and attacked index manifests, and persisted Chroma collection metadata.
- Require the persisted clean collection inventory to equal clean_index.json
  exactly before validating the attacked collection.
- Reject malformed clean or attacked manifest document entries instead of
  silently omitting them.
- Reject missing or mismatched attacked collections and synthetic IDs or filenames
  in the clean collection.
- Permit fully injected fake generator/retriever tests to bypass persisted
  collection checks while retaining output and configuration safety checks.
- Added regression tests proving all safety failures make zero generator calls.

## Verification

- Focused runner and safety tests: 19 passed.
- Full test suite: 67 passed.
- git diff --check: passed.
- No Gemini generation calls were made.
- Existing experiment results, project documentation, and PHASE_STATUS.md
  were not modified.

## Remaining concern

Production execution intentionally stops unless the local clean and attacked
indexes and their manifests are present and internally consistent. This is a
fail-closed guard for the fixed experiment and does not assert that a defense
is perfect.
