# Phase Status Tracker

Update this file at the end of every phase.

| Phase | Status | QA Passed | Commit | Pushed | Notes |
|---|---|---:|---:|---:|---|
| Phase 1 — Setup, Clean Data, Baseline RAG | Complete | Yes | Yes | Yes | 7 official PDFs; 200 pages; 307 chunks; 30 tests; 3 live Gemini QA calls passed |
| Phase 2 — Poisoning Attacks | Complete | Yes | Yes | Yes | 10 facts; 6 controlled local research PDFs (not manufacturer publications); 30 retrieval-only attack questions; 18 clean controls; 23/30 target-page retrievals; 6-call smoke used 5,551 provider-reported tokens |
| Phase 3 — Defenses | Complete | Yes | Yes | Yes | 48-question attack-blind local comparison; 23/23 retrieved targets removed by source trust/combined; 6-call smoke used 4,449 provider-reported tokens; published cross-mode latency is not a fair steady-state comparison because the fixed mode order assigns local embedder cold start to `similarity_filter` |
| Phase 4 — Evaluation and Error Analysis | Complete | Yes | Yes | Yes | 48 frozen questions × 6 conditions = 288 scored cells; 103 unique provider calls/attempts; 81,403 provider-reported tokens; 192 correct and 96 incorrect after 27 manual adjudications; 37 refusal-proxy rows; control accuracy 15/18 per condition; scoring and generation timing corrected from cache with zero new calls; final Phase 4 integration pushed to origin/main |
| Phase 5 — Frontend, Demo, Documentation | Not Started | No | No | No | |

## Allowed status values

- Not Started
- In Progress
- Blocked
- Complete

## Phase completion rule

A phase is only `Complete` when:
- its core deliverable works
- relevant tests pass
- a quick manual QA passes
- documentation is updated
- changes are committed
- changes are pushed

If an external requirement is missing, use `Blocked`.

Examples:
- missing brochure PDFs
- missing API configuration
- GitHub authentication unavailable
