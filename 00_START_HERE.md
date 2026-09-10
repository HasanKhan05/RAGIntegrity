# RAGIntegrity — Evaluating Retrieval Poisoning Attacks and Defenses

**Owner:** Muhammad Hasan Dad Khan  
**Project type:** Personal portfolio / research demonstration  
**Final Figma:** https://www.figma.com/design/m8D51hB7Q9KA8llSRHRBhb/RAG-Poisoning-Testbed-%E2%80%94-Muhammad-Hasan-Dad-Khan?node-id=1-169

## Read order for Codex

1. `AGENTS.md`
2. `PROJECT_PLAN.md`
3. `ARCHITECTURE_AND_THREAT_MODEL.md`
4. `DATA_AND_EVALUATION.md`
5. `FRONTEND_SPEC.md`
6. `SETUP.md`
7. `PHASE_STATUS.md`
8. `PHASE_1_PROMPT.md`

Then execute **Phase 1 only**.

## Non-negotiable project rule

> **DO NOT USE TOKENS HEAVILY.**

This project is not intended to become production software. The priorities are:

1. It works correctly.
2. The experiment is understandable and defensible.
3. Development is fast.
4. API/token usage stays low.
5. The code remains simple enough for the owner to understand.
6. Do not add production architecture, infrastructure, security layers, or abstractions unless explicitly requested.

## Important workflow rules

- Do not start a later phase automatically.
- At the end of each phase: run appropriate QA/tests, update `PHASE_STATUS.md`, commit, and push only if the phase is working.
- If official car brochure PDFs are required and not available, ask the user to provide/download them. Do not substitute random third-party documents.
- If a live LLM call is required and `.env` is not configured, ask the user to fill in the required API settings.
- Never print, expose, or commit the API key.
