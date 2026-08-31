# RAG Poisoning Testbed

**Muhammad Hasan Dad Khan**

A small portfolio project for demonstrating how retrieval-augmented generation can be influenced by synthetic documents added to a car-brochure knowledge base, and how simple defenses can reduce the effect.

## Final demo concept

```text
Clean RAG → Poisoned RAG → Defended RAG
```

The clean knowledge base uses readable official car brochure PDFs. The live demo accepts free-form questions.

The project keeps retrieval compromise separate from generation compromise and reports real experiment metrics rather than hardcoded percentages.

## Final Figma

https://www.figma.com/design/m8D51hB7Q9KA8llSRHRBhb/RAG-Poisoning-Testbed-%E2%80%94-Muhammad-Hasan-Dad-Khan?node-id=1-169

## Core rule

> **DO NOT USE TOKENS HEAVILY.**

This is a local portfolio/research demo, not production software. Keep the implementation simple, fast, understandable, and inexpensive to run.

## Development phases

1. Setup, Clean Data, and Baseline RAG
2. Poisoning Attacks
3. Defenses
4. Evaluation and Error Analysis
5. Frontend, Demo, and Documentation

See `PROJECT_PLAN.md` for details.

## Start

Codex should begin with:

`00_START_HERE.md`

and execute only:

`PHASE_1_PROMPT.md`
