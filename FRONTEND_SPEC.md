# Final Frontend Specification

## Design status

**FINALIZED — do not redesign unless the user explicitly asks.**

Figma:
https://www.figma.com/design/m8D51hB7Q9KA8llSRHRBhb/RAG-Poisoning-Testbed-%E2%80%94-Muhammad-Hasan-Dad-Khan?node-id=1-169

Owner name displayed in the UI:
**Muhammad Hasan Dad Khan**

## Final page order

1. About
2. Demo
3. Documents
4. Results

## Visual direction

- warm ivory page background
- clean light top navigation
- graphite/dark text
- controlled orange accent
- no heavy black navigation block
- active navigation item receives a subtle orange treatment
- simple premium automotive/research feel
- do not turn it into a cybersecurity dashboard

---

# About page

Purpose:
Explain the project before the visitor interacts with it.

Content:
- what RAG is
- how the experiment works
- simple threat model
- retrieval compromise versus generation compromise
- project focus

No extra features needed.

---

# Demo page

Purpose:
Show one live run.

Main flow:
1. User types any free-form question.
2. Run clean RAG.
3. Add/inject the selected synthetic test document.
4. Run attacked RAG.
5. Apply a defense.
6. Run defended RAG.
7. Compare.

Display:
- clean answer
- attacked answer
- defended answer
- retrieved documents
- source order/rank
- evaluation-only injected-document indicator
- current-run metrics

Current placeholders such as:
- `—`
- `— %`
- `— ms`
- `# —`
- `Yes / No`

must be populated later by real backend values.

### Poison rank

Meaning:
The position of the synthetic document in top-k retrieval.

Example:
`#2`

If not retrieved:
`Not Retrieved`

The RAG itself does not know the document is poison. The UI may mark it after retrieval using the hidden evaluation manifest.

---

# Documents page

Show:
- clean official brochure PDFs
- synthetic test PDFs in a clearly separate experiment section

The user should be able to open/read the PDFs.

The clean brochure list should reflect the actual files in `data/clean/`.

Synthetic PDFs should not contain internal “poison” labels inside the files themselves.

---

# Results page

Shows saved aggregate experiment metrics.

Important:
The Results page should **not** rerun the entire benchmark every time it loads.

Workflow:

```text
experiment runner
    ↓
real calculations
    ↓
JSON/CSV result file
    ↓
Results page
```

Do not hardcode fake values.

---

# Implementation scope

Frontend implementation occurs in Phase 5.

Use:
- React
- Vite
- TypeScript

Do not add:
- login
- dashboard pages
- admin pages
- charts that do not add clear value
- animation-heavy effects
- unnecessary configuration screens

The current feature set is sufficient.
