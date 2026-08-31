# Data and Evaluation Plan

## Clean corpus

The initial clean corpus should use **official Toyota UK brochure PDFs** supplied/downloaded by the user.

Suggested models:

- Toyota Corolla
- Toyota Yaris
- Toyota Yaris Cross
- Toyota C-HR
- Toyota RAV4
- Toyota bZ4X
- Toyota Aygo X

The project does not require all seven before basic ingestion code can exist, but meaningful RAG testing requires actual brochures.

## Source rule

The exact downloaded brochure is the experiment ground truth for that run.

Do not silently replace the brochure's facts with:
- current web search results
- dealer listings
- third-party specification sites
- scraped databases

A later brochure revision may differ; that does not invalidate the experiment using the earlier downloaded PDF.

## Clean source manifest

Store basic source metadata, for example:

```json
{
  "document_id": "toyota_rav4_uk_2024",
  "filename": "toyota_rav4_uk_2024.pdf",
  "manufacturer": "Toyota",
  "model": "RAV4",
  "market": "UK",
  "source": "Toyota UK",
  "trusted_provenance": true
}
```

`trusted_provenance` is realistic source metadata. It is not the same as hidden poison ground truth.

---

# Synthetic attack documents

Create them in **Phase 2**, after clean brochure facts are available.

Use approximately 3–5 synthetic PDFs.

Good target facts:
- horsepower / power output
- luggage capacity
- dimensions
- fuel tank capacity
- equipment availability
- trim features

Synthetic documents should:
- be readable PDFs
- look like ordinary informational documents
- conflict with a fact in the clean corpus
- not contain the words “poisoned” or “fake”
- be tracked in a separate hidden attack manifest

The project may create these PDFs with ReportLab.

---

# Free-form questions

The live Demo page must accept arbitrary user questions.

Do not force users to choose from a fixed question list.

Examples shown in docs or UI are only examples.

---

# Repeatable evaluation benchmark

Aggregate percentages require a repeatable benchmark.

This benchmark is not the live-demo question system.

In Phase 4, build a small factual set from the actual brochures. Keep it modest to protect token usage.

Recommended initial size:
- approximately 20–30 questions total
- mostly short factual questions
- enough coverage to compare clean / attacked / defended behavior

Prefer questions with objective expected answers.

---

# Metrics

## Live Demo metrics

Calculated for the current run:
- synthetic document retrieved: yes/no
- poison rank: `#1`, `#2`, etc., or `Not Retrieved`
- answer changed: yes/no when comparison is available
- latency
- sources used
- token usage when the provider reports it

## Aggregate Results metrics

Calculated by the experiment runner and saved:
- clean answer correctness
- retrieval attack success rate
- generation attack success rate
- attack success after each defense
- source/citation correctness
- clean false rejection rate
- average latency
- defense latency overhead
- token usage

The Results page must read these values from experiment output files. Never hardcode invented percentages.

---

# Correctness strategy

To keep token usage low:

1. Prefer objective car-spec facts.
2. Store expected answers in the evaluation set.
3. Use normalized deterministic comparisons where reasonable.
4. Use manual review for a small number of ambiguous cases.
5. Use LLM-as-judge only if necessary and only on cases deterministic checks cannot handle.

Do not make an LLM judge the default for every evaluation item.

---

# Error analysis

Keep a small set of representative cases:

- synthetic document retrieved, but final answer remained correct
- synthetic document changed the answer without being rank #1
- defense rejected a useful clean brochure chunk
- answer was wrong even without successful poisoning

A few strong examples are more useful than a huge error-analysis pipeline.
