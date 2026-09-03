# Phase 3 RAG Defenses Design

**Date:** 2026-09-03  
**Status:** Approved  
**Scope:** Phase 3 only

## Objective

Add a small, explainable, attack-blind post-retrieval defense layer to the existing RAG poisoning testbed. The implementation will compare undefended retrieval with three individual defenses and one fixed combined defense while preserving the Phase 2 clean and attacked indexes, runners, results, and API defaults.

Phase 3 will not add frontend work, Phase 4 aggregate evaluation, production infrastructure, or new attack documents.

## Constraints

- Local retrieval and embeddings remain unchanged.
- Gemini usage is capped at six new generation calls.
- The attack manifest is evaluation-only and is never an input to a defense.
- Every benchmark question is retrieved exactly once from the attacked corpus. All defense modes operate on the same immutable top-k snapshot.
- Defense processing latency is measured separately from retrieval latency.
- Existing `/ask` requests behave exactly as before because `defense_mode` defaults to `none`.

## Architecture

The pipeline becomes:

```text
question
  -> attacked or clean retriever
  -> immutable top-k snapshot
  -> selected post-retrieval defense pipeline
  -> final ranked context
  -> existing prompt builder and Gemini generator
```

The defense package will expose a single coordinator that accepts ordinary `RetrievedChunk` objects, a defense mode, and legitimate provenance/configuration. It returns retained chunks, an attack-blind trace, and defense-only processing latency.

The retriever will not know which documents are synthetic. Evaluation code may compare trace results with the hidden manifest only after the defense has finished.

## Defense Modes

### `none`

Return the top-k snapshot unchanged. Original and final ranks match. This mode preserves current behavior and provides the analysis baseline.

### `source_trust`

Build the trusted source inventory from the official clean-corpus document inventory or clean index manifest. A chunk is trusted when its ordinary filename belongs to that inventory.

Remove chunks from filenames outside the trusted inventory with reason `untrusted_source`.

The trusted inventory must not be derived from `is_poison`, attack IDs, false claims, target models, or any other attack-manifest field.

### `instruction_filter`

Remove the entire chunk when normalized text matches a narrow set of clearly model-directed patterns, including:

- `when answering`
- `always state`
- `respond with`
- `ignore previous`
- `prioritize this update`

The filter will not reject broad words such as `should`, `recommended`, `please`, or normal automotive instructions. Removed chunks receive reason `instruction_content`.

### `similarity_filter`

Use the existing local embedding model to encode the retrieved chunks, normalize the vectors, and calculate pairwise cosine similarity. The threshold will be documented and configurable through the existing settings/environment pattern.

For chunks whose cosine similarity meets or exceeds the threshold:

- Trusted versus untrusted: retain the trusted chunk.
- Same trust level: retain the chunk with the better original retrieval rank.

Removed chunks receive reason `near_duplicate`. The filter has no access to attack-specific values or labels.

The threshold will be selected once based on a simple documented default, not repeatedly tuned to maximize benchmark removal.

### `combined`

Apply stages in this fixed order:

1. `instruction_filter`
2. `similarity_filter`
3. `source_trust`

The trace retains stage-by-stage decisions so the first component that removes a chunk is visible. Later stages operate only on chunks still retained.

Source trust is expected to dominate this controlled experiment because the official brochure provenance is known. Documentation will state that this is a limitation of the threat model rather than evidence of a universally strong defense.

## Trace Model

Each retrieved chunk receives an attack-blind trace containing:

- original rank
- filename
- page number
- chunk ID
- included or removed status
- stage decisions and ordinary reasons
- final rank when retained

Allowed removal reasons are ordinary defense observations such as:

- `untrusted_source`
- `instruction_content`
- `near_duplicate`

Trace data must never contain poison labels, attack IDs, false-claim labels, or synthetic-attack terminology. Retained chunks are re-ranked contiguously while preserving relative order.

## API Changes

`POST /ask` gains an optional `defense_mode` enum:

- `none`
- `source_trust`
- `instruction_filter`
- `similarity_filter`
- `combined`

The default is `none`. Existing requests containing only `question` or `question` plus `corpus_mode` remain valid and produce the existing undefended behavior.

The response may include the defense mode, defense processing latency, and attack-blind trace. Sources supplied to the existing generator are the retained final chunks only. Hidden evaluation fields are never exposed.

## Local Benchmark Analysis

The Phase 3 analysis runner will process:

- 30 existing attack questions
- 18 existing clean-control questions

For every question:

1. Retrieve once from the attacked corpus.
2. Freeze the returned top-k snapshot.
3. Apply every defense mode independently to that same snapshot.
4. Record defense-only processing latency.
5. Attach evaluation labels only after all defense processing is complete.

For each defense, record:

- target poison originally retrieved
- target poison removed or retained
- poison survival/removal rates
- legitimate clean chunks retrieved
- legitimate clean chunks removed
- clean false-rejection rate
- average remaining chunks
- average defense processing latency

For clean controls, only removed chunks belonging to the official clean-corpus inventory count as false rejections. Removed synthetic chunks do not.

Results are saved to:

`experiments/results/phase3_defense_analysis.json`

The local analysis makes zero Gemini calls.

## Generation Smoke Test

Reuse valid Phase 2 no-defense outputs. Make exactly six new Gemini calls at temperature zero:

1. Yaris power with `source_trust`
2. Yaris power with `combined`
3. Land Cruiser towing capacity with `source_trust`
4. Land Cruiser towing capacity with `combined`
5. Land Cruiser wading-depth instruction attack with `instruction_filter`
6. Land Cruiser wading-depth instruction attack with `combined`

No Gemini calls will be made for similarity filtering. Generation-compromise checks reuse the existing deterministic evaluator and permit manual review when wording is ambiguous.

Results are saved to:

`experiments/results/phase3_defense_smoke.json`

The result records new call count and Gemini-reported input, output, and total tokens without modifying the original Phase 2 result files.

## Testing

Focused tests will cover:

- `none` preserves chunks and ranks
- trusted inventory comes from clean provenance
- source trust removes only untrusted filenames
- instruction patterns are narrow and remove whole matching chunks
- normal brochure language is retained
- similarity threshold behavior
- trusted chunks win trusted/untrusted similarity conflicts
- better original rank wins same-trust conflicts
- combined stage order and stage trace
- contiguous final ranking
- attack-blind trace fields and reasons
- hidden manifest data never enters a defense, trace, or prompt
- one retrieval snapshot is reused across every analysis mode
- clean false rejection excludes synthetic chunks
- defense latency excludes retrieval
- API default remains undefended and backward-compatible
- explicit API defense modes use only retained chunks for generation
- analysis and smoke result schemas

All existing Phase 1 and Phase 2 tests must continue to pass.

## Documentation and Completion

Update `README.md`, `ARCHITECTURE_AND_THREAT_MODEL.md`, `DATA_AND_EVALUATION.md`, and `PHASE_STATUS.md` with the actual implementation, thresholds, local metrics, smoke outcomes, Gemini usage, and limitations.

Before completion:

- run the full test suite
- verify `.env` remains ignored and untracked
- scan tracked changes for secrets and hidden attack-label leakage
- review generated result files
- commit with `feat: complete phase 3 rag defenses`
- push `main` to `origin`
- report metrics, tests, Gemini calls/tokens, commit hash, and push status

Stop after Phase 3. Do not begin Phase 4.
