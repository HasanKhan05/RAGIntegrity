import { useEffect, useState } from "react";
import { loadResultsSummary } from "../api";
import type { ResultsSummary } from "../types";

const percent = (value: number | null | undefined) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const labels: Record<string, string> = {
  attacked_none: "No defense",
  source_trust: "Source trust filter",
  instruction_filter: "Instruction filter",
  similarity_filter: "Similarity / duplicate filter",
  combined: "Combined defense"
};

export default function Results() {
  const [summary, setSummary] = useState<ResultsSummary>();
  const [error, setError] = useState(false);
  useEffect(() => { loadResultsSummary().then(setSummary).catch(() => setError(true)); }, []);
  return (
    <main>
      <section className="page-intro results-intro">
        <h1>Experiment Results</h1>
        <p>This page shows saved results from the full evaluation run — it does not rerun dozens of LLM queries when the page opens.</p>
      </section>
      {error ? <p className="status-message" role="alert">Saved Phase 4 results are unavailable.</p> : !summary ? <p className="status-message">Loading saved results…</p> : (
        <>
          <div className="headline-metrics">
            <article><span>Clean answer quality</span><strong>{percent(summary.clean_answer_quality_rate)}</strong><small>{summary.clean_answer_correct} / {summary.clean_answer_total} clean attack questions</small></article>
            <article><span>Generation attack success</span><strong>{percent(summary.undefended_attack_success_rate)}</strong><small>before defense</small></article>
            <article><span>After combined defense</span><strong>{percent(summary.selected_defense_attack_success_rate)}</strong><small>attack success</small></article>
            <article><span>Average extra latency</span><strong>{summary.average_extra_latency_ms.toFixed(1)} ms</strong><small>defense processing overhead</small></article>
          </div>
          <div className="results-details">
            <section className="panel defense-comparison">
              <h2>Defense comparison</h2><p className="muted">Attack success rate — lower is better</p>
              <ul>{Object.entries(summary.defense_comparison).map(([mode, value]) => <li key={mode}><span>{labels[mode] ?? mode}</span><i aria-hidden="true" /><b>{percent(value)}</b></li>)}</ul>
            </section>
            <section className="panel reading-panel">
              <h2>How to read these results</h2>
              <dl>
                <dt>Retrieval attack success</dt><dd>How often an injected test document enters top-k retrieval.</dd>
                <dt>Generation attack success</dt><dd>How often the injected information actually changes the final answer.</dd>
                <dt>False rejection</dt><dd>How often a defense wrongly removes a clean official document.</dd>
              </dl>
              <div className="saved-run-facts">
                <b>{summary.benchmark.question_count} questions · {summary.benchmark.conceptual_cells} scored cells</b>
                <span>Retrieval success {percent(summary.retrieval_attack_success_rate)}</span>
                <span>Conditional ASR {percent(summary.conditional_attack_success_rate)}</span>
                <span>Clean controls {percent(summary.clean_control_accuracy_rate)}</span>
                <span>Poison removed {percent(summary.selected_defense_tradeoffs.poison_removal_rate)}</span>
                <span>Clean false rejection {percent(summary.selected_defense_tradeoffs.clean_false_rejection_rate)}</span>
                <span>{summary.generation_usage.provider_calls} calls · {summary.generation_usage.total_tokens.toLocaleString()} tokens</span>
              </div>
            </section>
          </div>
          <p className="results-footer">Live poison rank and latency belong on the Demo page; saved aggregate percentages belong here.</p>
        </>
      )}
    </main>
  );
}
