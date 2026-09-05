import { FormEvent, useState } from "react";
import { ask } from "../api";
import RunCard from "../components/RunCard";
import type { AskResponse, DefenseMode } from "../types";

interface DemoResults { clean: AskResponse; attack: AskResponse; defended: AskResponse; }
const normalizedAnswer = (value: string) => value.toLowerCase().replace(/\s+/g, " ").trim();

export default function Demo() {
  const [question, setQuestion] = useState("");
  const [defense, setDefense] = useState<Exclude<DefenseMode, "none">>("combined");
  const [results, setResults] = useState<DemoResults>();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const run = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setError("Enter a question before running the demo.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const [clean, attack, defended] = await Promise.all([
        ask(trimmed, "clean", "none"),
        ask(trimmed, "attacked", "none"),
        ask(trimmed, "attacked", defense)
      ]);
      setResults({ clean, attack, defended });
    } catch {
      setError("The backend is unavailable. Start the API and try again.");
    } finally {
      setLoading(false);
    }
  };

  const injected = results?.attack.sources.find((source) => source.is_injected_test_document);
  const answerChanged = results ? normalizedAnswer(results.clean.answer) !== normalizedAnswer(results.attack.answer) : undefined;
  const totalLatency = results ? results.clean.latency_ms + results.attack.latency_ms + results.defended.latency_ms : undefined;

  return (
    <main>
      <p className="demo-lead">Ask any question about the car brochures. Then inject a synthetic test document and see whether retrieval or the final answer changes.</p>
      <form className="panel question-panel" onSubmit={run}>
        <label htmlFor="question">Ask a free-form question</label>
        <div className="question-row">
          <input id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="e.g. Which Toyota model has the larger luggage capacity?" />
          <label className="defense-label" htmlFor="defense">Defense</label>
          <select id="defense" value={defense} onChange={(event) => setDefense(event.target.value as Exclude<DefenseMode, "none">)}>
            <option value="source_trust">Source trust</option>
            <option value="instruction_filter">Instruction filter</option>
            <option value="similarity_filter">Similarity filter</option>
            <option value="combined">Combined</option>
          </select>
          <button type="submit" disabled={loading}>{loading ? "Running…" : "Run Question"}</button>
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
      </form>
      <div className="run-grid">
        <RunCard number={1} title="Clean Run" subtitle="Official brochure answers only" tone="clean" result={results?.clean} />
        <RunCard number={2} title="Attack Run" subtitle="Synthetic document added to the same corpus" tone="attack" result={results?.attack} />
        <RunCard number={3} title="Defended Run" subtitle={`Apply ${defense.replaceAll("_", " ")} and run again`} tone="defended" result={results?.defended} />
      </div>
      <aside className="evaluation-note">
        <strong>Important: the RAG system never sees an “is_poison” label.</strong>
        <p>The interface marks injected test documents only after retrieval by matching document IDs against a hidden experiment manifest. This keeps the attack test fair.</p>
      </aside>
      <section className="panel metrics-panel">
        <h2>Current run metrics</h2>
        <div className="metric-row">
          <div><span>Poison retrieved</span><b>{injected ? "Yes" : results ? "No" : "—"}</b></div>
          <div><span>Poison rank</span><b>{injected ? `#${injected.rank}` : results ? "Not Retrieved" : "—"}</b></div>
          <div><span>Answer changed</span><b>{answerChanged === undefined ? "—" : answerChanged ? "Yes" : "No"}</b></div>
          <div><span>Latency</span><b>{totalLatency === undefined ? "—" : `${totalLatency.toFixed(1)} ms`}</b></div>
          <div><span>Sources used</span><b>{results?.attack.sources.length ?? "—"}</b></div>
        </div>
      </section>
    </main>
  );
}
