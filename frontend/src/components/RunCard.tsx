import type { AskResponse } from "../types";

interface Props {
  number: number;
  title: string;
  subtitle: string;
  tone: "clean" | "attack" | "defended";
  result?: AskResponse;
}

export default function RunCard({ number, title, subtitle, tone, result }: Props) {
  return (
    <section className={`run-card run-card--${tone}`} aria-label={`${title} result`}>
      <h2>{number}&nbsp; {title}</h2>
      <p className="muted">{subtitle}</p>
      <h3>Retrieved documents</h3>
      {result?.sources.length ? (
        <ol className="source-list">
          {result.sources.map((source) => (
            <li key={source.chunk_id}>
              <span>#{source.rank} {source.filename}, p. {source.page_number}</span>
              {source.is_injected_test_document && <b>Injected Test Document</b>}
            </li>
          ))}
        </ol>
      ) : <p className="empty-value">No retrieved sources yet.</p>}
      <h3>Answer</h3>
      <p className="answer">{result?.answer ?? "Run a question to see the grounded answer."}</p>
      {result && <p className="latency">{result.latency_ms.toFixed(1)} ms · {result.defense_mode.replaceAll("_", " ")}</p>}
      {tone === "defended" && result?.defense_trace.some((entry) => !entry.included) && (
        <details>
          <summary>Defense trace</summary>
          <ul>{result.defense_trace.filter((entry) => !entry.included).map((entry) => <li key={entry.chunk_id}>{entry.filename}, p. {entry.page_number} — filtered</li>)}</ul>
        </details>
      )}
    </section>
  );
}
