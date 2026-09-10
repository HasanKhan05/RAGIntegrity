export default function About() {
  return (
    <main>
      <section className="page-intro about-intro">
        <h1>RAGIntegrity — Evaluating Retrieval Poisoning Attacks and Defenses</h1>
        <p>A research demo showing how synthetic documents can influence retrieval-augmented generation and how simple defenses can reduce that effect.</p>
      </section>
      <section className="panel rag-panel">
        <h2>What is RAG?</h2>
        <p>The system searches a collection of official vehicle PDFs, retrieves the most relevant passages for your question, and gives those passages to an LLM to produce the answer.</p>
      </section>
      <div className="about-grid">
        <section className="panel">
          <h2>How the experiment works</h2>
          <ol className="flow-list">
            <li>Load official Toyota UK brochure PDFs.</li>
            <li>Ask any free-form question about the cars.</li>
            <li>Add a synthetic PDF containing a conflicting claim.</li>
            <li>Ask the same question again.</li>
            <li>Apply a simple defense and compare the answer.</li>
          </ol>
        </section>
        <section className="panel threat-panel">
          <h2>Simple threat model</h2>
          <strong>The attacker can add a new document to the corpus.</strong>
          <p>The attacker cannot edit trusted brochures, change the LLM, or modify the retriever.</p>
          <h3>Important distinction</h3>
          <p><b>Retrieval compromise</b> = the injected document appears in top-k.</p>
          <p><b>Generation compromise</b> = the injected document actually changes the answer.</p>
        </section>
      </div>
      <section className="panel focus-panel">
        <h2>Project focus</h2>
        <p>Keep the system understandable: readable PDFs, free-form questions, visible retrieved sources, simple defenses, and real experiment metrics. The goal is a clear portfolio demonstration, not a production security platform.</p>
      </section>
    </main>
  );
}
