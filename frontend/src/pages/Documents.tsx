import { useEffect, useState } from "react";
import { apiUrl, loadDocumentCatalog } from "../api";
import type { CatalogDocument, DocumentCatalog } from "../types";

function DocumentList({ documents, synthetic = false }: { documents: CatalogDocument[]; synthetic?: boolean }) {
  if (!documents.length) return <p className="empty-value">No documents are available.</p>;
  return (
    <ul className="document-list">
      {documents.map((document) => (
        <li key={document.document_id}>
          <div>
            <strong>{document.display_name}</strong>
            <span>{document.page_count} {document.page_count === 1 ? "page" : "pages"}</span>
            {synthetic && <em>Synthetic / evaluation-controlled</em>}
          </div>
          <a href={apiUrl(document.pdf_url)} target="_blank" rel="noreferrer">View PDF</a>
        </li>
      ))}
    </ul>
  );
}

export default function Documents() {
  const [catalog, setCatalog] = useState<DocumentCatalog>();
  const [error, setError] = useState(false);
  useEffect(() => { loadDocumentCatalog().then(setCatalog).catch(() => setError(true)); }, []);
  return (
    <main>
      <section className="page-intro">
        <h1>Documents</h1>
        <p>The knowledge base is made from readable PDFs. Clean documents come from official manufacturer sources; synthetic documents are created only for controlled attack experiments.</p>
      </section>
      {error ? <p className="status-message" role="alert">The document catalog is unavailable.</p> : !catalog ? <p className="status-message">Loading documents…</p> : (
        <div className="documents-grid">
          <section className="panel">
            <h2>Official clean PDFs</h2>
            <p className="muted">Source rule: the downloaded official Toyota UK brochures are the experiment ground truth.</p>
            <DocumentList documents={catalog.official_clean} />
          </section>
          <section className="panel synthetic-panel">
            <h2>Synthetic test PDFs</h2>
            <p className="muted">Controlled research artifacts that conflict with facts in the clean brochures. They are not Toyota publications.</p>
            <DocumentList documents={catalog.synthetic_test} synthetic />
            <p className="synthetic-note">The hidden evaluation manifest keeps attack identity separate from RAG-visible metadata.</p>
          </section>
        </div>
      )}
      <p className="document-footer">You can open the PDFs yourself, read them, and ask any free-form question you want.</p>
    </main>
  );
}
