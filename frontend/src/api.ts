import type { AskResponse, DefenseMode, DocumentCatalog, ResultsSummary } from "./types";

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

export const apiUrl = (path: string) => `${apiBase}${path}`;

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const ask = (question: string, corpusMode: "clean" | "attacked", defenseMode: DefenseMode) =>
  fetchJson<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, corpus_mode: corpusMode, defense_mode: defenseMode })
  });

export const loadDocumentCatalog = () => fetchJson<DocumentCatalog>("/documents/catalog");
export const loadResultsSummary = () => fetchJson<ResultsSummary>("/results/summary");
