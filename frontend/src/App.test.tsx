import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const askResponse = (answer: string, injected = false) => ({
  answer,
  sources: [
    {
      rank: 1,
      document_id: injected ? "attack-doc" : "clean-doc",
      filename: injected ? "vehicle_update.pdf" : "rav4.pdf",
      page_number: 3,
      chunk_id: "chunk-1",
      relevance_score: 0.91,
      text: "context",
      is_injected_test_document: injected
    }
  ],
  latency_ms: 120.5,
  token_usage: null,
  defense_mode: "none",
  defense_latency_ms: 0.1,
  defense_trace: []
});

const jsonResponse = (value: unknown) =>
  Promise.resolve({ ok: true, json: () => Promise.resolve(value) } as Response);

afterEach(() => vi.unstubAllGlobals());

describe("portfolio application", () => {
  it("renders the About page and navigates through all four routes", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) =>
      String(input).includes("/results/")
        ? jsonResponse({
            benchmark: { question_count: 48, conceptual_cells: 288 },
            clean_answer_quality_rate: 0.933,
            clean_answer_correct: 28,
            clean_answer_total: 30,
            retrieval_attack_success_rate: 0.767,
            undefended_attack_success_rate: 0.733,
            conditional_attack_success_rate: 0.957,
            selected_defense: "combined",
            selected_defense_attack_success_rate: 0,
            average_extra_latency_ms: 143.05,
            clean_control_accuracy_rate: 0.833,
            defense_comparison: {},
            selected_defense_tradeoffs: { poison_removal_rate: 1, poison_survival_rate: 0, clean_false_rejection_rate: 0.017 },
            generation_usage: { provider_calls: 103, total_tokens: 81403 }
          })
        : jsonResponse({ official_clean: [], synthetic_test: [] })
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />, { wrapper: MemoryRouter });

    expect(screen.getByRole("heading", { name: "About the Project" })).toBeVisible();
    expect(screen.getByText("What is RAG?")).toBeVisible();

    await userEvent.click(screen.getByRole("link", { name: "Demo" }));
    expect(screen.getByLabelText("Ask a free-form question")).toBeVisible();
    await userEvent.click(screen.getByRole("link", { name: "Documents" }));
    expect(screen.getByRole("heading", { name: "Documents" })).toBeVisible();
    await userEvent.click(screen.getByRole("link", { name: "Results" }));
    expect(screen.getByRole("heading", { name: "Experiment Results" })).toBeVisible();
  });

  it("accepts a free-form question and requests clean, attacked, and defended runs", async () => {
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(() => jsonResponse(askResponse("Clean answer")))
      .mockImplementationOnce(() => jsonResponse(askResponse("Attack answer", true)))
      .mockImplementationOnce(() => jsonResponse({ ...askResponse("Defended answer"), defense_mode: "combined" }));
    vi.stubGlobal("fetch", fetchMock);
    render(<App />, { wrapper: ({ children }) => <MemoryRouter initialEntries={["/demo"]}>{children}</MemoryRouter> });

    await userEvent.type(screen.getByLabelText("Ask a free-form question"), "Which model has more cargo room?");
    await userEvent.click(screen.getByRole("button", { name: "Run Question" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    const bodies = fetchMock.mock.calls.map((call) => JSON.parse(String((call[1] as RequestInit).body)));
    expect(bodies).toEqual([
      { question: "Which model has more cargo room?", corpus_mode: "clean", defense_mode: "none" },
      { question: "Which model has more cargo room?", corpus_mode: "attacked", defense_mode: "none" },
      { question: "Which model has more cargo room?", corpus_mode: "attacked", defense_mode: "combined" }
    ]);
    expect(JSON.stringify(bodies)).not.toMatch(/is_poison|attack_type|false_claim/);
    expect(await screen.findByText("Clean answer")).toBeVisible();
    expect(screen.getByText("Attack answer")).toBeVisible();
    expect(screen.getByText("Defended answer")).toBeVisible();
    expect(screen.getByText("Injected Test Document")).toBeVisible();
    expect(screen.getByText("#1")).toBeVisible();
  });

  it("validates an empty question and exposes a simple request error", async () => {
    const fetchMock = vi.fn(() => Promise.reject(new Error("offline")));
    vi.stubGlobal("fetch", fetchMock);
    render(<App />, { wrapper: ({ children }) => <MemoryRouter initialEntries={["/demo"]}>{children}</MemoryRouter> });

    await userEvent.click(screen.getByRole("button", { name: "Run Question" }));
    expect(screen.getByText("Enter a question before running the demo.")).toBeVisible();
    await userEvent.type(screen.getByLabelText("Ask a free-form question"), "RAV4 capacity?");
    await userEvent.click(screen.getByRole("button", { name: "Run Question" }));
    expect(await screen.findByText("The backend is unavailable. Start the API and try again.")).toBeVisible();
  });

  it("shows a loading state while the three live requests are pending", async () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => undefined)));
    render(<App />, { wrapper: ({ children }) => <MemoryRouter initialEntries={["/demo"]}>{children}</MemoryRouter> });
    await userEvent.type(screen.getByLabelText("Ask a free-form question"), "Any question");
    await userEvent.click(screen.getByRole("button", { name: "Run Question" }));
    expect(screen.getByRole("button", { name: "Running…" })).toBeDisabled();
  });

  it("loads official and synthetic PDF links from the backend catalog", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({
      official_clean: [{ document_id: "c1", filename: "rav4.pdf", page_count: 56, display_name: "Toyota RAV4 — official brochure", pdf_url: "/documents/file/clean/rav4.pdf" }],
      synthetic_test: [{ document_id: "s1", filename: "vehicle_update.pdf", page_count: 1, display_name: "Vehicle Update", pdf_url: "/documents/file/synthetic/vehicle_update.pdf" }]
    })));
    render(<App />, { wrapper: ({ children }) => <MemoryRouter initialEntries={["/documents"]}>{children}</MemoryRouter> });

    expect(await screen.findByText("Toyota RAV4 — official brochure")).toBeVisible();
    expect(screen.getByText("Vehicle Update")).toBeVisible();
    const official = screen.getByText("Toyota RAV4 — official brochure").closest("li")!;
    expect(within(official).getByRole("link", { name: "View PDF" })).toHaveAttribute("href", "/api/documents/file/clean/rav4.pdf");
  });

  it("renders real saved metrics from the results endpoint", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({
      benchmark: { question_count: 48, conceptual_cells: 288 },
      clean_answer_quality_rate: 0.9333333333,
      clean_answer_correct: 28,
      clean_answer_total: 30,
      retrieval_attack_success_rate: 0.7666666667,
      undefended_attack_success_rate: 0.7333333333,
      conditional_attack_success_rate: 0.9565217391,
      selected_defense: "combined",
      selected_defense_attack_success_rate: 0,
      average_extra_latency_ms: 143.05,
      clean_control_accuracy_rate: 0.8333333333,
      defense_comparison: { attacked_none: 0.7333333333, source_trust: 0, instruction_filter: 0.6666666667, similarity_filter: 0.7333333333, combined: 0 },
      selected_defense_tradeoffs: { poison_removal_rate: 1, poison_survival_rate: 0, clean_false_rejection_rate: 0.0172413793 },
      generation_usage: { provider_calls: 103, total_tokens: 81403 }
    })));
    render(<App />, { wrapper: ({ children }) => <MemoryRouter initialEntries={["/results"]}>{children}</MemoryRouter> });

    expect(await screen.findByText("93.3%")).toBeVisible();
    expect(screen.getAllByText("73.3%").length).toBeGreaterThan(0);
    expect(screen.getByText("143.1 ms")).toBeVisible();
    expect(screen.getByText(/48 questions · 288 scored cells/)).toBeVisible();
    expect(screen.getByText(/Retrieval success 76.7%/)).toBeVisible();
    expect(screen.queryByText("— %")).not.toBeInTheDocument();
  });
});
