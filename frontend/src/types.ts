export type DefenseMode = "none" | "source_trust" | "instruction_filter" | "similarity_filter" | "combined";

export interface Source {
  rank: number;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_id: string;
  relevance_score: number | null;
  text: string;
  is_injected_test_document: boolean;
}

export interface DefenseTraceEntry {
  original_rank: number;
  filename: string;
  page_number: number;
  chunk_id: string;
  included: boolean;
  final_rank: number | null;
  stage_decisions: Array<{ stage: string; included: boolean; reason: string | null }>;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  latency_ms: number;
  defense_mode: DefenseMode;
  defense_latency_ms: number;
  defense_trace: DefenseTraceEntry[];
}

export interface CatalogDocument {
  document_id: string;
  filename: string;
  page_count: number;
  display_name: string;
  pdf_url: string;
}

export interface DocumentCatalog {
  official_clean: CatalogDocument[];
  synthetic_test: CatalogDocument[];
}

export interface ResultsSummary {
  benchmark: { question_count: number; conceptual_cells: number };
  clean_answer_quality_rate: number;
  clean_answer_correct: number;
  clean_answer_total: number;
  retrieval_attack_success_rate: number;
  undefended_attack_success_rate: number;
  conditional_attack_success_rate: number;
  selected_defense: "combined";
  selected_defense_attack_success_rate: number;
  average_extra_latency_ms: number;
  clean_control_accuracy_rate: number;
  defense_comparison: Record<string, number>;
  selected_defense_tradeoffs: {
    poison_removal_rate: number;
    poison_survival_rate: number;
    clean_false_rejection_rate: number;
  };
  generation_usage: { provider_calls: number; total_tokens: number };
}
