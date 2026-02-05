/**
 * Batch Answer Generation Types
 *
 * Types for batch answer generation from retrieval test results.
 * Answer-only mode: uses saved search results without performing new search.
 */

// ─── Search Result Types ───

export interface SearchResultItem {
  rank: number;
  doc_id: string;
  score: number;
  title: string;
  snippet: string;
  content: string; // Full content for LLM context
  chunk_id?: string;
  page?: number;
  device_name?: string;
  doc_type?: string;
  expanded_pages?: number[];
  expanded_page_urls?: string[];
}

// ─── Metrics Types ───

export interface RetrievalMetrics {
  hit_at_1: boolean;
  hit_at_3: boolean;
  hit_at_5: boolean;
  hit_at_10: boolean;
  reciprocal_rank: number | null;
  first_relevant_rank: number | null;
}

export interface RunProgress {
  total: number;
  completed: number;
  failed: number;
}

export interface RunMetrics {
  avg_rating: number | null;
  rating_count: number;
  avg_latency_ms: number | null;
  total_tokens: number;
  hit_at_1_ratio: number | null;
  hit_at_3_ratio: number | null;
  hit_at_5_ratio: number | null;
  mrr: number | null;
}

// ─── Source Config (from Retrieval Test) ───

export interface SourceConfig {
  dense_weight?: number;
  sparse_weight?: number;
  use_rrf?: boolean;
  rrf_k?: number;
  rerank?: boolean;
  rerank_top_k?: number;
  top_k?: number;
}

// ─── Run Types ───

export type RunStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface BatchAnswerRun {
  run_id: string;
  status: RunStatus;
  name: string | null;
  description: string | null;
  source_type: string;
  source_run_id: string | null;
  source_config: SourceConfig | null;
  progress: RunProgress;
  metrics: RunMetrics;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface RunListResponse {
  items: BatchAnswerRun[];
  total: number;
}

// ─── Result Types ───

export type ResultStatus = "pending" | "completed" | "failed";

export interface BatchAnswerResult {
  result_id: string;
  run_id: string;
  question_id: string;
  question: string;
  status: ResultStatus;
  answer: string;
  reasoning: string | null;
  search_results: SearchResultItem[];
  search_result_count: number;
  ground_truth_doc_ids: string[];
  category: string | null;
  retrieval_metrics: RetrievalMetrics;
  latency_ms: number | null;
  token_count: { input: number; output: number } | null;
  rating: number | null;
  rating_comment: string | null;
  rated_by: string | null;
  rated_at: string | null;
  error_message: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ResultListResponse {
  items: BatchAnswerResult[];
  total: number;
}

// ─── Request Types ───

export interface QuestionInput {
  id: string;
  question: string;
  ground_truth_doc_ids?: string[];
  groundTruthDocIds?: string[]; // Alternative naming
  search_results: SearchResultItem[];
  category?: string;
}

export interface CreateRunRequest {
  name?: string;
  description?: string;
  source_run_id?: string;
  source_config?: SourceConfig;
  questions: QuestionInput[];
}

export interface RatingRequest {
  rating: number; // 1-5
  comment?: string;
  rated_by?: string;
}

// ─── Execute Response ───

export interface ExecuteNextResponse {
  result_id: string;
  question_id: string;
  question: string;
  answer: string;
  reasoning: string | null;
  search_results: SearchResultItem[];
  metrics: RetrievalMetrics;
  progress: RunProgress;
  status: ResultStatus;
  error_message: string | null;
}
