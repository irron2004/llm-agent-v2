export interface TestQuestion {
  id: string;
  question: string;
  groundTruthDocIds: string[];
  category?: string;
  difficulty?: "easy" | "medium" | "hard";
}

export interface FieldConfig {
  field: string;
  label: string;
  enabled: boolean;
  weight: number;
}

export interface SearchConfig {
  denseWeight: number;
  sparseWeight: number;
  rerank: boolean;
  rerankModel: string;
  rerankTopK: number;
  multiQuery: boolean;
  multiQueryN: number;
  fieldWeights: FieldConfig[];
  size: number; // 검색 결과 개수
}

export interface SearchResult {
  rank: number;
  id: string;
  title: string;
  snippet: string;
  score: number;
  score_display: string;
  chunk_summary?: string;
  chunk_keywords?: string[];
  chapter?: string;
  doc_type?: string;
  device_name?: string;
  page?: number;
}

export interface ResultMetrics {
  hit_at_1: boolean;
  hit_at_3: boolean;
  hit_at_5: boolean;
  hit_at_10: boolean;
  reciprocal_rank: number | null;
  first_relevant_rank: number | null;
}

export interface RetrievalTestResult {
  questionId: string;
  question: string;
  searchResults: SearchResult[];
  groundTruthDocIds: string[];
  metrics: ResultMetrics;
  config: SearchConfig;
  timestamp: string;
}

export interface AggregatedMetrics {
  total_queries: number;
  hit_at_1: number;
  hit_at_3: number;
  hit_at_5: number;
  hit_at_10: number;
  mrr: number;
  avg_first_relevant_rank: number | null;
}
