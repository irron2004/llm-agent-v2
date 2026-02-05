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
  useRrf: boolean; // RRF 사용 시 가중치 무시됨
  rrfK: number; // RRF constant
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
  content?: string; // 본문 전체 (LLM 컨텍스트용, max 10,000자)
  score: number;
  score_display: string;
  chunk_summary?: string;
  chunk_keywords?: string[];
  chapter?: string;
  doc_type?: string;
  device_name?: string;
  page?: number;
  // Expanded pages (Chat/Search와 동일한 로직)
  expanded_pages?: number[];
  expanded_page_urls?: string[];
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

// Chat Pipeline 검색 요청/응답 타입
export interface ChatPipelineSearchRequest {
  query: string;
  search_override?: {
    dense_weight?: number;
    sparse_weight?: number;
    use_rrf?: boolean;
    rrf_k?: number;
    rerank?: boolean;
    rerank_top_k?: number;
    top_k?: number;
  };
  selected_devices?: string[];
  selected_doc_types?: string[];
}

export interface ChatPipelineSearchResponse {
  query: string;
  query_en?: string;
  query_ko?: string;
  route?: string;
  search_queries: string[];
  auto_parsed_device?: string;
  auto_parsed_doc_type?: string;
  items: SearchResult[];
  total: number;
}
