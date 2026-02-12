export type MessageRole = "user" | "assistant" | "system";

export type FeedbackRating = "up" | "down";

export type MessageFeedback = {
  rating: FeedbackRating;
  reason?: string | null;
  ts?: string | null;
  // Detailed feedback scores (from separate feedback index)
  accuracy?: number | null;      // 1-5
  completeness?: number | null;  // 1-5
  relevance?: number | null;     // 1-5
  avgScore?: number | null;
  comment?: string | null;
};

export type SuggestedDevice = {
  name: string;
  count: number;
};

export type Message = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt?: string;
  edited?: boolean;
  originalQuery?: string;
  reference?: Reference;
  rawAnswer?: string;
  retrievedDocs?: RetrievedDoc[];
  allRetrievedDocs?: RetrievedDoc[];  // 전체 검색 문서 (재생성용, 20개)
  logs?: string[];
  currentNode?: string | null;
  sessionId?: string;
  turnId?: number;
  feedback?: MessageFeedback | null;
  // Auto-parse and filter info for regeneration
  autoParse?: AutoParseResult | null;
  selectedDevices?: string[] | null;
  selectedDocTypes?: string[] | null;
  searchQueries?: string[] | null;
  // Device suggestion (장비 미지정 시)
  suggestedDevices?: SuggestedDevice[];
};

export type DeviceInfo = {
  name: string;
  doc_count: number;
};

export type DocTypeInfo = {
  name: string;
  doc_count: number;
};

export type DeviceCatalogResponse = {
  devices: DeviceInfo[];
  doc_types: DocTypeInfo[];
  vis?: string[];
};

export type ReferenceChunk = {
  id: string;
  content: string;
  documentId?: string;
  documentName?: string;
  similarity?: number;
};

export type Reference = {
  chunks: ReferenceChunk[];
  total?: number;
};

export type RetrievedDoc = {
  id: string;
  title: string;
  snippet: string;
  score?: number | null;
  score_percent?: number | null;
  metadata?: Record<string, unknown> | null;
  page?: number | null;
  page_image_url?: string | null;
  expanded_pages?: number[] | null;
  expanded_page_urls?: string[] | null;
};

export type ReviewDoc = {
  docId: string;
  rank: number;
  content: string;
  title?: string | null;
  page?: number | null;
  page_image_url?: string | null;
  score?: number | null;
  metadata?: Record<string, unknown> | null;
};

export type AutoParseResult = {
  device?: string | null;
  doc_type?: string | null;
  devices?: string[] | null;
  doc_types?: string[] | null;
  message?: string | null;
};

export type AgentResponse = {
  query: string;
  answer: string;
  judge?: Record<string, unknown>;
  retrieved_docs?: RetrievedDoc[];
  all_retrieved_docs?: RetrievedDoc[];  // 전체 검색 문서 (재생성용, 20개)
  metadata?: Record<string, unknown>;
  interrupted?: boolean;
  interrupt_payload?: Record<string, unknown> | null;
  thread_id?: string | null;
  // Auto-parse results
  auto_parse?: AutoParseResult | null;
  selected_devices?: string[] | null;
  selected_doc_types?: string[] | null;
  search_queries?: string[] | null;
  // Device suggestion (장비 미지정 시)
  suggested_devices?: SuggestedDevice[];
};

export type AgentRequest = {
  message: string;
  top_k?: number;
  max_attempts?: number;
  mode?: string;
  thread_id?: string | null;
  session_id?: string | null;
  save_history?: boolean;
  ask_user_after_retrieve?: boolean;
  ask_device_selection?: boolean;
  resume_decision?: unknown;
  auto_parse?: boolean;
  filter_devices?: string[] | null;
  filter_doc_types?: string[] | null;
  search_queries?: string[] | null;
  selected_doc_ids?: string[] | null;
};

export type Conversation = {
  id: string;
  title?: string;
  messages: Message[];
};

// ─── Conversation API Types ───

export type DocRefResponse = {
  slot: number;
  doc_id: string;
  title: string;
  snippet: string;
  page?: number | null;
  pages?: number[] | null;
  score?: number | null;
};

export type TurnResponse = {
  session_id: string;
  turn_id: number;
  user_text: string;
  assistant_text: string;
  doc_refs: DocRefResponse[];
  title?: string | null;
  ts: string;
  edited?: boolean | null;
  parent_session_id?: string | null;
  branched_from_turn_id?: number | null;
  is_branch?: boolean | null;
  feedback_rating?: FeedbackRating | null;
  feedback_reason?: string | null;
  feedback_ts?: string | null;
  auto_parse_message?: string | null;
};

export type SessionListItem = {
  session_id: string;  // API returns session_id, not id
  title: string;
  preview: string;
  turnCount: number;
  createdAt: string;
  updatedAt: string;
  isBranch?: boolean;
  parentSessionId?: string | null;
  branchedFromTurnId?: number | null;
};

export type SessionListResponse = {
  sessions: SessionListItem[];
  total: number;
};

export type SessionDetailResponse = {
  session_id: string;
  title: string;
  turns: TurnResponse[];
  turn_count: number;
  is_branch?: boolean;
  parent_session_id?: string | null;
  branched_from_turn_id?: number | null;
};

export type SaveTurnRequest = {
  user_text: string;
  assistant_text: string;
  doc_refs: DocRefResponse[];
  title?: string | null;
  edited?: boolean | null;
  parent_session_id?: string | null;
  branched_from_turn_id?: number | null;
  is_branch?: boolean | null;
  auto_parse_message?: string | null;
};

// --- Detailed Feedback Types (for feedback index) ---

export type DetailedFeedbackRequest = {
  accuracy: number;      // 1-5
  completeness: number;  // 1-5
  relevance: number;     // 1-5
  comment?: string | null;
  reviewer_name?: string | null;  // 피드백 제출자 이름 (선택)
  logs?: string[] | null;
  user_text?: string | null;
  assistant_text?: string | null;
};

export type FeedbackResponse = {
  session_id: string;
  turn_id: number;
  user_text: string;
  assistant_text: string;
  accuracy: number;
  completeness: number;
  relevance: number;
  avg_score: number;
  rating: string;
  comment?: string | null;
  reviewer_name?: string | null;
  logs?: string[] | null;
  ts: string;
};

export type FeedbackListResponse = {
  items: FeedbackResponse[];
  total: number;
};

export type FeedbackStatisticsResponse = {
  total_count: number;
  avg_accuracy?: number | null;
  avg_completeness?: number | null;
  avg_relevance?: number | null;
  avg_score?: number | null;
  rating_distribution: Record<string, number>;
};

// --- Retrieval Evaluation Types (query-unit storage) ---

/**
 * Individual document detail in evaluation
 */
export type DocDetail = {
  doc_id: string;
  doc_rank: number;           // 1-based
  doc_title: string;
  relevance_score: number;    // 1-5
  retrieval_score?: number | null;
  doc_snippet?: string;
  chunk_id?: string | null;
  page?: number | null;
};

/**
 * Request to save query-unit evaluation (batch save)
 * query_id: chat="{session_id}:{turn_id}", search="search:{timestamp}"
 */
export type RetrievalEvaluationRequest = {
  source: "chat" | "search";
  query: string;
  doc_details: DocDetail[];   // Required: individual document scores
  // Chat context (optional for search)
  session_id?: string | null;
  turn_id?: number | null;
  // Filter context for reproducibility
  filter_devices?: string[] | null;
  filter_doc_types?: string[] | null;
  search_queries?: string[] | null;  // Multi-query expansion results
  // Search page only
  search_params?: Record<string, unknown> | null;
  // Reviewer info
  reviewer_name?: string | null;
};

/**
 * Response for query-unit evaluation
 */
export type RetrievalEvaluationResponse = {
  query_id: string;
  source: "chat" | "search";
  query: string;
  relevant_docs: string[];      // Auto-generated: score >= 3
  irrelevant_docs: string[];    // Auto-generated: score < 3
  doc_details: DocDetail[];
  // Chat context
  session_id?: string | null;
  turn_id?: number | null;
  // Filter context
  filter_devices?: string[] | null;
  filter_doc_types?: string[] | null;
  search_queries?: string[] | null;
  // Search params
  search_params?: Record<string, unknown> | null;
  // Reviewer info
  reviewer_name?: string | null;
  ts: string;
};

export type RetrievalEvaluationListResponse = {
  items: RetrievalEvaluationResponse[];
  total: number;
};

// --- Legacy types (deprecated, for backwards compatibility) ---

/** @deprecated Use RetrievalEvaluationRequest instead */
export type DocRelevanceEvaluationRequest = {
  relevance_score: number;  // 1-5
  reviewer_name?: string | null;
  query: string;
  doc_rank: number;  // 1-based
  doc_title: string;
  doc_snippet: string;
  message_id?: string | null;
  chunk_id?: string | null;
  retrieval_score?: number | null;
  filter_devices?: string[] | null;
  filter_doc_types?: string[] | null;
  search_queries?: string[] | null;  // Multi-query expansion results
};

/** @deprecated Use RetrievalEvaluationResponse instead */
export type DocRelevanceEvaluationResponse = {
  session_id: string;
  turn_id: number;
  doc_id: string;
  relevance_score: number;
  is_relevant: boolean;
  query: string;
  doc_rank: number;
  doc_title?: string;
  doc_snippet?: string;
  message_id?: string | null;
  chunk_id?: string | null;
  retrieval_score?: number | null;
  reviewer_name?: string | null;
  filter_devices?: string[] | null;
  filter_doc_types?: string[] | null;
  search_queries?: string[] | null;  // Multi-query expansion results
  ts: string;
};

/** @deprecated Use RetrievalEvaluationListResponse instead */
export type DocRelevanceEvaluationListResponse = {
  items: DocRelevanceEvaluationResponse[];
  total: number;
};
