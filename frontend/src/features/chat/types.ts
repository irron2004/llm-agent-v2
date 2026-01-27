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

export type Message = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt?: string;
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
};

export type AgentRequest = {
  message: string;
  top_k?: number;
  max_attempts?: number;
  mode?: string;
  thread_id?: string | null;
  ask_user_after_retrieve?: boolean;
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
  feedback_rating?: FeedbackRating | null;
  feedback_reason?: string | null;
  feedback_ts?: string | null;
};

export type SessionListItem = {
  session_id: string;  // API returns session_id, not id
  title: string;
  preview: string;
  turnCount: number;
  createdAt: string;
  updatedAt: string;
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
};

export type SaveTurnRequest = {
  user_text: string;
  assistant_text: string;
  doc_refs: DocRefResponse[];
  title?: string | null;
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
