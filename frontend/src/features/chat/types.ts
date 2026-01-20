export type MessageRole = "user" | "assistant" | "system";

export type FeedbackRating = "up" | "down";

export type MessageFeedback = {
  rating: FeedbackRating;
  reason?: string | null;
  ts?: string | null;
};

export type Message = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt?: string;
  reference?: Reference;
  rawAnswer?: string;
  retrievedDocs?: RetrievedDoc[];
  logs?: string[];
  currentNode?: string | null;
  sessionId?: string;
  turnId?: number;
  feedback?: MessageFeedback | null;
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

export type AgentResponse = {
  query: string;
  answer: string;
  judge?: Record<string, unknown>;
  retrieved_docs?: RetrievedDoc[];
  metadata?: Record<string, unknown>;
  interrupted?: boolean;
  interrupt_payload?: Record<string, unknown> | null;
  thread_id?: string | null;
};

export type AgentRequest = {
  message: string;
  top_k?: number;
  max_attempts?: number;
  mode?: string;
  thread_id?: string | null;
  ask_user_after_retrieve?: boolean;
  resume_decision?: unknown;
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
