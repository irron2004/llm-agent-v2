export type MessageRole = "user" | "assistant" | "system";

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
};

export type SessionListItem = {
  id: string;
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
