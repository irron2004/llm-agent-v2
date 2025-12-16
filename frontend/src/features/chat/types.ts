export type MessageRole = "user" | "assistant" | "system";

export type Message = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt?: string;
  reference?: Reference;
  rawAnswer?: string;
  retrievedDocs?: RetrievedDoc[];
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
};

export type AgentResponse = {
  query: string;
  answer: string;
  judge?: Record<string, unknown>;
  retrieved_docs?: RetrievedDoc[];
  metadata?: Record<string, unknown>;
};

export type Conversation = {
  id: string;
  title?: string;
  messages: Message[];
};
