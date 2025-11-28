export type MessageRole = "user" | "assistant" | "system";

export type Message = {
  id: string;
  role: MessageRole;
  content: string;
  createdAt?: string;
  reference?: Reference;
  rawAnswer?: string;
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

export type Conversation = {
  id: string;
  title?: string;
  messages: Message[];
};
