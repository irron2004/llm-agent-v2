import { apiClient } from "../../lib/api-client";
import { env } from "../../config/env";
import { AgentResponse, Conversation, Message } from "./types";

export async function fetchConversations(): Promise<Conversation[]> {
  return apiClient.get<Conversation[]>("/api/conversations");
}

export async function sendChatMessage(
  payload: { message: string }
): Promise<AgentResponse> {
  // Agent endpoint returns JSON (non-SSE)
  return apiClient.post<AgentResponse>(env.chatPath || "/api/agent/run", payload);
}

export async function saveMessage(
  conversationId: string,
  message: Message
): Promise<void> {
  await apiClient.post(`/api/conversations/${conversationId}/messages`, message);
}
