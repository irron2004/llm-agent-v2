import { apiClient } from "../../lib/api-client";
import { connectSse } from "../../lib/sse";
import { env } from "../../config/env";
import { Conversation, Message } from "./types";

export async function fetchConversations(): Promise<Conversation[]> {
  return apiClient.get<Conversation[]>("/api/conversations");
}

export async function sendChatMessage(
  payload: { conversationId?: string; message: string },
  handlers: {
    onMessage: (text: string) => void;
    onDone?: () => void;
    onError?: (err: unknown) => void;
  }
) {
  return connectSse(
    {
      path: env.chatPath,
      method: "POST",
      body: payload,
    },
    {
      onMessage: handlers.onMessage,
      onClose: handlers.onDone,
      onError: handlers.onError,
    }
  );
}

export async function saveMessage(
  conversationId: string,
  message: Message
): Promise<void> {
  await apiClient.post(`/api/conversations/${conversationId}/messages`, message);
}
