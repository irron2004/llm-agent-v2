import { apiClient } from "../../lib/api-client";
import { connectSse } from "../../lib/sse";
import { env } from "../../config/env";
import { Conversation, Message } from "./types";

export async function fetchConversations(): Promise<Conversation[]> {
  return apiClient.get<Conversation[]>("/api/conversations");
}

export async function sendChatMessage(
  payload: { conversationId?: string; message: string; sessionId?: string },
  handlers: {
    onMessage: (text: string) => void;
    onDone?: () => void;
    onError?: (err: unknown) => void;
  }
) {
  // BE Agent API 형식으로 변환
  const body: Record<string, unknown> = {
    message: payload.message,
  };
  // session_id가 있으면 BE에서 history 자동 로드
  if (payload.sessionId) {
    body.session_id = payload.sessionId;
  }

  return connectSse(
    {
      path: env.chatPath,
      method: "POST",
      body,
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
