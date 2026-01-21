import { apiClient } from "../../lib/api-client";
import { env } from "../../config/env";
import {
  AgentRequest,
  AgentResponse,
  FeedbackRating,
  SessionListResponse,
  SessionDetailResponse,
  SaveTurnRequest,
  TurnResponse,
} from "./types";

// ─── Agent API ───

export async function sendChatMessage(
  payload: AgentRequest
): Promise<AgentResponse> {
  // Agent endpoint returns JSON (non-SSE)
  return apiClient.post<AgentResponse>(env.chatPath || "/api/agent/run", payload);
}

// ─── Conversations API ───

export async function fetchSessions(
  limit = 50,
  offset = 0
): Promise<SessionListResponse> {
  return apiClient.get<SessionListResponse>(
    `/api/conversations?limit=${limit}&offset=${offset}`
  );
}

export async function fetchSession(
  sessionId: string
): Promise<SessionDetailResponse> {
  return apiClient.get<SessionDetailResponse>(
    `/api/conversations/${sessionId}`
  );
}

export async function saveTurn(
  sessionId: string,
  turn: SaveTurnRequest
): Promise<TurnResponse> {
  return apiClient.post<TurnResponse>(
    `/api/conversations/${sessionId}/turns`,
    turn
  );
}

export async function saveFeedback(
  sessionId: string,
  turnId: number,
  payload: { rating: FeedbackRating; reason?: string | null }
): Promise<TurnResponse> {
  return apiClient.post<TurnResponse>(
    `/api/conversations/${sessionId}/turns/${turnId}/feedback`,
    payload
  );
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiClient.delete(`/api/conversations/${sessionId}`);
}

export async function hideSession(sessionId: string): Promise<void> {
  await apiClient.post(`/api/conversations/${sessionId}/hide`, {});
}

export async function unhideSession(sessionId: string): Promise<void> {
  await apiClient.post(`/api/conversations/${sessionId}/unhide`, {});
}
