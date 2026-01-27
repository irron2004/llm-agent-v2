import { apiClient } from "../../lib/api-client";
import { buildUrl } from "../../config/env";
import { env } from "../../config/env";
import {
  AgentRequest,
  AgentResponse,
  FeedbackRating,
  DeviceCatalogResponse,
  SessionListResponse,
  SessionDetailResponse,
  SaveTurnRequest,
  TurnResponse,
  DetailedFeedbackRequest,
  FeedbackResponse,
  FeedbackListResponse,
  FeedbackStatisticsResponse,
} from "./types";

// ─── Agent API ───

export async function sendChatMessage(
  payload: AgentRequest
): Promise<AgentResponse> {
  // Agent endpoint returns JSON (non-SSE)
  return apiClient.post<AgentResponse>(env.chatPath || "/api/agent/run", payload);
}

export async function fetchDeviceCatalog(): Promise<DeviceCatalogResponse> {
  return apiClient.get<DeviceCatalogResponse>("/api/device-catalog");
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

// --- Detailed Feedback API (separate index) ---

export async function saveDetailedFeedback(
  sessionId: string,
  turnId: number,
  data: DetailedFeedbackRequest
): Promise<FeedbackResponse> {
  return apiClient.post<FeedbackResponse>(
    `/api/feedback/${sessionId}/${turnId}`,
    data
  );
}

export async function getDetailedFeedback(
  sessionId: string,
  turnId: number
): Promise<FeedbackResponse | null> {
  try {
    return await apiClient.get<FeedbackResponse>(
      `/api/feedback/${sessionId}/${turnId}`
    );
  } catch {
    return null;
  }
}

export async function listFeedback(params: {
  limit?: number;
  offset?: number;
  rating?: string;
  minScore?: number;
  maxScore?: number;
}): Promise<FeedbackListResponse> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  if (params.rating) query.set("rating", params.rating);
  if (params.minScore !== undefined) query.set("min_score", String(params.minScore));
  if (params.maxScore !== undefined) query.set("max_score", String(params.maxScore));

  return apiClient.get<FeedbackListResponse>(
    `/api/feedback?${query.toString()}`
  );
}

export async function getFeedbackStatistics(): Promise<FeedbackStatisticsResponse> {
  return apiClient.get<FeedbackStatisticsResponse>("/api/feedback/statistics");
}

export async function exportFeedbackJson(minScore = 3.0): Promise<Blob> {
  const url = buildUrl(`/api/feedback/export/json?min_score=${minScore}`);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Export failed: ${res.status}`);
  }
  return res.blob();
}

export async function exportFeedbackCsv(minScore = 3.0): Promise<Blob> {
  const url = buildUrl(`/api/feedback/export/csv?min_score=${minScore}`);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Export failed: ${res.status}`);
  }
  return res.blob();
}
