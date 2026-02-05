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
  // Query-unit retrieval evaluation types
  RetrievalEvaluationRequest,
  RetrievalEvaluationResponse,
  RetrievalEvaluationListResponse,
  // Legacy types (deprecated)
  DocRelevanceEvaluationRequest,
  DocRelevanceEvaluationResponse,
  DocRelevanceEvaluationListResponse,
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

export async function truncateSessionTurns(
  sessionId: string,
  fromTurnId: number
): Promise<{ deleted: number; session_id: string; from_turn_id: number }> {
  return apiClient.post(
    `/api/conversations/${sessionId}/turns/${fromTurnId}/truncate`,
    {}
  );
}

export async function branchSessionTurns(
  sessionId: string,
  fromTurnId: number
): Promise<{ session_id: string; copied: number; from_turn_id: number }> {
  return apiClient.post(
    `/api/conversations/${sessionId}/turns/${fromTurnId}/branch`,
    {}
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

// --- Retrieval Evaluation API (query-unit storage) ---

/**
 * Save query-unit evaluation (batch save).
 * @param queryId - Query ID (chat: "{sessionId}:{turnId}", search: "search:{timestamp}")
 * @param data - Evaluation data with doc_details
 */
export async function saveRetrievalEvaluation(
  queryId: string,
  data: RetrievalEvaluationRequest
): Promise<RetrievalEvaluationResponse> {
  return apiClient.post<RetrievalEvaluationResponse>(
    `/api/retrieval-evaluation/query/${encodeURIComponent(queryId)}`,
    data
  );
}

/**
 * Get evaluation by query_id.
 * @param queryId - Query ID
 */
export async function getRetrievalEvaluation(
  queryId: string
): Promise<RetrievalEvaluationResponse | null> {
  try {
    return await apiClient.get<RetrievalEvaluationResponse>(
      `/api/retrieval-evaluation/query/${encodeURIComponent(queryId)}`
    );
  } catch {
    return null;
  }
}

/**
 * List all query evaluations with pagination.
 * @param params - Query parameters (limit, offset, source)
 */
export async function listRetrievalEvaluations(params?: {
  limit?: number;
  offset?: number;
  source?: "chat" | "search";
}): Promise<RetrievalEvaluationListResponse> {
  const query = new URLSearchParams();
  if (params?.limit !== undefined) query.set("limit", String(params.limit));
  if (params?.offset !== undefined) query.set("offset", String(params.offset));
  if (params?.source) query.set("source", params.source);

  return apiClient.get<RetrievalEvaluationListResponse>(
    `/api/retrieval-evaluation/list?${query.toString()}`
  );
}

/**
 * Delete evaluation by query_id.
 * @param queryId - Query ID to delete
 */
export async function deleteRetrievalEvaluation(
  queryId: string
): Promise<{ message: string; query_id: string }> {
  return apiClient.delete(
    `/api/retrieval-evaluation/query/${encodeURIComponent(queryId)}`
  );
}

/**
 * Export retrieval evaluation data as JSON.
 * @param minRelevance - Minimum relevance score for 'relevant' (default: 3)
 * @param limit - Maximum number of records (default: 10000)
 */
export async function exportRetrievalEvaluationJson(
  minRelevance = 3,
  limit = 10000
): Promise<Blob> {
  const url = buildUrl(
    `/api/retrieval-evaluation/export/json?min_relevance=${minRelevance}&limit=${limit}`
  );
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Export failed: ${res.status}`);
  }
  return res.blob();
}

/**
 * Export retrieval evaluation data as CSV.
 * @param minRelevance - Minimum relevance score for 'relevant' (default: 3)
 * @param limit - Maximum number of records (default: 10000)
 */
export async function exportRetrievalEvaluationCsv(
  minRelevance = 3,
  limit = 10000
): Promise<Blob> {
  const url = buildUrl(
    `/api/retrieval-evaluation/export/csv?min_relevance=${minRelevance}&limit=${limit}`
  );
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Export failed: ${res.status}`);
  }
  return res.blob();
}

// --- Legacy Retrieval Evaluation API (deprecated) ---

/**
 * @deprecated Use saveRetrievalEvaluation for query-unit storage
 */
export async function saveDocRelevanceEvaluation(
  sessionId: string,
  turnId: number,
  docId: string,
  data: DocRelevanceEvaluationRequest
): Promise<DocRelevanceEvaluationResponse> {
  return apiClient.post<DocRelevanceEvaluationResponse>(
    `/api/retrieval-evaluation/${sessionId}/${turnId}/${encodeURIComponent(docId)}`,
    data
  );
}

/**
 * @deprecated Use getRetrievalEvaluation for query-unit storage
 */
export async function getDocRelevanceEvaluation(
  sessionId: string,
  turnId: number,
  docId: string
): Promise<DocRelevanceEvaluationResponse | null> {
  try {
    return await apiClient.get<DocRelevanceEvaluationResponse>(
      `/api/retrieval-evaluation/${sessionId}/${turnId}/${encodeURIComponent(docId)}`
    );
  } catch {
    return null;
  }
}

/**
 * @deprecated Use listRetrievalEvaluations for query-unit storage
 */
export async function listDocRelevanceEvaluations(
  sessionId: string,
  turnId: number
): Promise<DocRelevanceEvaluationListResponse> {
  return apiClient.get<DocRelevanceEvaluationListResponse>(
    `/api/retrieval-evaluation/${sessionId}/${turnId}`
  );
}
