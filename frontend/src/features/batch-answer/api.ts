/**
 * Batch Answer API Client
 *
 * API functions for batch answer generation.
 */

import { apiClient } from "../../lib/api-client";
import {
  BatchAnswerRun,
  BatchAnswerResult,
  RunListResponse,
  ResultListResponse,
  CreateRunRequest,
  RatingRequest,
  ExecuteNextResponse,
} from "./types";

// ─── Run API ───

/**
 * Create a new batch answer run.
 * Creates pending results for each question.
 */
export async function createRun(
  request: CreateRunRequest
): Promise<BatchAnswerRun> {
  return apiClient.post<BatchAnswerRun>("/api/batch-answer/runs", request);
}

/**
 * List batch answer runs with pagination.
 */
export async function listRuns(params?: {
  limit?: number;
  offset?: number;
  status?: string;
}): Promise<RunListResponse> {
  const query = new URLSearchParams();
  if (params?.limit !== undefined) query.set("limit", String(params.limit));
  if (params?.offset !== undefined) query.set("offset", String(params.offset));
  if (params?.status) query.set("status", params.status);

  const queryString = query.toString();
  const path = queryString
    ? `/api/batch-answer/runs?${queryString}`
    : "/api/batch-answer/runs";
  return apiClient.get<RunListResponse>(path);
}

/**
 * Get a batch answer run by ID.
 */
export async function getRun(runId: string): Promise<BatchAnswerRun> {
  return apiClient.get<BatchAnswerRun>(`/api/batch-answer/runs/${runId}`);
}

/**
 * Delete a batch answer run and all its results.
 */
export async function deleteRun(
  runId: string
): Promise<{ status: string; run_id: string }> {
  return apiClient.delete(`/api/batch-answer/runs/${runId}`);
}

// ─── Execute API ───

/**
 * Execute answer generation for the next pending question.
 * Answer-only mode: uses saved search results, no new search.
 */
export async function executeNext(runId: string): Promise<ExecuteNextResponse> {
  return apiClient.post<ExecuteNextResponse>(
    `/api/batch-answer/runs/${runId}/execute-next`,
    {}
  );
}

// ─── Result API ───

/**
 * List results for a run.
 */
export async function listResults(
  runId: string,
  params?: {
    limit?: number;
    offset?: number;
    status?: string;
  }
): Promise<ResultListResponse> {
  const query = new URLSearchParams();
  if (params?.limit !== undefined) query.set("limit", String(params.limit));
  if (params?.offset !== undefined) query.set("offset", String(params.offset));
  if (params?.status) query.set("status", params.status);

  const queryString = query.toString();
  const path = queryString
    ? `/api/batch-answer/runs/${runId}/results?${queryString}`
    : `/api/batch-answer/runs/${runId}/results`;
  return apiClient.get<ResultListResponse>(path);
}

/**
 * Get a result by ID.
 */
export async function getResult(resultId: string): Promise<BatchAnswerResult> {
  return apiClient.get<BatchAnswerResult>(
    `/api/batch-answer/results/${resultId}`
  );
}

/**
 * Save rating for a result.
 */
export async function saveRating(
  resultId: string,
  request: RatingRequest
): Promise<{ status: string; result_id: string; rating: number }> {
  return apiClient.put(
    `/api/batch-answer/results/${resultId}/rating`,
    request
  );
}
