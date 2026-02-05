/**
 * Batch Answer Hook
 *
 * Manages batch answer generation state and execution.
 */

import { useState, useCallback } from "react";
import {
  BatchAnswerRun,
  BatchAnswerResult,
  CreateRunRequest,
  ExecuteNextResponse,
  QuestionInput,
  SourceConfig,
} from "../types";
import * as api from "../api";

interface UseBatchAnswerState {
  // Run state
  currentRun: BatchAnswerRun | null;
  runs: BatchAnswerRun[];
  runsTotal: number;
  // Result state
  results: BatchAnswerResult[];
  resultsTotal: number;
  // Execution state
  isCreating: boolean;
  isExecuting: boolean;
  isLoadingRuns: boolean;
  isLoadingResults: boolean;
  // Error state
  error: string | null;
}

export function useBatchAnswer() {
  const [state, setState] = useState<UseBatchAnswerState>({
    currentRun: null,
    runs: [],
    runsTotal: 0,
    results: [],
    resultsTotal: 0,
    isCreating: false,
    isExecuting: false,
    isLoadingRuns: false,
    isLoadingResults: false,
    error: null,
  });

  // ─── Run Operations ───

  /**
   * Create a new batch answer run from retrieval test results.
   */
  const createRun = useCallback(
    async (
      questions: QuestionInput[],
      options?: {
        name?: string;
        description?: string;
        sourceRunId?: string;
        sourceConfig?: SourceConfig;
      }
    ) => {
      setState((prev) => ({ ...prev, isCreating: true, error: null }));

      try {
        const request: CreateRunRequest = {
          name: options?.name,
          description: options?.description,
          source_run_id: options?.sourceRunId,
          source_config: options?.sourceConfig,
          questions,
        };

        const run = await api.createRun(request);
        setState((prev) => ({
          ...prev,
          currentRun: run,
          results: [],
          resultsTotal: questions.length,
          isCreating: false,
        }));
        return run;
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to create run";
        setState((prev) => ({ ...prev, error: message, isCreating: false }));
        throw err;
      }
    },
    []
  );

  /**
   * Load list of runs.
   */
  const loadRuns = useCallback(
    async (params?: { limit?: number; offset?: number; status?: string }) => {
      setState((prev) => ({ ...prev, isLoadingRuns: true, error: null }));

      try {
        const response = await api.listRuns(params);
        setState((prev) => ({
          ...prev,
          runs: response.items,
          runsTotal: response.total,
          isLoadingRuns: false,
        }));
        return response;
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to load run list";
        setState((prev) => ({ ...prev, error: message, isLoadingRuns: false }));
        throw err;
      }
    },
    []
  );

  /**
   * Load a specific run and its results.
   */
  const loadRun = useCallback(async (runId: string) => {
    setState((prev) => ({
      ...prev,
      isLoadingRuns: true,
      isLoadingResults: true,
      error: null,
    }));

    try {
      const [run, resultsResponse] = await Promise.all([
        api.getRun(runId),
        api.listResults(runId, { limit: 500 }),
      ]);

      setState((prev) => ({
        ...prev,
        currentRun: run,
        results: resultsResponse.items,
        resultsTotal: resultsResponse.total,
        isLoadingRuns: false,
        isLoadingResults: false,
      }));

      return { run, results: resultsResponse.items };
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load run";
      setState((prev) => ({
        ...prev,
        error: message,
        isLoadingRuns: false,
        isLoadingResults: false,
      }));
      throw err;
    }
  }, []);

  /**
   * Delete a run.
   */
  const deleteRun = useCallback(async (runId: string) => {
    try {
      await api.deleteRun(runId);
      setState((prev) => ({
        ...prev,
        runs: prev.runs.filter((r) => r.run_id !== runId),
        runsTotal: prev.runsTotal - 1,
        currentRun:
          prev.currentRun?.run_id === runId ? null : prev.currentRun,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to delete run";
      setState((prev) => ({ ...prev, error: message }));
      throw err;
    }
  }, []);

  // ─── Execution ───

  /**
   * Execute next pending question.
   * Returns the result or null if no more pending questions.
   */
  const executeNext = useCallback(async (): Promise<ExecuteNextResponse | null> => {
    if (!state.currentRun) {
      throw new Error("No current run");
    }

    setState((prev) => ({ ...prev, isExecuting: true, error: null }));

    try {
      const response = await api.executeNext(state.currentRun.run_id);

      // Update results list
      setState((prev) => {
        const existingIndex = prev.results.findIndex(
          (r) => r.result_id === response.result_id
        );

        const updatedResult: BatchAnswerResult = {
          result_id: response.result_id,
          run_id: prev.currentRun!.run_id,
          question_id: response.question_id,
          question: response.question,
          status: response.status,
          answer: response.answer,
          reasoning: response.reasoning,
          search_results: response.search_results,
          search_result_count: response.search_results.length,
          ground_truth_doc_ids: [], // Will be filled from original
          category: null,
          retrieval_metrics: response.metrics,
          latency_ms: null,
          token_count: null,
          rating: null,
          rating_comment: null,
          rated_by: null,
          rated_at: null,
          error_message: response.error_message,
          created_at: null,
          updated_at: null,
        };

        // Copy ground_truth from existing result if available
        if (existingIndex >= 0) {
          updatedResult.ground_truth_doc_ids =
            prev.results[existingIndex].ground_truth_doc_ids;
          updatedResult.category = prev.results[existingIndex].category;
        }

        const newResults =
          existingIndex >= 0
            ? [
                ...prev.results.slice(0, existingIndex),
                updatedResult,
                ...prev.results.slice(existingIndex + 1),
              ]
            : [...prev.results, updatedResult];

        // Update run progress
        const updatedRun = prev.currentRun
          ? {
              ...prev.currentRun,
              progress: response.progress,
              status:
                response.progress.completed + response.progress.failed >=
                response.progress.total
                  ? ("completed" as const)
                  : ("running" as const),
            }
          : null;

        return {
          ...prev,
          results: newResults,
          currentRun: updatedRun,
          isExecuting: false,
        };
      });

      return response;
    } catch (err: unknown) {
      // Check if it's "No pending questions" error
      if (
        err instanceof Error &&
        err.message.includes("No pending questions")
      ) {
        setState((prev) => ({
          ...prev,
          isExecuting: false,
          currentRun: prev.currentRun
            ? { ...prev.currentRun, status: "completed" as const }
            : null,
        }));
        return null;
      }

      const message =
        err instanceof Error ? err.message : "Failed to generate answer";
      setState((prev) => ({ ...prev, error: message, isExecuting: false }));
      throw err;
    }
  }, [state.currentRun]);

  /**
   * Execute all pending questions sequentially.
   * Calls onProgress callback after each execution.
   */
  const executeAll = useCallback(
    async (
      onProgress?: (
        completed: number,
        total: number,
        result: ExecuteNextResponse | null
      ) => void
    ) => {
      if (!state.currentRun) {
        throw new Error("No current run");
      }

      const total = state.currentRun.progress.total;
      let completed = state.currentRun.progress.completed;

      while (completed < total) {
        try {
          const result = await executeNext();
          if (result === null) {
            // No more pending
            break;
          }
          completed = result.progress.completed;
          onProgress?.(completed, total, result);
        } catch (err) {
          // Stop execution on error
          console.error("Batch execution error:", err);
          break;
        }
      }
    },
    [state.currentRun, executeNext]
  );

  // ─── Rating ───

  /**
   * Save rating for a result.
   */
  const saveRating = useCallback(
    async (
      resultId: string,
      rating: number,
      options?: { comment?: string; ratedBy?: string }
    ) => {
      try {
        await api.saveRating(resultId, {
          rating,
          comment: options?.comment,
          rated_by: options?.ratedBy,
        });

        setState((prev) => ({
          ...prev,
          results: prev.results.map((r) =>
            r.result_id === resultId
              ? {
                  ...r,
                  rating,
                  rating_comment: options?.comment || null,
                  rated_by: options?.ratedBy || null,
                  rated_at: new Date().toISOString(),
                }
              : r
          ),
        }));
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to save rating";
        setState((prev) => ({ ...prev, error: message }));
        throw err;
      }
    },
    []
  );

  // ─── Utility ───

  /**
   * Clear current run and results.
   */
  const clearCurrentRun = useCallback(() => {
    setState((prev) => ({
      ...prev,
      currentRun: null,
      results: [],
      resultsTotal: 0,
      error: null,
    }));
  }, []);

  /**
   * Clear error.
   */
  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  return {
    // State
    ...state,
    // Run operations
    createRun,
    loadRuns,
    loadRun,
    deleteRun,
    // Execution
    executeNext,
    executeAll,
    // Rating
    saveRating,
    // Utility
    clearCurrentRun,
    clearError,
  };
}
