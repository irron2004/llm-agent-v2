import { useState, useCallback } from "react";
import { buildUrl } from "@/config/env";
import {
  SearchConfig,
  TestQuestion,
  RetrievalTestResult,
  SearchResult,
} from "../types";
import { calculateMetrics } from "./use-metrics-calculation";

interface RetrievalDoc {
  rank: number;
  doc_id: string;
  title?: string | null;
  snippet?: string | null;
  score?: number | null;
  metadata?: Record<string, unknown>;
  page?: number;
}

interface RetrievalResponse {
  run_id: string;
  effective_config: Record<string, unknown>;
  effective_config_hash: string;
  warnings?: string[];
  steps: Record<string, unknown>;
  docs: RetrievalDoc[];
}

function getDefaultConfig(): SearchConfig {
  return {
    size: 20,
    deterministic: true,
    rerank: false,
    autoParse: true,
    skipMq: false,
  };
}

export function useRetrievalTest() {
  const [selectedQuestion, setSelectedQuestion] = useState<TestQuestion | null>(
    null
  );
  const [config, setConfig] = useState<SearchConfig>(getDefaultConfig());
  const [results, setResults] = useState<RetrievalTestResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRunId, setLastRunId] = useState<string | null>(null);

  const runSingleTest = useCallback(
    async (question: TestQuestion, replayRunId?: string) => {
      setLoading(true);
      setError(null);

      try {
        const url = buildUrl("/api/retrieval/run");

        const payload: Record<string, unknown> = {
          query: question.question,
          steps: ["retrieve"],
          deterministic: config.deterministic,
          final_top_k: config.size,
          rerank_enabled: config.rerank,
          auto_parse: config.autoParse,
          skip_mq: config.skipMq,
        };

        if (replayRunId) {
          payload.replay_run_id = replayRunId;
        }

        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error(`Search failed: ${response.status}`);
        }

        const data: RetrievalResponse = await response.json();

        // Store run_id for replay functionality
        setLastRunId(data.run_id);

        // Convert canonical docs to SearchResult format for display and metrics
        // Use the actual returned docs count (up to final_top_k) for metrics evaluation
        const searchResults: SearchResult[] = data.docs.map((doc, index) => {
          const score = typeof doc.score === "number" ? doc.score : null;
          return {
            rank: index + 1,
            id: doc.doc_id, // Canonical uses doc_id, map to id for compatibility
            title: doc.title ?? "Untitled",
            snippet: doc.snippet ?? "",
            score: score ?? 0,
            score_display: score === null ? "N/A" : score.toFixed(4),
            page: doc.page,
          };
        });

        // Evaluate metrics using the canonical doc_ids
        // Use the size from config (final_top_k) as the evaluation cutoff
        const docsToEvaluate = searchResults.slice(0, config.size);
        const metrics = calculateMetrics(
          docsToEvaluate,
          question.groundTruthDocIds
        );

        const result: RetrievalTestResult = {
          questionId: question.id,
          question: question.question,
          searchResults,
          groundTruthDocIds: question.groundTruthDocIds,
          metrics,
          config: { ...config },
          timestamp: new Date().toISOString(),
        };

        // 같은 질문의 기존 결과를 새 결과로 교체
        setResults((prev) => {
          const filtered = prev.filter((r) => r.questionId !== question.id);
          return [...filtered, result];
        });

        return result;
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [config]
  );

  const runBatchTest = useCallback(
    async (questions: TestQuestion[]) => {
      setLoading(true);
      setError(null);
      setResults([]);

      try {
        for (const question of questions) {
          await runSingleTest(question);
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
      } catch (err) {
        console.error("Batch test failed:", err);
      } finally {
        setLoading(false);
      }
    },
    [runSingleTest]
  );

  const replayLastRun = useCallback(
    async (question: TestQuestion) => {
      if (!lastRunId) {
        throw new Error("No previous run to replay");
      }
      return runSingleTest(question, lastRunId);
    },
    [lastRunId, runSingleTest]
  );

  const updateConfig = useCallback((updates: Partial<SearchConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  return {
    selectedQuestion,
    setSelectedQuestion,
    config,
    updateConfig,
    results,
    loading,
    error,
    runSingleTest,
    runBatchTest,
    replayLastRun,
    lastRunId,
    clearResults,
  };
}
