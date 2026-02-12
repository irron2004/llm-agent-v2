import { useState, useCallback, useEffect } from "react";
import { buildUrl } from "@/config/env";
import {
  SearchConfig,
  TestQuestion,
  RetrievalTestResult,
  SearchResult,
  ChatPipelineSearchRequest,
  ChatPipelineSearchResponse,
} from "../types";
import { calculateMetrics } from "./use-metrics-calculation";
import { listRetrievalEvaluations } from "@/features/chat/api";
import type { RetrievalEvaluationResponse } from "@/features/chat/types";

function getDefaultConfig(): SearchConfig {
  return {
    denseWeight: 0.7,
    sparseWeight: 0.3,
    useRrf: true, // RRF enabled by default (weights ignored)
    rrfK: 60, // RRF constant
    rerank: true,
    rerankModel: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerankTopK: 10,
    multiQuery: true,
    multiQueryN: 3,
    size: 20,
    fieldWeights: [
      {
        field: "search_text",
        label: "Body text",
        enabled: true,
        weight: 1.0,
      },
      {
        field: "chunk_summary",
        label: "Chunk summary",
        enabled: true,
        weight: 0.7,
      },
      {
        field: "chunk_keywords.text",
        label: "Keywords",
        enabled: true,
        weight: 0.8,
      },
      {
        field: "content",
        label: "Raw content",
        enabled: false,
        weight: 0.6,
      },
      // Note: chapter, doc_description, device_name, doc_type are not searchable
      // chapter/device_name/doc_type are keyword fields (not text-searchable)
      // doc_description has index: false in ES mapping
    ],
  };
}

/**
 * Convert saved evaluation to TestQuestion format.
 */
function evaluationToTestQuestion(eval_: RetrievalEvaluationResponse): TestQuestion {
  return {
    id: eval_.query_id,
    question: eval_.query,
    groundTruthDocIds: eval_.relevant_docs,
    category: eval_.source === "chat" ? "chat-evaluation" : "search-evaluation",
    difficulty: "medium",
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
  const [savedEvaluations, setSavedEvaluations] = useState<TestQuestion[]>([]);
  const [loadingEvaluations, setLoadingEvaluations] = useState(true); // Start with true for initial load
  const [initialLoadDone, setInitialLoadDone] = useState(false);

  // Auto-load evaluations on mount
  useEffect(() => {
    if (initialLoadDone) return;

    const loadInitial = async () => {
      setLoadingEvaluations(true);
      try {
        const response = await listRetrievalEvaluations({ limit: 100 });
        const validEvaluations = response.items.filter(
          (e) => e.relevant_docs && e.relevant_docs.length > 0
        );
        const questions = validEvaluations.map(evaluationToTestQuestion);
        setSavedEvaluations(questions);
      } catch (err) {
        console.error("Failed to load saved evaluations:", err);
        setError("Failed to load evaluation data.");
      } finally {
        setLoadingEvaluations(false);
        setInitialLoadDone(true);
      }
    };

    loadInitial();
  }, [initialLoadDone]);

  const runSingleTest = useCallback(
    async (question: TestQuestion) => {
      setLoading(true);
      setError(null);

      try {
        // Use chat pipeline API (translate → mq → retrieve → expand)
        const requestBody: ChatPipelineSearchRequest = {
          query: question.question,
          search_override: {
            top_k: config.size,
            use_rrf: config.useRrf,
            rrf_k: config.rrfK,
            rerank: config.rerank,
            rerank_top_k: config.rerankTopK,
          },
        };

        // Only send weights when RRF is disabled
        if (!config.useRrf) {
          requestBody.search_override!.dense_weight = config.denseWeight;
          requestBody.search_override!.sparse_weight = config.sparseWeight;
        }

        const url = buildUrl("/api/search/chat-pipeline");
        console.log("[RetrievalTest] Chat Pipeline API Request:", url);
        console.log("[RetrievalTest] Request Body:", requestBody);

        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Search failed: ${response.status} - ${errorText}`);
        }

        const data: ChatPipelineSearchResponse = await response.json();
        console.log("[RetrievalTest] Response:", {
          route: data.route,
          query_en: data.query_en,
          search_queries: data.search_queries?.length,
          items: data.items?.length,
        });

        const metrics = calculateMetrics(
          data.items,
          question.groundTruthDocIds
        );

        const result: RetrievalTestResult = {
          questionId: question.id,
          question: question.question,
          searchResults: data.items,
          groundTruthDocIds: question.groundTruthDocIds,
          metrics,
          config: { ...config },
          timestamp: new Date().toISOString(),
        };

        // Replace previous results for the same question
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

  const updateConfig = useCallback((updates: Partial<SearchConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  /**
   * Load saved evaluations from API and convert to TestQuestion format.
   */
  const loadSavedEvaluations = useCallback(async () => {
    setLoadingEvaluations(true);
    try {
      const response = await listRetrievalEvaluations({ limit: 100 });
      // Filter evaluations that have at least one relevant doc
      const validEvaluations = response.items.filter(
        (e) => e.relevant_docs && e.relevant_docs.length > 0
      );
      const questions = validEvaluations.map(evaluationToTestQuestion);
      setSavedEvaluations(questions);
      return questions;
    } catch (err) {
      console.error("Failed to load saved evaluations:", err);
      setError("Failed to load evaluation data.");
      return [];
    } finally {
      setLoadingEvaluations(false);
    }
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
    clearResults,
    // Saved evaluations
    savedEvaluations,
    loadingEvaluations,
    loadSavedEvaluations,
  };
}
