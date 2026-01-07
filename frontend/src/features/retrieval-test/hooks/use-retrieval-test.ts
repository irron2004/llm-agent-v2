import { useState, useCallback } from "react";
import { buildUrl } from "@/config/env";
import {
  SearchConfig,
  TestQuestion,
  RetrievalTestResult,
  SearchResult,
} from "../types";
import { calculateMetrics } from "./use-metrics-calculation";

interface SearchResponse {
  query: string;
  items: SearchResult[];
  total: number;
  page: number;
  size: number;
  has_next: boolean;
}

function getDefaultConfig(): SearchConfig {
  return {
    denseWeight: 0.7,
    sparseWeight: 0.3,
    rerank: false,
    rerankModel: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerankTopK: 10,
    multiQuery: false,
    multiQueryN: 3,
    size: 20,
    fieldWeights: [
      {
        field: "search_text",
        label: "본문 텍스트",
        enabled: true,
        weight: 1.0,
      },
      {
        field: "chunk_summary",
        label: "청크 요약",
        enabled: true,
        weight: 0.7,
      },
      {
        field: "chunk_keywords.text",
        label: "키워드",
        enabled: true,
        weight: 0.8,
      },
      {
        field: "content",
        label: "원본 콘텐츠",
        enabled: false,
        weight: 0.6,
      },
      {
        field: "chapter",
        label: "챕터명",
        enabled: false,
        weight: 1.2,
      },
      {
        field: "doc_description",
        label: "문서 설명",
        enabled: false,
        weight: 0.9,
      },
      {
        field: "device_name",
        label: "장비명",
        enabled: false,
        weight: 1.5,
      },
      {
        field: "doc_type",
        label: "문서 타입",
        enabled: false,
        weight: 1.0,
      },
    ],
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

  const runSingleTest = useCallback(
    async (question: TestQuestion) => {
      setLoading(true);
      setError(null);

      try {
        const fieldWeights = config.fieldWeights
          .filter((f) => f.enabled)
          .map((f) => `${f.field}^${f.weight.toFixed(1)}`)
          .join(",");

        const params = new URLSearchParams({
          q: question.question,
          field_weights: fieldWeights,
          size: config.size.toString(),
          dense_weight: config.denseWeight.toString(),
          sparse_weight: config.sparseWeight.toString(),
        });

        if (config.multiQuery) {
          params.append("multi_query", "true");
          params.append("multi_query_n", config.multiQueryN.toString());
        }

        if (config.rerank) {
          params.append("rerank", "true");
          params.append("rerank_top_k", config.rerankTopK.toString());
        }

        const url = buildUrl(`/api/search?${params.toString()}`);
        console.log("[RetrievalTest] API Request URL:", url);
        console.log("[RetrievalTest] Config:", { size: config.size, denseWeight: config.denseWeight, sparseWeight: config.sparseWeight });
        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(`Search failed: ${response.status}`);
        }

        const data: SearchResponse = await response.json();

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
    clearResults,
  };
}
