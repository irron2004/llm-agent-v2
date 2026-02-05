import { createContext, useContext, useState, useCallback, useEffect, useRef, ReactNode } from "react";
import { buildUrl } from "@/config/env";
import {
  SearchConfig,
  TestQuestion,
  RetrievalTestResult,
  ChatPipelineSearchRequest,
  ChatPipelineSearchResponse,
} from "../types";
import { calculateMetrics } from "../hooks/use-metrics-calculation";
import { listRetrievalEvaluations } from "@/features/chat/api";
import type { RetrievalEvaluationResponse } from "@/features/chat/types";
import { DEFAULT_TEST_QUESTIONS } from "../data/default-questions";

function getDefaultConfig(): SearchConfig {
  return {
    denseWeight: 0.7,
    sparseWeight: 0.3,
    useRrf: true,
    rrfK: 60,
    rerank: false,
    rerankModel: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerankTopK: 10,
    multiQuery: false,
    multiQueryN: 3,
    size: 20,
    fieldWeights: [
      { field: "search_text", label: "Body text", enabled: true, weight: 1.0 },
      { field: "chunk_summary", label: "Chunk summary", enabled: true, weight: 0.7 },
      { field: "chunk_keywords.text", label: "Keywords", enabled: true, weight: 0.8 },
      { field: "content", label: "Raw content", enabled: false, weight: 0.6 },
    ],
  };
}

function evaluationToTestQuestion(eval_: RetrievalEvaluationResponse): TestQuestion {
  return {
    id: eval_.query_id,
    question: eval_.query,
    groundTruthDocIds: eval_.relevant_docs,
    category: eval_.source === "chat" ? "chat-evaluation" : "search-evaluation",
    difficulty: "medium",
  };
}

/**
 * 기본 질문을 TestQuestion 형식으로 변환
 */
function defaultQuestionToTestQuestion(question: string, index: number): TestQuestion {
  return {
    id: `default-${index}`,
    question,
    groundTruthDocIds: [], // 기본 질문은 정답 문서 없음
    category: "default",
    difficulty: "medium",
  };
}

// 기본 질문을 TestQuestion 형식으로 변환
const DEFAULT_QUESTIONS: TestQuestion[] = DEFAULT_TEST_QUESTIONS.map(
  defaultQuestionToTestQuestion
);

interface BatchProgress {
  current: number;
  total: number;
  currentQuestion: string;
}

interface RetrievalTestContextValue {
  selectedQuestion: TestQuestion | null;
  setSelectedQuestion: (q: TestQuestion | null) => void;
  config: SearchConfig;
  updateConfig: (updates: Partial<SearchConfig>) => void;
  results: RetrievalTestResult[];
  loading: boolean;
  error: string | null;
  runSingleTest: (question: TestQuestion) => Promise<RetrievalTestResult>;
  runBatchTest: (questions: TestQuestion[]) => Promise<void>;
  stopBatchTest: () => void;
  clearResults: () => void;
  // 기본 질문 (groundTruthDocIds 없음)
  defaultQuestions: TestQuestion[];
  // 저장된 평가 (groundTruthDocIds 있음)
  savedEvaluations: TestQuestion[];
  loadingEvaluations: boolean;
  loadSavedEvaluations: () => Promise<TestQuestion[]>;
  // 배치 진행 상태
  batchProgress: BatchProgress | null;
}

const RetrievalTestContext = createContext<RetrievalTestContextValue | null>(null);

export function RetrievalTestProvider({ children }: { children: ReactNode }) {
  const [selectedQuestion, setSelectedQuestion] = useState<TestQuestion | null>(null);
  const [config, setConfig] = useState<SearchConfig>(getDefaultConfig());
  const [results, setResults] = useState<RetrievalTestResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedEvaluations, setSavedEvaluations] = useState<TestQuestion[]>([]);
  const [loadingEvaluations, setLoadingEvaluations] = useState(true);
  const [initialLoadDone, setInitialLoadDone] = useState(false);
  const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null);
  const stopRequestedRef = useRef(false);

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

        if (!config.useRrf) {
          requestBody.search_override!.dense_weight = config.denseWeight;
          requestBody.search_override!.sparse_weight = config.sparseWeight;
        }

        const url = buildUrl("/api/search/chat-pipeline");
        console.log("[RetrievalTest] Chat Pipeline API Request:", url);

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Search failed: ${response.status} - ${errorText}`);
        }

        const data: ChatPipelineSearchResponse = await response.json();
        console.log("[RetrievalTest] Response:", {
          route: data.route,
          items: data.items?.length,
        });

        const metrics = calculateMetrics(data.items, question.groundTruthDocIds);

        const result: RetrievalTestResult = {
          questionId: question.id,
          question: question.question,
          searchResults: data.items,
          groundTruthDocIds: question.groundTruthDocIds,
          metrics,
          config: { ...config },
          timestamp: new Date().toISOString(),
        };

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
      stopRequestedRef.current = false;
      setBatchProgress({ current: 0, total: questions.length, currentQuestion: "" });

      try {
        for (let i = 0; i < questions.length; i++) {
          // 중지 요청 체크
          if (stopRequestedRef.current) {
            console.log("[RetrievalTest] Batch stopped by user");
            break;
          }

          const question = questions[i];
          setBatchProgress({
            current: i + 1,
            total: questions.length,
            currentQuestion: question.question.slice(0, 50) + (question.question.length > 50 ? "..." : ""),
          });
          await runSingleTest(question);
          await new Promise((resolve) => setTimeout(resolve, 300));
        }
      } catch (err) {
        console.error("Batch test failed:", err);
      } finally {
        setLoading(false);
        setBatchProgress(null);
        stopRequestedRef.current = false;
      }
    },
    [runSingleTest]
  );

  const stopBatchTest = useCallback(() => {
    stopRequestedRef.current = true;
  }, []);

  const updateConfig = useCallback((updates: Partial<SearchConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  }, []);

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  const loadSavedEvaluations = useCallback(async () => {
    setLoadingEvaluations(true);
    try {
      const response = await listRetrievalEvaluations({ limit: 100 });
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

  return (
    <RetrievalTestContext.Provider
      value={{
        selectedQuestion,
        setSelectedQuestion,
        config,
        updateConfig,
        results,
        loading,
        error,
        runSingleTest,
        runBatchTest,
        stopBatchTest,
        clearResults,
        defaultQuestions: DEFAULT_QUESTIONS,
        savedEvaluations,
        loadingEvaluations,
        loadSavedEvaluations,
        batchProgress,
      }}
    >
      {children}
    </RetrievalTestContext.Provider>
  );
}

export function useRetrievalTestContext() {
  const context = useContext(RetrievalTestContext);
  if (!context) {
    throw new Error("useRetrievalTestContext must be used within RetrievalTestProvider");
  }
  return context;
}
