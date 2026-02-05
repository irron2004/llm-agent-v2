import { createContext, useContext, useState, useCallback, useEffect, useMemo, ReactNode } from "react";
import { message } from "antd";
import { saveRetrievalEvaluation, getRetrievalEvaluation } from "../api";
import type { RetrievedDoc, DocDetail, RetrievalEvaluationRequest } from "../types";

/**
 * Context for managing retrieval evaluation state across document items.
 *
 * This allows inline star ratings next to each document title while
 * maintaining batch save functionality with a single submit button.
 */

type RetrievalEvaluationContextValue = {
  scores: Map<string, number>;
  setScore: (docId: string, score: number) => void;
  isSubmitting: boolean;
  isSaved: boolean;
  isLoading: boolean;
  evaluatedCount: number;
  submit: () => Promise<void>;
};

const RetrievalEvaluationContext = createContext<RetrievalEvaluationContextValue | null>(null);

export function useRetrievalEvaluation() {
  const context = useContext(RetrievalEvaluationContext);
  if (!context) {
    throw new Error("useRetrievalEvaluation must be used within RetrievalEvaluationProvider");
  }
  return context;
}

// Optional hook that doesn't throw - for components that might be outside provider
export function useRetrievalEvaluationOptional() {
  return useContext(RetrievalEvaluationContext);
}

type RetrievalEvaluationProviderProps = {
  queryId: string;
  source: "chat" | "search";
  query: string;
  docs: RetrievedDoc[];
  sessionId?: string;
  turnId?: number;
  filterDevices?: string[] | null;
  filterDocTypes?: string[] | null;
  searchQueries?: string[] | null;
  searchParams?: Record<string, unknown> | null;
  onSaved?: () => void;
  children: ReactNode;
};

export function RetrievalEvaluationProvider({
  queryId,
  source,
  query,
  docs,
  sessionId,
  turnId,
  filterDevices,
  filterDocTypes,
  searchQueries,
  searchParams,
  onSaved,
  children,
}: RetrievalEvaluationProviderProps) {
  const [scores, setScores] = useState<Map<string, number>>(new Map());
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Load existing evaluation on mount
  useEffect(() => {
    let cancelled = false;
    const loadExisting = async () => {
      try {
        const existing = await getRetrievalEvaluation(queryId);
        if (!cancelled && existing) {
          const loadedScores = new Map<string, number>();
          for (const detail of existing.doc_details) {
            loadedScores.set(detail.doc_id, detail.relevance_score);
          }
          setScores(loadedScores);
          setIsSaved(true);
        }
      } catch {
        // Ignore - no existing evaluation
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };
    loadExisting();
    return () => {
      cancelled = true;
    };
  }, [queryId]);

  const setScore = useCallback((docId: string, score: number) => {
    setScores((prev) => {
      const next = new Map(prev);
      next.set(docId, score);
      return next;
    });
    setIsSaved(false);
  }, []);

  const evaluatedCount = useMemo(() => {
    return Array.from(scores.values()).filter((s) => s > 0).length;
  }, [scores]);

  const submit = useCallback(async () => {
    const evaluatedDocs = docs.filter((doc) => {
      const score = scores.get(doc.id);
      return score !== undefined && score > 0;
    });

    if (evaluatedDocs.length === 0) {
      message.warning("No evaluated documents.");
      return;
    }

    setIsSubmitting(true);

    try {
      const docDetails: DocDetail[] = evaluatedDocs.map((doc, index) => ({
        doc_id: doc.id,
        doc_rank: index + 1,
        doc_title: doc.title,
        relevance_score: scores.get(doc.id) || 0,
        retrieval_score: doc.score ?? undefined,
        doc_snippet: doc.snippet,
        chunk_id: undefined,
        page: doc.page ?? undefined,
      }));

      const data: RetrievalEvaluationRequest = {
        source,
        query,
        doc_details: docDetails,
        session_id: sessionId,
        turn_id: turnId,
        filter_devices: filterDevices ?? undefined,
        filter_doc_types: filterDocTypes ?? undefined,
        search_queries: searchQueries ?? undefined,
        search_params: searchParams ?? undefined,
      };

      await saveRetrievalEvaluation(queryId, data);
      setIsSaved(true);
      message.success("Saved.");
      onSaved?.();
    } catch (err) {
      console.error("Failed to save retrieval evaluation:", err);
      message.error("Failed to save.");
    } finally {
      setIsSubmitting(false);
    }
  }, [
    docs,
    scores,
    queryId,
    source,
    query,
    sessionId,
    turnId,
    filterDevices,
    filterDocTypes,
    searchQueries,
    searchParams,
    onSaved,
  ]);

  const value = useMemo<RetrievalEvaluationContextValue>(
    () => ({
      scores,
      setScore,
      isSubmitting,
      isSaved,
      isLoading,
      evaluatedCount,
      submit,
    }),
    [scores, setScore, isSubmitting, isSaved, isLoading, evaluatedCount, submit]
  );

  return (
    <RetrievalEvaluationContext.Provider value={value}>
      {children}
    </RetrievalEvaluationContext.Provider>
  );
}

export default RetrievalEvaluationProvider;
