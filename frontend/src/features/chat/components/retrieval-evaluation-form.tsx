import { useState, useCallback, useEffect, useMemo } from "react";
import { Button, message } from "antd";
import { CheckOutlined, LoadingOutlined } from "@ant-design/icons";
import { DocRelevanceRating } from "./doc-relevance-rating";
import { saveRetrievalEvaluation, getRetrievalEvaluation } from "../api";
import type { RetrievedDoc, DocDetail, RetrievalEvaluationRequest } from "../types";

/**
 * Form component for evaluating multiple documents at once.
 *
 * Features:
 * - Star rating (1-5) for each document
 * - Local state management until submit
 * - Batch save on submit button click
 * - Toast notification on success
 * - Loads existing evaluation on mount
 */

type RetrievalEvaluationFormProps = {
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
};

export function RetrievalEvaluationForm({
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
}: RetrievalEvaluationFormProps) {
  // Map<docId, score>
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

  // Handle score change from DocRelevanceRating
  const handleScoreChange = useCallback((docId: string, score: number) => {
    setScores((prev) => {
      const next = new Map(prev);
      next.set(docId, score);
      return next;
    });
    setIsSaved(false);
  }, []);

  // Count of evaluated documents
  const evaluatedCount = useMemo(() => {
    return Array.from(scores.values()).filter((s) => s > 0).length;
  }, [scores]);

  // Submit evaluation
  const handleSubmit = useCallback(async () => {
    // Filter only evaluated documents
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

  if (isLoading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "16px",
          color: "var(--color-text-secondary)",
        }}
      >
        <LoadingOutlined style={{ marginRight: 8 }} />
        Loading evaluation...
      </div>
    );
  }

  return (
    <div className="retrieval-evaluation-form">
      {/* Document list with ratings */}
      <div style={{ marginBottom: 16 }}>
        {docs.map((doc, index) => (
          <div
            key={doc.id}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "8px 0",
              borderBottom:
                index < docs.length - 1
                  ? "1px solid var(--color-border-light, #f0f0f0)"
                  : "none",
            }}
          >
            <div
              style={{
                flex: 1,
                minWidth: 0,
                marginRight: 12,
              }}
            >
              <div
                style={{
                  fontSize: 13,
                  fontWeight: 500,
                  color: "var(--color-text-primary)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
                title={doc.title}
              >
                {index + 1}. {doc.title}
              </div>
              {doc.score !== null && doc.score !== undefined && (
                <div
                  style={{
                    fontSize: 11,
                    color: "var(--color-text-secondary)",
                    marginTop: 2,
                  }}
                >
                  score: {doc.score.toFixed(3)}
                </div>
              )}
            </div>
            <DocRelevanceRating
              docId={doc.id}
              initialScore={scores.get(doc.id) || 0}
              onChange={handleScoreChange}
              disabled={isSubmitting}
              showLabel={false}
            />
          </div>
        ))}
      </div>

      {/* Submit button */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          gap: 8,
        }}
      >
        {evaluatedCount > 0 && (
          <span
            style={{
              fontSize: 12,
              color: "var(--color-text-secondary)",
            }}
          >
            {evaluatedCount} documents evaluated
          </span>
        )}
        <Button
          type="primary"
          size="small"
          onClick={handleSubmit}
          loading={isSubmitting}
          disabled={evaluatedCount === 0}
          icon={isSaved && !isSubmitting ? <CheckOutlined /> : undefined}
        >
          {isSaved && !isSubmitting ? "Saved" : "Submit evaluation"}
        </Button>
      </div>
    </div>
  );
}

export default RetrievalEvaluationForm;
