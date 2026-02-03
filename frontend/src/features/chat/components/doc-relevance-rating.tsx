import { useState, useCallback, useEffect } from "react";
import { StarOutlined, StarFilled, CheckOutlined, LoadingOutlined } from "@ant-design/icons";
import { saveDocRelevanceEvaluation, getDocRelevanceEvaluation } from "../api";
import { DocRelevanceEvaluationRequest } from "../types";

type DocRelevanceRatingProps = {
  sessionId: string;
  turnId: number;
  docId: string;
  docRank: number;
  docTitle: string;
  docSnippet: string;
  query: string;
  messageId?: string;
  chunkId?: string;
  retrievalScore?: number | null;
  filterDevices?: string[] | null;
  filterDocTypes?: string[] | null;
  searchQueries?: string[] | null;  // Multi-query expansion results
  initialScore?: number;
};

export function DocRelevanceRating({
  sessionId,
  turnId,
  docId,
  docRank,
  docTitle,
  docSnippet,
  query,
  messageId,
  chunkId,
  retrievalScore,
  filterDevices,
  filterDocTypes,
  searchQueries,
  initialScore = 0,
}: DocRelevanceRatingProps) {
  const [score, setScore] = useState<number>(initialScore);
  const [hoverScore, setHoverScore] = useState<number | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Load existing evaluation on mount
  useEffect(() => {
    let cancelled = false;
    const loadExisting = async () => {
      try {
        const existing = await getDocRelevanceEvaluation(sessionId, turnId, docId);
        if (!cancelled && existing) {
          setScore(existing.relevance_score);
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
    return () => { cancelled = true; };
  }, [sessionId, turnId, docId]);

  const handleClick = useCallback(async (starValue: number) => {
    if (isSaving) return;

    setScore(starValue);
    setIsSaving(true);
    setIsSaved(false);

    try {
      const data: DocRelevanceEvaluationRequest = {
        relevance_score: starValue,
        query,
        doc_rank: docRank,
        doc_title: docTitle,
        doc_snippet: docSnippet,
        message_id: messageId,
        chunk_id: chunkId,
        retrieval_score: retrievalScore ?? undefined,
        filter_devices: filterDevices ?? undefined,
        filter_doc_types: filterDocTypes ?? undefined,
        search_queries: searchQueries ?? undefined,
      };

      await saveDocRelevanceEvaluation(sessionId, turnId, docId, data);
      setIsSaved(true);
    } catch (err) {
      console.error("Failed to save doc relevance evaluation:", err);
      // Revert on error
      setScore(initialScore);
    } finally {
      setIsSaving(false);
    }
  }, [
    isSaving,
    sessionId,
    turnId,
    docId,
    docRank,
    docTitle,
    docSnippet,
    query,
    messageId,
    chunkId,
    retrievalScore,
    filterDevices,
    filterDocTypes,
    searchQueries,
    initialScore,
  ]);

  const handleMouseEnter = useCallback((starValue: number) => {
    if (!isSaving) {
      setHoverScore(starValue);
    }
  }, [isSaving]);

  const handleMouseLeave = useCallback(() => {
    setHoverScore(null);
  }, []);

  const displayScore = hoverScore ?? score;

  if (isLoading) {
    return (
      <div
        className="doc-relevance-rating"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          fontSize: 12,
          color: "var(--color-text-secondary)",
        }}
      >
        <LoadingOutlined style={{ fontSize: 12 }} />
      </div>
    );
  }

  return (
    <div
      className="doc-relevance-rating"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        fontSize: 12,
      }}
      onMouseLeave={handleMouseLeave}
    >
      <span style={{ color: "var(--color-text-secondary)", marginRight: 2 }}>
        관련성:
      </span>
      {[1, 2, 3, 4, 5].map((star) => {
        const filled = star <= displayScore;
        return (
          <button
            key={star}
            type="button"
            onClick={() => handleClick(star)}
            onMouseEnter={() => handleMouseEnter(star)}
            disabled={isSaving}
            style={{
              background: "none",
              border: "none",
              padding: 0,
              cursor: isSaving ? "not-allowed" : "pointer",
              color: filled
                ? "var(--color-accent-primary, #1890ff)"
                : "var(--color-border, #d9d9d9)",
              fontSize: 14,
              lineHeight: 1,
              transition: "color 0.2s, transform 0.1s",
              transform: hoverScore === star ? "scale(1.15)" : "scale(1)",
              opacity: isSaving ? 0.5 : 1,
            }}
            title={`${star}점`}
            aria-label={`관련성 ${star}점`}
          >
            {filled ? <StarFilled /> : <StarOutlined />}
          </button>
        );
      })}
      {score > 0 && (
        <span
          style={{
            marginLeft: 4,
            color: "var(--color-text-secondary)",
          }}
        >
          {score}점
        </span>
      )}
      {isSaving && (
        <LoadingOutlined
          style={{
            marginLeft: 4,
            fontSize: 12,
            color: "var(--color-accent-primary)",
          }}
        />
      )}
      {isSaved && !isSaving && (
        <CheckOutlined
          style={{
            marginLeft: 4,
            fontSize: 12,
            color: "var(--color-success, #52c41a)",
          }}
        />
      )}
    </div>
  );
}

export default DocRelevanceRating;
