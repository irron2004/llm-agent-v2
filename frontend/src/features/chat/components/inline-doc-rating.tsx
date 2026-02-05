import { useState, useCallback } from "react";
import { StarOutlined, StarFilled } from "@ant-design/icons";
import { useRetrievalEvaluationOptional } from "./retrieval-evaluation-context";

/**
 * Inline star rating component that integrates with RetrievalEvaluationContext.
 * Designed to be placed next to document titles.
 */

type InlineDocRatingProps = {
  docId: string;
  size?: "small" | "default";
};

export function InlineDocRating({ docId, size = "small" }: InlineDocRatingProps) {
  const context = useRetrievalEvaluationOptional();
  const [hoverScore, setHoverScore] = useState<number | null>(null);

  // If no context, don't render anything
  if (!context) {
    return null;
  }

  const { scores, setScore, isSubmitting, isLoading } = context;
  const score = scores.get(docId) || 0;
  const displayScore = hoverScore ?? score;
  const disabled = isSubmitting || isLoading;

  const handleClick = (starValue: number) => {
    if (disabled) return;
    setScore(docId, starValue);
  };

  const handleMouseEnter = (starValue: number) => {
    if (!disabled) {
      setHoverScore(starValue);
    }
  };

  const handleMouseLeave = () => {
    setHoverScore(null);
  };

  const starSize = size === "small" ? 12 : 14;

  return (
    <div
      className="inline-doc-rating"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 2,
        marginLeft: 8,
      }}
      onMouseLeave={handleMouseLeave}
    >
      {[1, 2, 3, 4, 5].map((star) => {
        const filled = star <= displayScore;
        return (
          <button
            key={star}
            type="button"
            onClick={() => handleClick(star)}
            onMouseEnter={() => handleMouseEnter(star)}
            disabled={disabled}
            style={{
              background: "none",
              border: "none",
              padding: 0,
              cursor: disabled ? "not-allowed" : "pointer",
              color: filled
                ? "var(--color-accent-primary, #1890ff)"
                : "var(--color-border, #d9d9d9)",
              fontSize: starSize,
              lineHeight: 1,
              transition: "color 0.2s, transform 0.1s",
              transform: hoverScore === star ? "scale(1.15)" : "scale(1)",
              opacity: disabled ? 0.5 : 1,
            }}
            title={`${star} stars`}
            aria-label={`Relevance ${star} stars`}
          >
            {filled ? <StarFilled /> : <StarOutlined />}
          </button>
        );
      })}
    </div>
  );
}

export default InlineDocRating;
