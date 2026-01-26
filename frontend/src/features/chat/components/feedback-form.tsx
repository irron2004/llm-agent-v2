import { useState, useCallback } from "react";
import { CloseOutlined, UserOutlined } from "@ant-design/icons";
import { RatingStars } from "./rating-stars";

type FeedbackFormData = {
  accuracy: number;
  completeness: number;
  relevance: number;
  comment?: string;
  reviewerName?: string;
};

type FeedbackFormProps = {
  onSubmit: (data: FeedbackFormData) => void;
  onCancel: () => void;
  initialValues?: Partial<FeedbackFormData>;
  isSubmitting?: boolean;
};

export function FeedbackForm({
  onSubmit,
  onCancel,
  initialValues,
  isSubmitting = false,
}: FeedbackFormProps) {
  const [accuracy, setAccuracy] = useState(initialValues?.accuracy ?? 0);
  const [completeness, setCompleteness] = useState(initialValues?.completeness ?? 0);
  const [relevance, setRelevance] = useState(initialValues?.relevance ?? 0);
  const [comment, setComment] = useState(initialValues?.comment ?? "");
  const [reviewerName, setReviewerName] = useState(initialValues?.reviewerName ?? "");
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(() => {
    // Validate
    if (accuracy === 0 || completeness === 0 || relevance === 0) {
      setError("모든 항목의 점수를 선택해 주세요.");
      return;
    }

    setError(null);
    onSubmit({
      accuracy,
      completeness,
      relevance,
      comment: comment.trim() || undefined,
      reviewerName: reviewerName.trim() || undefined,
    });
  }, [accuracy, completeness, relevance, comment, reviewerName, onSubmit]);

  const avgScore =
    accuracy > 0 && completeness > 0 && relevance > 0
      ? ((accuracy + completeness + relevance) / 3).toFixed(1)
      : "-";

  return (
    <div
      className="feedback-form"
      style={{
        background: "var(--color-bg-secondary)",
        borderRadius: 8,
        padding: 16,
        marginTop: 8,
        border: "1px solid var(--color-border)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <h4
          style={{
            margin: 0,
            fontSize: 14,
            fontWeight: 600,
            color: "var(--color-text-primary)",
          }}
        >
          답변 평가
        </h4>
        <button
          onClick={onCancel}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            color: "var(--color-text-secondary)",
            padding: 4,
          }}
          title="닫기"
        >
          <CloseOutlined />
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <RatingStars
          label="정확성 (Accuracy)"
          value={accuracy}
          onChange={setAccuracy}
          disabled={isSubmitting}
        />
        <RatingStars
          label="완성도 (Completeness)"
          value={completeness}
          onChange={setCompleteness}
          disabled={isSubmitting}
        />
        <RatingStars
          label="관련성 (Relevance)"
          value={relevance}
          onChange={setRelevance}
          disabled={isSubmitting}
        />
      </div>

      <div
        style={{
          marginTop: 12,
          padding: "8px 12px",
          background: "var(--color-bg-primary)",
          borderRadius: 4,
          fontSize: 12,
          color: "var(--color-text-secondary)",
        }}
      >
        평균 점수: <strong style={{ color: "var(--color-text-primary)" }}>{avgScore}</strong>
        {avgScore !== "-" && (
          <span style={{ marginLeft: 8 }}>
            ({Number(avgScore) >= 3 ? "만족" : "불만족"})
          </span>
        )}
      </div>

      <div style={{ marginTop: 12 }}>
        <label
          style={{
            display: "block",
            marginBottom: 4,
            fontSize: 12,
            color: "var(--color-text-secondary)",
            fontWeight: 500,
          }}
        >
          의견 (선택사항)
        </label>
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          rows={3}
          placeholder="추가 의견이 있다면 입력해 주세요."
          disabled={isSubmitting}
          style={{
            width: "100%",
            borderRadius: 6,
            border: "1px solid var(--color-border)",
            padding: "8px 10px",
            fontSize: 12,
            resize: "vertical",
            background: "var(--color-bg-primary)",
            color: "var(--color-text-primary)",
          }}
        />
      </div>

      <div style={{ marginTop: 12 }}>
        <label
          style={{
            display: "block",
            marginBottom: 4,
            fontSize: 12,
            color: "var(--color-text-secondary)",
            fontWeight: 500,
          }}
        >
          <UserOutlined style={{ marginRight: 4 }} />
          이름 (선택사항)
        </label>
        <input
          type="text"
          value={reviewerName}
          onChange={(e) => setReviewerName(e.target.value)}
          placeholder="피드백 제출자 이름"
          disabled={isSubmitting}
          style={{
            width: "100%",
            borderRadius: 6,
            border: "1px solid var(--color-border)",
            padding: "8px 10px",
            fontSize: 12,
            background: "var(--color-bg-primary)",
            color: "var(--color-text-primary)",
          }}
        />
      </div>

      {error && (
        <div
          style={{
            marginTop: 8,
            fontSize: 12,
            color: "var(--color-danger, #c0392b)",
          }}
        >
          {error}
        </div>
      )}

      <div
        style={{
          marginTop: 12,
          display: "flex",
          gap: 8,
          justifyContent: "flex-end",
        }}
      >
        <button
          className="action-button"
          onClick={onCancel}
          disabled={isSubmitting}
          style={{
            padding: "6px 12px",
            fontSize: 12,
          }}
        >
          취소
        </button>
        <button
          className="action-button"
          onClick={handleSubmit}
          disabled={isSubmitting}
          style={{
            padding: "6px 12px",
            fontSize: 12,
            background: "var(--color-accent-primary)",
            color: "white",
            border: "none",
          }}
        >
          {isSubmitting ? "저장 중..." : "저장"}
        </button>
      </div>
    </div>
  );
}

export default FeedbackForm;
