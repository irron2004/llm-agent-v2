import { Button } from "antd";
import { CheckOutlined, LoadingOutlined } from "@ant-design/icons";
import { useRetrievalEvaluationOptional } from "./retrieval-evaluation-context";

/**
 * Submit button for retrieval evaluation.
 * Uses the RetrievalEvaluationContext to get state and submit function.
 */
export function RetrievalEvaluationSubmit() {
  const context = useRetrievalEvaluationOptional();

  if (!context) {
    return null;
  }

  const { isLoading, isSubmitting, isSaved, evaluatedCount, submit } = context;

  if (isLoading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          padding: "8px 0",
          fontSize: 12,
          color: "var(--color-text-secondary)",
        }}
      >
        <LoadingOutlined style={{ marginRight: 6 }} />
        Loading evaluation...
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-end",
        gap: 8,
        paddingTop: 12,
        marginTop: 8,
        borderTop: "1px solid var(--color-border-light, #f0f0f0)",
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
        onClick={submit}
        loading={isSubmitting}
        disabled={evaluatedCount === 0}
        icon={isSaved && !isSubmitting ? <CheckOutlined /> : undefined}
      >
        {isSaved && !isSubmitting ? "Saved" : "Submit evaluation"}
      </Button>
    </div>
  );
}

export default RetrievalEvaluationSubmit;
