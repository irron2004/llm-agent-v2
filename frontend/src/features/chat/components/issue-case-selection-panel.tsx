import { useMemo, useState } from "react";
import { Card, Space, Typography } from "antd";

const { Text, Title } = Typography;

type IssueCase = {
  doc_id: string;
  title: string;
  summary: string;
};

type IssueCaseSelectionPanelProps = {
  question: string;
  instruction: string;
  cases: IssueCase[];
  onSelect: (selectedDocId: string) => void;
};

export function IssueCaseSelectionPanel({
  question,
  instruction,
  cases,
  onSelect,
}: IssueCaseSelectionPanelProps) {
  const [selectedDocId, setSelectedDocId] = useState<string>("");

  const indexedCases = useMemo(
    () => cases.map((item, idx) => ({ ...item, index: idx + 1 })),
    [cases]
  );

  return (
    <Card
      style={{
        margin: "16px 0 8px",
        borderRadius: 12,
        border: "1px solid var(--color-border)",
        backgroundColor: "var(--color-bg-secondary)",
        maxWidth: 720,
      }}
    >
      <Space direction="vertical" style={{ width: "100%" }} size="middle">
        <div>
          <Title level={5} style={{ margin: 0, marginBottom: 4 }}>
            이슈 사례 선택
          </Title>
          <Text type="secondary" style={{ fontSize: 13 }}>
            {instruction}
          </Text>
        </div>

        <div
          style={{
            backgroundColor: "var(--color-bg-tertiary)",
            padding: "12px 16px",
            borderRadius: 8,
            fontSize: 13,
          }}
        >
          <Text strong>질문:</Text> {question}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {indexedCases.map((item) => {
            const isSelected = selectedDocId === item.doc_id;
            return (
              <button
                key={item.doc_id}
                type="button"
                onClick={() => {
                  setSelectedDocId(item.doc_id);
                  onSelect(item.doc_id);
                }}
                style={{
                  textAlign: "left",
                  borderRadius: 8,
                  border: isSelected
                    ? "2px solid var(--color-accent-primary)"
                    : "1px solid var(--color-border)",
                  backgroundColor: isSelected
                    ? "var(--color-accent-primary-light)"
                    : "var(--color-bg-primary)",
                  padding: "10px 12px",
                  cursor: "pointer",
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 6 }}>
                  {item.index}. {item.title}
                </div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {item.summary}
                </Text>
              </button>
            );
          })}
        </div>

      </Space>
    </Card>
  );
}
