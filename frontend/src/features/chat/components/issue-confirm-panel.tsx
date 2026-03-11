import { Button, Card, Space, Typography } from "antd";

const { Text, Title } = Typography;

type IssueConfirmPanelProps = {
  question: string;
  instruction: string;
  prompt: string;
  stage: "post_summary" | "post_detail";
  onConfirm: (confirm: boolean) => void;
};

export function IssueConfirmPanel({
  question,
  instruction,
  prompt,
  stage,
  onConfirm,
}: IssueConfirmPanelProps) {
  const title = stage === "post_summary" ? "추가 이슈 확인" : "다른 이슈 확인";

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
            {title}
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

        <div
          style={{
            backgroundColor: "var(--color-bg-primary)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            padding: "12px 16px",
          }}
        >
          {prompt}
        </div>

        <Space size={8}>
          <Button type="primary" onClick={() => onConfirm(true)}>
            예
          </Button>
          <Button onClick={() => onConfirm(false)}>아니오</Button>
        </Space>
      </Space>
    </Card>
  );
}
