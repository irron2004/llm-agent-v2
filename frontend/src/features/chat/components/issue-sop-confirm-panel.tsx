import { Button, Card, Space, Typography } from "antd";

const { Text, Title } = Typography;

type IssueSopConfirmPanelProps = {
  question: string;
  instruction: string;
  prompt: string;
  hasSopRef: boolean;
  sopHint?: string | null;
  onConfirm: (confirm: boolean) => void;
};

export function IssueSopConfirmPanel({
  question,
  instruction,
  prompt,
  hasSopRef,
  sopHint,
  onConfirm,
}: IssueSopConfirmPanelProps) {
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
            SOP 확인
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
          <div style={{ marginBottom: 6 }}>{prompt}</div>
          {hasSopRef ? (
            <Text type="secondary" style={{ fontSize: 12 }}>
              감지된 SOP 힌트: {sopHint || "(없음)"}
            </Text>
          ) : (
            <Text type="secondary" style={{ fontSize: 12 }}>
              SOP 힌트가 없어도 확인을 진행할 수 있습니다.
            </Text>
          )}
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
