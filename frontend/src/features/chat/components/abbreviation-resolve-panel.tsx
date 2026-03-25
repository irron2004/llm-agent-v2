import { useMemo, useState } from "react";
import { Button, Card, Space, Typography } from "antd";

const { Text, Title } = Typography;

type AbbreviationOption = {
  value: string;
  label: string;
  eng?: string;
  kr?: string | null;
};

type AbbreviationItem = {
  token: string;
  abbr_key: string;
  options: AbbreviationOption[];
};

type AbbreviationResolvePanelProps = {
  question: string;
  instruction: string;
  abbreviations: AbbreviationItem[];
  onSubmit: (selections: Record<string, string>) => void;
};

export function AbbreviationResolvePanel({
  question,
  instruction,
  abbreviations,
  onSubmit,
}: AbbreviationResolvePanelProps) {
  const [selections, setSelections] = useState<Record<string, string>>({});

  const abbreviationKeys = useMemo(
    () => abbreviations.map((item) => item.abbr_key),
    [abbreviations]
  );

  const canSubmit = abbreviationKeys.every((key) => {
    const value = selections[key];
    return typeof value === "string" && value.trim().length > 0;
  });

  return (
    <Card
      style={{
        margin: "16px 0 8px",
        borderRadius: 14,
        border: "1px solid var(--color-accent-primary, #1677ff)",
        background:
          "linear-gradient(180deg, var(--color-bg-secondary) 0%, var(--color-accent-primary-light, rgba(22,119,255,0.08)) 100%)",
        boxShadow: "0 10px 24px rgba(15, 23, 42, 0.08)",
        maxWidth: 720,
      }}
    >
      <Space direction="vertical" style={{ width: "100%" }} size="middle">
        <div>
          <Title level={5} style={{ margin: 0, marginBottom: 4 }}>
            약어 의미 선택
          </Title>
          <Text type="secondary" style={{ fontSize: 13 }}>
            {instruction}
          </Text>
        </div>

        <div
          style={{
            backgroundColor: "var(--color-bg-primary)",
            border: "1px dashed var(--color-accent-primary, #1677ff)",
            padding: "12px 16px",
            borderRadius: 8,
            fontSize: 13,
          }}
        >
          <Text strong>질문:</Text> {question}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {abbreviations.map((item, itemIndex) => (
            <div
              key={item.abbr_key}
              style={{
                border: "1px solid var(--color-border)",
                borderRadius: 8,
                background: "var(--color-bg-primary)",
                padding: "10px 12px",
              }}
            >
              <Text strong>
                {itemIndex + 1}. {item.token || item.abbr_key}
              </Text>
              <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
                {item.options.map((option, optionIndex) => {
                  const checked = selections[item.abbr_key] === option.value;
                  return (
                    <button
                      type="button"
                      key={option.value}
                      aria-pressed={checked}
                      onClick={() => {
                        setSelections((prev) => ({
                          ...prev,
                          [item.abbr_key]: option.value,
                        }));
                      }}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        width: "100%",
                        cursor: "pointer",
                        fontSize: 13,
                        padding: "10px 12px",
                        textAlign: "left",
                        borderRadius: 8,
                        border: checked
                          ? "2px solid var(--color-accent-primary, #1677ff)"
                          : "1px solid var(--color-border)",
                        background: checked
                          ? "var(--color-accent-primary-light, rgba(22,119,255,0.12))"
                          : "var(--color-bg-primary)",
                      }}
                    >
                      <span style={{ fontWeight: checked ? 600 : 500 }}>
                        {optionIndex + 1}. {option.label}
                      </span>
                      {checked ? (
                        <Text style={{ color: "var(--color-accent-primary, #1677ff)", fontSize: 12 }}>
                          선택됨
                        </Text>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <Button
            type="primary"
            onClick={() => onSubmit(selections)}
            disabled={!canSubmit}
          >
            선택 완료
          </Button>
        </div>
      </Space>
    </Card>
  );
}
