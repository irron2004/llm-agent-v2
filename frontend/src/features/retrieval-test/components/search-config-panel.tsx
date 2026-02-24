import { Card, Slider, Switch, Space, Typography, InputNumber, Button, Divider } from "antd";
import { SearchConfig } from "../types";
import { ReloadOutlined } from "@ant-design/icons";

const { Text } = Typography;

interface Props {
  config: SearchConfig;
  onChange: (updates: Partial<SearchConfig>) => void;
  onReplay?: () => void;
  lastRunId?: string | null;
  disabled?: boolean;
}

export default function SearchConfigPanel({ config, onChange, onReplay, lastRunId, disabled }: Props) {
  const resetToDefaults = () => {
    onChange({
      size: 20,
      deterministic: true,
      rerank: false,
      autoParse: true,
      skipMq: false,
    });
  };

  return (
    <Card
      title="검색 설정"
      style={{
        background: "var(--color-bg-card)",
        borderColor: "var(--color-border)",
      }}
      styles={{
        header: { color: "var(--color-text-primary)", borderColor: "var(--color-border)" },
        body: { color: "var(--color-text-primary)" },
      }}
    >
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        {/* Search Size */}
        <div>
          <Text strong>검색 결과 개수 (final_top_k)</Text>
          <div style={{ marginTop: "8px" }}>
            <InputNumber
              style={{ width: "100%" }}
              min={1}
              max={100}
              value={config.size}
              onChange={(v) => v && onChange({ size: v })}
              addonAfter="개"
              disabled={disabled}
            />
          </div>
          <Text type="secondary" style={{ fontSize: "11px", fontStyle: "italic" }}>
            💡 평가에 사용할 문서 개수 (1~100)
          </Text>
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Deterministic Mode */}
        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text strong>Deterministic 모드</Text>
            <Switch
              checked={config.deterministic}
              onChange={(checked) => onChange({ deterministic: checked })}
              disabled={disabled}
            />
          </div>
          <Text type="secondary" style={{ fontSize: "11px", fontStyle: "italic" }}>
            {config.deterministic
              ? "💡 반복 가능한 일관된 결과를 반환합니다"
              : "💡 최적의 정확도를 위해 다양한 검색을 시도합니다"}
          </Text>
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Rerank */}
        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text strong>Reranking</Text>
            <Switch
              checked={config.rerank}
              onChange={(checked) => onChange({ rerank: checked })}
              disabled={disabled}
            />
          </div>
          <Text type="secondary" style={{ fontSize: "11px" }}>
            {config.rerank ? "활성화됨" : "비활성화됨"}
          </Text>
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Auto Parse */}
        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text strong>Auto Parse</Text>
            <Switch
              checked={config.autoParse}
              onChange={(checked) => onChange({ autoParse: checked })}
              disabled={disabled}
            />
          </div>
          <Text type="secondary" style={{ fontSize: "11px" }}>
            {config.autoParse ? "활성화됨" : "비활성화됨"}
          </Text>
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Skip MQ */}
        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text strong>Skip Multi-Query</Text>
            <Switch
              checked={config.skipMq}
              onChange={(checked) => onChange({ skipMq: checked })}
              disabled={disabled}
            />
          </div>
          <Text type="secondary" style={{ fontSize: "11px" }}>
            {config.skipMq ? "Multi-Query 생략" : "Multi-Query 사용"}
          </Text>
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Replay Button */}
        <div>
          <Button
            type="default"
            icon={<ReloadOutlined />}
            onClick={onReplay}
            disabled={!lastRunId || disabled}
            block
          >
            Replay last run
          </Button>
          {lastRunId && (
            <Text type="secondary" style={{ fontSize: "11px", display: "block", marginTop: "4px" }}>
              Last run_id: {lastRunId.slice(0, 12)}...
            </Text>
          )}
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Reset Button */}
        <div>
          <Button
            type="dashed"
            onClick={resetToDefaults}
            disabled={disabled}
            block
          >
            설정 초기화
          </Button>
        </div>
      </Space>
    </Card>
  );
}
