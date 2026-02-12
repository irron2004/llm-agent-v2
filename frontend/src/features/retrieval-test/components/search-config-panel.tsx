import { Card, Slider, Switch, Space, Typography, InputNumber, Checkbox, Button, Divider } from "antd";
import { SearchConfig, FieldConfig } from "../types";

const { Text } = Typography;

interface Props {
  config: SearchConfig;
  onChange: (updates: Partial<SearchConfig>) => void;
}

export default function SearchConfigPanel({ config, onChange }: Props) {
  const updateFieldWeight = (index: number, updates: Partial<FieldConfig>) => {
    const newFields = [...config.fieldWeights];
    newFields[index] = { ...newFields[index], ...updates };
    onChange({ fieldWeights: newFields });
  };

  const toggleAllFields = () => {
    const allEnabled = config.fieldWeights.every((f) => f.enabled);
    const newFields = config.fieldWeights.map((f) => ({
      ...f,
      enabled: !allEnabled,
    }));
    onChange({ fieldWeights: newFields });
  };

  const resetWeights = () => {
    const defaultFields: FieldConfig[] = [
      { field: "search_text", label: "Body text", enabled: true, weight: 1.0 },
      { field: "chunk_summary", label: "Chunk summary", enabled: true, weight: 0.7 },
      { field: "chunk_keywords.text", label: "Keywords", enabled: true, weight: 0.8 },
      { field: "content", label: "Raw content", enabled: false, weight: 0.6 },
    ];
    onChange({
      fieldWeights: defaultFields,
      denseWeight: 0.7,
      sparseWeight: 0.3,
      useRrf: true,
      rrfK: 60,
      rerank: true,
      rerankTopK: 10,
      multiQuery: true,
      multiQueryN: 3,
      size: 20,
    });
  };

  return (
    <Card
      title="Search settings"
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
          <Text strong>Number of results</Text>
          <div style={{ marginTop: "8px" }}>
            <InputNumber
              style={{ width: "100%" }}
              min={1}
              max={100}
              value={config.size}
              onChange={(v) => v && onChange({ size: v })}
              addonAfter="items"
            />
          </div>
          <Text type="secondary" style={{ fontSize: "11px", fontStyle: "italic" }}>
            💡 Number of documents to return (1-100)
          </Text>
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Hybrid Search Mode */}
        <div>
          <Text strong>Hybrid Search mode</Text>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginTop: "8px",
            }}
          >
            <Text type="secondary" style={{ fontSize: "12px" }}>
              RRF (Reciprocal Rank Fusion)
            </Text>
            <Switch
              checked={config.useRrf}
              onChange={(checked) => onChange({ useRrf: checked })}
            />
          </div>

          {config.useRrf ? (
            // RRF 모드: K 상수만 조절 가능
            <div style={{ marginTop: "12px", paddingLeft: "8px" }}>
              <Text type="secondary" style={{ fontSize: "12px" }}>
                RRF K (rank constant): {config.rrfK}
              </Text>
              <Slider
                min={1}
                max={100}
                step={1}
                value={config.rrfK}
                onChange={(v) => onChange({ rrfK: v })}
                marks={{ 1: "1", 60: "60", 100: "100" }}
              />
              <Text type="secondary" style={{ fontSize: "11px", fontStyle: "italic" }}>
                💡 RRF merges by rank and does not use weights
              </Text>
            </div>
          ) : (
            // 가중치 모드: Dense/Sparse 가중치 조절
            <>
              <div style={{ marginTop: "12px" }}>
                <Text type="secondary" style={{ fontSize: "12px" }}>
                  Dense (vector search): {config.denseWeight.toFixed(1)}
                </Text>
                <Slider
                  min={0}
                  max={1}
                  step={0.1}
                  value={config.denseWeight}
                  onChange={(v) => onChange({ denseWeight: v })}
                  marks={{ 0: "0.0", 0.5: "0.5", 1.0: "1.0" }}
                />
              </div>
              <div style={{ marginTop: "8px" }}>
                <Text type="secondary" style={{ fontSize: "12px" }}>
                  Sparse (BM25): {config.sparseWeight.toFixed(1)}
                </Text>
                <Slider
                  min={0}
                  max={1}
                  step={0.1}
                  value={config.sparseWeight}
                  onChange={(v) => onChange({ sparseWeight: v })}
                  marks={{ 0: "0.0", 0.5: "0.5", 1.0: "1.0" }}
                />
              </div>
              <Text type="secondary" style={{ fontSize: "11px", fontStyle: "italic" }}>
                💡 Dense=0, Sparse=1 → BM25-only mode
              </Text>
            </>
          )}
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Rerank Settings */}
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
            />
          </div>
          {config.rerank && (
            <div style={{ marginTop: "12px", paddingLeft: "8px" }}>
              <Space direction="vertical" style={{ width: "100%" }} size="small">
                <div>
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    Rerank Top K
                  </Text>
                  <InputNumber
                    style={{ width: "100%", marginTop: "4px" }}
                    min={1}
                    max={50}
                    value={config.rerankTopK}
                    onChange={(v) => v && onChange({ rerankTopK: v })}
                  />
                </div>
                <Text type="secondary" style={{ fontSize: "11px" }}>
                  Model: {config.rerankModel}
                </Text>
              </Space>
            </div>
          )}
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Multi-Query Settings */}
        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Text strong>Multi-Query Expansion</Text>
            <Switch
              checked={config.multiQuery}
              onChange={(checked) => onChange({ multiQuery: checked })}
            />
          </div>
          {config.multiQuery && (
            <div style={{ marginTop: "12px", paddingLeft: "8px" }}>
              <Text type="secondary" style={{ fontSize: "12px" }}>
                Number of expanded queries
              </Text>
              <InputNumber
                style={{ width: "100%", marginTop: "4px" }}
                min={1}
                max={10}
                value={config.multiQueryN}
                onChange={(v) => v && onChange({ multiQueryN: v })}
              />
            </div>
          )}
        </div>

        <Divider style={{ margin: "8px 0" }} />

        {/* Field Weights */}
        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "12px",
            }}
          >
            <Text strong>Field weights</Text>
            <Space size="small">
              <Button size="small" onClick={toggleAllFields}>
                {config.fieldWeights.every((f) => f.enabled)
                  ? "Clear all"
                  : "Select all"}
              </Button>
              <Button size="small" onClick={resetWeights}>
                Reset
              </Button>
            </Space>
          </div>

          {config.fieldWeights.map((field, index) => (
            <div key={field.field} style={{ marginBottom: "16px" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  marginBottom: "4px",
                }}
              >
                <Checkbox
                  checked={field.enabled}
                  onChange={(e) =>
                    updateFieldWeight(index, { enabled: e.target.checked })
                  }
                />
                <Text style={{ marginLeft: "8px", fontSize: "13px" }}>
                  {field.label}
                </Text>
                {field.enabled && (
                  <Text
                    type="secondary"
                    style={{ marginLeft: "auto", fontSize: "12px" }}
                  >
                    {field.weight.toFixed(1)}
                  </Text>
                )}
              </div>
              {field.enabled && (
                <div style={{ paddingLeft: "24px" }}>
                  <Slider
                    min={0}
                    max={3}
                    step={0.1}
                    value={field.weight}
                    onChange={(v) => updateFieldWeight(index, { weight: v })}
                    marks={{ 0: "0", 1.5: "1.5", 3: "3" }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      </Space>
    </Card>
  );
}
