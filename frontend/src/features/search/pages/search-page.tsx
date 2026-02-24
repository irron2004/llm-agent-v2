import { useState } from "react";
import { Input, Card, List, Typography, Row, Col, Button, Spin, Switch, Space, Divider, Tag } from "antd";
import { SearchOutlined, SettingOutlined, ExperimentOutlined } from "@ant-design/icons";
import { buildUrl } from "@/config/env";

const { Title, Text, Paragraph } = Typography;

interface RetrievalDoc {
  rank: number;
  doc_id: string;
  title?: string | null;
  snippet?: string | null;
  score?: number | null;
  metadata?: Record<string, unknown>;
  page?: number;
}

interface RetrievalStep {
  name: string;
  input?: unknown;
  output?: unknown;
  artifacts?: Record<string, unknown>;
}

interface RetrievalResponse {
  run_id: string;
  effective_config: Record<string, unknown>;
  effective_config_hash: string;
  warnings?: string[];
  steps: Record<string, RetrievalStep>;
  docs: RetrievalDoc[];
}

const DEFAULT_RETRIEVAL_REQUEST = {
  final_top_k: 20,
  rerank_enabled: false,
  auto_parse: true,
  skip_mq: false,
} as const;

const toSafeText = (value: unknown): string => {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    return value.map((item) => toSafeText(item)).filter(Boolean).join(", ");
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const toOptionalNumber = (value: unknown): number | undefined => {
  const num = typeof value === "number" ? value : Number(value);
  return Number.isFinite(num) ? num : undefined;
};

const normalizeDocs = (docs: unknown[]): RetrievalDoc[] => {
  if (!Array.isArray(docs)) return [];

  return docs.map((doc, index) => {
    const source = doc && typeof doc === "object" ? (doc as Record<string, unknown>) : {};
    return {
      rank: index + 1,
      doc_id: toSafeText(source.doc_id),
      title: toSafeText(source.title) || "Untitled",
      snippet: toSafeText(source.snippet),
      score: toOptionalNumber(source.score),
      metadata: (source.metadata as Record<string, unknown>) || undefined,
      page: toOptionalNumber(source.page),
    };
  });
};

const formatScore = (score: number | null | undefined): string => {
  return typeof score === "number" ? score.toFixed(3) : "N/A";
};

const formatJson = (value: unknown): string => {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [deterministic, setDeterministic] = useState(true);
  const [debug, setDebug] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RetrievalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const url = buildUrl("/api/retrieval/run");

      const payload = {
        query: searchQuery,
        steps: ["retrieve"],
        deterministic,
        debug,
        ...DEFAULT_RETRIEVAL_REQUEST,
      };

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[Search] Error response:", errorText);
        throw new Error(`Search failed: ${response.status} ${errorText}`);
      }

      const data: RetrievalResponse = await response.json();
      setResult({
        ...data,
        docs: normalizeDocs((data.docs as unknown[]) ?? []),
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      console.error("[Search] Exception:", err);
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  // Convert steps object to array for display
  const getStepsList = (): RetrievalStep[] => {
    if (!result?.steps) return [];
    return Object.entries(result.steps).map(([name, step]) => ({
      ...Object.fromEntries(
        Object.entries(step).filter(([key]) => key !== 'name')
      ),
      name,
    })) as RetrievalStep[];
  };

  // Extract key artifacts from steps
  const getRouteArtifact = () => {
    if (!result?.steps?.route?.artifacts) return null;
    return result.steps.route.artifacts.route as string | undefined;
  };

  const getSearchQueriesArtifact = () => {
    if (!result?.steps?.st_mq?.artifacts) return null;
    return result.steps.st_mq.artifacts.search_queries as string[] | undefined;
  };

  return (
    <div style={{ padding: "24px", maxWidth: "1400px", margin: "0 auto" }}>
      <Title level={2}>Retrieval 검색</Title>

      <Row gutter={24}>
        {/* Left Panel - Search Controls */}
        <Col xs={24} lg={8}>
          <Card title="검색 설정">
            <Space direction="vertical" style={{ width: "100%" }} size="large">
              <div>
                <Input.Search
                  placeholder="검색어를 입력하세요"
                  allowClear
                  enterButton={<SearchOutlined />}
                  size="large"
                  value={query}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
                  onSearch={handleSearch}
                  loading={loading}
                />
              </div>

              <div
                style={{
                  padding: "12px",
                  background: deterministic ? "#e6f7ff" : "#fff7e6",
                  border: `1px solid ${deterministic ? "#91d5ff" : "#ffd591"}`,
                  borderRadius: "4px",
                }}
              >
                <Space direction="vertical" style={{ width: "100%" }} size="small">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <Text strong>
                      <SettingOutlined /> Deterministic 모드
                    </Text>
                    <Switch
                      checked={deterministic}
                      onChange={setDeterministic}
                      checkedChildren="ON"
                      unCheckedChildren="OFF"
                    />
                  </div>
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    {deterministic
                      ? "반복 가능한 일관된 결과를 반환합니다"
                      : "최적의 정확도를 위해 다양한 검색을 시도합니다"}
                  </Text>
                </Space>
              </div>

              <div
                style={{
                  padding: "12px",
                  background: "#f5f5f5",
                  borderRadius: "4px",
                }}
              >
                <Space direction="vertical" style={{ width: "100%" }} size="small">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <Text strong>
                      <ExperimentOutlined /> 디버그 정보 표시
                    </Text>
                    <Switch
                      checked={debug}
                      onChange={setDebug}
                      checkedChildren="ON"
                      unCheckedChildren="OFF"
                    />
                  </div>
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    검색 파이프라인의 상세 정보를 표시합니다
                  </Text>
                </Space>
              </div>

              <Divider style={{ margin: "12px 0" }} />

              <div>
                <Text type="secondary">요청 파라미터:</Text>
                <pre
                  style={{
                    marginTop: "8px",
                    padding: "8px",
                    background: "#f0f0f0",
                    borderRadius: "4px",
                    fontSize: "11px",
                    overflow: "auto",
                  }}
                >
                  {JSON.stringify(
                    {
                      query: query || "(입력 대기 중)",
                      steps: ["retrieve"],
                      deterministic,
                      debug,
                      ...DEFAULT_RETRIEVAL_REQUEST,
                    },
                    null,
                    2
                  )}
                </pre>
              </div>
            </Space>
          </Card>

          {/* Debug Panel - Configuration & Hash */}
          {result && debug && (
            <Card
              title="설정 정보"
              style={{ marginTop: "16px" }}
              extra={
                <Button size="small" onClick={() => setShowDebug(!showDebug)}>
                  {showDebug ? "숨기기" : "상세"}
                </Button>
              }
            >
              <Space direction="vertical" style={{ width: "100%" }} size="small">
                <div>
                  <Text strong>Run ID: </Text>
                  <Text code>{result.run_id}</Text>
                </div>
                <div>
                  <Text strong>Config Hash: </Text>
                  <Text code copyable style={{ fontSize: "11px" }}>
                    {result.effective_config_hash}
                  </Text>
                </div>

                {showDebug && (
                  <>
                    <Divider style={{ margin: "8px 0" }} />
                    <div>
                      <Text strong>Effective Config:</Text>
                      <pre
                        style={{
                          marginTop: "8px",
                          padding: "8px",
                          background: "#f5f5f5",
                          borderRadius: "4px",
                          fontSize: "10px",
                          maxHeight: "200px",
                          overflow: "auto",
                        }}
                      >
                        {formatJson(result.effective_config)}
                      </pre>
                    </div>
                  </>
                )}
              </Space>
            </Card>
          )}
        </Col>

        {/* Right Panel - Search Results */}
        <Col xs={24} lg={16}>
          <Card
            title={
              <div>
                검색 결과
                {result && (
                  <Text type="secondary" style={{ marginLeft: "8px", fontSize: "14px" }}>
                    (총 {result.docs.length}건)
                  </Text>
                )}
              </div>
            }
          >
            {loading ? (
              <div style={{ textAlign: "center", padding: "40px" }}>
                <Spin size="large" />
                <div style={{ marginTop: "16px" }}>
                  <Text type="secondary">검색 중...</Text>
                </div>
              </div>
            ) : error ? (
              <div style={{ textAlign: "center", padding: "40px", color: "#ff4d4f" }}>
                <Text type="danger">{error}</Text>
              </div>
            ) : result ? (
              <>
                {/* Steps / Artifacts */}
                {debug && result.steps && Object.keys(result.steps).length > 0 && (
                  <Card
                    size="small"
                    title="검색 파이프라인"
                    style={{ marginBottom: "16px", background: "#fafafa" }}
                  >
                    <List
                      size="small"
                      dataSource={getStepsList()}
                      renderItem={(step: RetrievalStep, idx) => (
                        <List.Item style={{ padding: "8px 0" }}>
                          <Tag color="blue">{idx + 1}</Tag>
                          <Text strong>{step.name}</Text>
                          {step.artifacts && (
                            <div style={{ marginLeft: "16px", flex: 1 }}>
                              <Text type="secondary" style={{ fontSize: "11px" }}>
                                {JSON.stringify(step.artifacts).slice(0, 100)}
                              </Text>
                            </div>
                          )}
                        </List.Item>
                      )}
                    />

                    {/* Key Artifacts Display */}
                    {getRouteArtifact() && (
                      <div style={{ marginTop: "8px" }}>
                        <Tag color="purple">Route: {getRouteArtifact()}</Tag>
                      </div>
                    )}
                    {getSearchQueriesArtifact() && (
                      <div style={{ marginTop: "8px" }}>
                        <Text strong>Search Queries: </Text>
                        {getSearchQueriesArtifact()?.map((q, i) => (
                          <Tag key={i} color="green">{q}</Tag>
                        ))}
                      </div>
                    )}
                  </Card>
                )}

                {/* Document List */}
                <List
                  dataSource={result.docs}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                              <Text strong>
                               {item.rank}. {item.title ?? "Untitled"}
                              </Text>
                             <Tag color="green">Score: {formatScore(item.score)}</Tag>
                           </div>
                         }
                         description={
                          <div>
                            <Paragraph
                              ellipsis={{ rows: 3, expandable: true, symbol: "more" }}
                              style={{ marginBottom: "8px" }}
                            >
                              {item.snippet ?? ""}
                            </Paragraph>

                            <div style={{ marginTop: "8px" }}>
                              <Space split="|" size="small">
                                <Text type="secondary" style={{ fontSize: "12px" }}>
                                  ID: {item.doc_id}
                                </Text>
                                {item.page && (
                                  <Text type="secondary" style={{ fontSize: "12px" }}>
                                    페이지: {item.page}
                                  </Text>
                                )}
                                {item.metadata?.chapter !== undefined && (
                                  <Text type="secondary" style={{ fontSize: "12px" }}>
                                    챕터: {String(item.metadata?.chapter)}
                                  </Text>
                                )}
                                {item.metadata?.doc_type !== undefined && (
                                  <Text type="secondary" style={{ fontSize: "12px" }}>
                                    타입: {String(item.metadata?.doc_type)}
                                  </Text>
                                )}
                                {item.metadata?.device_name !== undefined && (
                                  <Text type="secondary" style={{ fontSize: "12px" }}>
                                    장비: {String(item.metadata?.device_name)}
                                  </Text>
                                )}
                              </Space>
                            </div>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              </>
            ) : (
              <div style={{ textAlign: "center", padding: "40px", color: "#999" }}>
                검색어를 입력하고 검색 버튼을 눌러주세요
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
