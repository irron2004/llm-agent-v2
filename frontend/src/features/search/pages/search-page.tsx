import { useState } from "react";
import { Input, Card, List, Slider, Space, Typography, Row, Col, Button, Spin, Checkbox, Switch, Alert } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import { buildUrl } from "@/config/env";

const { Title, Text } = Typography;
const { Search } = Input;

interface FieldConfig {
  field: string;
  label: string;
  description: string;
  defaultWeight: number;
  enabled: boolean;
  weight: number;
}

interface SearchResult {
  rank: number;
  id: string;
  title: string;
  snippet: string;
  score: number;
  score_display: string;
  chunk_summary?: string;
  chunk_keywords?: string[];
  chapter?: string;
  doc_type?: string;
  device_name?: string;
  page?: number;
}

interface SearchResponse {
  query: string;
  items: SearchResult[];
  total: number;
  page: number;
  size: number;
  has_next: boolean;
}

const AVAILABLE_FIELDS: FieldConfig[] = [
  {
    field: "search_text",
    label: "본문 텍스트",
    description: "문서의 주요 본문 내용",
    defaultWeight: 1.0,
    enabled: true,
    weight: 1.0,
  },
  {
    field: "chunk_summary",
    label: "청크 요약",
    description: "각 청크의 요약 내용",
    defaultWeight: 0.7,
    enabled: true,
    weight: 0.7,
  },
  {
    field: "chunk_keywords.text",
    label: "키워드",
    description: "추출된 키워드",
    defaultWeight: 0.8,
    enabled: true,
    weight: 0.8,
  },
  {
    field: "content",
    label: "원본 콘텐츠",
    description: "전처리 전 원본 텍스트",
    defaultWeight: 0.6,
    enabled: false,
    weight: 0.6,
  },
  {
    field: "chapter",
    label: "챕터명",
    description: "문서 챕터/섹션 제목",
    defaultWeight: 1.2,
    enabled: false,
    weight: 1.2,
  },
  {
    field: "doc_description",
    label: "문서 설명",
    description: "문서 전체 설명",
    defaultWeight: 0.9,
    enabled: false,
    weight: 0.9,
  },
  {
    field: "device_name",
    label: "장비명",
    description: "관련 장비 이름",
    defaultWeight: 1.5,
    enabled: false,
    weight: 1.5,
  },
  {
    field: "doc_type",
    label: "문서 타입",
    description: "문서 유형 (SOP, Setup 등)",
    defaultWeight: 1.0,
    enabled: false,
    weight: 1.0,
  },
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [fields, setFields] = useState<FieldConfig[]>(
    AVAILABLE_FIELDS.map((f) => ({ ...f }))
  );
  const [bm25Only, setBm25Only] = useState(true); // BM25 전용 모드 (기본 true)

  const buildFieldWeightsParam = () => {
    return fields
      .filter((f) => f.enabled)
      .map((f) => `${f.field}^${f.weight.toFixed(1)}`)
      .join(",");
  };

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;

    const enabledFields = fields.filter((f) => f.enabled);
    if (enabledFields.length === 0) {
      alert("최소 1개 이상의 필드를 선택해주세요");
      return;
    }

    setLoading(true);
    try {
      const fieldWeights = buildFieldWeightsParam();

      // Build search params
      const params = new URLSearchParams({
        q: searchQuery,
        field_weights: fieldWeights,
        size: "20",
      });

      // Add hybrid search weights if BM25-only mode is enabled
      if (bm25Only) {
        params.append("dense_weight", "0.0");
        params.append("sparse_weight", "1.0");
      }

      const url = buildUrl(`/api/search?${params.toString()}`);

      console.log("[Search] Request URL:", url);
      console.log("[Search] BM25 Only Mode:", bm25Only);

      const response = await fetch(url);
      console.log("[Search] Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[Search] Error response:", errorText);
        throw new Error(`Search failed: ${response.status} ${errorText}`);
      }

      const data: SearchResponse = await response.json();
      console.log("[Search] Results:", data.total, "items");
      setResults(data.items);
      setTotal(data.total);
    } catch (error) {
      console.error("[Search] Exception:", error);
      alert(`검색 중 오류가 발생했습니다: ${error}`);
      setResults([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  };

  const handleFieldToggle = (index: number) => {
    const newFields = [...fields];
    newFields[index].enabled = !newFields[index].enabled;
    setFields(newFields);
  };

  const handleWeightChange = (index: number, value: number) => {
    const newFields = [...fields];
    newFields[index].weight = value;
    setFields(newFields);
  };

  const toggleAll = () => {
    const allEnabled = fields.every((f) => f.enabled);
    setFields(fields.map((f) => ({ ...f, enabled: !allEnabled })));
  };

  const resetWeights = () => {
    setFields(AVAILABLE_FIELDS.map((f) => ({ ...f })));
  };

  return (
    <div style={{ padding: "24px", maxWidth: "1400px", margin: "0 auto" }}>
      <Title level={2}>Elasticsearch 필드별 가중치 검색 테스트</Title>

      <Row gutter={24}>
        {/* Left Panel - Search Controls */}
        <Col xs={24} lg={8}>
          <Card title="검색 설정">
            <Space direction="vertical" style={{ width: "100%" }} size="large">
              <div>
                <Search
                  placeholder="검색어를 입력하세요"
                  allowClear
                  enterButton={<SearchOutlined />}
                  size="large"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onSearch={handleSearch}
                />
              </div>

              <div
                style={{
                  padding: "12px",
                  background: bm25Only ? "#e6f7ff" : "#fff7e6",
                  border: `1px solid ${bm25Only ? "#91d5ff" : "#ffd591"}`,
                  borderRadius: "4px",
                }}
              >
                <Space direction="vertical" style={{ width: "100%" }} size="small">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <Text strong>검색 모드</Text>
                    <Switch
                      checked={bm25Only}
                      onChange={setBm25Only}
                      checkedChildren="BM25"
                      unCheckedChildren="하이브리드"
                    />
                  </div>
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    {bm25Only
                      ? "BM25 전용 모드: 필드 가중치 변경 효과가 명확하게 나타납니다"
                      : "하이브리드 모드: 벡터 검색(70%) + BM25(30%)"}
                  </Text>
                </Space>
              </div>

              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "16px",
                  }}
                >
                  <Text strong>검색 필드 선택</Text>
                  <Space size="small">
                    <Button size="small" onClick={toggleAll}>
                      {fields.every((f) => f.enabled) ? "전체 해제" : "전체 선택"}
                    </Button>
                    <Button size="small" onClick={resetWeights}>
                      초기화
                    </Button>
                  </Space>
                </div>

                {fields.map((field, index) => (
                  <div key={field.field} style={{ marginBottom: "16px" }}>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        marginBottom: "8px",
                      }}
                    >
                      <Checkbox
                        checked={field.enabled}
                        onChange={() => handleFieldToggle(index)}
                      />
                      <div style={{ marginLeft: "8px", flex: 1 }}>
                        <Text strong={field.enabled}>{field.label}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: "12px" }}>
                          {field.description}
                        </Text>
                      </div>
                      {field.enabled && (
                        <Text type="secondary" style={{ fontSize: "12px" }}>
                          {field.weight.toFixed(1)}
                        </Text>
                      )}
                    </div>

                    {field.enabled && (
                      <div style={{ paddingLeft: "32px" }}>
                        <Slider
                          min={0}
                          max={3}
                          step={0.1}
                          value={field.weight}
                          onChange={(value) => handleWeightChange(index, value)}
                          marks={{
                            0: "0.0",
                            1.0: "1.0",
                            2.0: "2.0",
                            3.0: "3.0",
                          }}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div
                style={{
                  padding: "12px",
                  background: "#f5f5f5",
                  borderRadius: "4px",
                  fontSize: "12px",
                }}
              >
                <div style={{ marginBottom: "8px" }}>
                  <Text type="secondary">
                    선택된 필드: {fields.filter((f) => f.enabled).length} / {fields.length}
                  </Text>
                </div>
                <div>
                  <Text type="secondary">쿼리 파라미터:</Text>
                  <br />
                  <Text code style={{ fontSize: "11px", wordBreak: "break-all" }}>
                    {buildFieldWeightsParam() || "(필드를 선택해주세요)"}
                  </Text>
                </div>
              </div>
            </Space>
          </Card>
        </Col>

        {/* Right Panel - Search Results */}
        <Col xs={24} lg={16}>
          <Card
            title={
              <div>
                검색 결과
                {total > 0 && (
                  <Text type="secondary" style={{ marginLeft: "8px", fontSize: "14px" }}>
                    (총 {total}건)
                  </Text>
                )}
              </div>
            }
          >
            {loading ? (
              <div style={{ textAlign: "center", padding: "40px" }}>
                <Spin size="large" />
              </div>
            ) : results.length > 0 ? (
              <List
                dataSource={results}
                renderItem={(item) => (
                  <List.Item>
                    <List.Item.Meta
                      title={
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <Text strong>
                            {item.rank}. {item.title}
                          </Text>
                          <Text type="secondary" style={{ fontSize: "12px" }}>
                            스코어: {item.score_display}
                          </Text>
                        </div>
                      }
                      description={
                        <div>
                          <Text>{item.snippet}</Text>
                          <br />

                          {item.chunk_summary && (
                            <div style={{ marginTop: "8px", padding: "8px", background: "#f0f7ff", borderRadius: "4px" }}>
                              <Text strong style={{ fontSize: "12px", color: "#1890ff" }}>청크 요약: </Text>
                              <Text style={{ fontSize: "12px" }}>{item.chunk_summary}</Text>
                            </div>
                          )}

                          {item.chunk_keywords && item.chunk_keywords.length > 0 && (
                            <div style={{ marginTop: "8px" }}>
                              <Text strong style={{ fontSize: "12px", color: "#52c41a" }}>키워드: </Text>
                              {item.chunk_keywords.map((kw, idx) => (
                                <span
                                  key={idx}
                                  style={{
                                    display: "inline-block",
                                    padding: "2px 8px",
                                    margin: "0 4px 4px 0",
                                    background: "#f6ffed",
                                    border: "1px solid #b7eb8f",
                                    borderRadius: "4px",
                                    fontSize: "11px",
                                  }}
                                >
                                  {kw}
                                </span>
                              ))}
                            </div>
                          )}

                          <div style={{ marginTop: "8px" }}>
                            <Space split="|" size="small">
                              <Text type="secondary" style={{ fontSize: "12px" }}>
                                ID: {item.id}
                              </Text>
                              {item.page && (
                                <Text type="secondary" style={{ fontSize: "12px" }}>
                                  페이지: {item.page}
                                </Text>
                              )}
                              {item.chapter && (
                                <Text type="secondary" style={{ fontSize: "12px" }}>
                                  챕터: {item.chapter}
                                </Text>
                              )}
                              {item.doc_type && (
                                <Text type="secondary" style={{ fontSize: "12px" }}>
                                  타입: {item.doc_type}
                                </Text>
                              )}
                              {item.device_name && (
                                <Text type="secondary" style={{ fontSize: "12px" }}>
                                  장비: {item.device_name}
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