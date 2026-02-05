import { useEffect, useState } from "react";
import { Card, Row, Col, Typography, Space, Button, Select, Alert, Spin, Badge, Empty } from "antd";
import { ExperimentOutlined, PlayCircleOutlined, ClearOutlined, ReloadOutlined, LoadingOutlined, StopOutlined } from "@ant-design/icons";
import { useRetrievalTestContext } from "../context/retrieval-test-context";
import { useMetricsCalculation } from "../hooks/use-metrics-calculation";
import SearchConfigPanel from "../components/search-config-panel";
import ResultsViewer from "../components/results-viewer";
import MetricsDisplay from "../components/metrics-display";
import { buildUrl } from "@/config/env";

const { Title, Text } = Typography;

export default function RetrievalTestPage() {
  const {
    selectedQuestion,
    setSelectedQuestion,
    config,
    updateConfig,
    results,
    loading,
    error,
    runSingleTest,
    runBatchTest,
    stopBatchTest,
    clearResults,
    // 기본 질문 (groundTruthDocIds 없음)
    defaultQuestions,
    // Saved evaluations (auto-loaded on mount)
    savedEvaluations,
    loadingEvaluations,
    loadSavedEvaluations,
    batchProgress,
  } = useRetrievalTestContext();

  // 서버 활성 요청 수 폴링
  const [activeRequests, setActiveRequests] = useState(0);

  useEffect(() => {
    const fetchActiveRequests = async () => {
      try {
        const response = await fetch(buildUrl("/api/search/active-requests"));
        if (response.ok) {
          const data = await response.json();
          setActiveRequests(data.active_count);
        }
      } catch {
        // 조용히 실패
      }
    };

    // 즉시 한 번 실행
    fetchActiveRequests();

    // 2초마다 폴링
    const interval = setInterval(fetchActiveRequests, 2000);
    return () => clearInterval(interval);
  }, []);

  // 기본 질문 + 저장된 평가 통합
  const allQuestions = [...defaultQuestions, ...savedEvaluations];

  const { aggregated } = useMetricsCalculation(results);

  const currentResult = results.find((r) => r.questionId === selectedQuestion?.id);

  return (
    <div style={{
      padding: "24px",
      width: "100%",
      maxWidth: "100%",
      margin: "0 auto",
      background: "var(--color-bg-canvas)",
      color: "var(--color-text-primary)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
        <Title level={2} style={{ color: "var(--color-text-primary)", margin: 0 }}>
          <ExperimentOutlined /> Retrieval Test Lab
        </Title>
        {activeRequests > 0 && (
          <Badge
            count={activeRequests}
            style={{ backgroundColor: "#1890ff" }}
            overflowCount={999}
          >
            <span style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 12px",
              background: "var(--color-bg-secondary)",
              borderRadius: 16,
              fontSize: 13,
              color: "var(--color-text-secondary)",
            }}>
              <LoadingOutlined spin />
              Server running
            </span>
          </Badge>
        )}
      </div>
      <Text style={{ color: "var(--color-text-secondary)" }}>
        Test different search settings and review evaluation metrics
      </Text>

      <Row gutter={24} style={{ marginTop: "24px" }}>
        <Col xs={24} lg={8}>
          <Card
            title={`Test questions (default ${defaultQuestions.length} + evaluated ${savedEvaluations.length})`}
            style={{
              marginBottom: "16px",
              background: "var(--color-bg-card)",
              borderColor: "var(--color-border)",
            }}
            styles={{
              header: { color: "var(--color-text-primary)", borderColor: "var(--color-border)" },
              body: { color: "var(--color-text-primary)" },
            }}
            extra={
              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={loadSavedEvaluations}
                loading={loadingEvaluations}
              >
                Refresh
              </Button>
            }
          >
            <Space direction="vertical" style={{ width: "100%" }} size="middle">
              {loadingEvaluations ? (
                <div style={{ textAlign: "center", padding: "20px" }}>
                  <Spin size="small" />
                  <div style={{ marginTop: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>
                    Loading evaluations...
                  </div>
                </div>
              ) : (
                <>
                  <Select
                    style={{ width: "100%" }}
                    placeholder="Select a question"
                    value={selectedQuestion?.id}
                    onChange={(id) => {
                      const q = allQuestions.find((q) => q.id === id);
                      setSelectedQuestion(q || null);
                    }}
                    showSearch
                    filterOption={(input, option) =>
                      (option?.label?.toString() ?? "").toLowerCase().includes(input.toLowerCase())
                    }
                    options={[
                      {
                        label: `Default questions (${defaultQuestions.length})`,
                        options: defaultQuestions.map((q) => ({
                          value: q.id,
                          label: q.question.slice(0, 70) + (q.question.length > 70 ? "..." : ""),
                        })),
                      },
                      {
                        label: `Saved evaluations (${savedEvaluations.length})`,
                        options: savedEvaluations.map((q) => ({
                          value: q.id,
                          label: `[${q.category}] ${q.question.slice(0, 60)}${q.question.length > 60 ? "..." : ""}`,
                        })),
                      },
                    ]}
                  />

                  <Space style={{ width: "100%", justifyContent: "space-between" }}>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      disabled={!selectedQuestion || loading}
                      loading={loading}
                      onClick={() => selectedQuestion && runSingleTest(selectedQuestion)}
                    >
                      Run single
                    </Button>

                    <Button
                      icon={<PlayCircleOutlined />}
                      disabled={loading || allQuestions.length === 0}
                      loading={loading}
                      onClick={() => runBatchTest(allQuestions)}
                    >
                      Run all ({allQuestions.length})
                    </Button>
                  </Space>
                </>
              )}

              {selectedQuestion && (
                <div
                  style={{
                    padding: "12px",
                    background: "var(--color-bg-secondary)",
                    borderRadius: "4px",
                    fontSize: "13px",
                  }}
                >
                  <div style={{ marginBottom: 8 }}>
                    <Text strong>Question:</Text>
                    <div style={{ marginTop: 4, color: "var(--color-text-secondary)" }}>
                      {selectedQuestion.question}
                    </div>
                  </div>
                  {selectedQuestion.groundTruthDocIds.length > 0 ? (
                    <>
                      <Text strong>Ground truth docs ({selectedQuestion.groundTruthDocIds.length}):</Text>
                      <div style={{ marginTop: 4, maxHeight: 120, overflowY: "auto" }}>
                        {selectedQuestion.groundTruthDocIds.map((id) => (
                          <div key={id}>
                            <Text code style={{ fontSize: "11px" }}>
                              {id}
                            </Text>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Default question (no ground truth docs - metrics unavailable)
                    </Text>
                  )}
                </div>
              )}
            </Space>
          </Card>

          <SearchConfigPanel config={config} onChange={updateConfig} />
        </Col>

        <Col xs={24} lg={16}>
          {results.length > 0 && (
            <Card
              title="Overall evaluation metrics"
              style={{
                marginBottom: "16px",
                background: "var(--color-bg-card)",
                borderColor: "var(--color-border)",
              }}
              styles={{
                header: { color: "var(--color-text-primary)", borderColor: "var(--color-border)" },
                body: { color: "var(--color-text-primary)" },
              }}
              extra={
                <Button size="small" icon={<ClearOutlined />} onClick={clearResults}>
                  Reset
                </Button>
              }
            >
              <MetricsDisplay metrics={aggregated} />
            </Card>
          )}

          {error && (
            <Alert
              message="Error"
              description={error}
              type="error"
              closable
              style={{ marginBottom: "16px" }}
            />
          )}

          {loading && (
            <Card
              style={{
                background: "var(--color-bg-card)",
                borderColor: "var(--color-border)",
              }}
            >
              <div style={{ textAlign: "center", padding: "40px 20px", color: "var(--color-text-primary)" }}>
                <Spin size="large" />
                {batchProgress ? (
                  <>
                    <div style={{ marginTop: "16px", fontSize: 16, fontWeight: 500 }}>
                      {batchProgress.current} / {batchProgress.total} running
                    </div>
                    <div style={{ marginTop: 8, fontSize: 13, color: "var(--color-text-secondary)" }}>
                      {batchProgress.currentQuestion}
                    </div>
                    <div style={{ marginTop: 16 }}>
                      <Button
                        danger
                        icon={<StopOutlined />}
                        onClick={stopBatchTest}
                      >
                        Stop
                      </Button>
                    </div>
                    <div style={{
                      marginTop: 16,
                      fontSize: 12,
                      color: "var(--color-text-tertiary)",
                      padding: "8px 16px",
                      background: "var(--color-bg-secondary)",
                      borderRadius: 4,
                      display: "inline-block",
                    }}>
                      Search continues on the server even if you leave the page
                    </div>
                  </>
                ) : (
                  <div style={{ marginTop: "16px" }}>Searching...</div>
                )}
              </div>
            </Card>
          )}

          {!loading && currentResult && <ResultsViewer result={currentResult} />}

          {!loading && !currentResult && results.length === 0 && (
            <Card
              style={{
                background: "var(--color-bg-card)",
                borderColor: "var(--color-border)",
              }}
            >
              <div
                style={{
                  textAlign: "center",
                  padding: "60px 20px",
                  color: "var(--color-text-secondary)",
                }}
              >
                <ExperimentOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
                <div>Select a question and run the search</div>
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
}
