import { Card, Row, Col, Typography, Space, Button, Select, Alert, Spin } from "antd";
import { ExperimentOutlined, PlayCircleOutlined, ClearOutlined } from "@ant-design/icons";
import { useRetrievalTest } from "../hooks/use-retrieval-test";
import { useMetricsCalculation } from "../hooks/use-metrics-calculation";
import { TEST_QUESTIONS } from "../data/test-questions";
import SearchConfigPanel from "../components/search-config-panel";
import ResultsViewer from "../components/results-viewer";
import MetricsDisplay from "../components/metrics-display";

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
    clearResults,
  } = useRetrievalTest();

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
      <Title level={2} style={{ color: "var(--color-text-primary)" }}>
        <ExperimentOutlined /> Retrieval Test Lab
      </Title>
      <Text style={{ color: "var(--color-text-secondary)" }}>
        다양한 검색 설정을 테스트하고 평가 메트릭을 확인하세요
      </Text>

      <Row gutter={24} style={{ marginTop: "24px" }}>
        <Col xs={24} lg={8}>
          <Card
            title="테스트 질문 선택"
            style={{
              marginBottom: "16px",
              background: "var(--color-bg-card)",
              borderColor: "var(--color-border)",
            }}
            styles={{
              header: { color: "var(--color-text-primary)", borderColor: "var(--color-border)" },
              body: { color: "var(--color-text-primary)" },
            }}
          >
            <Space direction="vertical" style={{ width: "100%" }} size="middle">
              <Select
                style={{ width: "100%" }}
                placeholder="질문을 선택하세요"
                value={selectedQuestion?.id}
                onChange={(id) => {
                  const q = TEST_QUESTIONS.find((q) => q.id === id);
                  setSelectedQuestion(q || null);
                }}
                options={TEST_QUESTIONS.map((q) => ({
                  value: q.id,
                  label: `[${q.category}] ${q.question}`,
                }))}
              />

              <Space style={{ width: "100%", justifyContent: "space-between" }}>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  disabled={!selectedQuestion || loading}
                  loading={loading}
                  onClick={() => selectedQuestion && runSingleTest(selectedQuestion)}
                >
                  단일 실행
                </Button>

                <Button
                  icon={<PlayCircleOutlined />}
                  disabled={loading}
                  loading={loading}
                  onClick={() => runBatchTest(TEST_QUESTIONS)}
                >
                  전체 실행
                </Button>
              </Space>

              {selectedQuestion && (
                <div
                  style={{
                    padding: "12px",
                    background: "var(--color-bg-secondary)",
                    borderRadius: "4px",
                    fontSize: "13px",
                  }}
                >
                  <Text strong>정답 문서:</Text>
                  <br />
                  {selectedQuestion.groundTruthDocIds.map((id) => (
                    <div key={id}>
                      <Text code style={{ fontSize: "12px" }}>
                        {id}
                      </Text>
                    </div>
                  ))}
                </div>
              )}
            </Space>
          </Card>

          <SearchConfigPanel config={config} onChange={updateConfig} />
        </Col>

        <Col xs={24} lg={16}>
          {results.length > 0 && (
            <Card
              title="전체 평가 메트릭"
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
                  초기화
                </Button>
              }
            >
              <MetricsDisplay metrics={aggregated} />
            </Card>
          )}

          {error && (
            <Alert
              message="오류"
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
              <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--color-text-primary)" }}>
                <Spin size="large" />
                <div style={{ marginTop: "16px" }}>검색 중...</div>
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
                <div>질문을 선택하고 검색을 실행하세요</div>
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
}
