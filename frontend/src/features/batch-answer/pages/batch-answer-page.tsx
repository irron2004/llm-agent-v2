import { useState, useEffect, useCallback } from "react";
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Alert,
  Spin,
  Empty,
  Progress,
  Statistic,
  Table,
  Tag,
  Modal,
  Rate,
  Input,
  Tabs,
} from "antd";
import {
  RobotOutlined,
  PlayCircleOutlined,
  StopOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  EyeOutlined,
  FileTextOutlined,
  BulbOutlined,
} from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import { useBatchAnswer } from "../hooks/use-batch-answer";
import { useRetrievalTestContext } from "../../retrieval-test/context/retrieval-test-context";
import {
  BatchAnswerResult,
  RetrievalMetrics,
  SearchResultItem,
} from "../types";

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

// ─── Results Table ───

interface ResultsTableProps {
  results: BatchAnswerResult[];
  onViewDetail: (result: BatchAnswerResult) => void;
  onRate: (resultId: string, rating: number) => void;
}

function ResultsTable({ results, onViewDetail, onRate }: ResultsTableProps) {
  const columns: ColumnsType<BatchAnswerResult> = [
    {
      title: "#",
      dataIndex: "question_id",
      width: 60,
      render: (_: string, __: BatchAnswerResult, index: number) => index + 1,
    },
    {
      title: "Question",
      dataIndex: "question",
      ellipsis: true,
      render: (text: string) => (
        <span title={text}>{text.slice(0, 60)}{text.length > 60 ? "..." : ""}</span>
      ),
    },
    {
      title: "Status",
      dataIndex: "status",
      width: 90,
      render: (status: string) => {
        if (status === "completed") {
          return <Tag color="success" icon={<CheckCircleOutlined />}>Completed</Tag>;
        }
        if (status === "failed") {
          return <Tag color="error" icon={<CloseCircleOutlined />}>Failed</Tag>;
        }
        return <Tag color="processing">Pending</Tag>;
      },
    },
    {
      title: "Hit@3",
      dataIndex: "retrieval_metrics",
      width: 80,
      render: (metrics: RetrievalMetrics) => (
        metrics.hit_at_3 ? (
          <Tag color="success">HIT</Tag>
        ) : (
          <Tag color="default">MISS</Tag>
        )
      ),
    },
    {
      title: "Answer",
      dataIndex: "answer",
      width: 80,
      render: (_: string, record: BatchAnswerResult) =>
        record.status === "completed" || record.status === "failed" ? (
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => onViewDetail(record)}
            danger={record.status === "failed"}
          >
            View
          </Button>
        ) : null,
    },
    {
      title: "Rating",
      dataIndex: "rating",
      width: 160,
      render: (rating: number | null, record: BatchAnswerResult) =>
        record.status === "completed" ? (
          <Rate
            value={rating || 0}
            onChange={(value) => onRate(record.result_id, value)}
            style={{ fontSize: 14 }}
          />
        ) : (
          <Text type="secondary">-</Text>
        ),
    },
  ];

  return (
    <Table
      dataSource={results}
      columns={columns}
      rowKey="result_id"
      size="small"
      pagination={false}
      scroll={{ y: 400 }}
    />
  );
}

// ─── Answer Detail Modal ───

interface AnswerDetailModalProps {
  visible: boolean;
  result: BatchAnswerResult | null;
  onClose: () => void;
  onSaveRating: (rating: number, comment: string) => void;
}

function AnswerDetailModal({
  visible,
  result,
  onClose,
  onSaveRating,
}: AnswerDetailModalProps) {
  const [rating, setRating] = useState<number>(0);
  const [comment, setComment] = useState<string>("");

  useEffect(() => {
    if (result) {
      setRating(result.rating || 0);
      setComment(result.rating_comment || "");
    }
  }, [result]);

  if (!result) return null;

  const tabs = [
    {
      key: "answer",
      label: (
        <span>
          <FileTextOutlined /> Generated Answer
        </span>
      ),
      children: (
        <div style={{ maxHeight: 400, overflow: "auto" }}>
          {result.status === "failed" && result.error_message && (
            <Alert
              type="error"
              message="Answer generation failed"
              description={result.error_message}
              style={{ marginBottom: 16 }}
            />
          )}
          <Paragraph style={{ whiteSpace: "pre-wrap" }}>
            {result.answer || "(No answer)"}
          </Paragraph>
        </div>
      ),
    },
    {
      key: "docs",
      label: (
        <span>
          <FileTextOutlined /> Retrieved Documents ({result.search_results.length})
        </span>
      ),
      children: (
        <div style={{ maxHeight: 400, overflow: "auto" }}>
          {result.search_results.map((doc: SearchResultItem, idx: number) => (
            <Card
              key={doc.doc_id}
              size="small"
              style={{ marginBottom: 8 }}
              title={
                <Space>
                  <Tag color="blue">#{idx + 1}</Tag>
                  <span>{doc.title}</span>
                  {doc.page && <Tag>p.{doc.page}</Tag>}
                </Space>
              }
            >
              <Text type="secondary" style={{ fontSize: 12 }}>
                {doc.snippet}
              </Text>
            </Card>
          ))}
        </div>
      ),
    },
  ];

  // Add reasoning tab only if reasoning exists
  if (result.reasoning) {
    tabs.push({
      key: "reasoning",
      label: (
        <span>
          <BulbOutlined /> Reasoning
        </span>
      ),
      children: (
        <div style={{ maxHeight: 400, overflow: "auto" }}>
          <Paragraph style={{ whiteSpace: "pre-wrap" }}>
            {result.reasoning}
          </Paragraph>
        </div>
      ),
    });
  }

  return (
    <Modal
      title={
        <Text ellipsis style={{ maxWidth: 500 }}>
          {result.question}
        </Text>
      }
      open={visible}
      onCancel={onClose}
      width={800}
      footer={
        <Space>
          <Text>Rating:</Text>
          <Rate value={rating} onChange={setRating} />
          <Button
            type="primary"
            disabled={!rating}
            onClick={() => {
              onSaveRating(rating, comment);
              onClose();
            }}
          >
            Save
          </Button>
        </Space>
      }
    >
      <Tabs items={tabs} />
      <div style={{ marginTop: 16 }}>
        <Text strong>Comment (optional):</Text>
        <TextArea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="Enter rating comment"
          rows={2}
          style={{ marginTop: 8 }}
        />
      </div>
    </Modal>
  );
}

// ─── Main Page ───

export default function BatchAnswerPage() {
  const {
    // Retrieval test context for source questions (shared state)
    results: retrievalResults,
    savedEvaluations,
    loadingEvaluations,
    config: retrievalConfig,
  } = useRetrievalTestContext();

  const {
    currentRun,
    results,
    isCreating,
    isExecuting,
    error,
    createRun,
    executeNext,
    saveRating,
    clearCurrentRun,
    clearError,
  } = useBatchAnswer();

  const [detailModal, setDetailModal] = useState<{
    visible: boolean;
    result: BatchAnswerResult | null;
  }>({ visible: false, result: null });

  const [isRunning, setIsRunning] = useState(false);

  // Use retrieval test results if available, otherwise use saved evaluations
  const sourceQuestions = retrievalResults.length > 0 ? retrievalResults : [];

  // Start batch execution
  const handleStartBatch = useCallback(async () => {
    if (sourceQuestions.length === 0) return;

    // Create run with questions from retrieval test
    const questions = sourceQuestions.map((r) => ({
      id: r.questionId,
      question: r.question,
      ground_truth_doc_ids: r.groundTruthDocIds,
      search_results: r.searchResults.map((sr, idx) => ({
        rank: sr.rank || idx + 1,
        doc_id: sr.id,
        score: sr.score,
        title: sr.title,
        snippet: sr.snippet,
        content: sr.content || "",
        page: sr.page,
        device_name: sr.device_name,
        doc_type: sr.doc_type,
        expanded_pages: sr.expanded_pages,
        expanded_page_urls: sr.expanded_page_urls,
      })),
    }));

    const sourceConfig = {
      use_rrf: retrievalConfig.useRrf,
      rrf_k: retrievalConfig.rrfK,
      rerank: retrievalConfig.rerank,
      rerank_top_k: retrievalConfig.rerankTopK,
      top_k: retrievalConfig.size,
      dense_weight: retrievalConfig.denseWeight,
      sparse_weight: retrievalConfig.sparseWeight,
    };

    try {
      await createRun(questions, {
        name: `Batch ${new Date().toLocaleString("en-US")}`,
        sourceConfig,
      });
      setIsRunning(true);
    } catch (err) {
      console.error("Failed to create run:", err);
    }
  }, [sourceQuestions, retrievalConfig, createRun]);

  // Execute questions one by one
  useEffect(() => {
    if (!isRunning || !currentRun || isExecuting) return;

    const runNext = async () => {
      try {
        const result = await executeNext();
        if (result === null) {
          // All done
          setIsRunning(false);
        }
      } catch (err) {
        console.error("Execution error:", err);
        setIsRunning(false);
      }
    };

    runNext();
  }, [isRunning, currentRun, isExecuting, executeNext]);

  // Stop execution
  const handleStop = useCallback(() => {
    setIsRunning(false);
  }, []);

  // View detail
  const handleViewDetail = useCallback((result: BatchAnswerResult) => {
    setDetailModal({ visible: true, result });
  }, []);

  // Rate
  const handleRate = useCallback(
    async (resultId: string, rating: number) => {
      await saveRating(resultId, rating);
    },
    [saveRating]
  );

  // Save rating from modal
  const handleSaveRatingFromModal = useCallback(
    async (rating: number, comment: string) => {
      if (!detailModal.result) return;
      await saveRating(detailModal.result.result_id, rating, { comment });
    },
    [detailModal.result, saveRating]
  );

  // Calculate metrics
  const completedResults = results.filter((r) => r.status === "completed");
  const hitAt3Count = completedResults.filter(
    (r) => r.retrieval_metrics.hit_at_3
  ).length;
  const ratedCount = completedResults.filter((r) => r.rating !== null).length;
  const avgRating =
    ratedCount > 0
      ? completedResults
          .filter((r) => r.rating !== null)
          .reduce((sum, r) => sum + (r.rating || 0), 0) / ratedCount
      : 0;

  return (
    <div
      style={{
        padding: "24px",
        width: "100%",
        maxWidth: "100%",
        margin: "0 auto",
        background: "var(--color-bg-canvas)",
        color: "var(--color-text-primary)",
      }}
    >
      <Title level={2} style={{ color: "var(--color-text-primary)" }}>
        <RobotOutlined /> Batch Answer Generation
      </Title>
      <Text style={{ color: "var(--color-text-secondary)" }}>
        Generate and evaluate batch answers using Retrieval Test results
      </Text>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          closable
          onClose={clearError}
          style={{ marginTop: 16, marginBottom: 16 }}
        />
      )}

      <Row gutter={24} style={{ marginTop: 24 }}>
        {/* Left Panel: Source & Controls */}
        <Col xs={24} lg={8}>
          <Card
            title="Source (Retrieval Test)"
            style={{
              marginBottom: 16,
              background: "var(--color-bg-card)",
              borderColor: "var(--color-border)",
            }}
            styles={{
              header: {
                color: "var(--color-text-primary)",
                borderColor: "var(--color-border)",
              },
              body: { color: "var(--color-text-primary)" },
            }}
          >
            {loadingEvaluations ? (
              <div style={{ textAlign: "center", padding: 20 }}>
                <Spin size="small" />
              </div>
            ) : sourceQuestions.length === 0 ? (
              <Empty
                description="Run Retrieval Test first"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Run questions in Retrieval Test to
                  <br />
                  generate answers here
                </Text>
              </Empty>
            ) : (
              <Space direction="vertical" style={{ width: "100%" }}>
                <Statistic
                  title="Available questions"
                  value={sourceQuestions.length}
                  suffix="items"
                />
                <div
                  style={{
                    fontSize: 12,
                    color: "var(--color-text-secondary)",
                  }}
                >
                  Search settings: RRF={retrievalConfig.useRrf ? "ON" : "OFF"},
                  Rerank={retrievalConfig.rerank ? "ON" : "OFF"},
                  Top-K={retrievalConfig.size}
                </div>
              </Space>
            )}
          </Card>

          <Card
            title="Run"
            style={{
              background: "var(--color-bg-card)",
              borderColor: "var(--color-border)",
            }}
            styles={{
              header: {
                color: "var(--color-text-primary)",
                borderColor: "var(--color-border)",
              },
              body: { color: "var(--color-text-primary)" },
            }}
          >
            <Space direction="vertical" style={{ width: "100%" }}>
              {!currentRun ? (
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  block
                  disabled={sourceQuestions.length === 0 || isCreating}
                  loading={isCreating}
                  onClick={handleStartBatch}
                >
                  Start generation
                </Button>
              ) : (
                <>
                  <Progress
                    percent={Math.round(
                      (currentRun.progress.completed /
                        currentRun.progress.total) *
                        100
                    )}
                    status={isRunning ? "active" : undefined}
                  />
                  <Text>
                    Progress: {currentRun.progress.completed} /{" "}
                    {currentRun.progress.total}
                    {currentRun.progress.failed > 0 && (
                      <Text type="danger">
                        {" "}
                        (Failed: {currentRun.progress.failed})
                      </Text>
                    )}
                  </Text>
                  <Space>
                    {isRunning ? (
                      <Button
                        danger
                        icon={<StopOutlined />}
                        onClick={handleStop}
                      >
                        Stop
                      </Button>
                    ) : (
                      <Button
                        type="primary"
                        icon={<PlayCircleOutlined />}
                        onClick={() => setIsRunning(true)}
                        disabled={
                          currentRun.progress.completed >=
                          currentRun.progress.total
                        }
                      >
                        Resume
                      </Button>
                    )}
                    <Button onClick={clearCurrentRun}>Reset</Button>
                  </Space>
                </>
              )}
            </Space>
          </Card>
        </Col>

        {/* Right Panel: Results */}
        <Col xs={24} lg={16}>
          {currentRun && (
            <Card
              title="Aggregate Metrics"
              style={{
                marginBottom: 16,
                background: "var(--color-bg-card)",
                borderColor: "var(--color-border)",
              }}
              styles={{
                header: {
                  color: "var(--color-text-primary)",
                  borderColor: "var(--color-border)",
                },
                body: { color: "var(--color-text-primary)" },
              }}
            >
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="Hit@3"
                    value={completedResults.length > 0 ? Math.round((hitAt3Count / completedResults.length) * 100) : 0}
                    suffix="%"
                    valueStyle={{ color: hitAt3Count > 0 ? "#52c41a" : undefined }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Average rating"
                    value={avgRating.toFixed(1)}
                    suffix="/ 5"
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Rated"
                    value={ratedCount}
                    suffix={`/ ${completedResults.length}`}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Completed"
                    value={completedResults.length}
                    suffix={`/ ${currentRun.progress.total}`}
                  />
                </Col>
              </Row>
            </Card>
          )}

          <Card
            title="Results"
            style={{
              background: "var(--color-bg-card)",
              borderColor: "var(--color-border)",
            }}
            styles={{
              header: {
                color: "var(--color-text-primary)",
                borderColor: "var(--color-border)",
              },
              body: { color: "var(--color-text-primary)" },
            }}
          >
            {results.length === 0 ? (
              <Empty
                description="No results"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Text type="secondary">
                  Results will appear once generation starts
                </Text>
              </Empty>
            ) : (
              <ResultsTable
                results={results}
                onViewDetail={handleViewDetail}
                onRate={handleRate}
              />
            )}
          </Card>
        </Col>
      </Row>

      <AnswerDetailModal
        visible={detailModal.visible}
        result={detailModal.result}
        onClose={() => setDetailModal({ visible: false, result: null })}
        onSaveRating={handleSaveRatingFromModal}
      />
    </div>
  );
}
