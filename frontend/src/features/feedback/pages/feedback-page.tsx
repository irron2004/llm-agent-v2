import { useState, useEffect, useCallback } from "react";
import {
  Table,
  Select,
  InputNumber,
  Button,
  Modal,
  Spin,
  message,
  Card,
  Statistic,
  Row,
  Col,
  Tag,
} from "antd";
import {
  DownloadOutlined,
  ReloadOutlined,
  StarFilled,
} from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import {
  listFeedback,
  getFeedbackStatistics,
  exportFeedbackJson,
  exportFeedbackCsv,
} from "../../chat/api";
import type {
  FeedbackResponse,
  FeedbackStatisticsResponse,
} from "../../chat/types";
import { MarkdownContent } from "../../chat/components/markdown-content";

export default function FeedbackPage() {
  const [loading, setLoading] = useState(false);
  const [items, setItems] = useState<FeedbackResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [ratingFilter, setRatingFilter] = useState<string | undefined>();
  const [minScore, setMinScore] = useState<number | undefined>();
  const [maxScore, setMaxScore] = useState<number | undefined>();
  const [stats, setStats] = useState<FeedbackStatisticsResponse | null>(null);
  const [selectedFeedback, setSelectedFeedback] =
    useState<FeedbackResponse | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [feedbackRes, statsRes] = await Promise.all([
        listFeedback({
          limit: pageSize,
          offset: (page - 1) * pageSize,
          rating: ratingFilter,
          minScore,
          maxScore,
        }),
        getFeedbackStatistics(),
      ]);
      setItems(feedbackRes.items);
      setTotal(feedbackRes.total);
      setStats(statsRes);
    } catch (err) {
      console.error("Failed to fetch feedback:", err);
      message.error("Failed to load feedback data.");
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, ratingFilter, minScore, maxScore]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleExportJson = async () => {
    try {
      const blob = await exportFeedbackJson(minScore ?? 3.0);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "feedback_export.json";
      a.click();
      URL.revokeObjectURL(url);
      message.success("JSON export complete.");
    } catch {
      message.error("JSON export failed.");
    }
  };

  const handleExportCsv = async () => {
    try {
      const blob = await exportFeedbackCsv(minScore ?? 3.0);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "feedback_export.csv";
      a.click();
      URL.revokeObjectURL(url);
      message.success("CSV export complete.");
    } catch {
      message.error("CSV export failed.");
    }
  };

  const columns: ColumnsType<FeedbackResponse> = [
    {
      title: "Time",
      dataIndex: "ts",
      key: "ts",
      width: 150,
      render: (ts: string) =>
        ts ? new Date(ts).toLocaleString("en-US") : "-",
    },
    {
      title: "Avg",
      dataIndex: "avg_score",
      key: "avg_score",
      width: 80,
      render: (score: number) => (
        <span
          style={{
            color: score >= 3 ? "var(--color-success, #52c41a)" : "var(--color-danger, #ff4d4f)",
            fontWeight: 600,
          }}
        >
          <StarFilled style={{ marginRight: 4 }} />
          {score.toFixed(1)}
        </span>
      ),
      sorter: (a, b) => a.avg_score - b.avg_score,
    },
    {
      title: "Acc",
      dataIndex: "accuracy",
      key: "accuracy",
      width: 60,
      align: "center",
    },
    {
      title: "Comp",
      dataIndex: "completeness",
      key: "completeness",
      width: 60,
      align: "center",
    },
    {
      title: "Rel",
      dataIndex: "relevance",
      key: "relevance",
      width: 60,
      align: "center",
    },
    {
      title: "Rating",
      dataIndex: "rating",
      key: "rating",
      width: 80,
      render: (rating: string) => (
        <Tag color={rating === "up" ? "green" : "red"}>
          {rating === "up" ? "Satisfied" : "Dissatisfied"}
        </Tag>
      ),
    },
    {
      title: "Question",
      dataIndex: "user_text",
      key: "user_text",
      ellipsis: true,
      render: (text: string) => (
        <span title={text}>{text?.slice(0, 50)}...</span>
      ),
    },
    {
      title: "Comment",
      dataIndex: "comment",
      key: "comment",
      width: 150,
      ellipsis: true,
      render: (comment: string) =>
        comment ? <span title={comment}>{comment.slice(0, 30)}...</span> : "-",
    },
    {
      title: "Reviewer",
      dataIndex: "reviewer_name",
      key: "reviewer_name",
      width: 100,
      render: (name: string) => name || "-",
    },
    {
      title: "",
      key: "actions",
      width: 80,
      render: (_: unknown, record: FeedbackResponse) => (
        <Button
          type="link"
          size="small"
          onClick={() => {
            setSelectedFeedback(record);
            setDetailModalOpen(true);
          }}
        >
          Details
        </Button>
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ marginBottom: 24 }}>Feedback Management</h1>

      {/* Statistics Cards */}
      {stats && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic title="Total feedback" value={stats.total_count} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Average score"
                value={stats.avg_score?.toFixed(2) ?? "-"}
                suffix="/ 5"
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Satisfied"
                value={stats.rating_distribution?.up ?? 0}
                valueStyle={{ color: "#52c41a" }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="Dissatisfied"
                value={stats.rating_distribution?.down ?? 0}
                valueStyle={{ color: "#ff4d4f" }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* Filters */}
      <div
        style={{
          display: "flex",
          gap: 16,
          marginBottom: 16,
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <Select
          placeholder="Rating filter"
          allowClear
          style={{ width: 120 }}
          value={ratingFilter}
          onChange={(v) => {
            setRatingFilter(v);
            setPage(1);
          }}
          options={[
            { label: "Satisfied", value: "up" },
            { label: "Dissatisfied", value: "down" },
          ]}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>Score:</span>
          <InputNumber
            placeholder="Min"
            min={1}
            max={5}
            step={0.5}
            value={minScore}
            onChange={(v) => {
              setMinScore(v ?? undefined);
              setPage(1);
            }}
            style={{ width: 80 }}
          />
          <span>~</span>
          <InputNumber
            placeholder="Max"
            min={1}
            max={5}
            step={0.5}
            value={maxScore}
            onChange={(v) => {
              setMaxScore(v ?? undefined);
              setPage(1);
            }}
            style={{ width: 80 }}
          />
        </div>
        <Button icon={<ReloadOutlined />} onClick={fetchData}>
          Refresh
        </Button>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <Button icon={<DownloadOutlined />} onClick={handleExportJson}>
            Export JSON
          </Button>
          <Button icon={<DownloadOutlined />} onClick={handleExportCsv}>
            Export CSV
          </Button>
        </div>
      </div>

      {/* Table */}
      <Table
        columns={columns}
        dataSource={items}
        rowKey={(r) => `${r.session_id}:${r.turn_id}`}
        loading={loading}
        pagination={{
          current: page,
          pageSize: pageSize,
          total: total,
          showSizeChanger: true,
          showTotal: (t) => `Total ${t}`,
          onChange: (p, ps) => {
            setPage(p);
            setPageSize(ps);
          },
        }}
        size="small"
      />

      {/* Detail Modal */}
      <Modal
        title="Feedback Details"
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={null}
        width={800}
      >
        {selectedFeedback && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <h4 style={{ marginBottom: 8 }}>Scores</h4>
              <div style={{ display: "flex", gap: 24 }}>
                <div>
                  Accuracy: <strong>{selectedFeedback.accuracy}</strong>/5
                </div>
                <div>
                  Completeness: <strong>{selectedFeedback.completeness}</strong>/5
                </div>
                <div>
                  Relevance: <strong>{selectedFeedback.relevance}</strong>/5
                </div>
                <div>
                  Average:{" "}
                  <strong
                    style={{
                      color:
                        selectedFeedback.avg_score >= 3
                          ? "#52c41a"
                          : "#ff4d4f",
                    }}
                  >
                    {selectedFeedback.avg_score.toFixed(1)}
                  </strong>
                  /5
                </div>
              </div>
            </div>

            {selectedFeedback.comment && (
              <div style={{ marginBottom: 16 }}>
                <h4 style={{ marginBottom: 8 }}>Comment</h4>
                <p>{selectedFeedback.comment}</p>
              </div>
            )}

            {selectedFeedback.reviewer_name && (
              <div style={{ marginBottom: 16 }}>
                <h4 style={{ marginBottom: 8 }}>Reviewer</h4>
                <p>{selectedFeedback.reviewer_name}</p>
              </div>
            )}

            <div style={{ marginBottom: 16 }}>
              <h4 style={{ marginBottom: 8 }}>Question</h4>
              <div
                style={{
                  background: "var(--color-bg-secondary, #f5f5f5)",
                  padding: 12,
                  borderRadius: 8,
                }}
              >
                {selectedFeedback.user_text}
              </div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <h4 style={{ marginBottom: 8 }}>Answer</h4>
              <div
                style={{
                  background: "var(--color-bg-secondary, #f5f5f5)",
                  padding: 12,
                  borderRadius: 8,
                  maxHeight: 300,
                  overflowY: "auto",
                }}
              >
                <MarkdownContent content={selectedFeedback.assistant_text} />
              </div>
            </div>

            {selectedFeedback.logs && selectedFeedback.logs.length > 0 && (
              <div>
                <h4 style={{ marginBottom: 8 }}>Activity Log</h4>
                <div
                  style={{
                    background: "#1e1e1e",
                    color: "#d4d4d4",
                    padding: 12,
                    borderRadius: 8,
                    maxHeight: 200,
                    overflowY: "auto",
                    fontFamily: "monospace",
                    fontSize: 12,
                  }}
                >
                  {selectedFeedback.logs.map((log, idx) => (
                    <div key={idx}>{log}</div>
                  ))}
                </div>
              </div>
            )}

            <div
              style={{
                marginTop: 16,
                fontSize: 12,
                color: "var(--color-text-secondary)",
              }}
            >
              Session: {selectedFeedback.session_id} | Turn:{" "}
              {selectedFeedback.turn_id} |{" "}
              {new Date(selectedFeedback.ts).toLocaleString("en-US")}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
