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
      message.error("피드백 데이터를 불러오는데 실패했습니다.");
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
      message.success("JSON 내보내기 완료");
    } catch {
      message.error("JSON 내보내기 실패");
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
      message.success("CSV 내보내기 완료");
    } catch {
      message.error("CSV 내보내기 실패");
    }
  };

  const columns: ColumnsType<FeedbackResponse> = [
    {
      title: "시간",
      dataIndex: "ts",
      key: "ts",
      width: 150,
      render: (ts: string) =>
        ts ? new Date(ts).toLocaleString("ko-KR") : "-",
    },
    {
      title: "평균",
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
      title: "정확",
      dataIndex: "accuracy",
      key: "accuracy",
      width: 60,
      align: "center",
    },
    {
      title: "완성",
      dataIndex: "completeness",
      key: "completeness",
      width: 60,
      align: "center",
    },
    {
      title: "관련",
      dataIndex: "relevance",
      key: "relevance",
      width: 60,
      align: "center",
    },
    {
      title: "평가",
      dataIndex: "rating",
      key: "rating",
      width: 80,
      render: (rating: string) => (
        <Tag color={rating === "up" ? "green" : "red"}>
          {rating === "up" ? "만족" : "불만족"}
        </Tag>
      ),
    },
    {
      title: "질문",
      dataIndex: "user_text",
      key: "user_text",
      ellipsis: true,
      render: (text: string) => (
        <span title={text}>{text?.slice(0, 50)}...</span>
      ),
    },
    {
      title: "의견",
      dataIndex: "comment",
      key: "comment",
      width: 150,
      ellipsis: true,
      render: (comment: string) =>
        comment ? <span title={comment}>{comment.slice(0, 30)}...</span> : "-",
    },
    {
      title: "평가자",
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
          상세
        </Button>
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ marginBottom: 24 }}>피드백 관리</h1>

      {/* Statistics Cards */}
      {stats && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic title="전체 피드백" value={stats.total_count} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="평균 점수"
                value={stats.avg_score?.toFixed(2) ?? "-"}
                suffix="/ 5"
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="만족"
                value={stats.rating_distribution?.up ?? 0}
                valueStyle={{ color: "#52c41a" }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="불만족"
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
          placeholder="평가 필터"
          allowClear
          style={{ width: 120 }}
          value={ratingFilter}
          onChange={(v) => {
            setRatingFilter(v);
            setPage(1);
          }}
          options={[
            { label: "만족", value: "up" },
            { label: "불만족", value: "down" },
          ]}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>점수:</span>
          <InputNumber
            placeholder="최소"
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
            placeholder="최대"
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
          새로고침
        </Button>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <Button icon={<DownloadOutlined />} onClick={handleExportJson}>
            JSON 내보내기
          </Button>
          <Button icon={<DownloadOutlined />} onClick={handleExportCsv}>
            CSV 내보내기
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
          showTotal: (t) => `총 ${t}개`,
          onChange: (p, ps) => {
            setPage(p);
            setPageSize(ps);
          },
        }}
        size="small"
      />

      {/* Detail Modal */}
      <Modal
        title="피드백 상세"
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={null}
        width={800}
      >
        {selectedFeedback && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <h4 style={{ marginBottom: 8 }}>점수</h4>
              <div style={{ display: "flex", gap: 24 }}>
                <div>
                  정확성: <strong>{selectedFeedback.accuracy}</strong>/5
                </div>
                <div>
                  완성도: <strong>{selectedFeedback.completeness}</strong>/5
                </div>
                <div>
                  관련성: <strong>{selectedFeedback.relevance}</strong>/5
                </div>
                <div>
                  평균:{" "}
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
                <h4 style={{ marginBottom: 8 }}>의견</h4>
                <p>{selectedFeedback.comment}</p>
              </div>
            )}

            {selectedFeedback.reviewer_name && (
              <div style={{ marginBottom: 16 }}>
                <h4 style={{ marginBottom: 8 }}>평가자</h4>
                <p>{selectedFeedback.reviewer_name}</p>
              </div>
            )}

            <div style={{ marginBottom: 16 }}>
              <h4 style={{ marginBottom: 8 }}>질문</h4>
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
              <h4 style={{ marginBottom: 8 }}>답변</h4>
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
                <h4 style={{ marginBottom: 8 }}>실행 로그</h4>
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
              {new Date(selectedFeedback.ts).toLocaleString("ko-KR")}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
