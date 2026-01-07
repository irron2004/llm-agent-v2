import { Row, Col, Statistic, Progress, Card, Space, Typography } from "antd";
import {
  TrophyOutlined,
  RiseOutlined,
  CheckCircleOutlined,
} from "@ant-design/icons";
import { AggregatedMetrics } from "../types";

const { Text } = Typography;

interface Props {
  metrics: AggregatedMetrics;
}

export default function MetricsDisplay({ metrics }: Props) {
  const getColor = (value: number, threshold: number) => {
    return value >= threshold ? "#3f8600" : value >= threshold * 0.7 ? "#faad14" : "#cf1322";
  };

  const getProgressColor = (value: number, threshold: number) => {
    return value >= threshold ? "#52c41a" : value >= threshold * 0.7 ? "#faad14" : "#ff4d4f";
  };

  return (
    <div>
      {/* Primary Metrics */}
      <Row gutter={16}>
        <Col xs={24} sm={12} md={6}>
          <Card size="small" bordered={false} style={{ background: "var(--color-bg-card)" }}>
            <Statistic
              title={
                <Space style={{ color: "var(--color-text-secondary)" }}>
                  <CheckCircleOutlined />
                  <span>Total Queries</span>
                </Space>
              }
              value={metrics.total_queries}
              valueStyle={{ fontSize: "24px", fontWeight: "bold", color: "var(--color-text-primary)" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small" bordered={false} style={{ background: "var(--color-bg-card)" }}>
            <Statistic
              title={
                <Space style={{ color: "var(--color-text-secondary)" }}>
                  <TrophyOutlined />
                  <span>Hit@1</span>
                </Space>
              }
              value={(metrics.hit_at_1 * 100).toFixed(1)}
              suffix="%"
              valueStyle={{
                color: getColor(metrics.hit_at_1, 0.6),
                fontSize: "24px",
                fontWeight: "bold",
              }}
            />
            <Progress
              percent={metrics.hit_at_1 * 100}
              showInfo={false}
              strokeColor={getProgressColor(metrics.hit_at_1, 0.6)}
              style={{ marginTop: "8px" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small" bordered={false} style={{ background: "var(--color-bg-card)" }}>
            <Statistic
              title={
                <Space style={{ color: "var(--color-text-secondary)" }}>
                  <RiseOutlined />
                  <span>Hit@3</span>
                </Space>
              }
              value={(metrics.hit_at_3 * 100).toFixed(1)}
              suffix="%"
              valueStyle={{
                color: getColor(metrics.hit_at_3, 0.75),
                fontSize: "24px",
                fontWeight: "bold",
              }}
            />
            <Progress
              percent={metrics.hit_at_3 * 100}
              showInfo={false}
              strokeColor={getProgressColor(metrics.hit_at_3, 0.75)}
              style={{ marginTop: "8px" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card size="small" bordered={false} style={{ background: "var(--color-bg-card)" }}>
            <Statistic
              title={<span style={{ color: "var(--color-text-secondary)" }}>MRR</span>}
              value={metrics.mrr.toFixed(3)}
              precision={3}
              valueStyle={{
                color: getColor(metrics.mrr, 0.6),
                fontSize: "24px",
                fontWeight: "bold",
              }}
            />
            <Text type="secondary" style={{ fontSize: "11px" }}>
              Mean Reciprocal Rank
            </Text>
          </Card>
        </Col>
      </Row>

      {/* Secondary Metrics */}
      <Row gutter={16} style={{ marginTop: "16px" }}>
        <Col xs={24} sm={8}>
          <Card size="small" bordered={false} style={{ background: "var(--color-bg-secondary)" }}>
            <Statistic
              title={<span style={{ color: "var(--color-text-secondary)" }}>Hit@5</span>}
              value={(metrics.hit_at_5 * 100).toFixed(1)}
              suffix="%"
              valueStyle={{
                color: getColor(metrics.hit_at_5, 0.85),
                fontSize: "18px",
              }}
            />
            <Progress
              percent={metrics.hit_at_5 * 100}
              showInfo={false}
              strokeColor={getProgressColor(metrics.hit_at_5, 0.85)}
              size="small"
              style={{ marginTop: "4px" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card size="small" bordered={false} style={{ background: "var(--color-bg-secondary)" }}>
            <Statistic
              title={<span style={{ color: "var(--color-text-secondary)" }}>Hit@10</span>}
              value={(metrics.hit_at_10 * 100).toFixed(1)}
              suffix="%"
              valueStyle={{
                color: getColor(metrics.hit_at_10, 0.9),
                fontSize: "18px",
              }}
            />
            <Progress
              percent={metrics.hit_at_10 * 100}
              showInfo={false}
              strokeColor={getProgressColor(metrics.hit_at_10, 0.9)}
              size="small"
              style={{ marginTop: "4px" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card size="small" bordered={false} style={{ background: "var(--color-bg-secondary)" }}>
            <Statistic
              title={<span style={{ color: "var(--color-text-secondary)" }}>Avg First Relevant Rank</span>}
              value={
                metrics.avg_first_relevant_rank
                  ? metrics.avg_first_relevant_rank.toFixed(1)
                  : "N/A"
              }
              valueStyle={{ fontSize: "18px", color: "var(--color-text-primary)" }}
            />
            <Text type="secondary" style={{ fontSize: "11px" }}>
              낮을수록 좋음
            </Text>
          </Card>
        </Col>
      </Row>

      {/* Performance Interpretation */}
      <div
        style={{
          marginTop: "16px",
          padding: "12px",
          background: "var(--color-bg-secondary)",
          borderRadius: "4px",
          border: "1px solid var(--color-border)",
        }}
      >
        <Space direction="vertical" size="small" style={{ width: "100%" }}>
          <Text strong style={{ fontSize: "12px", color: "var(--color-text-primary)" }}>
            📊 성능 해석:
          </Text>
          <ul
            style={{
              margin: 0,
              paddingLeft: "20px",
              fontSize: "11px",
              color: "var(--color-text-secondary)",
            }}
          >
            <li>
              Hit@1 ≥ 60%: 첫 번째 결과에 정답이 있는 비율 (
              {metrics.hit_at_1 >= 0.6 ? "✅ 우수" : "❌ 개선 필요"})
            </li>
            <li>
              Hit@3 ≥ 75%: 상위 3개 결과에 정답이 있는 비율 (
              {metrics.hit_at_3 >= 0.75 ? "✅ 우수" : "❌ 개선 필요"})
            </li>
            <li>
              MRR ≥ 0.6: 정답 순위의 역수 평균 (
              {metrics.mrr >= 0.6 ? "✅ 우수" : "❌ 개선 필요"})
            </li>
            <li>
              평균 정답 순위: {metrics.avg_first_relevant_rank?.toFixed(1) ?? "N/A"} (
              {metrics.avg_first_relevant_rank && metrics.avg_first_relevant_rank <= 3
                ? "✅ 우수"
                : "⚠️ 보통"}
              )
            </li>
          </ul>
        </Space>
      </div>
    </div>
  );
}
