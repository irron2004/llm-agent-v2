import { Card, List, Typography, Space, Tag, Badge, Collapse } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  TrophyOutlined,
} from "@ant-design/icons";
import { RetrievalTestResult } from "../types";

const { Text, Paragraph } = Typography;

interface Props {
  result: RetrievalTestResult;
}

export default function ResultsViewer({ result }: Props) {
  const { searchResults, groundTruthDocIds, metrics } = result;

  return (
    <Card
      title={`검색 결과: ${result.question}`}
      style={{
        background: "var(--color-bg-card)",
        borderColor: "var(--color-border)",
      }}
      styles={{
        header: { color: "var(--color-text-primary)", borderColor: "var(--color-border)" },
        body: { color: "var(--color-text-primary)" },
      }}
    >
      {/* Metrics Summary */}
      <div
        style={{
          padding: "16px",
          background: "var(--color-bg-secondary)",
          borderRadius: "4px",
          marginBottom: "16px",
        }}
      >
        <Space size="large" wrap>
          <div>
            <Text type="secondary" style={{ fontSize: "12px" }}>
              Hit@1
            </Text>
            <div>
              {metrics.hit_at_1 ? (
                <Tag color="success" icon={<CheckCircleOutlined />}>
                  HIT
                </Tag>
              ) : (
                <Tag color="error" icon={<CloseCircleOutlined />}>
                  MISS
                </Tag>
              )}
            </div>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: "12px" }}>
              Hit@3
            </Text>
            <div>
              {metrics.hit_at_3 ? (
                <Tag color="success" icon={<CheckCircleOutlined />}>
                  HIT
                </Tag>
              ) : (
                <Tag color="error" icon={<CloseCircleOutlined />}>
                  MISS
                </Tag>
              )}
            </div>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: "12px" }}>
              Hit@5
            </Text>
            <div>
              {metrics.hit_at_5 ? (
                <Tag color="success" icon={<CheckCircleOutlined />}>
                  HIT
                </Tag>
              ) : (
                <Tag color="error" icon={<CloseCircleOutlined />}>
                  MISS
                </Tag>
              )}
            </div>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: "12px" }}>
              Hit@10
            </Text>
            <div>
              {metrics.hit_at_10 ? (
                <Tag color="success" icon={<CheckCircleOutlined />}>
                  HIT
                </Tag>
              ) : (
                <Tag color="error" icon={<CloseCircleOutlined />}>
                  MISS
                </Tag>
              )}
            </div>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: "12px" }}>
              First Relevant Rank
            </Text>
            <div>
              <Text strong style={{ fontSize: "16px" }}>
                {metrics.first_relevant_rank !== null ? (
                  <>
                    <TrophyOutlined style={{ color: "#faad14" }} />{" "}
                    {metrics.first_relevant_rank}
                  </>
                ) : (
                  "N/A"
                )}
              </Text>
            </div>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: "12px" }}>
              Reciprocal Rank
            </Text>
            <div>
              <Text strong style={{ fontSize: "16px" }}>
                {metrics.reciprocal_rank?.toFixed(3) ?? "N/A"}
              </Text>
            </div>
          </div>
        </Space>
      </div>

      {/* Ground Truth Info */}
      <Collapse
        ghost
        size="small"
        style={{ marginBottom: "16px" }}
        items={[
          {
            key: "ground-truth",
            label: (
              <Text type="secondary" style={{ fontSize: "12px" }}>
                정답 문서 ID 보기 ({groundTruthDocIds.length}개)
              </Text>
            ),
            children: (
              <div style={{ paddingLeft: "12px" }}>
                {groundTruthDocIds.map((id) => (
                  <div key={id}>
                    <Text code style={{ fontSize: "11px" }}>
                      {id}
                    </Text>
                  </div>
                ))}
              </div>
            ),
          },
        ]}
      />

      {/* Search Results List */}
      <List
        dataSource={searchResults.slice(0, 20)}
        renderItem={(item) => {
          const isGroundTruth = groundTruthDocIds.includes(item.id);
          const isTopRank = item.rank <= 3;

          return (
            <List.Item
              style={{
                background: isGroundTruth ? "#f6ffed" : "transparent",
                border: isGroundTruth
                  ? "2px solid #52c41a"
                  : isTopRank
                  ? "1px solid #d9d9d9"
                  : "none",
                borderRadius: "4px",
                padding: "12px",
                marginBottom: "8px",
              }}
            >
              <List.Item.Meta
                title={
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      flexWrap: "wrap",
                      gap: "8px",
                    }}
                  >
                    <Space>
                      <Text strong style={{ fontSize: "14px" }}>
                        #{item.rank}
                      </Text>
                      {item.rank === 1 && (
                        <TrophyOutlined
                          style={{ color: "#faad14", fontSize: "16px" }}
                        />
                      )}
                      {isGroundTruth && (
                        <Badge
                          status="success"
                          text="정답 문서"
                          style={{ fontWeight: "bold" }}
                        />
                      )}
                      <Text style={{ fontSize: "13px" }}>
                        {item.title || "제목 없음"}
                      </Text>
                    </Space>
                    <Space size="small">
                      <Tag color={item.score > 10 ? "green" : "blue"}>
                        스코어: {item.score_display}
                      </Tag>
                    </Space>
                  </div>
                }
                description={
                  <div style={{ marginTop: "8px" }}>
                    <Paragraph
                      ellipsis={{ rows: 2, expandable: true, symbol: "더보기" }}
                      style={{ marginBottom: "8px", fontSize: "13px" }}
                    >
                      {item.snippet || item.chunk_summary || "내용 없음"}
                    </Paragraph>
                    <Space
                      size="small"
                      wrap
                      style={{ fontSize: "11px", color: "#8c8c8c" }}
                    >
                      <Text type="secondary" style={{ fontSize: "11px" }}>
                        ID: {item.id}
                      </Text>
                      {item.chapter && (
                        <Tag style={{ fontSize: "10px" }}>{item.chapter}</Tag>
                      )}
                      {item.device_name && (
                        <Tag color="blue" style={{ fontSize: "10px" }}>
                          {item.device_name}
                        </Tag>
                      )}
                      {item.doc_type && (
                        <Tag color="cyan" style={{ fontSize: "10px" }}>
                          {item.doc_type}
                        </Tag>
                      )}
                      {item.page && (
                        <Text type="secondary" style={{ fontSize: "11px" }}>
                          Page: {item.page}
                        </Text>
                      )}
                    </Space>
                  </div>
                }
              />
            </List.Item>
          );
        }}
      />

      {searchResults.length > 20 && (
        <div style={{ textAlign: "center", marginTop: "16px" }}>
          <Text type="secondary" style={{ fontSize: "12px" }}>
            상위 20개 결과만 표시 (전체 {searchResults.length}개)
          </Text>
        </div>
      )}
    </Card>
  );
}
