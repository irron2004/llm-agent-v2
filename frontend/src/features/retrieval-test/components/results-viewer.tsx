import { useState, useMemo } from "react";
import { Card, List, Typography, Space, Tag, Badge, Collapse, Spin, Button, message } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  TrophyOutlined,
  LikeOutlined,
  DislikeOutlined,
  ExpandAltOutlined,
} from "@ant-design/icons";
import { RetrievalTestResult, SearchResult } from "../types";
import { buildUrl } from "@/config/env";
import { ImagePreviewModal, ImagePreviewItem } from "@/components/image-preview-modal";

const { Text, Paragraph } = Typography;

// Build page image URL from doc_id and page
function buildPageImageUrl(docId: string, page: number): string {
  return buildUrl(`/api/assets/docs/${docId}/pages/${page}`);
}

interface Props {
  result: RetrievalTestResult;
}

// Individual result item with image support
function ResultItem({
  item,
  isGroundTruth,
  onImageClick,
  onRate,
}: {
  item: SearchResult;
  isGroundTruth: boolean;
  onImageClick: (url: string, title: string, page?: number) => void;
  onRate?: (docId: string, rating: "relevant" | "irrelevant") => void;
}) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const isTopRank = item.rank <= 3;

  // Use expanded_page_urls if available, otherwise fall back to single page URL
  const pageUrls = item.expanded_page_urls && item.expanded_page_urls.length > 0
    ? item.expanded_page_urls.map(url => buildUrl(url))
    : item.page !== null && item.page !== undefined
      ? [buildPageImageUrl(item.id, item.page)]
      : [];

  const pageNumbers = item.expanded_pages && item.expanded_pages.length > 0
    ? item.expanded_pages
    : item.page !== null && item.page !== undefined
      ? [item.page]
      : [];

  const hasImages = pageUrls.length > 0;
  const isMultiPage = pageUrls.length > 1;

  // 문서 타입별 제목 표시 방식
  // - SOP, set_up_manual, trouble_shooting_guide → doc_id 표시
  // - myservice, gcb → summary 표시
  const docTypeLower = item.doc_type?.toLowerCase() || "";
  const isDocIdType = ["sop", "set_up_manual", "trouble_shooting_guide"].includes(docTypeLower);
  const isSummaryType = ["myservice", "gcb"].includes(docTypeLower);

  let displayTitle: string;
  if (isDocIdType) {
    displayTitle = item.id;
  } else if (isSummaryType) {
    displayTitle = item.chunk_summary || item.title || item.id;
  } else {
    displayTitle = item.title || item.id;
  }

  // Format page display text
  const pageDisplayText = pageNumbers.length > 0
    ? pageNumbers.length === 1
      ? `p.${pageNumbers[0]}`
      : `p.${pageNumbers[0]}-${pageNumbers[pageNumbers.length - 1]}`
    : null;

  return (
    <List.Item
      style={{
        display: "block",
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
      {/* Title row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "8px",
          marginBottom: 8,
        }}
      >
        <Space>
          <Text strong style={{ fontSize: "14px" }}>
            #{item.rank}
          </Text>
          {item.rank === 1 && (
            <TrophyOutlined style={{ color: "#faad14", fontSize: "16px" }} />
          )}
          {isGroundTruth && (
            <Badge
              status="success"
              text="Ground truth"
              style={{ fontWeight: "bold" }}
            />
          )}
          <Text style={{ fontSize: "13px" }}>{displayTitle}</Text>
          {pageDisplayText && (
            <Text type="secondary" style={{ fontSize: "12px" }}>
              {pageDisplayText}
            </Text>
          )}
        </Space>
        <Space>
          <Tag color={item.score > 10 ? "green" : "blue"}>
            Score: {item.score_display}
          </Tag>
          <Button
            size="small"
            type="text"
            icon={<ExpandAltOutlined />}
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? "Collapse" : "Expand"}
          </Button>
        </Space>
      </div>

      {/* Expanded content with title bar and rating buttons */}
      {expanded && (
        <div
          style={{
            background: "var(--color-bg-secondary)",
            borderRadius: 4,
            padding: 12,
            marginBottom: 8,
          }}
        >
          {/* Title bar with rating buttons */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 12,
              paddingBottom: 8,
              borderBottom: "1px solid var(--color-border)",
            }}
          >
            <div>
              <Text strong style={{ fontSize: 14 }}>{displayTitle}</Text>
              <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                ID: {item.id}
              </Text>
            </div>
            {onRate && (
              <Space>
                <Button
                  size="small"
                  type="primary"
                  icon={<LikeOutlined />}
                  onClick={() => onRate(item.id, "relevant")}
                  style={{ background: "#52c41a", borderColor: "#52c41a" }}
                >
                  Relevant
                </Button>
                <Button
                  size="small"
                  danger
                  icon={<DislikeOutlined />}
                  onClick={() => onRate(item.id, "irrelevant")}
                >
                  Not relevant
                </Button>
              </Space>
            )}
          </div>

          {/* Images (expanded pages) */}
          {hasImages && !imageError && (
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 8,
                marginBottom: 8,
              }}
            >
              {pageUrls.map((url, idx) => (
                <div key={idx} style={{ position: "relative", display: "inline-block" }}>
                  <img
                    src={url}
                    alt={`${displayTitle} page ${pageNumbers[idx] || idx + 1}`}
                    style={{
                      maxWidth: isMultiPage ? 150 : 250,
                      maxHeight: isMultiPage ? 200 : 300,
                      borderRadius: 4,
                      border: "1px solid #d9d9d9",
                      cursor: "pointer",
                      display: imageLoaded ? "block" : "none",
                    }}
                    title="Click to zoom"
                    onClick={() => onImageClick(url, displayTitle, pageNumbers[idx])}
                    onLoad={() => setImageLoaded(true)}
                    onError={() => setImageError(true)}
                  />
                </div>
              ))}
              {!imageLoaded && !imageError && (
                <div style={{ padding: "20px", textAlign: "center", background: "#fafafa", borderRadius: 4 }}>
                  <Spin size="small" />
                </div>
              )}
            </div>
          )}

          {/* Full content */}
          <div style={{ fontSize: 13, lineHeight: 1.6 }}>
            {item.content || item.snippet || item.chunk_summary || "No content"}
          </div>
        </div>
      )}

      {/* Collapsed view: Snippet only */}
      {!expanded && (
        <>
          {/* Small thumbnail if has images */}
          {hasImages && !imageError && (
            <div style={{ marginBottom: 8 }}>
              <img
                src={pageUrls[0]}
                alt={displayTitle}
                style={{
                  maxWidth: 100,
                  maxHeight: 80,
                  borderRadius: 4,
                  border: "1px solid #d9d9d9",
                  cursor: "pointer",
                  display: imageLoaded ? "inline-block" : "none",
                }}
                title="Click to zoom"
                onClick={() => onImageClick(pageUrls[0], displayTitle, pageNumbers[0])}
                onLoad={() => setImageLoaded(true)}
                onError={() => setImageError(true)}
              />
            </div>
          )}

          {/* Snippet */}
          <Paragraph
            ellipsis={{ rows: 2 }}
            style={{ marginBottom: "8px", fontSize: "13px", color: "var(--color-text-secondary)" }}
          >
            {item.snippet || item.chunk_summary || "No content"}
          </Paragraph>
        </>
      )}

      {/* Metadata */}
      <Space size="small" wrap style={{ fontSize: "11px", color: "#8c8c8c" }}>
        {!expanded && (
          <Text type="secondary" style={{ fontSize: "11px" }}>
            ID: {item.id}
          </Text>
        )}
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
      </Space>
    </List.Item>
  );
}

export default function ResultsViewer({ result }: Props) {
  const { searchResults, groundTruthDocIds, metrics } = result;

  // Image preview modal state
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewImages, setPreviewImages] = useState<ImagePreviewItem[]>([]);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [ratedDocs, setRatedDocs] = useState<Record<string, "relevant" | "irrelevant">>({});

  const handleRate = (docId: string, rating: "relevant" | "irrelevant") => {
    setRatedDocs((prev) => ({ ...prev, [docId]: rating }));
    message.success(
      rating === "relevant"
        ? `Marked document ${docId} as relevant.`
        : `Marked document ${docId} as not relevant.`
    );
  };

  const handleImageClick = (url: string, title: string, page?: number) => {
    // Build preview images from all results with expanded pages
    const images: ImagePreviewItem[] = [];
    searchResults.forEach((r) => {
      // 문서 타입별 제목 표시 방식
      const docTypeLower = r.doc_type?.toLowerCase() || "";
      const isDocIdType = ["sop", "set_up_manual", "trouble_shooting_guide"].includes(docTypeLower);
      const isSummaryType = ["myservice", "gcb"].includes(docTypeLower);

      let displayTitle: string;
      if (isDocIdType) {
        displayTitle = r.id;
      } else if (isSummaryType) {
        displayTitle = r.chunk_summary || r.title || r.id;
      } else {
        displayTitle = r.title || r.id;
      }

      const pageUrls = r.expanded_page_urls && r.expanded_page_urls.length > 0
        ? r.expanded_page_urls.map(u => buildUrl(u))
        : r.page !== null && r.page !== undefined
          ? [buildPageImageUrl(r.id, r.page)]
          : [];

      const pageNumbers = r.expanded_pages && r.expanded_pages.length > 0
        ? r.expanded_pages
        : r.page !== null && r.page !== undefined
          ? [r.page]
          : [];

      pageUrls.forEach((pageUrl, idx) => {
        images.push({
          url: pageUrl,
          title: displayTitle,
          page: pageNumbers[idx],
          docId: r.id,
        });
      });
    });

    const clickedIndex = images.findIndex((img) => img.url === url);
    setPreviewImages(images);
    setPreviewIndex(clickedIndex >= 0 ? clickedIndex : 0);
    setPreviewVisible(true);
  };

  return (
    <Card
      title={`Search results: ${result.question}`}
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
                Show ground truth doc IDs ({groundTruthDocIds.length})
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
      <div style={{ maxHeight: "60vh", overflowY: "auto" }}>
        <List
          dataSource={searchResults}
          renderItem={(item) => (
            <ResultItem
              item={item}
              isGroundTruth={groundTruthDocIds.includes(item.id) || ratedDocs[item.id] === "relevant"}
              onImageClick={handleImageClick}
              onRate={handleRate}
            />
          )}
        />
      </div>

      {searchResults.length > 0 && (
        <div style={{ textAlign: "center", marginTop: "16px" }}>
          <Text type="secondary" style={{ fontSize: "12px" }}>
            Total {searchResults.length} results
          </Text>
        </div>
      )}

      {/* Image Preview Modal */}
      {previewImages.length > 0 && (
        <ImagePreviewModal
          visible={previewVisible}
          images={previewImages}
          currentIndex={previewIndex}
          onIndexChange={setPreviewIndex}
          onClose={() => setPreviewVisible(false)}
        />
      )}
    </Card>
  );
}
