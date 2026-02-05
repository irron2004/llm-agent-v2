import { useState, useMemo } from "react";
import { Input, Card, List, Slider, Space, Typography, Row, Col, Button, Spin, Checkbox, Switch, Alert, message, Select } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import { buildUrl } from "@/config/env";
import { RetrievalEvaluationProvider } from "@/features/chat/components/retrieval-evaluation-context";
import { InlineDocRating } from "@/features/chat/components/inline-doc-rating";
import { RetrievalEvaluationSubmit } from "@/features/chat/components/retrieval-evaluation-submit";
import { ImagePreviewModal, ImagePreviewItem } from "@/components/image-preview-modal";
import { DEFAULT_TEST_QUESTIONS } from "@/features/retrieval-test/data/default-questions";
import type { RetrievedDoc } from "@/features/chat/types";

const { Title, Text } = Typography;
const { Search } = Input;

// Build page image URL from doc_id and page
function buildPageImageUrl(docId: string, page: number): string {
  return buildUrl(`/api/assets/docs/${docId}/pages/${page}`);
}

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
  expanded_pages?: number[];
  expanded_page_urls?: string[];
}

interface SearchResponse {
  query: string;
  query_en?: string;
  query_ko?: string;
  route?: string;
  search_queries?: string[];
  auto_parsed_device?: string;
  auto_parsed_doc_type?: string;
  items: SearchResult[];
  total: number;
}

const AVAILABLE_FIELDS: FieldConfig[] = [
  {
    field: "search_text",
    label: "Body text",
    description: "Main body content of the document",
    defaultWeight: 1.0,
    enabled: true,
    weight: 1.0,
  },
  {
    field: "chunk_summary",
    label: "Chunk summary",
    description: "Summary of each chunk",
    defaultWeight: 0.7,
    enabled: true,
    weight: 0.7,
  },
  {
    field: "chunk_keywords.text",
    label: "Keywords",
    description: "Extracted keywords",
    defaultWeight: 0.8,
    enabled: true,
    weight: 0.8,
  },
  {
    field: "content",
    label: "Raw content",
    description: "Original text before preprocessing",
    defaultWeight: 0.6,
    enabled: false,
    weight: 0.6,
  },
  {
    field: "chapter",
    label: "Chapter",
    description: "Document chapter/section title",
    defaultWeight: 1.2,
    enabled: false,
    weight: 1.2,
  },
  {
    field: "doc_description",
    label: "Document description",
    description: "Overall document description",
    defaultWeight: 0.9,
    enabled: false,
    weight: 0.9,
  },
  {
    field: "device_name",
    label: "Equipment",
    description: "Related equipment name",
    defaultWeight: 1.5,
    enabled: false,
    weight: 1.5,
  },
  {
    field: "doc_type",
    label: "Document type",
    description: "Document type (SOP, Setup, etc.)",
    defaultWeight: 1.0,
    enabled: false,
    weight: 1.0,
  },
];

// Component for displaying a single search result item with expanded images
function SearchResultItem({
  item,
  onImageClick,
}: {
  item: SearchResult;
  onImageClick: (url: string, title: string, page?: number) => void;
}) {
  const [imagesLoadSuccess, setImagesLoadSuccess] = useState(false);
  const [imagesLoadError, setImagesLoadError] = useState(false);

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
    <List.Item style={{ display: "block", padding: "12px 0" }}>
      {/* Title row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
        <Text strong style={{ flex: 1, marginRight: 12 }}>
          {item.rank}. {displayTitle}
        </Text>
        <Text type="secondary" style={{ fontSize: 12, whiteSpace: "nowrap" }}>
          {item.score_display}
        </Text>
      </div>

      {/* Page number and Rating row */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        {pageDisplayText && (
          <Text type="secondary" style={{ fontSize: 12 }}>
            {pageDisplayText}
          </Text>
        )}
        <InlineDocRating docId={item.id} size="small" />
      </div>

      {/* Images (expanded pages) */}
      {hasImages && !imagesLoadError && (
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
                  maxWidth: isMultiPage ? 150 : "100%",
                  maxHeight: isMultiPage ? 200 : 300,
                  borderRadius: 4,
                  border: "1px solid #d9d9d9",
                  cursor: "pointer",
                }}
                title="Click to zoom"
                onClick={() => onImageClick(url, displayTitle, pageNumbers[idx])}
                onLoad={() => setImagesLoadSuccess(true)}
                onError={() => setImagesLoadError(true)}
              />
            </div>
          ))}
        </div>
      )}

      {/* Snippet (show if no image or image failed) */}
      {(!hasImages || imagesLoadError || !imagesLoadSuccess) && (
        <div style={{ fontSize: 13, color: "rgba(0, 0, 0, 0.65)", marginBottom: 8 }}>
          {item.snippet}
        </div>
      )}

      {/* Chunk summary */}
      {item.chunk_summary && (
        <div style={{ marginBottom: 8, padding: 8, background: "#f0f7ff", borderRadius: 4 }}>
          <Text strong style={{ fontSize: 12, color: "#1890ff" }}>Chunk summary: </Text>
          <Text style={{ fontSize: 12 }}>{item.chunk_summary}</Text>
        </div>
      )}

      {/* Keywords */}
      {item.chunk_keywords && item.chunk_keywords.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <Text strong style={{ fontSize: 12, color: "#52c41a" }}>Keywords: </Text>
          {item.chunk_keywords.map((kw, idx) => (
            <span
              key={idx}
              style={{
                display: "inline-block",
                padding: "2px 8px",
                margin: "0 4px 4px 0",
                background: "#f6ffed",
                border: "1px solid #b7eb8f",
                borderRadius: 4,
                fontSize: 11,
              }}
            >
              {kw}
            </span>
          ))}
        </div>
      )}

      {/* Metadata */}
      <div style={{ fontSize: 11, color: "rgba(0, 0, 0, 0.45)" }}>
        <Space split="|" size="small" wrap>
          <span>ID: {item.id}</span>
          {item.chapter && <span>Chapter: {item.chapter}</span>}
          {item.doc_type && <span>Type: {item.doc_type}</span>}
          {item.device_name && <span>Equipment: {item.device_name}</span>}
        </Space>
      </div>
    </List.Item>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [searchedQuery, setSearchedQuery] = useState("");  // Query at search time (for evaluation)
  const [results, setResults] = useState<SearchResult[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [fields, setFields] = useState<FieldConfig[]>(
    AVAILABLE_FIELDS.map((f) => ({ ...f }))
  );
  const [bm25Only, setBm25Only] = useState(true); // BM25-only mode (default true)
  const [queryId, setQueryId] = useState<string | null>(null);  // Query ID for evaluation

  // Image preview modal state
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewImages, setPreviewImages] = useState<ImagePreviewItem[]>([]);
  const [previewIndex, setPreviewIndex] = useState(0);

  const handleImageClick = (url: string, title: string, page?: number) => {
    // Build preview images from all results with expanded pages
    const images: ImagePreviewItem[] = [];
    results.forEach((r) => {
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

      // Use expanded_page_urls if available
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

  // Convert SearchResult[] to RetrievedDoc[] for RetrievalEvaluationForm
  const resultsAsDocs = useMemo<RetrievedDoc[]>(() => {
    return results.map((r) => {
      // Build full URLs for expanded pages
      const expandedPageUrls = r.expanded_page_urls && r.expanded_page_urls.length > 0
        ? r.expanded_page_urls.map(url => buildUrl(url))
        : r.page !== null && r.page !== undefined
          ? [buildPageImageUrl(r.id, r.page)]
          : null;

      const expandedPages = r.expanded_pages && r.expanded_pages.length > 0
        ? r.expanded_pages
        : r.page !== null && r.page !== undefined
          ? [r.page]
          : null;

      return {
        id: r.id,
        title: r.title,
        snippet: r.snippet,
        score: r.score,
        score_percent: null,
        metadata: {
          doc_type: r.doc_type,
          device_name: r.device_name,
          chapter: r.chapter,
        },
        page: r.page ?? null,
        page_image_url: expandedPageUrls && expandedPageUrls.length > 0 ? expandedPageUrls[0] : null,
        expanded_pages: expandedPages,
        expanded_page_urls: expandedPageUrls,
      };
    });
  }, [results]);

  // Build search params object for evaluation reproducibility
  const buildSearchParams = () => {
    return {
      pipeline: "chat-pipeline",
      bm25_only: bm25Only,
      use_rrf: !bm25Only,
      rrf_k: 60,
      dense_weight: bm25Only ? 0.0 : 0.7,
      sparse_weight: bm25Only ? 1.0 : 0.3,
      top_k: 20,
      rerank: false,
    };
  };

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
      alert("Select at least one field.");
      return;
    }

    setLoading(true);
    // Generate query_id at search execution time
    const newQueryId = `search:${Date.now()}`;
    setQueryId(newQueryId);
    setSearchedQuery(searchQuery.trim());

    try {
      // Use chat-pipeline for consistent search logic with Retrieval Test
      const url = buildUrl("/api/search/chat-pipeline");

      // Build search override (same parameters as Retrieval Test)
      const searchOverride: Record<string, unknown> = {
        top_k: 20,
        use_rrf: !bm25Only,  // Use RRF when not in BM25-only mode
        rrf_k: 60,
        rerank: false,
      };

      // BM25-only mode
      if (bm25Only) {
        searchOverride.dense_weight = 0.0;
        searchOverride.sparse_weight = 1.0;
        searchOverride.use_rrf = false;
      }

      const requestBody = {
        query: searchQuery.trim(),
        search_override: searchOverride,
      };

      console.log("[Search] Chat Pipeline Request:", requestBody);
      console.log("[Search] BM25 Only Mode:", bm25Only);

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      console.log("[Search] Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[Search] Error response:", errorText);
        throw new Error(`Search failed: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      console.log("[Search] Results:", data.total, "items");

      // Convert chat-pipeline response to SearchResult format
      const items: SearchResult[] = data.items.map((item: {
        rank: number;
        id: string;
        title: string;
        snippet: string;
        score: number;
        score_display: string;
        chunk_summary?: string;
        chapter?: string;
        doc_type?: string;
        device_name?: string;
        page?: number;
        expanded_pages?: number[];
        expanded_page_urls?: string[];
      }) => ({
        rank: item.rank,
        id: item.id,
        title: item.title,
        snippet: item.snippet,
        score: item.score,
        score_display: item.score_display,
        chunk_summary: item.chunk_summary,
        chunk_keywords: [],  // chat-pipeline doesn't return this
        chapter: item.chapter,
        doc_type: item.doc_type,
        device_name: item.device_name,
        page: item.page,
        expanded_pages: item.expanded_pages,
        expanded_page_urls: item.expanded_page_urls,
      }));

      setResults(items);
      setTotal(data.total);
    } catch (error) {
      console.error("[Search] Exception:", error);
      alert(`Search error occurred: ${error}`);
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
    <div style={{ padding: "16px", width: "100%" }}>
      <Row gutter={16}>
        {/* Left Panel - Search Controls */}
        <Col xs={24} lg={8}>
          <Card title="Search settings" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: "100%" }} size="middle">
              {/* Default question selection */}
              <div>
                <Text type="secondary" style={{ fontSize: 12, marginBottom: 4, display: "block" }}>
                  Select a default question
                </Text>
                <Select
                  style={{ width: "100%" }}
                  placeholder="Select a question"
                  showSearch
                  allowClear
                  value={query || undefined}
                  onChange={(value) => {
                    if (value) {
                      setQuery(value);
                    }
                  }}
                  filterOption={(input, option) =>
                    (option?.label?.toString() ?? "").toLowerCase().includes(input.toLowerCase())
                  }
                  options={DEFAULT_TEST_QUESTIONS.map((q, idx) => ({
                    value: q,
                    label: q.slice(0, 60) + (q.length > 60 ? "..." : ""),
                  }))}
                />
              </div>

              <div>
                <Search
                  placeholder="Enter a query"
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
                    <Text strong>Search mode</Text>
                    <Switch
                      checked={bm25Only}
                      onChange={setBm25Only}
                      checkedChildren="BM25"
                      unCheckedChildren="Hybrid"
                    />
                  </div>
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    {bm25Only
                      ? "BM25-only mode: field weight changes have clear effects"
                      : "Hybrid mode: vector search (70%) + BM25 (30%)"}
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
                  <Text strong>Select search fields</Text>
                  <Space size="small">
                    <Button size="small" onClick={toggleAll}>
                      {fields.every((f) => f.enabled) ? "Clear all" : "Select all"}
                    </Button>
                    <Button size="small" onClick={resetWeights}>
                      Reset
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
                    Selected fields: {fields.filter((f) => f.enabled).length} / {fields.length}
                  </Text>
                </div>
                <div>
                  <Text type="secondary">Query parameters:</Text>
                  <br />
                  <Text code style={{ fontSize: "11px", wordBreak: "break-all" }}>
                    {buildFieldWeightsParam() || "(Select fields)"}
                  </Text>
                </div>
              </div>
            </Space>
          </Card>
        </Col>

        {/* Right Panel - Search Results */}
        <Col xs={24} lg={16} style={{ minHeight: "calc(100vh - 100px)" }}>
          <Card
            title={
              <div>
                Search results
                {total > 0 && (
                  <Text type="secondary" style={{ marginLeft: "8px", fontSize: "14px" }}>
                    (Total {total})
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
              <RetrievalEvaluationProvider
                queryId={queryId || `search:${Date.now()}`}
                source="search"
                query={searchedQuery}
                docs={resultsAsDocs}
                searchParams={buildSearchParams()}
              >
                <List
                  dataSource={results}
                  renderItem={(item) => (
                    <SearchResultItem
                      item={item}
                      onImageClick={handleImageClick}
                    />
                  )}
                />

                {/* Retrieval Evaluation Submit */}
                {queryId && searchedQuery && (
                  <div style={{ marginTop: 24, paddingTop: 16, borderTop: "1px solid #f0f0f0" }}>
                    <RetrievalEvaluationSubmit />
                  </div>
                )}
              </RetrievalEvaluationProvider>
            ) : (
              <div style={{ textAlign: "center", padding: "40px", color: "#999" }}>
                Enter a query and press the search button
              </div>
            )}
          </Card>
        </Col>
      </Row>

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
    </div>
  );
}
