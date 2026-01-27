import { useState, useMemo, useCallback } from "react";
import { CopyOutlined, CheckOutlined, ReloadOutlined, FilterOutlined, LikeOutlined, LikeFilled, DislikeOutlined, DislikeFilled, EditOutlined } from "@ant-design/icons";
import { Message, FeedbackRating, RetrievedDoc, MessageFeedback } from "../types";
import { MarkdownContent } from "./markdown-content";
import { Collapse, Tag } from "antd";
import { ImagePreviewModal, ImagePreviewItem } from "../../../components/image-preview-modal";
import { FeedbackForm } from "./feedback-form";

// Preprocess snippet for better markdown rendering
function preprocessSnippet(snippet: string): string {
  if (!snippet) return "";

  let processed = snippet;

  // Escape markdown emphasis characters to prevent unintended formatting
  // This handles: _italic_, __bold__, *italic*, **bold**, ~~strikethrough~~
  // Escape underscores that are part of words (e.g., file_name, __init__)
  processed = processed.replace(/(\w)_(\w)/g, "$1\\_$2");
  // Escape tildes for strikethrough (~~text~~)
  processed = processed.replace(/~~/g, "\\~\\~");
  // Escape asterisks that might trigger bold/italic (but preserve list items)
  processed = processed.replace(/(\w)\*(\w)/g, "$1\\*$2");

  // Remove page_X or page_\d+ prefixes at the start
  processed = processed.replace(/^page_[X\d]+\s*/gi, "");

  // Remove leading prefixes like ">>>> " or ">>> "
  processed = processed.replace(/^>+\s*/gm, "");

  // Remove markdown code block wrappers more thoroughly
  // Handle cases like: ```markdown, ```, ```json, etc.
  // Match code blocks that span multiple lines
  processed = processed.replace(/```[a-z]*\s*\n?/gi, "");
  processed = processed.replace(/\n?```\s*/g, "");
  
  // Also handle inline code blocks that might be at the start
  processed = processed.replace(/^`{1,3}[a-z]*\s*/gi, "");
  processed = processed.replace(/`{1,3}\s*$/gi, "");

  // Remove any remaining backticks that are standalone
  processed = processed.replace(/^`+\s*/gm, "");
  processed = processed.replace(/\s*`+$/gm, "");

  // Convert HTML <br> tags to newlines
  processed = processed.replace(/<br\s*\/?>/gi, "\n");

  // Remove other common HTML tags that shouldn't be rendered
  processed = processed.replace(/<\/?(?:div|span|p)[^>]*>/gi, "\n");

  // Clean up lines that start with just markdown syntax characters
  processed = processed.split("\n")
    .map(line => {
      // Remove lines that are just markdown syntax
      const trimmed = line.trim();
      if (/^[#`|>\-\*_]+$/.test(trimmed) && trimmed.length < 5) {
        return "";
      }
      return line;
    })
    .join("\n");

  // Format markdown tables that are on single line
  if (processed.includes("|") && /\|\s*:?-{2,}:?\s*\|/.test(processed) && !processed.includes("\n|")) {
    processed = processed.replace(/\s*\|\s*\|\s*/g, " |\n| ");
  }

  // Clean up excessive newlines (more than 2 consecutive)
  processed = processed.replace(/\n{3,}/g, "\n\n");

  // Remove leading/trailing whitespace from each line
  processed = processed.split("\n")
    .map(line => line.trimEnd())
    .join("\n");

  // Trim overall whitespace
  processed = processed.trim();

  return processed;
}

// 이미지 URL이 유효한지 확인하는 헬퍼
function hasValidImageUrl(url: string | null | undefined): url is string {
  if (typeof url !== 'string') return false;
  const trimmed = url.trim();
  if (trimmed.length === 0 || trimmed === 'null' || trimmed === 'undefined') return false;
  return true;
}

// 개별 참고 문서 아이템 - 이미지 로드 상태를 자체 관리
type ReferenceItemProps = {
  doc: RetrievedDoc;
  idx: number;
  onImageClick: (docIndex: number, pageIndex: number) => void;
};

function ReferenceItem({ doc, idx, onImageClick }: ReferenceItemProps) {
  const [imageLoadSuccess, setImageLoadSuccess] = useState(false);
  const [imageLoadError, setImageLoadError] = useState(false);

  const pageNumbers = doc.expanded_pages && doc.expanded_pages.length > 0
    ? doc.expanded_pages
    : doc.page !== null && doc.page !== undefined
      ? [doc.page]
      : [];

  const pageUrls = doc.expanded_page_urls && doc.expanded_page_urls.length > 0
    ? doc.expanded_page_urls.filter(url => hasValidImageUrl(url))
    : hasValidImageUrl(doc.page_image_url)
      ? [doc.page_image_url]
      : [];

  const hasImageUrls = pageUrls.length > 0;

  // sop, ts, setup 타입은 {doc_type}_{id} 형식으로 표시
  const docType = (doc.metadata as Record<string, unknown>)?.doc_type as string | undefined;
  const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
  const displayTitle = isSpecialDocType
    ? `${docType}_${doc.id}`
    : (doc.title || `Document ${idx + 1}`);

  // 이미지가 있고 로드에 성공했으면 텍스트 숨김
  const showText = !hasImageUrls || imageLoadError || !imageLoadSuccess;

  return (
    <div className="reference-item" style={{ marginBottom: 16 }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>
        {displayTitle}
        {pageNumbers.length > 0 && (
          <span style={{ fontWeight: 400, marginLeft: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>
            {pageNumbers.length === 1
              ? `p.${pageNumbers[0]}`
              : `p.${pageNumbers[0]}-${pageNumbers[pageNumbers.length - 1]}`}
          </span>
        )}
      </div>
      {/* 이미지가 있으면 표시 */}
      {hasImageUrls && !imageLoadError && (
        <div
          className="reference-images-container"
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 8,
            marginBottom: showText ? 8 : 0,
          }}
        >
          {pageUrls.map((url, pageIdx) => (
            <div key={pageIdx} className="reference-image-wrapper">
              <div style={{ position: "relative", display: "inline-block" }}>
                <img
                  src={url}
                  alt={`${displayTitle} page ${pageNumbers[pageIdx] || pageIdx + 1}`}
                  style={{
                    maxWidth: pageUrls.length > 1 ? 150 : "100%",
                    maxHeight: pageUrls.length > 1 ? 200 : 300,
                    borderRadius: 4,
                    border: "1px solid var(--color-border)",
                    cursor: "pointer",
                  }}
                  title={`클릭하여 확대`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onImageClick(idx, pageIdx);
                  }}
                  onLoad={() => setImageLoadSuccess(true)}
                  onError={() => setImageLoadError(true)}
                />
              </div>
            </div>
          ))}
        </div>
      )}
      {/* 텍스트: 이미지가 없거나 로드 실패 시에만 표시 */}
      {showText && doc.snippet && (
        <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
          <MarkdownContent content={preprocessSnippet(doc.snippet)} />
        </div>
      )}
      {(doc.score !== null && doc.score !== undefined) && (
        <div style={{ fontSize: 12, opacity: 0.6 }}>
          score: {doc.score.toFixed(3)} {doc.score_percent ? `(${doc.score_percent}%)` : ""}
        </div>
      )}
    </div>
  );
}

export type RegeneratePayload = {
  messageId: string;
  originalQuery: string;
  retrievedDocs: RetrievedDoc[];
  selectedDevices?: string[] | null;
  selectedDocTypes?: string[] | null;
  searchQueries?: string[] | null;
};

type DetailedFeedbackPayload = {
  messageId: string;
  sessionId?: string;
  turnId?: number;
  accuracy: number;
  completeness: number;
  relevance: number;
  comment?: string;
  reviewerName?: string;
  logs?: string[];
};

type MessageItemProps = {
  message: Message;
  isStreaming?: boolean;
  onFeedback?: (payload: {
    messageId: string;
    sessionId?: string;
    turnId?: number;
    rating: FeedbackRating;
    reason?: string | null;
  }) => void;
  onDetailedFeedback?: (payload: DetailedFeedbackPayload) => void;
  onRegenerate?: (payload: RegeneratePayload) => void;
  originalQuery?: string;  // Original user query for regeneration
};

export function MessageItem({ message, isStreaming, onFeedback, onDetailedFeedback, onRegenerate, originalQuery }: MessageItemProps) {
  const [copied, setCopied] = useState(false);
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [showFilterInfo, setShowFilterInfo] = useState(false);

  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const hasFeedback = Boolean(message.feedback?.accuracy || message.feedback?.rating);
  const canFeedback = Boolean(message.sessionId && message.turnId);
  const regenerateQuery = (originalQuery || message.originalQuery || "").trim();

  // Check if regeneration info is available
  const hasFilterInfo = Boolean(
    message.selectedDevices?.length ||
    message.selectedDocTypes?.length ||
    message.searchQueries?.length ||
    message.autoParse
  );

  const handleRegenerate = () => {
    if (!onRegenerate || !regenerateQuery) return;
    // 재생성용으로 allRetrievedDocs (20개) 사용, 없으면 retrievedDocs fallback
    onRegenerate({
      messageId: message.id,
      originalQuery: regenerateQuery,
      retrievedDocs: message.allRetrievedDocs ?? message.retrievedDocs ?? [],
      selectedDevices: message.selectedDevices,
      selectedDocTypes: message.selectedDocTypes,
      searchQueries: message.searchQueries,
    });
  };

  // 이미지 URL이 유효한지 확인
  const hasValidImageUrl = (url: string | null | undefined): url is string => {
    if (typeof url !== 'string') return false;
    const trimmed = url.trim();
    // 빈 문자열, "null", "undefined" 등 무효한 값 필터링
    if (trimmed.length === 0 || trimmed === 'null' || trimmed === 'undefined') return false;
    return true;
  };

  // 이미지 미리보기용 배열 생성 (실제 이미지 URL이 있는 문서만)
  const previewImages: ImagePreviewItem[] = useMemo(() => {
    if (!message.retrievedDocs) return [];

    const images: ImagePreviewItem[] = [];
    message.retrievedDocs.forEach((doc) => {
      // sop, ts, setup 타입은 {doc_type}_{id} 형식으로 표시
      const docType = (doc.metadata as Record<string, unknown>)?.doc_type as string | undefined;
      const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
      const displayTitle = isSpecialDocType
        ? `${docType}_${doc.id}`
        : (doc.title || undefined);

      // expanded_page_urls가 있으면 사용
      if (doc.expanded_page_urls && doc.expanded_page_urls.length > 0) {
        const pageNumbers = doc.expanded_pages && doc.expanded_pages.length > 0
          ? doc.expanded_pages
          : [];
        doc.expanded_page_urls.forEach((url, idx) => {
          if (hasValidImageUrl(url)) {
            images.push({
              url,
              title: displayTitle,
              page: pageNumbers[idx] || undefined,
              docId: doc.id,
            });
          }
        });
      }
      // page_image_url만 있으면 사용
      else if (hasValidImageUrl(doc.page_image_url)) {
        images.push({
          url: doc.page_image_url,
          title: displayTitle,
          page: doc.page || undefined,
          docId: doc.id,
        });
      }
      // 실제 이미지 URL이 없으면 추가하지 않음 (동적 URL 생성 안 함)
    });
    return images;
  }, [message.retrievedDocs]);

  // 문서 인덱스와 페이지 인덱스에서 전체 미리보기 인덱스로 매핑
  const getPreviewIndex = (docIndex: number, pageIndex: number): number => {
    if (!message.retrievedDocs) return 0;
    let totalIdx = 0;
    for (let i = 0; i < docIndex; i++) {
      const doc = message.retrievedDocs[i];
      // 실제 이미지 URL이 있는 경우만 카운트
      const pageCount = doc.expanded_page_urls && doc.expanded_page_urls.length > 0
        ? doc.expanded_page_urls.filter(url => hasValidImageUrl(url)).length
        : hasValidImageUrl(doc.page_image_url)
          ? 1
          : 0;
      totalIdx += pageCount;
    }
    return totalIdx + pageIndex;
  };

  const handleImageClick = (docIndex: number, pageIndex: number) => {
    const previewIdx = getPreviewIndex(docIndex, pageIndex);
    setPreviewIndex(previewIdx);
    setPreviewVisible(true);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const handleFeedbackClick = () => {
    if (!canFeedback) return;
    setShowFeedbackForm(true);
  };

  const handleFeedbackSubmit = async (data: {
    accuracy: number;
    completeness: number;
    relevance: number;
    comment?: string;
    reviewerName?: string;
  }) => {
    if (!canFeedback || !onDetailedFeedback) return;

    setIsSubmittingFeedback(true);
    try {
      await onDetailedFeedback({
        messageId: message.id,
        sessionId: message.sessionId,
        turnId: message.turnId,
        accuracy: data.accuracy,
        completeness: data.completeness,
        relevance: data.relevance,
        comment: data.comment,
        reviewerName: data.reviewerName,
        logs: message.logs,
      });
      setShowFeedbackForm(false);
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  const handleFeedbackCancel = () => {
    setShowFeedbackForm(false);
  };

  // Calculate average score for display
  const feedbackAvgScore = message.feedback?.avgScore ??
    (message.feedback?.accuracy && message.feedback?.completeness && message.feedback?.relevance
      ? (message.feedback.accuracy + message.feedback.completeness + message.feedback.relevance) / 3
      : null);

  return (
    <div className="message-item">
      <div className={`message-content ${isUser ? "user" : ""}`}>
        <div className={`message-avatar ${isUser ? "user" : "assistant"}`}>
          {isUser ? "U" : "RTM"}
        </div>
        <div className="message-body">
          <div className={`message-bubble ${isUser ? "user" : "assistant"}`}>
            {isAssistant ? (
              <MarkdownContent content={message.content} />
            ) : (
              <div className="message-text">{message.content}</div>
            )}
            {isStreaming && (
              <span className="typing-indicator">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </span>
            )}
          </div>

          {isAssistant && isStreaming && message.currentNode && (
            <div className="node-indicator">
              <span className="node-indicator-dot" aria-hidden="true" />
              <span>처리중: {message.currentNode}</span>
            </div>
          )}

          {/* Retrieved documents (collapsible) */}
          {isAssistant && message.retrievedDocs && message.retrievedDocs.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <Collapse
                size="small"
                items={[
                  {
                    key: "retrieved",
                    label: `확장 문서/참고 문서 (${message.retrievedDocs.length})`,
                    children: (
                      <div className="reference-list">
                        {message.retrievedDocs.map((doc, idx) => {
                          const pageNumbers = doc.expanded_pages && doc.expanded_pages.length > 0
                            ? doc.expanded_pages
                            : doc.page !== null && doc.page !== undefined
                              ? [doc.page]
                              : [];
                          // 실제 이미지 URL만 사용 (동적 URL 생성 안 함)
                          const pageUrls = doc.expanded_page_urls && doc.expanded_page_urls.length > 0
                            ? doc.expanded_page_urls.filter(url => hasValidImageUrl(url))
                            : hasValidImageUrl(doc.page_image_url)
                              ? [doc.page_image_url]
                              : [];

                          const hasImageUrls = pageUrls.length > 0;

                          // sop, ts, setup 타입은 {doc_type}_{id} 형식으로 표시
                          const docType = (doc.metadata as Record<string, unknown>)?.doc_type as string | undefined;
                          const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
                          const displayTitle = isSpecialDocType
                            ? `${docType}_${doc.id}`
                            : (doc.title || `Document ${idx + 1}`);

                          return (
                            <div key={doc.id || idx} className="reference-item" style={{ marginBottom: 16 }}>
                              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                                {displayTitle}
                                {pageNumbers.length > 0 && (
                                  <span style={{ fontWeight: 400, marginLeft: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>
                                    {pageNumbers.length === 1
                                      ? `p.${pageNumbers[0]}`
                                      : `p.${pageNumbers[0]}-${pageNumbers[pageNumbers.length - 1]}`}
                                  </span>
                                )}
                              </div>
                              {/* 이미지와 텍스트 래퍼 */}
                              <div className="reference-content-wrapper" style={{ position: "relative" }}>
                                {/* 이미지가 있으면 표시 */}
                                {hasImageUrls && (
                                  <div
                                    className="reference-images-container"
                                    style={{
                                      display: "flex",
                                      flexWrap: "wrap",
                                      gap: 8,
                                      marginBottom: 8,
                                    }}
                                  >
                                    {pageUrls.map((url, pageIdx) => (
                                      <div key={pageIdx} className="reference-image-wrapper">
                                        <div style={{ position: "relative", display: "inline-block" }}>
                                          <img
                                            src={url}
                                            alt={`${displayTitle} page ${pageNumbers[pageIdx] || pageIdx + 1}`}
                                            style={{
                                              maxWidth: pageUrls.length > 1 ? 150 : "100%",
                                              maxHeight: pageUrls.length > 1 ? 200 : 300,
                                              borderRadius: 4,
                                              border: "1px solid var(--color-border)",
                                              cursor: "pointer",
                                            }}
                                            title={`클릭하여 확대`}
                                            onClick={(e) => {
                                              e.preventDefault();
                                              e.stopPropagation();
                                              handleImageClick(idx, pageIdx);
                                            }}
                                            onLoad={(e) => {
                                              // 이미지 로드 성공 시 텍스트 숨기기
                                              const wrapper = e.currentTarget.closest(".reference-content-wrapper");
                                              const textContent = wrapper?.querySelector(".reference-text-content") as HTMLElement;
                                              if (textContent) textContent.style.display = "none";
                                            }}
                                            onError={(e) => {
                                              // 이미지 로드 실패 시 이미지 숨기기
                                              e.currentTarget.style.display = "none";
                                            }}
                                          />
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                {/* 텍스트 콘텐츠 (항상 렌더링, 이미지 로드 성공 시 숨김) */}
                                {doc.snippet && (
                                  <div
                                    className="reference-text-content"
                                    style={{ fontSize: 12, color: "var(--color-text-secondary)" }}
                                  >
                                    <MarkdownContent content={preprocessSnippet(doc.snippet)} />
                                  </div>
                                )}
                              </div>
                              {(doc.score !== null && doc.score !== undefined) && (
                                <div style={{ fontSize: 12, opacity: 0.6 }}>
                                  score: {doc.score.toFixed(3)} {doc.score_percent ? `(${doc.score_percent}%)` : ""}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    ),
                  },
                ]}
              />
              {onRegenerate && regenerateQuery && (
                <div style={{ marginTop: 8, display: "flex", justifyContent: "flex-end" }}>
                  <button
                    className="action-button regenerate-button"
                    onClick={handleRegenerate}
                    title="답변 재생성"
                    disabled={isStreaming}
                  >
                    <ReloadOutlined />
                    <span style={{ marginLeft: 4, fontSize: 12 }}>재생성</span>
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Execution logs moved to right sidebar */}

          {/* Action buttons - only for assistant messages */}
          {isAssistant && !isStreaming && (
            <div className="message-actions">
              <button
                className={`action-button ${copied ? "active" : ""}`}
                onClick={handleCopy}
                title="Copy"
              >
                {copied ? <CheckOutlined /> : <CopyOutlined />}
              </button>
              {/* 피드백 버튼: 피드백이 없을 때만 표시 */}
              {!hasFeedback && (
                <button
                  className="action-button"
                  onClick={handleFeedbackClick}
                  title="답변 평가"
                  disabled={!canFeedback}
                >
                  <LikeOutlined />
                </button>
              )}
              {hasFilterInfo && (
                <button
                  className={`action-button ${showFilterInfo ? "active" : ""}`}
                  onClick={() => setShowFilterInfo(!showFilterInfo)}
                  title="검색 필터 정보"
                >
                  <FilterOutlined />
                </button>
              )}
            </div>
          )}

          {/* Filter info display */}
          {isAssistant && !isStreaming && showFilterInfo && hasFilterInfo && (
            <div className="filter-info" style={{ marginTop: 8, padding: "8px 12px", background: "var(--color-bg-secondary)", borderRadius: 6, fontSize: 12 }}>
              {message.autoParse?.message && (
                <div style={{ marginBottom: 4, color: "var(--color-accent-primary)" }}>
                  🔍 {message.autoParse.message}
                </div>
              )}
              {message.selectedDevices && message.selectedDevices.length > 0 && (
                <div style={{ marginBottom: 4 }}>
                  <span style={{ color: "var(--color-text-secondary)" }}>장비: </span>
                  {message.selectedDevices.map((d, i) => (
                    <Tag key={i} style={{ marginRight: 4 }}>{d}</Tag>
                  ))}
                </div>
              )}
              {message.selectedDocTypes && message.selectedDocTypes.length > 0 && (
                <div style={{ marginBottom: 4 }}>
                  <span style={{ color: "var(--color-text-secondary)" }}>문서 종류: </span>
                  {message.selectedDocTypes.map((d, i) => (
                    <Tag key={i} style={{ marginRight: 4 }}>{d}</Tag>
                  ))}
                </div>
              )}
              {message.searchQueries && message.searchQueries.length > 0 && (
                <div>
                  <span style={{ color: "var(--color-text-secondary)" }}>검색 쿼리: </span>
                  {message.searchQueries.slice(0, 3).map((q, i) => (
                    <div key={i} style={{ marginLeft: 8, color: "var(--color-text-secondary)" }}>• {q}</div>
                  ))}
                  {message.searchQueries.length > 3 && (
                    <div style={{ marginLeft: 8, color: "var(--color-text-secondary)" }}>...외 {message.searchQueries.length - 3}개</div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Feedback Form - only show when no existing feedback */}
          {isAssistant && showFeedbackForm && !hasFeedback && (
            <FeedbackForm
              onSubmit={handleFeedbackSubmit}
              onCancel={handleFeedbackCancel}
              isSubmitting={isSubmittingFeedback}
              initialValues={{
                accuracy: message.feedback?.accuracy ?? undefined,
                completeness: message.feedback?.completeness ?? undefined,
                relevance: message.feedback?.relevance ?? undefined,
                comment: message.feedback?.comment ?? undefined,
              }}
            />
          )}

          {/* Feedback Summary (read-only) */}
          {isAssistant && hasFeedback && (
            <div
              style={{
                marginTop: 8,
                padding: "8px 12px",
                background: "var(--color-bg-secondary)",
                borderRadius: 6,
                fontSize: 12,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                {message.feedback?.accuracy && (
                  <span>
                    정확성: <strong>{message.feedback.accuracy}</strong>
                  </span>
                )}
                {message.feedback?.completeness && (
                  <span>
                    완성도: <strong>{message.feedback.completeness}</strong>
                  </span>
                )}
                {message.feedback?.relevance && (
                  <span>
                    관련성: <strong>{message.feedback.relevance}</strong>
                  </span>
                )}
                {feedbackAvgScore !== null && (
                  <span style={{
                    marginLeft: "auto",
                    color: feedbackAvgScore >= 3 ? "var(--color-success, #52c41a)" : "var(--color-danger, #ff4d4f)",
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                  }}>
                    {feedbackAvgScore >= 3 ? <LikeFilled /> : <DislikeFilled />}
                    <strong>{feedbackAvgScore.toFixed(1)}</strong>/5
                  </span>
                )}
              </div>
              {message.feedback?.comment && (
                <div style={{ marginTop: 4, color: "var(--color-text-secondary)" }}>
                  의견: {message.feedback.comment}
                </div>
              )}
            </div>
          )}

          {/* Raw answer block (optional) */}
          {message.rawAnswer && (
            <details style={{ marginTop: 12 }}>
              <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--color-text-secondary)" }}>
                Show raw response
              </summary>
              <div className="raw-block">{message.rawAnswer}</div>
            </details>
          )}
        </div>
      </div>

      {/* 이미지 미리보기 모달 (조회 전용) */}
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
