import { useState, useMemo, useCallback } from "react";
import { CopyOutlined, CheckOutlined, ReloadOutlined, FilterOutlined, LikeOutlined, LikeFilled, DislikeOutlined, DislikeFilled, EditOutlined } from "@ant-design/icons";
import { Message, FeedbackRating, RetrievedDoc, MessageFeedback } from "../types";
import { MarkdownContent } from "./markdown-content";
import { Button, Collapse, Tag } from "antd";
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

function containsKorean(text: string): boolean {
  return /[가-힣]/.test(text || "");
}

function extractSearchQueriesFromRawAnswer(rawAnswer?: string): string[] {
  if (!rawAnswer) return [];
  try {
    const parsed = JSON.parse(rawAnswer) as {
      search_queries?: unknown;
      metadata?: { search_queries?: unknown };
    };
    const fromTopLevel = Array.isArray(parsed?.search_queries) ? parsed.search_queries : [];
    const fromMetadata = Array.isArray(parsed?.metadata?.search_queries) ? parsed.metadata.search_queries : [];
    const merged = [...fromTopLevel, ...fromMetadata];
    const cleaned = merged
      .map((value) => (typeof value === "string" ? value.trim() : ""))
      .filter((value) => value.length > 0);
    return Array.from(new Set(cleaned));
  } catch {
    return [];
  }
}

type ParsedRawAnswer = {
  query?: unknown;
  suggest_additional_device_search?: unknown;
  metadata?: {
    selected_task_mode?: unknown;
    applied_doc_type_scope?: unknown;
  } | null;
  expanded_docs?: Array<{
    rank?: unknown;
    doc_id?: unknown;
  }>;
  auto_parse?: {
    device?: unknown;
    devices?: unknown;
  } | null;
};

function extractExpandedDocIdsFromRawAnswer(rawAnswer?: string): string[] {
  if (!rawAnswer) return [];
  try {
    const parsed = JSON.parse(rawAnswer) as ParsedRawAnswer;
    const expanded = Array.isArray(parsed.expanded_docs) ? parsed.expanded_docs : [];
    return expanded
      .map((item) => (typeof item?.doc_id === "string" ? item.doc_id.trim() : ""))
      .filter((docId) => docId.length > 0);
  } catch {
    return [];
  }
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
  onEdit?: (editedText: string) => void;
  issueCases?: Array<{ doc_id: string; title: string; summary: string }>;
  onIssueCaseSelect?: (docId: string) => void;
  showIssueSopButtons?: boolean;
  onIssueSopConfirm?: (confirm: boolean) => void;
  originalQuery?: string;  // Original user query for regeneration
};

export function MessageItem({
  message,
  isStreaming,
  onFeedback,
  onDetailedFeedback,
  onRegenerate,
  onEdit,
  issueCases,
  onIssueCaseSelect,
  showIssueSopButtons,
  onIssueSopConfirm,
  originalQuery,
}: MessageItemProps) {
  const [copied, setCopied] = useState(false);
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [showFilterInfo, setShowFilterInfo] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);

  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const hasFeedback = Boolean(message.feedback?.accuracy || message.feedback?.rating);
  const canFeedback = Boolean(message.sessionId && message.turnId);
  const parsedRawAnswer = useMemo<ParsedRawAnswer | null>(() => {
    if (!message.rawAnswer) return null;
    try {
      const parsed = JSON.parse(message.rawAnswer);
      return typeof parsed === "object" && parsed !== null ? (parsed as ParsedRawAnswer) : null;
    } catch {
      return null;
    }
  }, [message.rawAnswer]);

  const queryFromRawAnswer = useMemo(() => {
    const q = parsedRawAnswer?.query;
    return typeof q === "string" ? q.trim() : "";
  }, [parsedRawAnswer]);

  const shouldSuggestDeviceSearch = useMemo(() => {
    if (typeof message.suggestAdditionalDeviceSearch === "boolean") {
      return message.suggestAdditionalDeviceSearch;
    }
    return parsedRawAnswer?.suggest_additional_device_search === true;
  }, [message.suggestAdditionalDeviceSearch, parsedRawAnswer]);

  const regenerateQuery = (originalQuery || message.originalQuery || queryFromRawAnswer || "").trim();
  const effectiveSearchQueries = useMemo(() => {
    const direct = Array.isArray(message.searchQueries)
      ? message.searchQueries.map((q) => (typeof q === "string" ? q.trim() : "")).filter((q) => q.length > 0)
      : [];
    if (direct.length > 0) return direct;
    return extractSearchQueriesFromRawAnswer(message.rawAnswer);
  }, [message.searchQueries, message.rawAnswer]);
  const englishSearchQueries = useMemo(
    () => effectiveSearchQueries.filter((q) => !containsKorean(q)).slice(0, 3),
    [effectiveSearchQueries]
  );
  const koreanSearchQueries = useMemo(
    () => effectiveSearchQueries.filter((q) => containsKorean(q)).slice(0, 3),
    [effectiveSearchQueries]
  );

  const isSopTaskContext = useMemo(() => {
    const selectedDocTypes = Array.isArray(message.selectedDocTypes)
      ? message.selectedDocTypes
      : [];
    const hasSopSelection = selectedDocTypes.some(
      (docType) => typeof docType === "string" && docType.trim().toLowerCase() === "sop"
    );

    const rawTaskMode = parsedRawAnswer?.metadata?.selected_task_mode;
    const normalizedRawTaskMode = typeof rawTaskMode === "string" ? rawTaskMode.trim().toLowerCase() : "";
    return hasSopSelection || normalizedRawTaskMode === "sop";
  }, [message.selectedDocTypes, parsedRawAnswer]);

  const answerSourceDocIds = useMemo(() => {
    const direct = Array.isArray(message.expandedDocs)
      ? message.expandedDocs
          .map((doc) => (typeof doc?.doc_id === "string" ? doc.doc_id.trim() : ""))
          .filter((docId) => docId.length > 0)
      : [];
    if (direct.length > 0) return direct;
    return extractExpandedDocIdsFromRawAnswer(message.rawAnswer);
  }, [message.expandedDocs, message.rawAnswer]);

  const sopPresentation = useMemo(() => {
    const fallbackDocs = message.retrievedDocs ?? [];
    if (!isAssistant || fallbackDocs.length === 0 || !isSopTaskContext) {
      return {
        enabled: false,
        docs: fallbackDocs,
        flowChartUrl: null as string | null,
      };
    }

    const normalizeDocType = (doc: RetrievedDoc): string => {
      const value = (doc.metadata as Record<string, unknown> | null | undefined)?.doc_type;
      return typeof value === "string" ? value.trim().toLowerCase() : "";
    };

    const finalSourceDocId = answerSourceDocIds.length > 0
      ? answerSourceDocIds[answerSourceDocIds.length - 1]
      : null;

    if (!finalSourceDocId) {
      return {
        enabled: false,
        docs: fallbackDocs,
        flowChartUrl: null as string | null,
      };
    }

    const fullDocCandidates = (message.allRetrievedDocs ?? fallbackDocs).filter(
      (doc) => doc.id === finalSourceDocId
    );
    const sameDocChunks = fullDocCandidates.length > 0
      ? fullDocCandidates
      : fallbackDocs.filter((doc) => doc.id === finalSourceDocId);

    const finalSource = sameDocChunks[0] ?? null;
    if (!finalSource || normalizeDocType(finalSource) !== "sop") {
      return {
        enabled: false,
        docs: fallbackDocs,
        flowChartUrl: null as string | null,
      };
    }

    const mergedPages: number[] = [];
    const mergedPageUrls: string[] = [];
    for (const chunk of sameDocChunks) {
      const pages = chunk.expanded_pages && chunk.expanded_pages.length > 0
        ? chunk.expanded_pages
        : chunk.page !== null && chunk.page !== undefined
          ? [chunk.page]
          : [];
      const urls = chunk.expanded_page_urls && chunk.expanded_page_urls.length > 0
        ? chunk.expanded_page_urls.filter((url) => hasValidImageUrl(url))
        : hasValidImageUrl(chunk.page_image_url)
          ? [chunk.page_image_url]
          : [];

      urls.forEach((url, idx) => {
        if (mergedPageUrls.includes(url)) return;
        mergedPageUrls.push(url);
        if (pages[idx] !== undefined) {
          mergedPages.push(pages[idx]);
        }
      });
    }

    const flowChartChunk = sameDocChunks.find((chunk) => {
      const meta = (chunk.metadata as Record<string, unknown> | null | undefined) ?? {};
      const chapter = `${String(meta.section_chapter ?? "")} ${String(meta.chapter ?? "")} ${String(meta.section ?? "")}`.toLowerCase();
      return chapter.includes("flow chart");
    });
    const flowChartUrl = flowChartChunk
      ? (
          flowChartChunk.expanded_page_urls?.find((url) => hasValidImageUrl(url))
          ?? (hasValidImageUrl(flowChartChunk.page_image_url) ? flowChartChunk.page_image_url : null)
        )
      : null;

    const fullDocEntry: RetrievedDoc = {
      ...finalSource,
      expanded_pages: mergedPages.length > 0 ? mergedPages : finalSource.expanded_pages,
      expanded_page_urls: mergedPageUrls.length > 0 ? mergedPageUrls : finalSource.expanded_page_urls,
      page: mergedPages.length === 1 ? mergedPages[0] : finalSource.page,
      page_image_url: mergedPageUrls.length > 0 ? mergedPageUrls[0] : finalSource.page_image_url,
    };

    return {
      enabled: true,
      docs: [fullDocEntry],
      flowChartUrl,
    };
  }, [isAssistant, message.retrievedDocs, message.allRetrievedDocs, isSopTaskContext, answerSourceDocIds]);

  const referenceDocs = sopPresentation.docs;
  const referenceLabel = sopPresentation.enabled
    ? "답변에 사용된 문서 (1)"
    : `확장 문서/참고 문서 (${referenceDocs.length})`;

  // Check if regeneration info is available
  const hasFilterInfo = Boolean(
    message.selectedDevices?.length ||
    message.selectedDocTypes?.length ||
    effectiveSearchQueries.length ||
    message.autoParse ||
    shouldSuggestDeviceSearch
  );

  const handleRegenerate = () => {
    if (!onRegenerate || !regenerateQuery) return;
    // 재생성용으로 allRetrievedDocs(최대 retrieval_top_k개) 사용, 없으면 retrievedDocs fallback
    onRegenerate({
      messageId: message.id,
      originalQuery: regenerateQuery,
      retrievedDocs: message.allRetrievedDocs ?? message.retrievedDocs ?? [],
      selectedDevices: message.selectedDevices,
      selectedDocTypes: message.selectedDocTypes,
      searchQueries: effectiveSearchQueries.length > 0 ? effectiveSearchQueries : null,
    });
  };

  // 이미지 미리보기용 배열 생성 (실제 이미지 URL이 있는 문서만)
  const previewImages: ImagePreviewItem[] = useMemo(() => {
    if (referenceDocs.length === 0) return [];

    const images: ImagePreviewItem[] = [];
    referenceDocs.forEach((doc) => {
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
  }, [referenceDocs]);

  // 문서 인덱스와 페이지 인덱스에서 전체 미리보기 인덱스로 매핑
  const getPreviewIndex = (docIndex: number, pageIndex: number): number => {
    if (referenceDocs.length === 0) return 0;
    let totalIdx = 0;
    for (let i = 0; i < docIndex; i++) {
      const doc = referenceDocs[i];
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
            {isAssistant && message.autoParse?.message && (
              <div className="message-parse-label">
                {message.autoParse.message}
              </div>
            )}
            {isAssistant ? (
              <>
                {sopPresentation.flowChartUrl && (
                  <div style={{ marginBottom: 12 }}>
                    <img
                      src={sopPresentation.flowChartUrl}
                      alt="SOP flow chart"
                      style={{
                        maxWidth: "100%",
                        maxHeight: 280,
                        borderRadius: 6,
                        border: "1px solid var(--color-border)",
                      }}
                    />
                  </div>
                )}
                <MarkdownContent content={message.content} />

                {Array.isArray(issueCases) && issueCases.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <div style={{ marginBottom: 8, fontSize: 13, color: "var(--color-text-secondary)" }}>
                      자세히 검색하고 싶은 문서를 선택하세요
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {issueCases.map((item, idx) => (
                        <Button
                          key={item.doc_id}
                          type="default"
                          size="middle"
                          style={{
                            textAlign: "left",
                            whiteSpace: "normal",
                            height: "auto",
                            padding: "8px 12px",
                            borderColor: "var(--color-accent-primary, #1677ff)",
                            color: "var(--color-accent-primary, #1677ff)",
                          }}
                          onClick={() => onIssueCaseSelect?.(item.doc_id)}
                        >
                          {idx + 1}. {item.title}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                {showIssueSopButtons && (
                  <div style={{ marginTop: 16 }}>
                    <div style={{ marginBottom: 8, fontSize: 13, color: "var(--color-text-secondary)" }}>
                      추가로 조회할 항목을 선택하세요
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <Button
                        type="primary"
                        size="small"
                        onClick={() => onIssueSopConfirm?.(true)}
                      >
                        관련 SOP 조회
                      </Button>
                      <Button
                        type="default"
                        size="small"
                        style={{
                          borderColor: "var(--color-accent-primary, #1677ff)",
                          color: "var(--color-accent-primary, #1677ff)",
                        }}
                        onClick={() => onIssueSopConfirm?.(false)}
                      >
                        다른 이슈 문서 선택
                      </Button>
                    </div>
                  </div>
                )}
              </>
            ) : isEditing ? (
              <div className="message-edit-area">
                <textarea
                  className="message-edit-input"
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      const trimmed = editText.trim();
                      if (trimmed && onEdit) {
                        onEdit(trimmed);
                        setIsEditing(false);
                      }
                    }
                    if (e.key === "Escape") {
                      setEditText(message.content);
                      setIsEditing(false);
                    }
                  }}
                  autoFocus
                  rows={Math.min(editText.split("\n").length + 1, 6)}
                />
                <div className="message-edit-actions">
                  <button
                    className="message-edit-cancel"
                    onClick={() => { setEditText(message.content); setIsEditing(false); }}
                  >
                    취소
                  </button>
                  <button
                    className="message-edit-submit"
                    onClick={() => {
                      const trimmed = editText.trim();
                      if (trimmed && onEdit) {
                        onEdit(trimmed);
                        setIsEditing(false);
                      }
                    }}
                    disabled={!editText.trim()}
                  >
                    제출
                  </button>
                </div>
              </div>
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

          {/* Edit button for user messages */}
          {isUser && !isStreaming && !isEditing && onEdit && (
            <div className="message-actions">
              <button
                className="action-button"
                onClick={() => { setEditText(message.content); setIsEditing(true); }}
                title="메시지 수정"
              >
                <EditOutlined />
              </button>
            </div>
          )}

          {isAssistant && isStreaming && message.currentNode && (
            <div className="node-indicator">
              <span className="node-indicator-dot" aria-hidden="true" />
              <span>처리중: {message.currentNode}</span>
            </div>
          )}

          {/* Retrieved documents (collapsible) - hidden by design */}
          {false && isAssistant && referenceDocs.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <Collapse
                size="small"
                items={[
                  {
                    key: "retrieved",
                    label: referenceLabel,
                    children: (
                      <div className="reference-list">
                        {referenceDocs.map((doc, idx) => {
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

                          const meta = (doc.metadata || {}) as Record<string, unknown>;
                          const chunkSummary = meta.chunk_summary as string | undefined;
                          const chunkKeywords = meta.chunk_keywords as string[] | undefined;
                          const metaChapter = meta.chapter as string | undefined;
                          const metaDocType = meta.doc_type as string | undefined;
                          const metaDeviceName = meta.device_name as string | undefined;

                          return (
                            <div key={doc.id || idx} className="reference-item" style={{ marginBottom: 16 }}>
                              {/* Title + Score (search page 동일) */}
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                                <span style={{ fontWeight: 600 }}>
                                  {idx + 1}. {displayTitle}
                                  {pageNumbers.length > 0 && (
                                    <span style={{ fontWeight: 400, marginLeft: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>
                                      {pageNumbers.length === 1
                                        ? `p.${pageNumbers[0]}`
                                        : `p.${pageNumbers[0]}-${pageNumbers[pageNumbers.length - 1]}`}
                                    </span>
                                  )}
                                </span>
                                {(doc.score !== null && doc.score !== undefined) && (
                                  <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
                                    스코어: {doc.score.toFixed(3)} {doc.score_percent ? `(${doc.score_percent}%)` : ""}
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
                              {/* 청크 요약 (search page 동일 - 파란 박스) */}
                              {chunkSummary && (
                                <div style={{ marginTop: 8, padding: 8, background: "#f0f7ff", borderRadius: 4 }}>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: "#1890ff" }}>청크 요약: </span>
                                  <span style={{ fontSize: 12 }}>{chunkSummary}</span>
                                </div>
                              )}
                              {/* 키워드 (search page 동일 - 녹색 태그) */}
                              {chunkKeywords && chunkKeywords.length > 0 && (
                                <div style={{ marginTop: 8 }}>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: "#52c41a" }}>키워드: </span>
                                  {chunkKeywords.map((kw, kwIdx) => (
                                    <span
                                      key={kwIdx}
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
                              {/* 메타데이터 행 (search page 동일 - ID | Chapter | Doc Type | Device Name) */}
                              <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center", fontSize: 12, color: "var(--color-text-secondary)" }}>
                                <span>ID: {doc.id}</span>
                                {metaChapter && (<><span style={{ margin: "0 4px" }}>|</span><span>챕터: {metaChapter}</span></>)}
                                {metaDocType && (<><span style={{ margin: "0 4px" }}>|</span><span>타입: {metaDocType}</span></>)}
                                {metaDeviceName && (<><span style={{ margin: "0 4px" }}>|</span><span>장비: {metaDeviceName}</span></>)}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ),
                  },
                ]}
              />
              {onRegenerate && regenerateQuery && !shouldSuggestDeviceSearch && (
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
              {effectiveSearchQueries.length > 0 && (
                <div>
                  <span style={{ color: "var(--color-text-secondary)" }}>검색 쿼리(MQ): </span>
                  {englishSearchQueries.length > 0 && (
                    <div style={{ marginTop: 4 }}>
                      <span style={{ color: "var(--color-text-secondary)" }}>EN</span>
                      {englishSearchQueries.map((q, i) => (
                        <div key={`en-${i}`} style={{ marginLeft: 8, color: "var(--color-text-secondary)" }}>• {q}</div>
                      ))}
                    </div>
                  )}
                  {koreanSearchQueries.length > 0 && (
                    <div style={{ marginTop: 4 }}>
                      <span style={{ color: "var(--color-text-secondary)" }}>KO</span>
                      {koreanSearchQueries.map((q, i) => (
                        <div key={`ko-${i}`} style={{ marginLeft: 8, color: "var(--color-text-secondary)" }}>• {q}</div>
                      ))}
                    </div>
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
