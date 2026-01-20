import { useState, useCallback, useMemo, useRef, useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { MenuOutlined, MenuUnfoldOutlined, FileTextOutlined } from "@ant-design/icons";
import LeftSidebar from "./left-sidebar";
import MainContent from "./main-content";
import RightSidebar from "./right-sidebar";
import { EmptyState } from "../empty-state";
import { GlobalSearch } from "../global-search";
import { MarkdownContent } from "../../features/chat/components/markdown-content";
import { useChatLogs } from "../../features/chat/context/chat-logs-context";
import { useChatReview } from "../../features/chat/context/chat-review-context";
import "./layout.css";

export default function Layout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isRightSidebarCollapsed, setIsRightSidebarCollapsed] = useState(false);
  const location = useLocation();
  const isChatPage = location.pathname === "/";

  // Get logs from context
  const { logs } = useChatLogs();

  // Get review data from context
  const {
    pendingReview,
    completedRetrievedDocs,
    selectedRanks,
    editableQueries,
    isEditingQueries,
    isStreaming,
    setSelectedRanks,
    setEditableQueries,
    setIsEditingQueries,
    submitReview,
    submitSearchQueries,
  } = useChatReview();

  // Show right sidebar when there are logs, pending review, or completed retrieved docs
  const shouldShowRightSidebar = isChatPage && (
    logs.length > 0 ||
    pendingReview !== null ||
    (completedRetrievedDocs !== null && completedRetrievedDocs.length > 0)
  );

  // Debug logging
  console.log("[Layout] pendingReview:", pendingReview, "shouldShowRightSidebar:", shouldShowRightSidebar);

  const handleOpenSidebar = useCallback(() => {
    setIsSidebarOpen(true);
  }, []);

  const handleCloseSidebar = useCallback(() => {
    setIsSidebarOpen(false);
  }, []);

  const handleToggleCollapse = useCallback(() => {
    setIsSidebarCollapsed((prev) => !prev);
  }, []);

  const handleToggleRightSidebar = useCallback(() => {
    setIsRightSidebarCollapsed((prev) => !prev);
  }, []);

  // Determine right sidebar title and content based on state
  const rightSidebarContent = useMemo(() => {
    if (pendingReview) {
      return {
        title: "검색 결과 확인",
        subtitle: pendingReview.instruction,
      };
    }
    if (completedRetrievedDocs && completedRetrievedDocs.length > 0) {
      return {
        title: "검색된 문서",
        subtitle: `${completedRetrievedDocs.length}개 문서`,
      };
    }
    return {
      title: "실행 로그",
      subtitle: `${logs.length}개 항목`,
    };
  }, [pendingReview, completedRetrievedDocs, logs.length]);

  return (
    <div
      className={`gpt-layout ${isSidebarCollapsed ? "sidebar-collapsed" : ""}`}
    >
      {/* Mobile menu toggle */}
      <button
        className="mobile-menu-toggle"
        onClick={handleOpenSidebar}
        aria-label="Open menu"
      >
        <MenuOutlined />
      </button>

      {/* Desktop menu toggle (when sidebar is collapsed) */}
      <button
        className="desktop-menu-toggle"
        onClick={handleToggleCollapse}
        aria-label="Expand sidebar"
      >
        <MenuUnfoldOutlined />
      </button>

      {/* Mobile backdrop */}
      <div
        className={`sidebar-backdrop ${isSidebarOpen ? "visible" : ""}`}
        onClick={handleCloseSidebar}
      />

      {/* Left Sidebar */}
      <LeftSidebar
        isOpen={isSidebarOpen}
        onClose={handleCloseSidebar}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={handleToggleCollapse}
      />

      {/* Main Content */}
      <MainContent isFullWidth={location.pathname === "/retrieval-test"}>
        <Outlet />
      </MainContent>

      {/* Right Sidebar - only show when conversation has started */}
      {shouldShowRightSidebar && (
        <RightSidebar
          isOpen={true}
          isCollapsed={isRightSidebarCollapsed}
          onClose={() => {}}
          onToggleCollapse={handleToggleRightSidebar}
          title={rightSidebarContent.title}
          subtitle={rightSidebarContent.subtitle}
        >
          {pendingReview ? (
            <ReviewPanelContent
              pendingReview={pendingReview}
              selectedRanks={selectedRanks}
              editableQueries={editableQueries}
              isEditingQueries={isEditingQueries}
              isStreaming={isStreaming}
              setSelectedRanks={setSelectedRanks}
              setEditableQueries={setEditableQueries}
              setIsEditingQueries={setIsEditingQueries}
              submitReview={submitReview}
              submitSearchQueries={submitSearchQueries}
            />
          ) : completedRetrievedDocs && completedRetrievedDocs.length > 0 ? (
            <>
              {console.log("[RightSidebar] completedRetrievedDocs:", completedRetrievedDocs)}
              <RetrievedDocsContent docs={completedRetrievedDocs} />
            </>
          ) : (
            <ChatLogsContent logs={logs} />
          )}
        </RightSidebar>
      )}

      {/* Right Sidebar Expand Button (when collapsed) */}
      {shouldShowRightSidebar && isRightSidebarCollapsed && (
        <button
          className="right-sidebar-toggle"
          onClick={handleToggleRightSidebar}
          aria-label="Expand right sidebar"
          title={rightSidebarContent.title}
        >
          <MenuUnfoldOutlined style={{ transform: "scaleX(-1)" }} />
        </button>
      )}

      {/* Global Search Modal */}
      <GlobalSearch />
    </div>
  );
}

// Review Panel Component
function ReviewPanelContent({
  pendingReview,
  selectedRanks,
  editableQueries,
  isEditingQueries,
  isStreaming,
  setSelectedRanks,
  setEditableQueries,
  setIsEditingQueries,
  submitReview,
  submitSearchQueries,
}: {
  pendingReview: {
    threadId: string;
    question: string;
    instruction: string;
    docs: Array<{
      docId: string;
      rank: number;
      content: string;
      title?: string | null;
      page?: number | null;
      page_image_url?: string | null;
      score?: number | null;
      metadata?: Record<string, unknown> | null;
    }>;
    searchQueries: string[];
  };
  selectedRanks: number[];
  editableQueries: string[];
  isEditingQueries: boolean;
  isStreaming: boolean;
  setSelectedRanks: (value: number[] | ((prev: number[]) => number[])) => void;
  setEditableQueries: (value: string[] | ((prev: string[]) => string[])) => void;
  setIsEditingQueries: (value: boolean | ((prev: boolean) => boolean)) => void;
  submitReview: (selection: { docIds: string[]; ranks: number[] }) => void;
  submitSearchQueries: (queries: string[]) => void;
}) {
  const allSelected =
    pendingReview.docs.length > 0 &&
    selectedRanks.length === pendingReview.docs.length;

  const toggleDoc = (rank: number) => {
    setSelectedRanks((prev: number[]) =>
      prev.includes(rank) ? prev.filter((id: number) => id !== rank) : [...prev, rank]
    );
  };

  const toggleAll = () => {
    if (allSelected) {
      setSelectedRanks([]);
    } else {
      setSelectedRanks(pendingReview.docs.map((doc) => doc.rank));
    }
  };

  const handleReviewSubmit = () => {
    const selectedDocIds = pendingReview.docs
      .filter((doc) => selectedRanks.includes(doc.rank))
      .map((doc) => doc.docId)
      .filter(Boolean);
    submitReview({ docIds: selectedDocIds, ranks: selectedRanks });
  };

  const handleQueryChange = (index: number, value: string) => {
    setEditableQueries((prev: string[]) => {
      const updated = [...prev];
      updated[index] = value;
      return updated;
    });
  };

  const handleAddQuery = () => {
    setEditableQueries((prev: string[]) => [...prev, ""]);
  };

  const handleRemoveQuery = (index: number) => {
    setEditableQueries((prev: string[]) => prev.filter((_: string, i: number) => i !== index));
  };

  const handleSearchWithQueries = () => {
    submitSearchQueries(editableQueries);
  };

  const toggleEditMode = () => {
    setIsEditingQueries((prev: boolean) => !prev);
  };

  console.log("[ReviewPanel] isEditingQueries:", isEditingQueries, "docs:", pendingReview.docs.length, "selectedRanks:", selectedRanks.length);
  console.log("[ReviewPanel] First doc:", pendingReview.docs[0]);
  console.log("[ReviewPanel] First doc page_image_url:", pendingReview.docs[0]?.page_image_url);

  return (
    <div className="review-panel-sidebar">
      {/* Search Query Editor Section */}
      <div className="review-queries">
        <div className="review-queries-header">
          <span className="review-queries-label">검색어</span>
          <button
            className="action-button"
            onClick={toggleEditMode}
            disabled={isStreaming}
            style={{ fontSize: "var(--font-size-xs)", padding: "4px 12px" }}
          >
            {isEditingQueries ? "편집 완료" : "검색어 수정"}
          </button>
        </div>

        {isEditingQueries ? (
          <div className="review-queries-editor">
            {editableQueries.map((query, idx) => (
              <div key={idx} className="review-query-input-row">
                <input
                  type="text"
                  className="review-query-input"
                  value={query}
                  onChange={(e) => handleQueryChange(idx, e.target.value)}
                  placeholder="검색어 입력"
                />
                {editableQueries.length > 1 && (
                  <button
                    className="review-query-remove"
                    onClick={() => handleRemoveQuery(idx)}
                    title="삭제"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
            {editableQueries.length < 5 && (
              <button className="review-query-add" onClick={handleAddQuery}>
                + 검색어 추가
              </button>
            )}
          </div>
        ) : (
          <div className="review-queries-display">
            {editableQueries.map((query, idx) => (
              <span key={idx} className="review-query-tag">
                {query}
              </span>
            ))}
          </div>
        )}
      </div>

      {pendingReview.docs.length === 0 ? (
        <div className="review-empty">
          검색 결과가 없습니다. 키워드를 입력해 재검색하세요.
        </div>
      ) : (
        <>
          <div className="review-controls">
            <label className="review-select-all">
              <input type="checkbox" checked={allSelected} onChange={toggleAll} />
              전체 선택
            </label>
            <span className="review-count">
              {selectedRanks.length}/{pendingReview.docs.length} 선택
            </span>
          </div>
          <div className="review-docs">
            {pendingReview.docs.map((doc, idx) => (
              <label key={`${doc.rank}-${doc.docId}`} className="review-doc">
                <input
                  type="checkbox"
                  checked={selectedRanks.includes(doc.rank)}
                  onChange={() => toggleDoc(doc.rank)}
                />
                <div className="review-doc-body">
                  <div className="review-doc-title">
                    {doc.title || `문서 ${doc.rank ?? idx + 1}`}
                    {doc.page && <span style={{ fontWeight: 400, marginLeft: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>p.{doc.page}</span>}
                  </div>
                  {doc.page_image_url ? (
                    <div className="review-doc-image-wrapper" style={{ marginBottom: 8 }}>
                      <img
                        src={doc.page_image_url}
                        alt={`${doc.title || "Document"} page ${doc.page || ""}`}
                        style={{
                          maxWidth: "100%",
                          maxHeight: 300,
                          borderRadius: 4,
                          border: "1px solid var(--color-border)",
                        }}
                        onError={(e) => {
                          e.currentTarget.style.display = "none";
                          const fallback = e.currentTarget.nextElementSibling as HTMLElement;
                          if (fallback) fallback.style.display = "block";
                        }}
                      />
                      <div className="review-doc-content" style={{ display: "none" }}>
                        <MarkdownContent content={preprocessSnippet(doc.content)} />
                      </div>
                    </div>
                  ) : (
                    <div className="review-doc-content">
                      <MarkdownContent content={preprocessSnippet(doc.content)} />
                    </div>
                  )}
                </div>
              </label>
            ))}
          </div>
        </>
      )}

      <div className="review-actions">
        {isEditingQueries ? (
          <button
            className="action-button"
            onClick={handleSearchWithQueries}
            disabled={isStreaming || editableQueries.every((q) => !q.trim())}
          >
            재검색
          </button>
        ) : (
          <>
            <button
              className="action-button"
              onClick={handleSearchWithQueries}
              disabled={isStreaming}
            >
              재검색
            </button>
            <button
              className="action-button"
              onClick={handleReviewSubmit}
              disabled={
                isStreaming || pendingReview.docs.length === 0 || selectedRanks.length === 0
              }
            >
              선택 문서로 답변
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// Format markdown tables that are on a single line
function formatMarkdownTable(text: string): string {
  // Check if this looks like a table (has pipe characters and alignment pattern)
  if (!text.includes("|")) return text;

  // Detect alignment row pattern (| :--- | or | --- |)
  const hasAlignmentRow = /\|\s*:?-{2,}:?\s*\|/.test(text);
  if (!hasAlignmentRow) return text;

  // If already has proper newlines, return as is
  if (text.includes("\n|")) return text;

  // Pattern: " | | " indicates row boundary (end of one row, start of next)
  // Replace " | | " with " |\n| " to create proper rows
  let formatted = text.replace(/\s*\|\s*\|\s*/g, " |\n| ");

  return formatted;
}

// Preprocess snippet for better markdown rendering
function preprocessSnippet(snippet: string): string {
  if (!snippet) return "";

  let processed = snippet;

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

  // Try to format markdown tables that are on single line
  processed = formatMarkdownTable(processed);

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

// Retrieved Docs Component
function RetrievedDocsContent({ docs }: { docs: Array<{
  id: string;
  title: string;
  snippet: string;
  score?: number | null;
  score_percent?: number | null;
  metadata?: Record<string, unknown> | null;
  page?: number | null;
  page_image_url?: string | null;
}> }) {
  return (
    <div className="retrieved-docs-container">
      {docs.map((doc, index) => (
        <div key={doc.id || index} className="retrieved-doc-item">
          <div className="retrieved-doc-header">
            <span className="retrieved-doc-rank">#{index + 1}</span>
            {doc.title && (
              <span className="retrieved-doc-title">{doc.title}</span>
            )}
            {doc.page && (
              <span className="retrieved-doc-page">p.{doc.page}</span>
            )}
            {doc.score !== null && doc.score !== undefined && (
              <span className="retrieved-doc-score">
                {typeof doc.score_percent === "number"
                  ? `${doc.score_percent.toFixed(1)}%`
                  : doc.score.toFixed(3)}
              </span>
            )}
          </div>
          {doc.page_image_url ? (
            <div className="retrieved-doc-image-wrapper">
              <img
                src={doc.page_image_url}
                alt={`${doc.title || "Document"} page ${doc.page || ""}`}
                className="retrieved-doc-image"
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                  const fallback = e.currentTarget.nextElementSibling as HTMLElement;
                  if (fallback) fallback.style.display = "block";
                }}
              />
              <div className="retrieved-doc-snippet" style={{ display: "none" }}>
                <MarkdownContent content={preprocessSnippet(doc.snippet)} />
              </div>
            </div>
          ) : (
            <div className="retrieved-doc-snippet">
              <MarkdownContent content={preprocessSnippet(doc.snippet)} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// Chat Logs Component
function ChatLogsContent({
  logs,
}: {
  logs: Array<{
    id: string;
    messageId: string;
    timestamp: number;
    content: string;
    node?: string | null;
  }>;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs are added
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs.length]);

  if (logs.length === 0) {
    return (
      <EmptyState
        icon={<FileTextOutlined />}
        title="로그가 없습니다"
        description="대화를 시작하면 실행 로그가 표시됩니다"
        size="medium"
      />
    );
  }

  return (
    <div className="chat-logs-container" ref={containerRef}>
      {logs.map((log) => (
        <div key={log.id} className="chat-log-entry">
          {log.node && <div className="chat-log-node">{log.node}</div>}
          <pre className="chat-log-content">{log.content}</pre>
          <div className="chat-log-timestamp">
            {new Date(log.timestamp).toLocaleTimeString()}
          </div>
        </div>
      ))}
    </div>
  );
}
