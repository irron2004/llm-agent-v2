import { useState, useCallback, useMemo, useRef, useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { MenuOutlined, MenuUnfoldOutlined, FileTextOutlined, ZoomInOutlined } from "@ant-design/icons";
import LeftSidebar from "./left-sidebar";
import MainContent from "./main-content";
import RightSidebar from "./right-sidebar";
import { EmptyState } from "../empty-state";
import { GlobalSearch } from "../global-search";
import { MarkdownContent } from "../../features/chat/components/markdown-content";
import { useChatLogs } from "../../features/chat/context/chat-logs-context";
import { useChatReview } from "../../features/chat/context/chat-review-context";
import { fetchDeviceCatalog } from "../../features/chat/api";
import type { DeviceInfo, DocTypeInfo, RetrievedDoc } from "../../features/chat/types";
import { ImagePreviewModal, ImagePreviewItem } from "../image-preview-modal";
import "./layout.css";

export default function Layout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isRightSidebarCollapsed, setIsRightSidebarCollapsed] = useState(false);
  const location = useLocation();
  const isChatPage = location.pathname === "/";

  // Get logs from context
  const { logs, activeMessageId, clearLogs } = useChatLogs();

  const visibleLogs = useMemo(() => {
    if (!activeMessageId) return logs;
    return logs.filter((log) => log.messageId === activeMessageId);
  }, [logs, activeMessageId]);

  // Get review data from context
  const {
    pendingReview,
    pendingRegeneration,
    completedRetrievedDocs,
    selectedRanks,
    editableQueries,
    isEditingQueries,
    isStreaming,
    setSelectedRanks,
    setEditableQueries,
    setIsEditingQueries,
    setPendingReview,
    setCompletedRetrievedDocs,
    submitReview,
    submitSearchQueries,
    submitRegeneration,
    setPendingRegeneration,
    setIsStreaming,
  } = useChatReview();

  // Show right sidebar when there are logs, pending review, or completed retrieved docs
  const shouldShowRightSidebar = isChatPage && (
    visibleLogs.length > 0 ||
    pendingReview !== null ||
    pendingRegeneration !== null ||
    (completedRetrievedDocs !== null && completedRetrievedDocs.length > 0)
  );

  // Auto-expand right sidebar when regeneration or review panel is triggered
  useEffect(() => {
    if (pendingRegeneration !== null || pendingReview !== null) {
      setIsRightSidebarCollapsed(false);
    }
  }, [pendingRegeneration, pendingReview]);

  // Reset right sidebar state and content when starting a new chat
  useEffect(() => {
    const handleNewChat = () => {
      setIsRightSidebarCollapsed(false);
      clearLogs();
      setPendingReview(null);
      setPendingRegeneration(null);
      setCompletedRetrievedDocs(null);
      setIsStreaming(false);
    };
    window.addEventListener("pe-agent:new-chat", handleNewChat);
    return () => {
      window.removeEventListener("pe-agent:new-chat", handleNewChat);
    };
  }, [
    clearLogs,
    setPendingReview,
    setPendingRegeneration,
    setCompletedRetrievedDocs,
    setIsStreaming,
  ]);

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
  // Priority: streaming (show logs) > regeneration > review > docs > logs
  const rightSidebarContent = useMemo(() => {
    // 새 요청 스트리밍 중이면 로그 표시
    if (isStreaming && visibleLogs.length > 0) {
      return {
        title: "Activity Log",
        subtitle: `${visibleLogs.length} items`,
      };
    }
    if (pendingRegeneration) {
      return {
        title: "Regenerate Answer",
        subtitle: "Select filters/documents to re-search",
      };
    }
    if (pendingReview) {
      return {
        title: "Review Search Results",
        subtitle: pendingReview.instruction,
      };
    }
    if (completedRetrievedDocs && completedRetrievedDocs.length > 0) {
      return {
        title: "Reference Documents",
        subtitle: `${completedRetrievedDocs.length} documents`,
      };
    }
    return {
      title: "Activity Log",
      subtitle: `${visibleLogs.length} items`,
    };
  }, [isStreaming, pendingRegeneration, pendingReview, completedRetrievedDocs, visibleLogs.length]);

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
          {/* 스트리밍 중이면 로그 표시 (최우선) */}
          {isStreaming && visibleLogs.length > 0 ? (
            <ChatLogsContent logs={visibleLogs} />
          ) : pendingRegeneration ? (
            <RegeneratePanelContent
              pendingRegeneration={pendingRegeneration}
              submitRegeneration={submitRegeneration}
              onClose={() => setPendingRegeneration(null)}
            />
          ) : pendingReview ? (
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
            <RetrievedDocsContent docs={completedRetrievedDocs} />
          ) : (
            <ChatLogsContent logs={visibleLogs} />
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
  // 이미지 미리보기 모달 상태
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewIndex, setPreviewIndex] = useState(0);

  // 이미지 URL이 유효한지 확인하는 헬퍼 함수
  const hasValidImageUrl = (url: string | null | undefined): url is string => {
    if (typeof url !== 'string') return false;
    const trimmed = url.trim();
    if (trimmed.length === 0 || trimmed === 'null' || trimmed === 'undefined') return false;
    return true;
  };

  // 모든 문서를 미리보기 배열로 생성 (이미지 또는 텍스트)
  // content는 항상 포함 (이미지 로드 실패 시 대체용)
  const previewImages: ImagePreviewItem[] = useMemo(() => {
    return pendingReview.docs.map((doc) => {
      // sop, ts, setup 타입은 {doc_type}_{id} 형식으로 표시
      const docType = (doc.metadata as Record<string, unknown>)?.doc_type as string | undefined;
      const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
      const displayTitle = isSpecialDocType
        ? `${docType}_${doc.docId}`
        : (doc.title || undefined);

      return {
        url: hasValidImageUrl(doc.page_image_url) ? doc.page_image_url : undefined,
        content: doc.content || undefined,
        title: displayTitle,
        page: doc.page || undefined,
        docId: doc.docId,
        rank: doc.rank,
      };
    });
  }, [pendingReview.docs]);

  const handleDocClick = (docIndex: number) => {
    setPreviewIndex(docIndex);
    setPreviewVisible(true);
  };

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

  return (
    <div className="review-panel-sidebar">
      <div className="review-panel-scrollable">
        {/* Search Query Editor Section */}
        <div className="review-queries">
          <div className="review-queries-header">
            <span className="review-queries-label">Search queries</span>
          <button
            className="action-button"
            onClick={toggleEditMode}
            disabled={isStreaming}
            style={{ fontSize: "var(--font-size-xs)", padding: "4px 12px" }}
          >
            {isEditingQueries ? "Done" : "Edit queries"}
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
                  placeholder="Enter query"
                />
                {editableQueries.length > 1 && (
                  <button
                    className="review-query-remove"
                    onClick={() => handleRemoveQuery(idx)}
                    title="Remove"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
            {editableQueries.length < 5 && (
              <button className="review-query-add" onClick={handleAddQuery}>
                + Add query
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
          No search results. Enter keywords to search again.
        </div>
      ) : (
        <>
          <div className="review-controls">
            <label className="review-select-all">
              <input type="checkbox" checked={allSelected} onChange={toggleAll} />
              Select all
            </label>
            <span className="review-count">
              {selectedRanks.length}/{pendingReview.docs.length} selected
            </span>
          </div>
          <div className="review-docs">
            {pendingReview.docs.map((doc, idx) => {
              // sop, ts, setup 타입은 {doc_type}_{id} 형식으로 표시
              const docType = (doc.metadata as Record<string, unknown>)?.doc_type as string | undefined;
              const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
              const displayTitle = isSpecialDocType
                ? `${docType}_${doc.docId}`
                : (doc.title || `Document ${doc.rank ?? idx + 1}`);

              return (
              <label key={`${doc.rank}-${doc.docId}`} className="review-doc">
                <input
                  type="checkbox"
                  checked={selectedRanks.includes(doc.rank)}
                  onChange={() => toggleDoc(doc.rank)}
                />
                <div className="review-doc-body">
                  <div className="review-doc-title">
                    {displayTitle}
                    {doc.page && <span style={{ fontWeight: 400, marginLeft: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>p.{doc.page}</span>}
                  </div>
                  <div className="review-doc-content-wrapper" style={{ position: "relative" }}>
                    {/* 이미지가 있으면 표시 (로드 실패 시 숨김) */}
                    {hasValidImageUrl(doc.page_image_url) && (
                      <img
                        src={doc.page_image_url}
                        alt={`${doc.title || "Document"} page ${doc.page || ""}`}
                        className="review-doc-image"
                        style={{
                          maxWidth: "100%",
                          maxHeight: 300,
                          borderRadius: 4,
                          border: "1px solid var(--color-border)",
                          marginBottom: 8,
                        }}
                        onLoad={(e) => {
                          const img = e.currentTarget;
                          // 이미지가 실제로 유효한지 확인 (naturalWidth > 0)
                          if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                            const wrapper = img.parentElement;
                            const textContent = wrapper?.querySelector(".review-doc-content") as HTMLElement;
                            if (textContent) textContent.style.display = "none";
                          } else {
                            // 유효하지 않은 이미지는 숨김
                            img.style.display = "none";
                          }
                        }}
                        onError={(e) => {
                          // 이미지 로드 실패 시 이미지 숨김
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    )}
                    {/* 텍스트 콘텐츠 (항상 렌더링, 이미지 로드 성공 시 숨김) */}
                    <div className="review-doc-content">
                      <MarkdownContent content={preprocessSnippet(doc.content)} />
                    </div>
                    {/* 확대 버튼 */}
                    <button
                      className="review-doc-zoom"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        handleDocClick(idx);
                      }}
                      title="Zoom"
                      style={{
                        position: "absolute",
                        top: 4,
                        right: 4,
                        width: 28,
                        height: 28,
                        borderRadius: 4,
                        border: "1px solid var(--color-border)",
                        background: "var(--color-bg-secondary, #f5f5f5)",
                        color: "var(--color-text-secondary)",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 14,
                        zIndex: 10,
                      }}
                    >
                      <ZoomInOutlined />
                    </button>
                  </div>
                </div>
              </label>
            );})}
          </div>
        </>
      )}
      </div>

      <div className="review-actions">
        {isEditingQueries ? (
          <button
            className="action-button"
            onClick={handleSearchWithQueries}
            disabled={isStreaming || editableQueries.every((q) => !q.trim())}
          >
            Search again
          </button>
        ) : (
          <>
            <button
              className="action-button"
              onClick={handleSearchWithQueries}
              disabled={isStreaming}
            >
              Search again
            </button>
            <button
              className="action-button"
              onClick={handleReviewSubmit}
              disabled={
                isStreaming || pendingReview.docs.length === 0 || selectedRanks.length === 0
              }
            >
              Answer with selected documents
            </button>
          </>
        )}
      </div>

      {/* 이미지 미리보기 모달 */}
      <ImagePreviewModal
        visible={previewVisible}
        images={previewImages}
        currentIndex={previewIndex}
        selectedRanks={selectedRanks}
        onIndexChange={setPreviewIndex}
        onClose={() => setPreviewVisible(false)}
        onToggleSelect={toggleDoc}
      />
    </div>
  );
}

function RegeneratePanelContent({
  pendingRegeneration,
  submitRegeneration,
  onClose,
}: {
  pendingRegeneration: {
    messageId: string;
    originalQuery: string;
    docs: RetrievedDoc[];
    searchQueries: string[];
    selectedDevices: string[];
    selectedDocTypes: string[];
  };
  submitRegeneration: (payload: {
    originalQuery: string;
    searchQueries: string[];
    selectedDevices: string[];
    selectedDocTypes: string[];
    selectedDocIds: string[];
  }) => void;
  onClose: () => void;
}) {
  const [editableQueries, setEditableQueries] = useState<string[]>([]);
  // Use index-based selection to handle duplicate docIds correctly
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);
  const [selectedDevices, setSelectedDevices] = useState<string[]>([]);
  const [selectedDocTypes, setSelectedDocTypes] = useState<string[]>([]);
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [docTypes, setDocTypes] = useState<DocTypeInfo[]>([]);
  const [visibleDeviceNames, setVisibleDeviceNames] = useState<string[] | null>(null);
  const [deviceFilter, setDeviceFilter] = useState("");
  const [docTypeFilter, setDocTypeFilter] = useState("");
  const [error, setError] = useState<string | null>(null);
  // 이미지 미리보기 모달 상태
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewIndex, setPreviewIndex] = useState(0);

  // 이미지 URL이 유효한지 확인하는 헬퍼 함수
  const hasValidImageUrl = (url: string | null | undefined): url is string => {
    if (typeof url !== 'string') return false;
    const trimmed = url.trim();
    if (trimmed.length === 0 || trimmed === 'null' || trimmed === 'undefined') return false;
    return true;
  };

  // 모든 문서를 미리보기 배열로 생성 (이미지 또는 텍스트)
  const previewImages: ImagePreviewItem[] = useMemo(() => {
    return pendingRegeneration.docs.map((doc) => {
      const docType = (doc.metadata as Record<string, unknown> | undefined)?.doc_type as string | undefined;
      const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
      const displayTitle = isSpecialDocType
        ? `${docType}_${doc.id}`
        : (doc.title || undefined);

      return {
        url: hasValidImageUrl(doc.page_image_url) ? doc.page_image_url : undefined,
        content: doc.snippet || undefined,
        title: displayTitle,
        page: doc.page || undefined,
        docId: doc.id,
      };
    });
  }, [pendingRegeneration.docs]);

  const handleDocClick = (docIndex: number) => {
    setPreviewIndex(docIndex);
    setPreviewVisible(true);
  };

  const allowedDocTypes = useMemo(
    () => ["myservice", "ts", "gcb", "sop", "setup"],
    []
  );

  useEffect(() => {
    const baseQueries = pendingRegeneration.searchQueries.length > 0
      ? pendingRegeneration.searchQueries
      : (pendingRegeneration.originalQuery ? [pendingRegeneration.originalQuery] : []);
    setEditableQueries(baseQueries);
    // Select all documents by index initially
    const allIndices = pendingRegeneration.docs.map((_, idx) => idx);
    setSelectedIndices(allIndices);
    setSelectedDevices(pendingRegeneration.selectedDevices ?? []);
    const allowedSet = new Set(allowedDocTypes.map((d) => d.toLowerCase()));
    const initialDocTypes = (pendingRegeneration.selectedDocTypes ?? [])
      .map((d) => d.toLowerCase())
      .filter((d) => allowedSet.has(d));
    setSelectedDocTypes(initialDocTypes);
    setError(null);
  }, [pendingRegeneration, allowedDocTypes]);

  useEffect(() => {
    let active = true;
    fetchDeviceCatalog()
      .then((res) => {
        if (!active) return;
        setDevices(Array.isArray(res.devices) ? res.devices : []);
        setDocTypes(Array.isArray(res.doc_types) ? res.doc_types : []);
        setVisibleDeviceNames(Array.isArray(res.vis) ? res.vis : null);
      })
      .catch(() => {
        if (!active) return;
        setError("Failed to load equipment/document types.");
      });
    return () => {
      active = false;
    };
  }, [pendingRegeneration]);

  // Use index-based selection to handle duplicate docIds
  const allDocsSelected =
    pendingRegeneration.docs.length > 0 && selectedIndices.length === pendingRegeneration.docs.length;

  const toggleDocByIndex = (index: number) => {
    setSelectedIndices((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index]
    );
  };

  const toggleAllDocs = () => {
    if (allDocsSelected) {
      setSelectedIndices([]);
      return;
    }
    setSelectedIndices(pendingRegeneration.docs.map((_, idx) => idx));
  };

  const handleQueryChange = (index: number, value: string) => {
    setEditableQueries((prev) => {
      const updated = [...prev];
      updated[index] = value;
      return updated;
    });
  };

  const handleAddQuery = () => {
    setEditableQueries((prev) => [...prev, ""]);
  };

  const handleRemoveQuery = (index: number) => {
    setEditableQueries((prev) => prev.filter((_, i) => i !== index));
  };

  const toggleDevice = (name: string) => {
    setSelectedDevices((prev) =>
      prev.includes(name) ? prev.filter((d) => d !== name) : [...prev, name]
    );
  };

  const toggleDocType = (name: string) => {
    setSelectedDocTypes((prev) =>
      prev.includes(name) ? prev.filter((d) => d !== name) : [...prev, name]
    );
  };

  const mergedDevices = useMemo(() => {
    const map = new Map<string, DeviceInfo>();
    devices.forEach((d) => map.set(d.name, d));
    selectedDevices.forEach((name) => {
      if (!map.has(name)) {
        map.set(name, { name, doc_count: 0 });
      }
    });
    return Array.from(map.values());
  }, [devices, selectedDevices]);

  const mergedDocTypes = useMemo(() => {
    const map = new Map<string, DocTypeInfo>();
    docTypes.forEach((d) => map.set(d.name.toLowerCase(), d));
    return allowedDocTypes.map((name) => {
      const existing = map.get(name.toLowerCase());
      return existing ?? { name, doc_count: 0 };
    });
  }, [docTypes, allowedDocTypes]);

  const sortedDevices = useMemo(() => {
    const list = [...mergedDevices];
    list.sort((a, b) => {
      const aSelected = selectedDevices.includes(a.name);
      const bSelected = selectedDevices.includes(b.name);
      if (aSelected !== bSelected) {
        return aSelected ? -1 : 1;
      }
      return (b.doc_count || 0) - (a.doc_count || 0);
    });
    return list;
  }, [mergedDevices, selectedDevices]);

  const filteredDevices = sortedDevices.filter((d) =>
    d.name.toLowerCase().includes(deviceFilter.toLowerCase())
  );
  const allDeviceNames = useMemo(() => mergedDevices.map((d) => d.name), [mergedDevices]);
  const allDevicesSelected = allDeviceNames.length > 0 && selectedDevices.length === allDeviceNames.length;
  // 화면에 표시할 기기:
  // - 검색어 입력 시: 전체 기기에서 검색
  // - 검색어 없을 때: vis 배열 (상위 10개)만 표시
  const visibleDevices = useMemo(() => {
    // 검색어가 있으면 전체 기기에서 검색
    if (deviceFilter.trim()) {
      return filteredDevices;
    }
    // 검색어가 없으면 vis 배열의 기기만 표시
    if (visibleDeviceNames && visibleDeviceNames.length > 0) {
      const allowed = new Set(visibleDeviceNames.map((name) => name.toLowerCase()));
      return filteredDevices.filter((d) => allowed.has(d.name.toLowerCase()));
    }
    return filteredDevices.slice(0, 10);
  }, [filteredDevices, visibleDeviceNames, deviceFilter]);

  const filteredDocTypes = mergedDocTypes.filter((d) =>
    d.name.toLowerCase().includes(docTypeFilter.toLowerCase())
  );

  const allDocTypesSelected = allowedDocTypes.length > 0 && selectedDocTypes.length === allowedDocTypes.length;

  const handleSubmit = () => {
    const queries = editableQueries.map((q) => q.trim()).filter((q) => q.length > 0);
    if (selectedIndices.length === 0) {
      setError("Select at least one document to re-search.");
      return;
    }
    // Convert selected indices to doc IDs
    const selectedDocIds = selectedIndices
      .map((idx) => pendingRegeneration.docs[idx]?.id)
      .filter((id): id is string => typeof id === "string" && id.trim().length > 0);
    // 전체 선택 시 빈 배열로 전달 (필터 없이 검색)
    const deviceFilter = allDevicesSelected ? [] : selectedDevices;
    const docTypeFilter = allDocTypesSelected ? [] : selectedDocTypes;
    submitRegeneration({
      originalQuery: pendingRegeneration.originalQuery,
      searchQueries: queries.length > 0 ? queries : [pendingRegeneration.originalQuery],
      selectedDevices: deviceFilter,
      selectedDocTypes: docTypeFilter,
      selectedDocIds,
    });
  };

  return (
    <div className="review-panel-sidebar">
      <div className="review-panel-scrollable">
        <div className="review-queries">
          <div className="review-queries-header">
            <span className="review-queries-label">Search queries (MQ)</span>
          </div>
        <div className="review-queries-editor">
          {editableQueries.map((query, idx) => (
            <div key={idx} className="review-query-input-row">
              <input
                type="text"
                className="review-query-input"
                value={query}
                onChange={(e) => handleQueryChange(idx, e.target.value)}
                placeholder="Enter query"
              />
              {editableQueries.length > 1 && (
                <button
                  className="review-query-remove"
                  onClick={() => handleRemoveQuery(idx)}
                  title="Remove"
                >
                  ×
                </button>
              )}
            </div>
          ))}
          {editableQueries.length < 5 && (
            <button className="review-query-add" onClick={handleAddQuery}>
              + Add query
            </button>
          )}
        </div>
      </div>

      <div style={{
        padding: "12px",
        borderRadius: 8,
        border: "1px solid var(--color-border)",
        background: "var(--color-bg-secondary)",
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Select equipment</div>
        <input
          type="text"
          value={deviceFilter}
          onChange={(e) => setDeviceFilter(e.target.value)}
          placeholder="Search equipment"
          className="review-query-input"
          style={{ marginBottom: 8 }}
        />
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
            {allDevicesSelected ? "All selected" : null}
          </span>
          <button
            className="action-button"
            style={{ padding: "4px 10px", fontSize: 12 }}
            onClick={() => {
              setSelectedDevices(allDevicesSelected ? [] : allDeviceNames);
            }}
          >
            {allDevicesSelected ? "Clear all" : "Select all"}
          </button>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {visibleDevices.length === 0 && (
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              No equipment to display.
            </span>
          )}
          {visibleDevices.map((device) => {
            const isSelected = selectedDevices.includes(device.name);
            return (
              <button
                key={device.name}
                className="review-query-tag"
                onClick={() => toggleDevice(device.name)}
                style={{
                  border: isSelected ? "1px solid var(--color-accent-primary)" : "1px solid var(--color-border)",
                  background: isSelected ? "var(--color-accent-primary-light)" : "var(--color-bg-primary)",
                }}
              >
                {device.name}
                {device.doc_count > 0 && (
                  <span style={{ marginLeft: 4, fontSize: 10, opacity: 0.6 }}>
                    ({device.doc_count})
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{
        padding: "12px",
        borderRadius: 8,
        border: "1px solid var(--color-border)",
        background: "var(--color-bg-secondary)",
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Select document types</div>
        <input
          type="text"
          value={docTypeFilter}
          onChange={(e) => setDocTypeFilter(e.target.value)}
          placeholder="Search document types"
          className="review-query-input"
          style={{ marginBottom: 8 }}
        />
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
            {allDocTypesSelected ? "All selected" : null}
          </span>
          <button
            className="action-button"
            style={{ padding: "4px 10px", fontSize: 12 }}
            onClick={() => {
              setSelectedDocTypes(allDocTypesSelected ? [] : allowedDocTypes.map((d) => d.toLowerCase()));
            }}
          >
            {allDocTypesSelected ? "Clear all" : "Select all"}
          </button>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {filteredDocTypes.length === 0 && (
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              No document types to display.
            </span>
          )}
          {filteredDocTypes.map((docType) => {
            const docTypeKey = docType.name.toLowerCase();
            const isSelected = selectedDocTypes.includes(docTypeKey);
            return (
              <button
                key={docType.name}
                className="review-query-tag"
                onClick={() => toggleDocType(docTypeKey)}
                style={{
                  border: isSelected ? "1px solid var(--color-accent-primary)" : "1px solid var(--color-border)",
                  background: isSelected ? "var(--color-accent-primary-light)" : "var(--color-bg-primary)",
                }}
              >
                {docType.name}
                {docType.doc_count > 0 && (
                  <span style={{ marginLeft: 4, fontSize: 10, opacity: 0.6 }}>
                    ({docType.doc_count})
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div className="review-controls">
        <label className="review-select-all">
          <input type="checkbox" checked={allDocsSelected} onChange={toggleAllDocs} />
          Select all previous documents
        </label>
        <span className="review-count">
          {selectedIndices.length}/{pendingRegeneration.docs.length} selected
        </span>
      </div>

      <div className="review-docs">
        {pendingRegeneration.docs.map((doc, idx) => {
          const docId = typeof doc.id === "string" ? doc.id : "";
          const docType = (doc.metadata as Record<string, unknown> | undefined)?.doc_type as string | undefined;
          const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
          const displayTitle = isSpecialDocType
            ? `${docType}_${docId}`
            : (doc.title || `Document ${idx + 1}`);
          return (
            <label key={`doc-${idx}`} className="review-doc">
              <input
                type="checkbox"
                checked={selectedIndices.includes(idx)}
                onChange={() => toggleDocByIndex(idx)}
              />
              <div className="review-doc-body">
                <div className="review-doc-title">
                  {displayTitle}
                  {doc.page && (
                    <span style={{ fontWeight: 400, marginLeft: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>
                      p.{doc.page}
                    </span>
                  )}
                </div>
                <div className="review-doc-content-wrapper" style={{ position: "relative" }}>
                  {/* 이미지가 있으면 표시 (로드 실패 시 숨김) */}
                  {hasValidImageUrl(doc.page_image_url) && (
                    <img
                      src={doc.page_image_url}
                      alt={`${displayTitle} page ${doc.page || ""}`}
                      className="review-doc-image"
                      style={{
                        maxWidth: "100%",
                        maxHeight: 300,
                        borderRadius: 4,
                        border: "1px solid var(--color-border)",
                        marginBottom: 8,
                      }}
                      onLoad={(e) => {
                        const img = e.currentTarget;
                        if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                          const wrapper = img.parentElement;
                          const textContent = wrapper?.querySelector(".review-doc-content") as HTMLElement;
                          if (textContent) textContent.style.display = "none";
                        } else {
                          img.style.display = "none";
                        }
                      }}
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                    />
                  )}
                  {/* 텍스트 콘텐츠 (항상 렌더링, 이미지 로드 성공 시 숨김) */}
                  {doc.snippet && (
                    <div className="review-doc-content">
                      <MarkdownContent content={preprocessSnippet(doc.snippet)} />
                    </div>
                  )}
                  {/* 확대 버튼 */}
                  <button
                    className="review-doc-zoom"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      handleDocClick(idx);
                    }}
                    title="Zoom"
                    style={{
                      position: "absolute",
                      top: 4,
                      right: 4,
                      width: 28,
                      height: 28,
                      borderRadius: 4,
                      border: "1px solid var(--color-border)",
                      background: "var(--color-bg-secondary, #f5f5f5)",
                      color: "var(--color-text-secondary)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 14,
                      zIndex: 10,
                    }}
                  >
                    <ZoomInOutlined />
                  </button>
                </div>
              </div>
            </label>
          );
        })}
      </div>

        {error && (
          <div style={{ color: "var(--color-danger, #d32f2f)", fontSize: 12 }}>
            {error}
          </div>
        )}
      </div>

      <div className="review-actions">
        <button className="action-button" onClick={onClose}>
          Close
        </button>
        <button className="action-button" onClick={handleSubmit}>
          Search again
        </button>
      </div>

      {/* 이미지 미리보기 모달 */}
      <ImagePreviewModal
        visible={previewVisible}
        images={previewImages}
        currentIndex={previewIndex}
        selectedRanks={selectedIndices.map((idx) => idx + 1)}
        onIndexChange={setPreviewIndex}
        onClose={() => setPreviewVisible(false)}
        onToggleSelect={(rank) => {
          toggleDocByIndex(rank - 1);
        }}
      />
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
  expanded_pages?: number[] | null;
  expanded_page_urls?: string[] | null;
}> }) {
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewIndex, setPreviewIndex] = useState(0);

  // 이미지 URL이 유효한지 확인
  const hasValidImageUrl = (url: string | null | undefined): url is string => {
    return typeof url === 'string' && url.trim().length > 0;
  };

  // 모든 문서의 모든 페이지를 미리보기 배열로 생성 (expanded 포함)
  // 각 문서의 시작 인덱스와 렌더링용 데이터도 함께 계산
  const { previewImages, docStartIndices, docRenderData } = useMemo(() => {
    const images: ImagePreviewItem[] = [];
    const startIndices: number[] = [];
    const renderData: Array<{
      pageUrls: string[];
      pageNumbers: number[];
      displayTitle: string | undefined;
      hasImageUrls: boolean;
    }> = [];

    docs.forEach((doc) => {
      // 현재 문서의 시작 인덱스 저장
      startIndices.push(images.length);

      const docType = doc.metadata?.doc_type as string | undefined;
      const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
      const displayTitle = isSpecialDocType
        ? `${docType}_${doc.id}`
        : (doc.title || undefined);

      // expanded_page_urls가 있으면 모든 페이지 추가
      const pageUrls = doc.expanded_page_urls && doc.expanded_page_urls.length > 0
        ? doc.expanded_page_urls.filter(url => hasValidImageUrl(url))
        : hasValidImageUrl(doc.page_image_url)
          ? [doc.page_image_url]
          : [];

      const pageNumbers = doc.expanded_pages && doc.expanded_pages.length > 0
        ? doc.expanded_pages
        : doc.page !== null && doc.page !== undefined
          ? [doc.page]
          : [];

      const hasImageUrls = pageUrls.length > 0;

      // 렌더링용 데이터 저장
      renderData.push({ pageUrls, pageNumbers, displayTitle, hasImageUrls });

      if (pageUrls.length > 0) {
        // 이미지가 있는 경우: 각 페이지별로 항목 추가
        pageUrls.forEach((url, idx) => {
          images.push({
            url,
            content: idx === 0 ? doc.snippet || undefined : undefined, // 첫 페이지에만 snippet
            title: displayTitle,
            page: pageNumbers[idx] || undefined,
            docId: doc.id,
          });
        });
      } else {
        // 이미지가 없는 경우: snippet만 추가
        images.push({
          url: undefined,
          content: doc.snippet || undefined,
          title: displayTitle,
          page: doc.page || undefined,
          docId: doc.id,
        });
      }
    });

    return { previewImages: images, docStartIndices: startIndices, docRenderData: renderData };
  }, [docs]);

  const handleDocClick = (docIndex: number) => {
    // 문서의 첫 번째 이미지로 이동
    setPreviewIndex(docStartIndices[docIndex] || 0);
    setPreviewVisible(true);
  };

  const handleImageClick = (docIndex: number, pageIndex: number) => {
    // 해당 문서의 특정 페이지 이미지로 이동
    const startIdx = docStartIndices[docIndex] || 0;
    const targetIndex = startIdx + pageIndex;
    console.log("[RetrievedDocs] handleImageClick:", { docIndex, pageIndex, startIdx, targetIndex, totalImages: previewImages.length });
    setPreviewIndex(targetIndex);
    setPreviewVisible(true);
  };

  return (
    <div className="retrieved-docs-container">
      {docs.map((doc, index) => {
        // useMemo에서 계산된 동일한 데이터 사용
        const { pageUrls, pageNumbers, displayTitle, hasImageUrls } = docRenderData[index];

        return (
          <div key={doc.id || index} className="retrieved-doc-item">
            <div className="retrieved-doc-header">
              <span className="retrieved-doc-rank">#{index + 1}</span>
              {displayTitle && (
                <span className="retrieved-doc-title">{displayTitle}</span>
              )}
              {pageNumbers.length > 0 && (
                <span className="retrieved-doc-page">
                  {pageNumbers.length === 1
                    ? `p.${pageNumbers[0]}`
                    : `p.${pageNumbers[0]}-${pageNumbers[pageNumbers.length - 1]}`}
                </span>
              )}
            </div>
            {/* 이미지와 텍스트 래퍼 */}
            <div className="retrieved-doc-content-wrapper" style={{ position: "relative" }}>
              {/* 이미지가 있으면 표시 */}
              {hasImageUrls && (
                <div className="retrieved-doc-image-wrapper">
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {pageUrls.map((url, pageIdx) => (
                      <img
                        key={`${url}-${pageIdx}`}
                        src={url}
                        alt={`${displayTitle || "Document"} page ${pageNumbers[pageIdx] || pageIdx + 1}`}
                        className="retrieved-doc-image"
                        style={{ cursor: "pointer" }}
                        onClick={() => handleImageClick(index, pageIdx)}
                        onLoad={(e) => {
                          // 이미지 로드 성공 시 텍스트 숨기기
                          const wrapper = e.currentTarget.closest(".retrieved-doc-content-wrapper");
                          const textContent = wrapper?.querySelector(".retrieved-doc-snippet") as HTMLElement;
                          if (textContent) textContent.style.display = "none";
                        }}
                        onError={(e) => {
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    ))}
                  </div>
                </div>
              )}
              {/* 텍스트 콘텐츠 (항상 렌더링, 이미지 로드 성공 시 숨김) */}
              {doc.snippet && (
                <div className="retrieved-doc-snippet">
                  <MarkdownContent content={preprocessSnippet(doc.snippet)} />
                </div>
              )}
              {/* 확대 버튼 */}
              <button
                className="retrieved-doc-zoom"
                onClick={() => handleDocClick(index)}
                title="Zoom"
                style={{
                  position: "absolute",
                  top: 4,
                  right: 4,
                  width: 28,
                  height: 28,
                  borderRadius: 4,
                  border: "1px solid var(--color-border)",
                  background: "var(--color-bg-secondary, #f5f5f5)",
                  color: "var(--color-text-secondary)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 14,
                  zIndex: 10,
                }}
              >
                <ZoomInOutlined />
              </button>
            </div>
          </div>
        );
      })}

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
        title="No logs"
        description="Activity log will be displayed when you start a conversation"
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
