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
  const { logs } = useChatLogs();

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
    submitReview,
    submitSearchQueries,
    submitRegeneration,
    setPendingRegeneration,
  } = useChatReview();

  // Show right sidebar when there are logs, pending review, or completed retrieved docs
  const shouldShowRightSidebar = isChatPage && (
    logs.length > 0 ||
    pendingReview !== null ||
    pendingRegeneration !== null ||
    (completedRetrievedDocs !== null && completedRetrievedDocs.length > 0)
  );


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
    if (isStreaming && logs.length > 0) {
      return {
        title: "실행 로그",
        subtitle: `${logs.length}개 항목`,
      };
    }
    if (pendingRegeneration) {
      return {
        title: "답변 재생성",
        subtitle: "필터/문서 선택 후 재검색",
      };
    }
    if (pendingReview) {
      return {
        title: "검색 결과 확인",
        subtitle: pendingReview.instruction,
      };
    }
    if (completedRetrievedDocs && completedRetrievedDocs.length > 0) {
      return {
        title: "확장 문서/참고 문서",
        subtitle: `${completedRetrievedDocs.length}개 문서`,
      };
    }
    return {
      title: "실행 로그",
      subtitle: `${logs.length}개 항목`,
    };
  }, [isStreaming, pendingRegeneration, pendingReview, completedRetrievedDocs, logs.length]);

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
          {isStreaming && logs.length > 0 ? (
            <ChatLogsContent logs={logs} />
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
            {pendingReview.docs.map((doc, idx) => {
              // sop, ts, setup 타입은 {doc_type}_{id} 형식으로 표시
              const docType = (doc.metadata as Record<string, unknown>)?.doc_type as string | undefined;
              const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
              const displayTitle = isSpecialDocType
                ? `${docType}_${doc.docId}`
                : (doc.title || `문서 ${doc.rank ?? idx + 1}`);

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
                      title="확대"
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
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
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
    const docIds = Array.from(new Set(
      pendingRegeneration.docs.map((d) => d.id).filter((id) => typeof id === "string" && id.trim())
    ));
    setSelectedDocIds(docIds);
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
        setError("장비/문서 종류 목록을 불러오지 못했습니다.");
      });
    return () => {
      active = false;
    };
  }, [pendingRegeneration]);

  const docIdList = useMemo(() => {
    return Array.from(new Set(
      pendingRegeneration.docs
        .map((d) => d.id)
        .filter((id) => typeof id === "string" && id.trim())
    ));
  }, [pendingRegeneration.docs]);

  const allDocsSelected =
    docIdList.length > 0 && selectedDocIds.length === docIdList.length;

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const toggleAllDocs = () => {
    if (allDocsSelected) {
      setSelectedDocIds([]);
      return;
    }
    setSelectedDocIds(docIdList);
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
    if (selectedDocIds.length === 0) {
      setError("재검색할 문서를 1개 이상 선택해 주세요.");
      return;
    }
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
      <div className="review-queries">
        <div className="review-queries-header">
          <span className="review-queries-label">검색어 (MQ)</span>
        </div>
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
      </div>

      <div style={{
        padding: "12px",
        borderRadius: 8,
        border: "1px solid var(--color-border)",
        background: "var(--color-bg-secondary)",
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>기기 선택</div>
        <input
          type="text"
          value={deviceFilter}
          onChange={(e) => setDeviceFilter(e.target.value)}
          placeholder="기기 검색"
          className="review-query-input"
          style={{ marginBottom: 8 }}
        />
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
            {allDevicesSelected ? "전체선택됨" : null}
          </span>
          <button
            className="action-button"
            style={{ padding: "4px 10px", fontSize: 12 }}
            onClick={() => {
              setSelectedDevices(allDevicesSelected ? [] : allDeviceNames);
            }}
          >
            {allDevicesSelected ? "전체 해제" : "전체 선택"}
          </button>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {visibleDevices.length === 0 && (
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              표시할 기기가 없습니다.
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
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>문서 종류 선택</div>
        <input
          type="text"
          value={docTypeFilter}
          onChange={(e) => setDocTypeFilter(e.target.value)}
          placeholder="문서 종류 검색"
          className="review-query-input"
          style={{ marginBottom: 8 }}
        />
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
            {allDocTypesSelected ? "전체선택됨" : null}
          </span>
          <button
            className="action-button"
            style={{ padding: "4px 10px", fontSize: 12 }}
            onClick={() => {
              setSelectedDocTypes(allDocTypesSelected ? [] : allowedDocTypes.map((d) => d.toLowerCase()));
            }}
          >
            {allDocTypesSelected ? "전체 해제" : "전체 선택"}
          </button>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {filteredDocTypes.length === 0 && (
            <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
              표시할 문서 종류가 없습니다.
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
          이전 문서 전체 선택
        </label>
        <span className="review-count">
          {selectedDocIds.length}/{pendingRegeneration.docs.length} 선택
        </span>
      </div>

      <div className="review-docs">
        {pendingRegeneration.docs.map((doc, idx) => {
          const docId = typeof doc.id === "string" ? doc.id : "";
          const docType = (doc.metadata as Record<string, unknown> | undefined)?.doc_type as string | undefined;
          const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
          const displayTitle = isSpecialDocType
            ? `${docType}_${docId}`
            : (doc.title || `문서 ${idx + 1}`);
          return (
            <label key={`${docId}-${idx}`} className="review-doc">
              <input
                type="checkbox"
                checked={docId ? selectedDocIds.includes(docId) : false}
                onChange={() => docId && toggleDoc(docId)}
                disabled={!docId}
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
                    title="확대"
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

      <div className="review-actions">
        <button className="action-button" onClick={onClose}>
          닫기
        </button>
        <button className="action-button" onClick={handleSubmit}>
          재검색
        </button>
      </div>

      {/* 이미지 미리보기 모달 */}
      <ImagePreviewModal
        visible={previewVisible}
        images={previewImages}
        currentIndex={previewIndex}
        selectedRanks={selectedDocIds.map((id) => {
          const idx = pendingRegeneration.docs.findIndex((d) => d.id === id);
          return idx >= 0 ? idx + 1 : -1;
        }).filter((r) => r > 0)}
        onIndexChange={setPreviewIndex}
        onClose={() => setPreviewVisible(false)}
        onToggleSelect={(rank) => {
          const doc = pendingRegeneration.docs[rank - 1];
          if (doc?.id) toggleDoc(doc.id);
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

  // 모든 문서를 미리보기 배열로 생성 (이미지 또는 텍스트)
  const previewImages: ImagePreviewItem[] = useMemo(() => {
    return docs.map((doc) => {
      // sop, ts, setup 타입은 {doc_type}_{id} 형식으로 표시
      const docType = doc.metadata?.doc_type as string | undefined;
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
  }, [docs]);

  const handleDocClick = (docIndex: number) => {
    setPreviewIndex(docIndex);
    setPreviewVisible(true);
  };

  const handleImageClick = (docIndex: number, _pageIndex: number) => {
    setPreviewIndex(docIndex);
    setPreviewVisible(true);
  };

  return (
    <div className="retrieved-docs-container">
      {docs.map((doc, index) => {
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
        const docType = doc.metadata?.doc_type as string | undefined;
        const isSpecialDocType = docType && ["sop", "ts", "setup"].includes(docType.toLowerCase());
        const displayTitle = isSpecialDocType
          ? `${docType}_${doc.id}`
          : doc.title;

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
              {doc.score !== null && doc.score !== undefined && (
                <span className="retrieved-doc-score">
                  {typeof doc.score_percent === "number"
                    ? `${doc.score_percent.toFixed(1)}%`
                    : doc.score.toFixed(3)}
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
                title="확대"
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
