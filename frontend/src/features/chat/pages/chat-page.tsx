import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useChatSession } from "../hooks/use-chat-session";
import { useChatHistoryContext } from "../context/chat-history-context";
import { useChatReview } from "../context/chat-review-context";
import {
  ChatContainer,
  MessageList,
  InputArea,
  MessageItem,
  ChatInput,
  DeviceSelectionPanel,
  GuidedSelectionPanel,
} from "../components";
import type { RegeneratePayload } from "../components/message-item";
import { fetchDeviceCatalog } from "../api";
import { Alert, Spin } from "antd";

export default function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const sessionParam = searchParams.get("session");

  const { refresh: refreshHistory } = useChatHistoryContext();
  const {
    setPendingReview,
    pendingRegeneration,
    setIsStreaming,
    registerSubmitHandlers,
    setPendingRegeneration,
    registerRegenerationHandlers,
  } = useChatReview();
  const [suggestedDevices, setSuggestedDevices] = useState<Array<{ name: string; doc_count: number }>>([]);
  const [isLoadingSuggestedDevices, setIsLoadingSuggestedDevices] = useState(false);
  const [suggestedDeviceError, setSuggestedDeviceError] = useState<string | null>(null);
  const [showSuggestedDevicePanel, setShowSuggestedDevicePanel] = useState(false);
  const [lastSelectedDevice, setLastSelectedDevice] = useState<string | null>(null);

  // Callback when a turn is saved
  const handleTurnSaved = useCallback(() => {
    refreshHistory();
  }, [refreshHistory]);

  const {
    sessionId,
    messages,
    send,
    stop,
    isStreaming,
    isLoadingSession,
    error,
    reset,
    loadSession,
    inputPlaceholder,
    pendingReview,
    pendingDeviceSelection,
    pendingGuidedSelection,
    submitGuidedSelectionNumber,
    submitGuidedSelectionFinal,
    submitReview,
    submitSearchQueries,
    submitDeviceSelection,
    submitFeedback,
    submitDetailedFeedback,
  } = useChatSession({ onTurnSaved: handleTurnSaved });

  // Load session from URL parameter
  useEffect(() => {
    if (sessionParam) {
      // Clear the URL parameter first to prevent re-triggering
      setSearchParams({}, { replace: true });
      // Then load the session
      loadSession(sessionParam);
    }
  }, [sessionParam, loadSession, setSearchParams]);

  // Register submit handlers for right sidebar to use
  useEffect(() => {
    registerSubmitHandlers({ submitReview, submitSearchQueries });
  }, [submitReview, submitSearchQueries, registerSubmitHandlers]);

  // Reset showSuggestedDevicePanel when pendingRegeneration changes
  useEffect(() => {
    setShowSuggestedDevicePanel(false);
  }, [pendingRegeneration?.messageId, pendingRegeneration?.reason]);

  // 이전에 기기를 선택한 적이 있으면 패널 표시 없이 무시
  useEffect(() => {
    if (
      pendingRegeneration?.reason === "missing_device_parse" &&
      lastSelectedDevice
    ) {
      setPendingRegeneration(null);
    }
  }, [pendingRegeneration?.messageId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let active = true;
    const shouldLoadCatalog =
      pendingRegeneration?.reason === "missing_device_parse" &&
      showSuggestedDevicePanel;
    if (!shouldLoadCatalog) {
      setSuggestedDevices([]);
      setSuggestedDeviceError(null);
      setIsLoadingSuggestedDevices(false);
      return () => { active = false; };
    }

    setIsLoadingSuggestedDevices(true);
    setSuggestedDeviceError(null);
    fetchDeviceCatalog()
      .then((res) => {
        if (!active) return;
        setSuggestedDevices(Array.isArray(res.devices) ? res.devices : []);
      })
      .catch(() => {
        if (!active) return;
        setSuggestedDevices([]);
        setSuggestedDeviceError("장비 목록을 불러오지 못했습니다.");
      })
      .finally(() => {
        if (!active) return;
        setIsLoadingSuggestedDevices(false);
      });

    return () => { active = false; };
  }, [pendingRegeneration?.reason, pendingRegeneration?.messageId, showSuggestedDevicePanel]);

  const sanitizeRegenerationQuery = useCallback((query: string): string => {
    let normalized = query.trim();
    if (!normalized) return "";

    normalized = normalized.replace(/^(?:\[\s*regenerate with[^\]]*\]\s*)+/gi, "").trim();
    normalized = normalized.replace(/^(?:regenerate with\b[^:\n]*[:\-]?\s*)+/gi, "").trim();
    normalized = normalized.replace(/^(?:재검색\s*(?:조건|필터)?\s*[:\-]?\s*)+/g, "").trim();

    if (/^[.\s…·•\-_~=]+$/.test(normalized)) return "";
    return normalized;
  }, []);

  const submitRegeneration = useCallback((payload: {
    originalQuery: string;
    searchQueries: string[];
    selectedDevices: string[];
    selectedDocTypes: string[];
    selectedDocIds: string[];
  }) => {
    const normalizedOriginalQuery = sanitizeRegenerationQuery(payload.originalQuery);
    const normalizedQueries = payload.searchQueries
      .map((query) => sanitizeRegenerationQuery(query))
      .filter((query) => query.length > 0);

    setPendingRegeneration(null);
    send({
      text: normalizedOriginalQuery || payload.originalQuery,
      overrides: {
        filterDevices: payload.selectedDevices,
        filterDocTypes: payload.selectedDocTypes,
        searchQueries: normalizedQueries.length > 0
          ? normalizedQueries
          : (normalizedOriginalQuery ? [normalizedOriginalQuery] : []),
        selectedDocIds: payload.selectedDocIds,
        autoParse: false,
      },
    });
  }, [send, setPendingRegeneration, sanitizeRegenerationQuery]);

  useEffect(() => {
    registerRegenerationHandlers({ submitRegeneration });
  }, [submitRegeneration, registerRegenerationHandlers]);

  // Sync streaming state with context
  useEffect(() => {
    setIsStreaming(isStreaming);
  }, [isStreaming, setIsStreaming]);

  // Sync pending review with context for right sidebar
  useEffect(() => {
    console.log("[ChatPage] pendingReview from hook:", pendingReview);
    if (pendingReview) {
      const queries = pendingReview.payload?.search_queries;
      const searchQueries = Array.isArray(queries)
        ? queries.map((q) => String(q))
        : [pendingReview.question];

      console.log("[ChatPage] Setting pendingReview in context:", {
        threadId: pendingReview.threadId,
        docs: pendingReview.docs?.length,
        searchQueries,
      });
      setPendingReview({
        threadId: pendingReview.threadId,
        question: pendingReview.question,
        instruction: pendingReview.instruction,
        docs: pendingReview.docs,
        searchQueries,
      });
    } else {
      setPendingReview(null);
    }
  }, [pendingReview, setPendingReview]);

  // Listen for new chat event from sidebar
  useEffect(() => {
    const handleNewChat = () => {
      reset();
      setPendingRegeneration(null);
    };
    window.addEventListener("pe-agent:new-chat", handleNewChat);
    return () => {
      window.removeEventListener("pe-agent:new-chat", handleNewChat);
    };
  }, [reset, setPendingRegeneration]);

  const handleDeviceSelect = (deviceIndex: number) => {
    if (!pendingRegeneration) return;
    const deviceName = suggestedDevices[deviceIndex].name;
    const originalQuery = pendingRegeneration.originalQuery;
    const displayText = `[${deviceName}] ${originalQuery}`;
    const searchQueries =
      pendingRegeneration.searchQueries?.length > 0
        ? pendingRegeneration.searchQueries
        : [originalQuery].filter((q) => q?.trim());

    setLastSelectedDevice(deviceName);
    setPendingRegeneration(null);
    send({
      text: displayText,
      overrides: {
        filterDevices: [deviceName],
        searchQueries,
        autoParse: false,
      },
    });
  };

  const handleSend = async (text: string) => {
    if (pendingGuidedSelection) {
      const trimmed = text.trim();
      if (trimmed === "0" || /^[1-9][0-9]*$/.test(trimmed)) {
        submitGuidedSelectionNumber(trimmed);
      }
      return;
    }

    if (
      pendingRegeneration?.reason === "missing_device_parse" &&
      !lastSelectedDevice
    ) {
      const trimmed = text.trim();

      // 1단계: 확인 다이얼로그 (1. 예 / 2. 아니오)
      if (!showSuggestedDevicePanel) {
        if (trimmed === "1") {
          setShowSuggestedDevicePanel(true);
          return;
        }
        if (trimmed === "2") {
          setPendingRegeneration(null);
          return;
        }
        // 그 외 입력 → 무시
        return;
      }

      // 2단계: 기기 번호 선택
      if (suggestedDevices.length > 0 && !isLoadingSuggestedDevices) {
        const num = parseInt(trimmed, 10);

        if (num === 0) {
          setPendingRegeneration(null);
          return;
        }

        if (!isNaN(num) && num >= 1 && num <= suggestedDevices.length) {
          handleDeviceSelect(num - 1);
          return;
        }
      }
      // 유효하지 않은 입력 → 무시
      return;
    }

    await send({ text });
  };

  // Handle regeneration request
  const handleRegenerate = useCallback((payload: RegeneratePayload) => {
    const fallbackQueries = payload.originalQuery ? [payload.originalQuery] : [];
    setPendingRegeneration({
      messageId: payload.messageId,
      originalQuery: payload.originalQuery,
      docs: payload.retrievedDocs ?? [],
      searchQueries: payload.searchQueries && payload.searchQueries.length > 0
        ? payload.searchQueries
        : fallbackQueries,
      selectedDevices: payload.selectedDevices ?? [],
      selectedDocTypes: payload.selectedDocTypes ?? [],
      reason: "manual",
    });
  }, [setPendingRegeneration]);

  // Get the original user query for the last assistant message
  const getOriginalQuery = useMemo(() => {
    const queryMap = new Map<string, string>();
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === "user") {
        // The next assistant message gets this user's query
        if (i + 1 < messages.length && messages[i + 1].role === "assistant") {
          queryMap.set(messages[i + 1].id, messages[i].content);
        }
      }
    }
    return (assistantId: string) => queryMap.get(assistantId) || "";
  }, [messages]);

  return (
    <div className="chat-layout">
      {/* Main Chat Area */}
      <main className="chat-main">
        <ChatContainer>
          {isLoadingSession ? (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                color: "var(--color-text-secondary)",
              }}
            >
              <Spin size="large" tip="대화를 불러오는 중..." />
            </div>
          ) : (
            <MessageList autoScrollToBottom>
              {messages.length === 0 ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    color: "var(--color-text-secondary)",
                    gap: 16,
                    padding: 48,
                  }}
                >
                  <div
                    style={{
                      width: 64,
                      height: 64,
                      borderRadius: "50%",
                      backgroundColor: "var(--color-accent-primary)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "white",
                      fontSize: 24,
                      fontWeight: 600,
                    }}
                  >
                    PE
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <h2 style={{ margin: "0 0 8px", color: "var(--color-text-primary)" }}>
                      PE Agent에 오신 것을 환영합니다
                    </h2>
                    <p style={{ margin: 0 }}>
                      질문을 입력하면 AI가 답변해 드립니다
                    </p>
                  </div>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <MessageItem
                    key={msg.id}
                    message={msg}
                    isStreaming={isStreaming && idx === messages.length - 1 && msg.role === "assistant"}
                    onFeedback={submitFeedback}
                    onDetailedFeedback={submitDetailedFeedback}
                    onRegenerate={msg.role === "assistant" ? handleRegenerate : undefined}
                    onEdit={msg.role === "user" && !isStreaming ? (editedText) => send({ text: editedText }) : undefined}
                    originalQuery={msg.role === "assistant" ? getOriginalQuery(msg.id) : undefined}
                  />
                ))
              )}
            </MessageList>
          )}

          {/* Device selection panel for missing_device_parse */}
          {pendingRegeneration?.reason === "missing_device_parse" && !lastSelectedDevice && (
            <div
              style={{
                margin: "16px 0 8px",
                padding: "12px 16px",
                borderRadius: 8,
                border: "1px solid var(--color-border)",
                background: "var(--color-bg-secondary)",
                maxWidth: 480,
              }}
            >
              {/* 1단계: 확인 다이얼로그 */}
              {!showSuggestedDevicePanel && (
                <>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
                    장비를 자동으로 파싱하지 못했습니다. 기기를 검색하시겠습니까?
                  </div>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button
                      className="action-button regenerate-button"
                      onClick={() => setShowSuggestedDevicePanel(true)}
                      type="button"
                    >
                      1. 예
                    </button>
                    <button
                      className="action-button"
                      onClick={() => setPendingRegeneration(null)}
                      type="button"
                    >
                      2. 아니오
                    </button>
                  </div>
                </>
              )}

              {/* 2단계: 기기 목록 */}
              {showSuggestedDevicePanel && isLoadingSuggestedDevices && (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Spin size="small" />
                  <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                    장비 목록을 불러오는 중입니다...
                  </span>
                </div>
              )}
              {showSuggestedDevicePanel && !isLoadingSuggestedDevices && suggestedDeviceError && (
                <Alert type="warning" showIcon message={suggestedDeviceError} />
              )}
              {showSuggestedDevicePanel && !isLoadingSuggestedDevices && !suggestedDeviceError && suggestedDevices.length > 0 && (
                <>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
                    아래에서 장비 번호를 입력해 주세요.
                  </div>
                  <div
                    style={{
                      maxHeight: 300,
                      overflowY: "auto",
                      display: "flex",
                      flexDirection: "column",
                      gap: 4,
                    }}
                  >
                    {suggestedDevices.map((device, idx) => (
                      <div
                        key={device.name}
                        style={{
                          fontSize: 13,
                          padding: "4px 8px",
                          borderRadius: 4,
                          cursor: "pointer",
                          transition: "background-color 0.15s ease",
                        }}
                        onMouseEnter={(e) =>
                          (e.currentTarget.style.backgroundColor = "var(--color-action-bg)")
                        }
                        onMouseLeave={(e) =>
                          (e.currentTarget.style.backgroundColor = "transparent")
                        }
                        onClick={() => handleDeviceSelect(idx)}
                      >
                        <span style={{ fontWeight: 600, color: "var(--color-accent-primary)" }}>
                          {idx + 1}.
                        </span>{" "}
                        {device.name}
                        <span style={{ color: "var(--color-text-secondary)", marginLeft: 8, fontSize: 12 }}>
                          ({device.doc_count.toLocaleString()} 문서)
                        </span>
                      </div>
                    ))}
                  </div>
                  <div
                    style={{
                      marginTop: 8,
                      fontSize: 12,
                      color: "var(--color-text-secondary)",
                    }}
                  >
                    번호를 입력하거나 클릭하여 선택하세요. (0: 취소)
                  </div>
                </>
              )}
            </div>
          )}

          {pendingGuidedSelection && (
            <GuidedSelectionPanel
              question={pendingGuidedSelection.question}
              instruction={pendingGuidedSelection.instruction}
              payload={pendingGuidedSelection.payload}
              onComplete={submitGuidedSelectionFinal}
            />
          )}

          {pendingDeviceSelection && (
            (pendingDeviceSelection.devices && pendingDeviceSelection.devices.length > 0) ||
            (pendingDeviceSelection.docTypes && pendingDeviceSelection.docTypes.length > 0)
          ) && (
            <DeviceSelectionPanel
              question={pendingDeviceSelection.question}
              devices={pendingDeviceSelection.devices ?? []}
              docTypes={pendingDeviceSelection.docTypes ?? []}
              instruction={pendingDeviceSelection.instruction}
              onSelect={submitDeviceSelection}
            />
          )}

          {/* Review panel moved to right sidebar */}

          {error && (
            <Alert
              type="error"
              message={error}
              showIcon
              closable
              style={{ margin: "0 0 16px" }}
            />
          )}

          <InputArea>
            <ChatInput
              onSend={handleSend}
              onStop={stop}
              isStreaming={isStreaming}
              placeholder={inputPlaceholder}
              disabled={isLoadingSession || !!pendingGuidedSelection}
            />
          </InputArea>
        </ChatContainer>
      </main>
    </div>
  );
}
