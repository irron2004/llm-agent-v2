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
  // [LEGACY] DeviceSelectionPanel - 채팅 화면 내 기기/문서 선택 패널 (사용 안 함)
  // DeviceSelectionPanel,
} from "../components";
import type { RegeneratePayload } from "../components/message-item";
import { Alert, Spin } from "antd";

export default function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const sessionParam = searchParams.get("session");

  const { refresh: refreshHistory } = useChatHistoryContext();
  const {
    setPendingReview,
    setIsStreaming,
    registerSubmitHandlers,
    setPendingRegeneration,
    registerRegenerationHandlers,
  } = useChatReview();

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
    branchSessionFromMessageId,
    markNextTurnEdited,
    inputPlaceholder,
    pendingReview,
    // [LEGACY] 채팅 화면 내 기기/문서 선택 패널 (사용 안 함)
    // pendingDeviceSelection,
    submitReview,
    submitSearchQueries,
    // submitDeviceSelection,
    submitFeedback,
    submitDetailedFeedback,
  } = useChatSession({ onTurnSaved: handleTurnSaved });

  // 기기 선택 활성화 상태 (ESC로 닫을 수 있음)
  const [isDeviceSelectionDismissed, setIsDeviceSelectionDismissed] = useState(false);

  // 마지막 assistant 메시지에 suggestedDevices가 있는지 확인
  const lastAssistantMessage = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") {
        return messages[i];
      }
    }
    return null;
  }, [messages]);

  const hasDeviceSuggestions = !isStreaming &&
    lastAssistantMessage?.suggestedDevices &&
    lastAssistantMessage.suggestedDevices.length > 0 &&
    !isDeviceSelectionDismissed;

  // 새 메시지가 오면 dismiss 상태 리셋
  useEffect(() => {
    setIsDeviceSelectionDismissed(false);
  }, [messages.length]);

  const handleDeviceDismiss = useCallback(() => {
    setIsDeviceSelectionDismissed(true);
  }, []);

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

  const submitRegeneration = useCallback((payload: {
    originalQuery: string;
    searchQueries: string[];
    selectedDevices: string[];
    selectedDocTypes: string[];
    selectedDocIds: string[];
  }) => {
    setPendingRegeneration(null);
    send({
      text: payload.originalQuery,
      askDeviceSelection: false,  // 이미 기기/문서 선택 완료
      overrides: {
        filterDevices: payload.selectedDevices,
        filterDocTypes: payload.selectedDocTypes,
        searchQueries: payload.searchQueries,
        selectedDocIds: payload.selectedDocIds,
        autoParse: false,
      },
    });
  }, [send, setPendingRegeneration]);

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

  const handleSend = async (text: string) => {
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
    });
  }, [setPendingRegeneration]);

  // Handle device selection from suggested devices
  const handleDeviceSelect = useCallback((messageId: string, deviceName: string) => {
    // Find the original query for this assistant message
    let originalQuery = "";
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === "user" && i + 1 < messages.length && messages[i + 1].id === messageId) {
        originalQuery = messages[i].content;
        break;
      }
    }
    if (!originalQuery) return;

    // 원본 쿼리 + 기기 필터로 재검색
    // MQ는 원본 쿼리 기반으로 다시 생성됨
    send({
      text: `[${deviceName}] ${originalQuery}`,
      overrides: {
        filterDevices: [deviceName],
      },
    });
  }, [messages, send]);

  const handleEditAndResend = useCallback(async (payload: { messageId: string; content: string }) => {
    const trimmed = payload.content.trim();
    if (!trimmed) return;
    const newSessionId = await branchSessionFromMessageId(payload.messageId);
    if (!newSessionId) return;
    markNextTurnEdited();
    send({ text: trimmed });
  }, [branchSessionFromMessageId, markNextTurnEdited, send]);

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
                    onEditAndResend={msg.role === "user" ? handleEditAndResend : undefined}
                    onDeviceSelect={msg.role === "assistant" ? handleDeviceSelect : undefined}
                    onDeviceDismiss={handleDeviceDismiss}
                    isDeviceSelectionActive={hasDeviceSuggestions && msg.id === lastAssistantMessage?.id}
                    originalQuery={msg.role === "assistant" ? getOriginalQuery(msg.id) : undefined}
                  />
                ))
              )}
            </MessageList>
          )}

          {/* Review panel moved to right sidebar */}

          {/* [LEGACY] Device selection panel - 채팅 화면 내 기기/문서 선택 패널 (사용 안 함)
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
          */}

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
              placeholder={hasDeviceSuggestions ? "기기를 선택하거나 ESC를 눌러주세요" : inputPlaceholder}
              disabled={isLoadingSession || hasDeviceSuggestions}
            />
          </InputArea>
        </ChatContainer>
      </main>
    </div>
  );
}
