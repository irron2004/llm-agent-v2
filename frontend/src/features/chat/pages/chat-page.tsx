import { useCallback, useEffect } from "react";
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
} from "../components";
import { Alert, Spin } from "antd";

export default function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const sessionParam = searchParams.get("session");

  const { refresh: refreshHistory } = useChatHistoryContext();
  const {
    setPendingReview,
    setIsStreaming,
    registerSubmitHandlers,
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
    inputPlaceholder,
    pendingReview,
    pendingDeviceSelection,
    submitReview,
    submitSearchQueries,
    submitDeviceSelection,
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
    };
    window.addEventListener("pe-agent:new-chat", handleNewChat);
    return () => {
      window.removeEventListener("pe-agent:new-chat", handleNewChat);
    };
  }, [reset]);

  const handleSend = async (text: string) => {
    await send({ text });
  };

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
                  />
                ))
              )}
            </MessageList>
          )}

          {/* Device selection panel */}
          {pendingDeviceSelection && pendingDeviceSelection.devices && pendingDeviceSelection.devices.length > 0 && (
            <DeviceSelectionPanel
              question={pendingDeviceSelection.question}
              devices={pendingDeviceSelection.devices}
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
              disabled={isLoadingSession}
            />
          </InputArea>
        </ChatContainer>
      </main>
    </div>
  );
}
