import { useEffect, useMemo, useState } from "react";
import { useChatSession } from "../hooks/use-chat-session";
import {
  ChatContainer,
  MessageList,
  InputArea,
  MessageItem,
  ChatInput,
} from "../components";
import { ThemeToggle } from "../../../components/theme-toggle";
import { Alert } from "antd";

export default function ChatPage() {
  const {
    messages,
    send,
    stop,
    isStreaming,
    error,
    reset,
    inputPlaceholder,
    pendingReview,
    submitReview,
  } = useChatSession();
  const [selectedRanks, setSelectedRanks] = useState<number[]>([]);

  useEffect(() => {
    if (pendingReview) {
      setSelectedRanks(pendingReview.docs.map((doc) => doc.rank));
    } else {
      setSelectedRanks([]);
    }
  }, [pendingReview?.threadId]);

  const allSelected = useMemo(() => {
    if (!pendingReview || pendingReview.docs.length === 0) return false;
    return selectedRanks.length === pendingReview.docs.length;
  }, [pendingReview, selectedRanks]);

  const toggleDoc = (rank: number) => {
    setSelectedRanks((prev) =>
      prev.includes(rank) ? prev.filter((id) => id !== rank) : [...prev, rank]
    );
  };

  const toggleAll = () => {
    if (!pendingReview) return;
    if (allSelected) {
      setSelectedRanks([]);
    } else {
      setSelectedRanks(pendingReview.docs.map((doc) => doc.rank));
    }
  };

  const handleReviewSubmit = () => {
    if (!pendingReview) return;
    const selectedDocIds = pendingReview.docs
      .filter((doc) => selectedRanks.includes(doc.rank))
      .map((doc) => doc.docId)
      .filter(Boolean);
    submitReview({ docIds: selectedDocIds, ranks: selectedRanks });
  };

  const handleSend = async (text: string) => {
    await send({ text });
  };

  return (
    <div className="chat-layout">
      {/* Header */}
      <header className="chat-header">
        <h1 className="chat-header-title">PE Agent</h1>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button className="action-button" onClick={reset}>
            New Chat
          </button>
          <ThemeToggle />
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="chat-main">
        <ChatContainer>
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

          {pendingReview && (
            <div className="review-panel">
              <div className="review-header">
                <div className="review-title">검색 결과 확인</div>
                <div className="review-subtitle">{pendingReview.instruction}</div>
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
                            문서 {doc.rank ?? idx + 1}
                            {doc.docId ? ` · ${doc.docId}` : ""}
                          </div>
                          <div className="review-doc-content">{doc.content}</div>
                        </div>
                      </label>
                    ))}
                  </div>
                </>
              )}

              <div className="review-actions">
                <button
                  className="action-button"
                  onClick={handleReviewSubmit}
                  disabled={
                    isStreaming || pendingReview.docs.length === 0 || selectedRanks.length === 0
                  }
                >
                  선택 문서로 답변
                </button>
              </div>
            </div>
          )}

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
            />
          </InputArea>
        </ChatContainer>
      </main>
    </div>
  );
}
