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
  const { messages, send, stop, isStreaming, error, reset, editAndResend } = useChatSession();

  const handleSend = async (text: string) => {
    await send({ text });
  };

  const handleEdit = async (id: string, newContent: string) => {
    await editAndResend(id, newContent);
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
                  onEdit={handleEdit}
                />
              ))
            )}
          </MessageList>

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
            />
          </InputArea>
        </ChatContainer>
      </main>
    </div>
  );
}
