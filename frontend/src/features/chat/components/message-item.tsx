import { useState } from "react";
import { CopyOutlined, LikeOutlined, DislikeOutlined, CheckOutlined } from "@ant-design/icons";
import { Message } from "../types";
import { MarkdownContent } from "./markdown-content";
import { Collapse } from "antd";

type MessageItemProps = {
  message: Message;
  isStreaming?: boolean;
  onLike?: (id: string) => void;
  onDislike?: (id: string) => void;
};

export function MessageItem({ message, isStreaming, onLike, onDislike }: MessageItemProps) {
  const [copied, setCopied] = useState(false);
  const [liked, setLiked] = useState(false);
  const [disliked, setDisliked] = useState(false);

  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const handleLike = () => {
    setLiked(!liked);
    setDisliked(false);
    onLike?.(message.id);
  };

  const handleDislike = () => {
    setDisliked(!disliked);
    setLiked(false);
    onDislike?.(message.id);
  };

  return (
    <div className="message-item">
      <div className={`message-content ${isUser ? "user" : ""}`}>
        <div className={`message-avatar ${isUser ? "user" : "assistant"}`}>
          {isUser ? "U" : "A"}
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
                    label: `검색 문서 (${message.retrievedDocs.length})`,
                    children: (
                      <div className="reference-list">
                        {message.retrievedDocs.map((doc, idx) => (
                          <div key={doc.id || idx} className="reference-item" style={{ marginBottom: 8 }}>
                            <div style={{ fontWeight: 600 }}>{doc.title || `Document ${idx + 1}`}</div>
                            <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
                              {doc.snippet || ""}
                            </div>
                            {(doc.score !== null && doc.score !== undefined) && (
                              <div style={{ fontSize: 12, opacity: 0.6 }}>
                                score: {doc.score.toFixed(3)} {doc.score_percent ? `(${doc.score_percent}%)` : ""}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ),
                  },
                ]}
              />
            </div>
          )}

          {/* Execution logs (collapsible) */}
          {isAssistant && message.logs && message.logs.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <Collapse
                size="small"
                defaultActiveKey={isStreaming ? ["logs"] : undefined}
                items={[
                  {
                    key: "logs",
                    label: `실행 로그 (${message.logs.length})`,
                    children: (
                      <pre
                        style={{
                          margin: 0,
                          maxHeight: 240,
                          overflow: "auto",
                          fontSize: 12,
                          background: "var(--color-code-bg)",
                          color: "var(--color-code-text)",
                          borderRadius: 6,
                          padding: 12,
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-word",
                        }}
                      >
                        {message.logs.join("\n")}
                      </pre>
                    ),
                  },
                ]}
              />
            </div>
          )}

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
              <button
                className={`action-button ${liked ? "active" : ""}`}
                onClick={handleLike}
                title="Like"
              >
                <LikeOutlined />
              </button>
              <button
                className={`action-button ${disliked ? "active" : ""}`}
                onClick={handleDislike}
                title="Dislike"
              >
                <DislikeOutlined />
              </button>
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
    </div>
  );
}
