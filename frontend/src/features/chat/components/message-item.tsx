import { useState } from "react";
import { CopyOutlined, LikeOutlined, DislikeOutlined, CheckOutlined } from "@ant-design/icons";
import { Message } from "../types";
import { MarkdownContent } from "./markdown-content";

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

          {/* Reference section */}
          {message.reference && message.reference.chunks.length > 0 && (
            <div className="reference-section">
              <div className="reference-title">
                <span>References ({message.reference.chunks.length})</span>
              </div>
              <div className="reference-list">
                {message.reference.chunks.map((chunk, idx) => (
                  <div key={chunk.id || idx} className="reference-item">
                    <span>{chunk.documentName || `Document ${idx + 1}`}</span>
                    {chunk.similarity !== undefined && (
                      <span style={{ opacity: 0.6 }}>
                        {(chunk.similarity * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                ))}
              </div>
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
