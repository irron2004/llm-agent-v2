import { useState } from "react";
import { CopyOutlined, LikeOutlined, DislikeOutlined, CheckOutlined } from "@ant-design/icons";
import { Message } from "../types";
import { MarkdownContent } from "./markdown-content";
import { Collapse } from "antd";

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

  // Format markdown tables that are on single line
  if (processed.includes("|") && /\|\s*:?-{2,}:?\s*\|/.test(processed) && !processed.includes("\n|")) {
    processed = processed.replace(/\s*\|\s*\|\s*/g, " |\n| ");
  }

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
                          <div key={doc.id || idx} className="reference-item" style={{ marginBottom: 12 }}>
                            <div style={{ fontWeight: 600, marginBottom: 4 }}>
                              {doc.title || `Document ${idx + 1}`}
                              {doc.page && <span style={{ fontWeight: 400, marginLeft: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>p.{doc.page}</span>}
                            </div>
                            {doc.page_image_url ? (
                              <div className="reference-image-wrapper" style={{ marginBottom: 8 }}>
                                <img
                                  src={doc.page_image_url}
                                  alt={`${doc.title || "Document"} page ${doc.page || ""}`}
                                  style={{
                                    maxWidth: "100%",
                                    maxHeight: 200,
                                    borderRadius: 4,
                                    border: "1px solid var(--color-border)",
                                  }}
                                  onError={(e) => {
                                    e.currentTarget.style.display = "none";
                                    const fallback = e.currentTarget.nextElementSibling as HTMLElement;
                                    if (fallback) fallback.style.display = "block";
                                  }}
                                />
                                <div style={{ display: "none", fontSize: 12, color: "var(--color-text-secondary)" }}>
                                  <MarkdownContent content={preprocessSnippet(doc.snippet || "")} />
                                </div>
                              </div>
                            ) : (
                              <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
                                <MarkdownContent content={preprocessSnippet(doc.snippet || "")} />
                              </div>
                            )}
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

          {/* Execution logs moved to right sidebar */}

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
