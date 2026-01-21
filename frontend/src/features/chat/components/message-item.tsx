import { useState } from "react";
import { CopyOutlined, LikeOutlined, DislikeOutlined, CheckOutlined } from "@ant-design/icons";
import { Message, FeedbackRating } from "../types";
import { MarkdownContent } from "./markdown-content";
import { Collapse } from "antd";

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
  onFeedback?: (payload: {
    messageId: string;
    sessionId?: string;
    turnId?: number;
    rating: FeedbackRating;
    reason?: string | null;
  }) => void;
};

export function MessageItem({ message, isStreaming, onFeedback }: MessageItemProps) {
  const [copied, setCopied] = useState(false);
  const [showReasonInput, setShowReasonInput] = useState(false);
  const [reasonText, setReasonText] = useState("");
  const [reasonError, setReasonError] = useState<string | null>(null);

  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const rating = message.feedback?.rating ?? null;
  const canFeedback = Boolean(message.sessionId && message.turnId);

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
    if (!canFeedback) return;
    setShowReasonInput(false);
    setReasonError(null);
    onFeedback?.({
      messageId: message.id,
      sessionId: message.sessionId,
      turnId: message.turnId,
      rating: "up",
    });
  };

  const handleDislike = () => {
    if (!canFeedback) return;
    const existingReason = message.feedback?.rating === "down" ? message.feedback.reason ?? "" : "";
    setReasonText(existingReason);
    setReasonError(null);
    setShowReasonInput(true);
  };

  const handleReasonSubmit = () => {
    if (!canFeedback) return;
    const reason = reasonText.trim();
    if (!reason) {
      setReasonError("불만족 사유를 입력해 주세요.");
      return;
    }
    onFeedback?.({
      messageId: message.id,
      sessionId: message.sessionId,
      turnId: message.turnId,
      rating: "down",
      reason,
    });
    setShowReasonInput(false);
    setReasonError(null);
  };

  const handleReasonCancel = () => {
    setShowReasonInput(false);
    setReasonError(null);
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
                    label: `확장 문서/참고 문서 (${message.retrievedDocs.length})`,
                    children: (
                      <div className="reference-list">
                        {message.retrievedDocs.map((doc, idx) => {
                          const pageNumbers = doc.expanded_pages && doc.expanded_pages.length > 0
                            ? doc.expanded_pages
                            : doc.page !== null && doc.page !== undefined
                              ? [doc.page]
                              : [];
                          const pageUrls = doc.expanded_page_urls && doc.expanded_page_urls.length > 0
                            ? doc.expanded_page_urls
                            : doc.page_image_url
                              ? [doc.page_image_url]
                              : doc.id && pageNumbers.length > 0
                                ? pageNumbers.map((p) => `/api/assets/docs/${doc.id}/pages/${p}`)
                                : [];

                          return (
                            <div key={doc.id || idx} className="reference-item" style={{ marginBottom: 16 }}>
                              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                                {doc.title || `Document ${idx + 1}`}
                                {pageNumbers.length > 0 && (
                                  <span style={{ fontWeight: 400, marginLeft: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>
                                    {pageNumbers.length === 1
                                      ? `p.${pageNumbers[0]}`
                                      : `p.${pageNumbers[0]}-${pageNumbers[pageNumbers.length - 1]}`}
                                  </span>
                                )}
                              </div>
                              {/* Show page images if available */}
                              {pageUrls.length > 0 && (
                                <div
                                  className="reference-images-container"
                                  style={{
                                    display: "flex",
                                    flexWrap: "wrap",
                                    gap: 8,
                                    marginBottom: 8,
                                  }}
                                >
                                  {pageUrls.map((url, pageIdx) => (
                                    <div key={pageIdx} className="reference-image-wrapper">
                                      <img
                                        src={url}
                                        alt={`${doc.title || "Document"} page ${pageNumbers[pageIdx] || pageIdx + 1}`}
                                        style={{
                                          maxWidth: pageUrls.length > 1 ? 150 : "100%",
                                          maxHeight: pageUrls.length > 1 ? 200 : 300,
                                          borderRadius: 4,
                                          border: "1px solid var(--color-border)",
                                          cursor: "pointer",
                                        }}
                                        title={`페이지 ${pageNumbers[pageIdx] || pageIdx + 1}`}
                                        onError={(e) => {
                                          e.currentTarget.style.display = "none";
                                        }}
                                      />
                                    </div>
                                  ))}
                                </div>
                              )}
                              {/* Always show snippet text */}
                              {doc.snippet && (
                                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: pageUrls.length > 0 ? 8 : 0 }}>
                                  <MarkdownContent content={preprocessSnippet(doc.snippet)} />
                                </div>
                              )}
                              {(doc.score !== null && doc.score !== undefined) && (
                                <div style={{ fontSize: 12, opacity: 0.6 }}>
                                  score: {doc.score.toFixed(3)} {doc.score_percent ? `(${doc.score_percent}%)` : ""}
                                </div>
                              )}
                            </div>
                          );
                        })}
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
                className={`action-button ${rating === "up" ? "active" : ""}`}
                onClick={handleLike}
                title="만족"
                disabled={!canFeedback}
              >
                <LikeOutlined />
              </button>
              <button
                className={`action-button ${rating === "down" ? "active" : ""}`}
                onClick={handleDislike}
                title="불만족"
                disabled={!canFeedback}
              >
                <DislikeOutlined />
              </button>
            </div>
          )}

          {isAssistant && showReasonInput && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, marginBottom: 6, color: "var(--color-text-secondary)" }}>
                불만족 사유
              </div>
              <textarea
                value={reasonText}
                onChange={(e) => {
                  setReasonText(e.target.value);
                  if (reasonError) setReasonError(null);
                }}
                rows={3}
                placeholder="어떤 점이 불만족이었는지 입력해 주세요."
                style={{
                  width: "100%",
                  borderRadius: 6,
                  border: "1px solid var(--color-border)",
                  padding: "8px 10px",
                  fontSize: 12,
                  resize: "vertical",
                  background: "var(--color-bg-secondary)",
                  color: "var(--color-text-primary)",
                }}
              />
              {reasonError && (
                <div style={{ marginTop: 6, fontSize: 12, color: "var(--color-danger, #c0392b)" }}>
                  {reasonError}
                </div>
              )}
              <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
                <button className="action-button" onClick={handleReasonSubmit} disabled={!canFeedback}>
                  저장
                </button>
                <button className="action-button" onClick={handleReasonCancel}>
                  취소
                </button>
              </div>
            </div>
          )}

          {isAssistant && message.feedback?.rating && (
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 6 }}>
              만족도: {message.feedback.rating === "up" ? "만족" : "불만족"}
            </div>
          )}
          {isAssistant && message.feedback?.rating === "down" && message.feedback?.reason && (
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 4 }}>
              불만족 사유: {message.feedback.reason}
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
