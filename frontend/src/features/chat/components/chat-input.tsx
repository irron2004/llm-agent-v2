import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { SendOutlined, StopOutlined } from "@ant-design/icons";

type ChatInputProps = {
  onSend: (message: string) => void;
  onStop?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  placeholder?: string;
  onDisabledClick?: () => void;  // 비활성화 상태에서 클릭 시 호출
};

export function ChatInput({
  onSend,
  onStop,
  isStreaming = false,
  disabled = false,
  placeholder = "Enter your message...",
  onDisabledClick,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [value]);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled || isStreaming) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleWrapperClick = () => {
    if (disabled && onDisabledClick) {
      onDisabledClick();
    }
  };

  return (
    <div className="input-wrapper" onClick={handleWrapperClick}>
      <textarea
        ref={textareaRef}
        className="input-textarea"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        style={disabled ? { pointerEvents: "none" } : undefined}
      />
      {isStreaming ? (
        <button
          className="send-button"
          onClick={onStop}
          style={{ backgroundColor: "var(--color-error)" }}
        >
          <StopOutlined />
          <span>Stop</span>
        </button>
      ) : (
        <button
          className="send-button"
          onClick={handleSend}
          disabled={!value.trim() || disabled}
        >
          <SendOutlined />
          <span>Send</span>
        </button>
      )}
    </div>
  );
}
