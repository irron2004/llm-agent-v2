import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { SendOutlined, StopOutlined } from "@ant-design/icons";

type ChatInputProps = {
  onSend: (message: string) => void;
  onStop?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  placeholder?: string;
};

export function ChatInput({
  onSend,
  onStop,
  isStreaming = false,
  disabled = false,
  placeholder = "메시지를 입력하세요...",
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

  return (
    <div className="input-wrapper">
      <textarea
        ref={textareaRef}
        className="input-textarea"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
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
