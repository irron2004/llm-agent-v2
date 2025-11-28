import { useRef, useEffect, PropsWithChildren } from "react";

type ChatContainerProps = PropsWithChildren<{
  title?: string;
  subtitle?: string;
  onScrollToBottom?: () => void;
}>;

export function ChatContainer({ children, title, subtitle }: ChatContainerProps) {
  return (
    <div className="chat-container">
      {(title || subtitle) && (
        <div className="chat-container-header">
          {title && <h2 className="chat-container-title">{title}</h2>}
          {subtitle && <span className="chat-container-subtitle">{subtitle}</span>}
        </div>
      )}
      {children}
    </div>
  );
}

type MessageListProps = PropsWithChildren<{
  autoScrollToBottom?: boolean;
}>;

export function MessageList({ children, autoScrollToBottom = true }: MessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      isAtBottomRef.current = scrollHeight - scrollTop - clientHeight < 50;
    };

    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    if (autoScrollToBottom && isAtBottomRef.current) {
      const container = containerRef.current;
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    }
  }, [children, autoScrollToBottom]);

  return (
    <div ref={containerRef} className="chat-messages">
      {children}
    </div>
  );
}

type InputAreaProps = PropsWithChildren;

export function InputArea({ children }: InputAreaProps) {
  return <div className="chat-input-area">{children}</div>;
}
