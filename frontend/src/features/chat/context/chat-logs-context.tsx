import { createContext, useContext, useState, useCallback, ReactNode } from "react";

export interface ChatLogEntry {
  id: string;
  messageId: string;
  timestamp: number;
  content: string;
  node?: string | null;
}

interface ChatLogsContextValue {
  logs: ChatLogEntry[];
  activeMessageId: string | null;
  addLog: (messageId: string, content: string, node?: string | null) => void;
  clearLogs: () => void;
  setActiveMessageId: (messageId: string | null) => void;
}

const ChatLogsContext = createContext<ChatLogsContextValue | undefined>(undefined);

export function ChatLogsProvider({ children }: { children: ReactNode }) {
  const [logs, setLogs] = useState<ChatLogEntry[]>([]);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);

  const addLog = useCallback((messageId: string, content: string, node?: string | null) => {
    setLogs((prev) => [
      ...prev,
      {
        id: `${messageId}-${Date.now()}-${Math.random()}`,
        messageId,
        timestamp: Date.now(),
        content,
        node,
      },
    ]);
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
    setActiveMessageId(null);
  }, []);

  return (
    <ChatLogsContext.Provider value={{ logs, activeMessageId, addLog, clearLogs, setActiveMessageId }}>
      {children}
    </ChatLogsContext.Provider>
  );
}

export function useChatLogs() {
  const context = useContext(ChatLogsContext);
  if (context === undefined) {
    throw new Error("useChatLogs must be used within ChatLogsProvider");
  }
  return context;
}
