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
  addLog: (messageId: string, content: string, node?: string | null) => void;
  clearLogs: () => void;
}

const ChatLogsContext = createContext<ChatLogsContextValue | undefined>(undefined);

export function ChatLogsProvider({ children }: { children: ReactNode }) {
  const [logs, setLogs] = useState<ChatLogEntry[]>([]);

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
  }, []);

  return (
    <ChatLogsContext.Provider value={{ logs, addLog, clearLogs }}>
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
