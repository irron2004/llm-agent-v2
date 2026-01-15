import { createContext, useContext, PropsWithChildren } from "react";
import { useChatHistory, ChatHistoryItem } from "../hooks/use-chat-history";

interface ChatHistoryContextValue {
  history: ChatHistoryItem[];
  isLoading: boolean;
  error: Error | null;
  deleteChat: (id: string) => Promise<void>;
  getChat: (id: string) => ChatHistoryItem | null;
  refresh: () => void;
}

const ChatHistoryContext = createContext<ChatHistoryContextValue | null>(null);

export function ChatHistoryProvider({ children }: PropsWithChildren) {
  const chatHistory = useChatHistory();

  return (
    <ChatHistoryContext.Provider value={chatHistory}>
      {children}
    </ChatHistoryContext.Provider>
  );
}

export function useChatHistoryContext() {
  const context = useContext(ChatHistoryContext);
  if (!context) {
    throw new Error("useChatHistoryContext must be used within a ChatHistoryProvider");
  }
  return context;
}
