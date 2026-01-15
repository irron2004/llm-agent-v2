import { useState, useEffect, useCallback, useMemo } from "react";
import { fetchSessions, deleteSession as deleteSessionApi, hideSession as hideSessionApi } from "../api";
import { SessionListItem } from "../types";

export interface ChatHistoryItem {
  id: string;
  title: string;
  preview: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export function useChatHistory() {
  const [history, setHistory] = useState<ChatHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Fetch sessions from API on mount
  const loadSessions = useCallback(async () => {
    console.log("[useChatHistory] Loading sessions from API...");
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetchSessions(50, 0);
      console.log("[useChatHistory] API response:", response);
      const items: ChatHistoryItem[] = response.sessions.map(
        (session: SessionListItem) => ({
          id: session.id,
          title: session.title,
          preview: session.preview,
          createdAt: session.createdAt,
          updatedAt: session.updatedAt,
          messageCount: session.turnCount,
        })
      );
      setHistory(items);
    } catch (err) {
      console.error("[useChatHistory] Error loading sessions:", err);
      setError(err instanceof Error ? err : new Error("Failed to load sessions"));
      // Keep existing history on error
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const deleteChat = useCallback(
    async (id: string) => {
      try {
        await deleteSessionApi(id);
        setHistory((prev) => prev.filter((item) => item.id !== id));
      } catch (err) {
        console.error("Failed to delete session:", err);
        throw err;
      }
    },
    []
  );

  // Soft delete - hides from UI but keeps in DB
  const hideChat = useCallback(
    async (id: string) => {
      try {
        await hideSessionApi(id);
        setHistory((prev) => prev.filter((item) => item.id !== id));
      } catch (err) {
        console.error("Failed to hide session:", err);
        throw err;
      }
    },
    []
  );

  const getChat = useCallback(
    (id: string) => {
      return history.find((item) => item.id === id) || null;
    },
    [history]
  );

  // Refresh the history list (call after saving a new turn)
  const refresh = useCallback(() => {
    loadSessions();
  }, [loadSessions]);

  return useMemo(
    () => ({
      history,
      isLoading,
      error,
      deleteChat,
      hideChat,
      getChat,
      refresh,
    }),
    [history, isLoading, error, deleteChat, hideChat, getChat, refresh]
  );
}
