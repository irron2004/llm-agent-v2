import { useCallback, useMemo, useState } from "react";
import { nanoid } from "nanoid";
import { sendChatMessage } from "../api";
import { Message } from "../types";

type SendOptions = {
  conversationId?: string;
  text: string;
};

export function useChatSession() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const appendMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const stop = useCallback(() => {
    setIsStreaming(false);
  }, []);

  const send = useCallback(
    async ({ conversationId, text }: SendOptions) => {
      stop();
      setError(null);
      appendMessage({
        id: nanoid(),
        role: "user",
        content: text,
      });
      setIsStreaming(true);

      try {
        const res = await sendChatMessage({ message: text });
        setMessages((prev) => [
          ...prev,
          {
            id: nanoid(),
            role: "assistant",
            content: res.answer || "",
            retrievedDocs: res.retrieved_docs || [],
            rawAnswer: JSON.stringify(res.metadata || {}, null, 2),
          },
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "요청 실패");
      } finally {
        setIsStreaming(false);
      }
    },
    [appendMessage, stop]
  );

  const reset = useCallback(() => {
    stop();
    setMessages([]);
    setError(null);
  }, [stop]);

  return useMemo(
    () => ({
    messages,
    isStreaming,
    error,
    send,
    stop,
      reset,
    }),
    [messages, isStreaming, error, send, stop, reset]
  );
}
