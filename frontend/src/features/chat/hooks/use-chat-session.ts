import { useCallback, useMemo, useRef, useState } from "react";
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
  const controllerRef = useRef<{ close: () => void } | null>(null);

  const appendMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const updateLastAssistant = useCallback((text: string) => {
    setMessages((prev) => {
      const clone = [...prev];
      for (let i = clone.length - 1; i >= 0; i -= 1) {
        if (clone[i].role === "assistant") {
          clone[i] = { ...clone[i], content: text };
          return clone;
        }
      }
      clone.push({
        id: nanoid(),
        role: "assistant",
        content: text,
      });
      return clone;
    });
  }, []);

  const stop = useCallback(() => {
    controllerRef.current?.close();
    controllerRef.current = null;
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

      controllerRef.current = await sendChatMessage(
        { conversationId, message: text },
        {
          onMessage: (delta) => updateLastAssistant(delta),
          onDone: () => setIsStreaming(false),
          onError: (err) => {
            setError(err instanceof Error ? err.message : "Streaming error");
            setIsStreaming(false);
          },
        }
      );
    },
    [appendMessage, stop, updateLastAssistant]
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
