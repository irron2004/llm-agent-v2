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

  const editAndResend = useCallback(
    async (messageId: string, newContent: string, conversationId?: string) => {
      stop();
      setError(null);

      // 해당 메시지의 인덱스 찾기
      const messageIndex = messages.findIndex((msg) => msg.id === messageId);
      if (messageIndex === -1) return;

      // 해당 메시지 이후의 모든 메시지 삭제하고, 해당 메시지 내용 업데이트
      setMessages((prev) => {
        const updated = prev.slice(0, messageIndex + 1);
        updated[messageIndex] = { ...updated[messageIndex], content: newContent };
        return updated;
      });

      // 새 대화 요청 전송
      setIsStreaming(true);
      controllerRef.current = await sendChatMessage(
        { conversationId, message: newContent },
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
    [messages, stop, updateLastAssistant]
  );

  return useMemo(
    () => ({
      messages,
      isStreaming,
      error,
      send,
      stop,
      reset,
      editAndResend,
    }),
    [messages, isStreaming, error, send, stop, reset, editAndResend]
  );
}
