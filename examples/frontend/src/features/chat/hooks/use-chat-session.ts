import { useCallback, useMemo, useRef, useState } from "react";
import { nanoid } from "nanoid";
import { sendChatMessage, FinalResult } from "../api";
import { Message, SuggestedDevice } from "../types";

type SendOptions = {
  text: string;
};

export function useChatSession() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<{ close: () => void } | null>(null);
  // 세션 ID: 대화 시작 시 생성, reset 시 갱신
  const sessionIdRef = useRef<string>(nanoid());

  // Device suggestion 관련 상태
  const lastQueryRef = useRef<string>("");
  const lastSuggestedDevicesRef = useRef<SuggestedDevice[]>([]);

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

  const updateLastAssistantSuggestions = useCallback((suggestedDevices: SuggestedDevice[]) => {
    setMessages((prev) => {
      const clone = [...prev];
      for (let i = clone.length - 1; i >= 0; i -= 1) {
        if (clone[i].role === "assistant") {
          clone[i] = { ...clone[i], suggestedDevices };
          return clone;
        }
      }
      return clone;
    });
    // 상태 저장
    lastSuggestedDevicesRef.current = suggestedDevices;
  }, []);

  const stop = useCallback(() => {
    controllerRef.current?.close();
    controllerRef.current = null;
    setIsStreaming(false);
  }, []);

  const sendWithOptions = useCallback(
    async (message: string, filterDevices?: string[]) => {
      setIsStreaming(true);
      const sessionId = sessionIdRef.current;

      controllerRef.current = await sendChatMessage(
        { message, sessionId, filterDevices },
        {
          onMessage: (delta) => updateLastAssistant(delta),
          onDone: () => setIsStreaming(false),
          onError: (err) => {
            setError(err instanceof Error ? err.message : "Streaming error");
            setIsStreaming(false);
          },
          onFinalResult: (result: FinalResult) => {
            if (result.suggestedDevices && result.suggestedDevices.length > 0) {
              updateLastAssistantSuggestions(result.suggestedDevices);
            }
          },
        }
      );
    },
    [updateLastAssistant, updateLastAssistantSuggestions]
  );

  const send = useCallback(
    async ({ text }: SendOptions) => {
      stop();
      setError(null);

      // 숫자만 입력인지 확인
      const numMatch = text.trim().match(/^\d+$/);
      if (numMatch && lastSuggestedDevicesRef.current.length > 0) {
        const index = parseInt(text.trim()) - 1;
        if (index >= 0 && index < lastSuggestedDevicesRef.current.length) {
          // 장비 선택: 직전 질문을 해당 장비로 재검색
          const selectedDevice = lastSuggestedDevicesRef.current[index].name;
          const originalQuery = lastQueryRef.current;

          // UI에는 숫자 그대로 표시
          appendMessage({
            id: nanoid(),
            role: "user",
            content: text,
          });

          // 재검색 (직전 질문 + 장비 필터)
          await sendWithOptions(originalQuery, [selectedDevice]);
          return;
        }
      }

      // 일반 질문 처리
      lastQueryRef.current = text; // 직전 질문 저장
      lastSuggestedDevicesRef.current = []; // 이전 추천 초기화

      appendMessage({
        id: nanoid(),
        role: "user",
        content: text,
      });

      await sendWithOptions(text);
    },
    [appendMessage, stop, sendWithOptions]
  );

  const reset = useCallback(() => {
    stop();
    setMessages([]);
    setError(null);
    // 새 세션 시작
    sessionIdRef.current = nanoid();
    // 추천 상태 초기화
    lastQueryRef.current = "";
    lastSuggestedDevicesRef.current = [];
  }, [stop]);

  const editAndResend = useCallback(
    async (messageId: string, newContent: string) => {
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

      // 직전 질문 저장
      lastQueryRef.current = newContent;
      lastSuggestedDevicesRef.current = [];

      // 새 대화 요청 전송
      await sendWithOptions(newContent);
    },
    [messages, stop, sendWithOptions]
  );

  // 장비 선택 핸들러 (버튼 클릭 시)
  const selectDevice = useCallback(
    async (index: number, deviceName: string) => {
      if (lastQueryRef.current && lastSuggestedDevicesRef.current.length > 0) {
        stop();
        setError(null);

        // UI에 선택 표시
        appendMessage({
          id: nanoid(),
          role: "user",
          content: `${index}`,
        });

        // 재검색 (직전 질문 + 장비 필터)
        await sendWithOptions(lastQueryRef.current, [deviceName]);
      }
    },
    [appendMessage, stop, sendWithOptions]
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
      selectDevice,
    }),
    [messages, isStreaming, error, send, stop, reset, editAndResend, selectDevice]
  );
}
