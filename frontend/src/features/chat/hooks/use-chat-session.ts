import { useCallback, useMemo, useRef, useState } from "react";
import { nanoid } from "nanoid";
import { sendChatMessage } from "../api";
import { AgentResponse, Message, ReviewDoc } from "../types";
import { connectSse } from "../../../lib/sse";
import { env } from "../../../config/env";

type SendOptions = {
  conversationId?: string;
  text: string;
  decisionOverride?: unknown;
};

type InterruptKind = "retrieval_review" | "human_review" | "unknown";

type PendingInterrupt = {
  threadId: string;
  question: string;
  instruction: string;
  docs: ReviewDoc[];
  kind: InterruptKind;
  payload?: Record<string, unknown> | null;
};

const APPROVE_TOKENS = ["true", "yes", "y", "ok", "okay", "승인", "확인", "approve"];
const REJECT_TOKENS = ["false", "no", "n", "거절", "reject", "decline"];

const resolveDecision = (text: string): boolean | string => {
  const trimmed = text.trim();
  const lowered = trimmed.toLowerCase();
  if (APPROVE_TOKENS.includes(lowered) || APPROVE_TOKENS.includes(trimmed)) return true;
  if (REJECT_TOKENS.includes(lowered) || REJECT_TOKENS.includes(trimmed)) return false;
  return trimmed;
};

const resolveInterruptKind = (payload?: Record<string, unknown> | null): InterruptKind => {
  if (payload?.type === "retrieval_review") return "retrieval_review";
  if (payload?.type === "human_review") return "human_review";
  return "unknown";
};

const buildInterruptPrompt = (kind: InterruptKind, instruction?: string) => {
  if (kind === "retrieval_review") {
    return "검색 결과가 준비되었습니다. 아래에서 문서를 선택하거나 추가 키워드를 입력해 주세요.";
  }
  if (instruction && instruction.trim()) return instruction.trim();
  return "추가 입력이 필요합니다. 승인/거절 또는 수정 답변을 입력해 주세요.";
};

const normalizeReviewDocs = (payload?: Record<string, unknown> | null): ReviewDoc[] => {
  const raw = payload?.retrieved_docs;
  if (!Array.isArray(raw)) return [];

  return raw.map((doc, index) => {
    const rank = typeof doc?.rank === "number" ? doc.rank : index + 1;
    const docId =
      typeof doc?.doc_id === "string" && doc.doc_id.trim()
        ? doc.doc_id.trim()
        : "";
    const content =
      typeof doc?.content === "string"
        ? doc.content
        : typeof doc?.snippet === "string"
          ? doc.snippet
          : "";
    return {
      docId,
      rank,
      content,
      score: typeof doc?.score === "number" ? doc.score : null,
      metadata: typeof doc?.metadata === "object" ? doc.metadata : null,
    };
  });
};

export function useChatSession() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingInterrupt, setPendingInterrupt] = useState<PendingInterrupt | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const appendMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const updateMessage = useCallback((id: string, updater: (prev: Message) => Message) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? updater(m) : m)));
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  const handleAgentResponse = useCallback(
    (res: AgentResponse, assistantId: string, fallbackQuestion: string) => {
      if (res.interrupted) {
        const threadId = res.thread_id ?? "";
        if (!threadId) {
          setError("thread_id가 없어 검색 결과 확인을 이어갈 수 없습니다.");
        }

        const payload = res.interrupt_payload ?? null;
        const kind = resolveInterruptKind(payload);
        const question =
          typeof payload?.question === "string" && payload.question.trim()
            ? payload.question
            : res.query || fallbackQuestion;
        const instruction =
          typeof payload?.instruction === "string" && payload.instruction.trim()
            ? payload.instruction.trim()
            : "검색 결과를 확인한 뒤 승인/거절/키워드를 입력하세요.";
        const docs = normalizeReviewDocs(payload);

        if (threadId) {
          setPendingInterrupt({
            threadId,
            question,
            instruction,
            docs,
            kind,
            payload,
          });
        }

        updateMessage(assistantId, (m) => ({
          ...m,
          content: buildInterruptPrompt(kind, instruction),
          retrievedDocs: res.retrieved_docs || [],
          rawAnswer: JSON.stringify(res, null, 2),
          currentNode: null,
        }));
        setIsStreaming(false);
        return;
      }

      setPendingInterrupt(null);
      updateMessage(assistantId, (m) => ({
        ...m,
        content: res.answer || "",
        retrievedDocs: res.retrieved_docs || [],
        rawAnswer: JSON.stringify(res, null, 2),
        currentNode: null,
      }));
    },
    [updateMessage, setIsStreaming]
  );

  const send = useCallback(
    async ({ conversationId, text, decisionOverride }: SendOptions) => {
      stop();
      setError(null);
      const pending = pendingInterrupt;
      const isResume = Boolean(pending);
      if (isResume && !pending?.threadId) {
        setError("thread_id가 없어 검색 결과 확인을 이어갈 수 없습니다.");
        return;
      }

      const userId = nanoid();
      const assistantId = nanoid();

      appendMessage({
        id: userId,
        role: "user",
        content: text,
      });

      // Assistant placeholder so the UI shows progress immediately.
      appendMessage({
        id: assistantId,
        role: "assistant",
        content: "처리 중...",
        logs: [],
        currentNode: null,
      });
      setIsStreaming(true);

      try {
        const requestMessage = isResume && pending ? pending.question : text;
        const decision = isResume ? (decisionOverride ?? resolveDecision(text)) : undefined;
        const payload = {
          message: requestMessage,
          ask_user_after_retrieve: true,
          ...(isResume && pending
            ? {
                thread_id: pending.threadId,
                resume_decision: decision,
              }
            : {}),
        };
        const canStream = env.chatPath.endsWith("/run");
        if (!canStream) {
          const res = await sendChatMessage(payload);
          handleAgentResponse(res, assistantId, requestMessage);
          return;
        }

        const controller = new AbortController();
        abortRef.current = controller;

        await connectSse(
          {
            path: `${env.chatPath}/stream`,
            body: payload,
            signal: controller.signal,
          },
          {
            onMessage: (data) => {
              let evt: any;
              try {
                evt = JSON.parse(data);
              } catch {
                return;
              }

              if (evt?.type === "log") {
                updateMessage(assistantId, (m) => {
                  const logs =
                    typeof evt?.message === "string"
                      ? [...(m.logs ?? []), evt.message]
                      : m.logs;
                  let currentNode = m.currentNode ?? null;
                  if (typeof evt?.node === "string") {
                    if (evt?.phase === "start") {
                      currentNode = evt.node;
                    } else if (evt?.phase === "done" && currentNode === evt.node) {
                      currentNode = null;
                    }
                  }
                  return {
                    ...m,
                    logs,
                    currentNode,
                  };
                });
                return;
              }

              if (evt?.type === "error") {
                const detail = typeof evt?.detail === "string" ? evt.detail : "요청 실패";
                setError(detail);
                updateMessage(assistantId, (m) => ({
                  ...m,
                  content: "오류가 발생했습니다.",
                  currentNode: null,
                }));
                return;
              }

              if (evt?.type === "final" && evt?.result) {
                const res = evt.result;
                handleAgentResponse(res, assistantId, requestMessage);
                return;
              }
            },
            onError: (err) => {
              // Abort is expected when user clicks Stop.
              if (err instanceof DOMException && err.name === "AbortError") return;
              setError(err instanceof Error ? err.message : "요청 실패");
            },
            onClose: () => {
              abortRef.current = null;
              setIsStreaming(false);
            },
          }
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "요청 실패");
      } finally {
        setIsStreaming(false);
      }
    },
    [appendMessage, stop, updateMessage, handleAgentResponse, pendingInterrupt]
  );

  const submitReview = useCallback(
    (selection: { docIds: string[]; ranks: number[] }) => {
      if (!pendingInterrupt || pendingInterrupt.kind !== "retrieval_review") return;
      const uniqueIds = Array.from(new Set(selection.docIds)).filter(Boolean);
      const uniqueRanks = Array.from(new Set(selection.ranks)).filter((n) => Number.isFinite(n));
      const label =
        uniqueIds.length > 0
          ? uniqueIds.join(", ")
          : uniqueRanks.length > 0
            ? uniqueRanks.join(", ")
            : "없음";
      const summary = `선택 문서: ${label}`;
      // 버튼 클릭 시 즉시 문서 선택 UI 숨기기
      setPendingInterrupt(null);
      send({
        text: summary,
        decisionOverride: {
          selected_doc_ids: uniqueIds,
          selected_ranks: uniqueRanks,
        },
      });
    },
    [pendingInterrupt, send]
  );

  const reset = useCallback(() => {
    stop();
    setMessages([]);
    setError(null);
    setPendingInterrupt(null);
  }, [stop]);

  return useMemo(
    () => ({
      messages,
      isStreaming,
      error,
      send,
      stop,
      pendingReview: pendingInterrupt?.kind === "retrieval_review" ? pendingInterrupt : null,
      submitReview,
      inputPlaceholder: pendingInterrupt
        ? pendingInterrupt.kind === "retrieval_review"
          ? "검색 결과 승인/거절 또는 추가 키워드를 입력하세요..."
          : "승인/거절 또는 수정 답변을 입력하세요..."
        : "메시지를 입력하세요...",
      reset,
    }),
    [messages, isStreaming, error, send, stop, pendingInterrupt, submitReview, reset]
  );
}
