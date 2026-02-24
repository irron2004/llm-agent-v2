import { useCallback, useMemo, useRef, useState } from "react";
import { nanoid } from "nanoid";
import { sendChatMessage, saveTurn, fetchSession, saveFeedback, saveDetailedFeedback, getDetailedFeedback, resolveChatPaths } from "../api";
import {
  AgentResponse,
  Message,
  ReviewDoc,
  DocRefResponse,
  RetrievedDoc,
  MessageFeedback,
  FeedbackRating,
  TurnResponse,
  AgentRequest,
} from "../types";
import { connectSse } from "../../../lib/sse";
import { env } from "../../../config/env";
import { useChatLogs } from "../context/chat-logs-context";
import { useChatReview } from "../context/chat-review-context";

// Session change callback type
export type SessionChangeCallback = (info: { sessionId: string; title: string; isNew: boolean }) => void;

type SendOptions = {
  text: string;
  decisionOverride?: unknown;
  overrides?: {
    filterDevices?: string[];
    filterDocTypes?: string[];
    searchQueries?: string[];
    selectedDocIds?: string[];
    autoParse?: boolean;
  };
};

type InterruptKind = "device_selection" | "retrieval_review" | "human_review" | "unknown";

type DeviceInfo = {
  name: string;
  doc_count: number;
};

type DocTypeInfo = {
  name: string;
  doc_count: number;
};

type FeedbackPayload = {
  messageId: string;
  sessionId?: string;
  turnId?: number;
  rating: FeedbackRating;
  reason?: string | null;
};

type DetailedFeedbackPayload = {
  messageId: string;
  sessionId?: string;
  turnId?: number;
  accuracy: number;
  completeness: number;
  relevance: number;
  comment?: string;
  reviewerName?: string;
  logs?: string[];
};

type PendingInterrupt = {
  threadId: string;
  question: string;
  instruction: string;
  docs: ReviewDoc[];
  devices?: DeviceInfo[];
  docTypes?: DocTypeInfo[];
  kind: InterruptKind;
  payload?: Record<string, unknown> | null;
};

const APPROVE_TOKENS = ["true", "yes", "y", "ok", "okay", "승인", "확인", "approve"];
const REJECT_TOKENS = ["false", "no", "n", "거절", "reject", "decline"];

const isEffectiveParsedDevice = (value: string): boolean => {
  const normalized = value.replace(/[\s\-_.\/]+/g, "").trim().toUpperCase();
  if (!normalized) return false;
  if (/^[A-Z]+$/.test(normalized) && normalized.length <= 4) return false;
  return true;
};

const resolveDecision = (text: string): boolean | string => {
  const trimmed = text.trim();
  const lowered = trimmed.toLowerCase();
  if (APPROVE_TOKENS.includes(lowered) || APPROVE_TOKENS.includes(trimmed)) return true;
  if (REJECT_TOKENS.includes(lowered) || REJECT_TOKENS.includes(trimmed)) return false;
  return trimmed;
};

const resolveInterruptKind = (payload?: Record<string, unknown> | null): InterruptKind => {
  if (payload?.type === "device_selection") return "device_selection";
  if (payload?.type === "retrieval_review") return "retrieval_review";
  if (payload?.type === "human_review") return "human_review";
  return "unknown";
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

// Type guard for AgentResponse - checks required fields
const isAgentResponseLike = (value: unknown): value is AgentResponse => {
  if (!isRecord(value)) return false;
  return typeof value.query === "string" && typeof value.answer === "string";
};

const toStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
};

const buildInterruptPrompt = (kind: InterruptKind, instruction?: string) => {
  if (kind === "device_selection") {
    return "검색에 사용할 기기와 문서 종류를 각각 1개 이상 선택하세요.";
  }
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
    const title = typeof doc?.title === "string" ? doc.title : null;
    const page = typeof doc?.page === "number" ? doc.page : null;
    const page_image_url = typeof doc?.page_image_url === "string" ? doc.page_image_url : null;
    return {
      docId,
      rank,
      content,
      title,
      page,
      page_image_url,
      score: typeof doc?.score === "number" ? doc.score : null,
      metadata: typeof doc?.metadata === "object" ? doc.metadata : null,
    };
  });
};

// Convert RetrievedDoc[] to DocRefResponse[] for API
const toDocRefs = (docs: RetrievedDoc[]): DocRefResponse[] => {
  return docs.map((doc, index) => ({
    slot: index + 1,
    doc_id: doc.id,
    title: doc.title,
    snippet: doc.snippet,
    page: doc.page ?? null,
    pages: Array.isArray(doc.expanded_pages) && doc.expanded_pages.length > 0
      ? doc.expanded_pages
      : doc.page !== null && doc.page !== undefined
        ? [doc.page]
        : null,
    score: doc.score ?? null,
  }));
};

const extractFeedback = (turn?: TurnResponse | null): MessageFeedback | null => {
  if (!turn?.feedback_rating) return null;
  return {
    rating: turn.feedback_rating,
    reason: turn.feedback_reason ?? null,
    ts: turn.feedback_ts ?? null,
  };
};

export type UseChatSessionOptions = {
  onSessionChange?: SessionChangeCallback;
  onTurnSaved?: () => void;
};

export function useChatSession(options: UseChatSessionOptions = {}) {
  const { onSessionChange, onTurnSaved } = options;
  const [sessionId, setSessionId] = useState<string>(() => nanoid());
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingInterrupt, setPendingInterrupt] = useState<PendingInterrupt | null>(null);
  const [isLoadingSession, setIsLoadingSession] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const isFirstMessageRef = useRef(true);
  const currentUserTextRef = useRef<string>("");
  const sessionTitleRef = useRef<string | null>(null);
  const turnCountRef = useRef(0);
  const streamedAutoParseRef = useRef<Record<string, {
    device?: string | null;
    devices?: string[] | null;
    doc_type?: string | null;
    doc_types?: string[] | null;
    language?: string | null;
    message?: string | null;
  }>>({});
  const onSessionChangeRef = useRef(onSessionChange);
  const onTurnSavedRef = useRef(onTurnSaved);
  onSessionChangeRef.current = onSessionChange;
  onTurnSavedRef.current = onTurnSaved;

  // Get chat logs context (Provider is always available in AppProviders)
  const { addLog, clearLogs } = useChatLogs();
  
  // Get chat review context for right-sidebar review/regeneration states
  const { setCompletedRetrievedDocs, setPendingRegeneration } = useChatReview();

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
      const metadataQueries = toStringArray(res.metadata?.search_queries);
      const responseQueries = Array.isArray(res.search_queries)
        ? res.search_queries.filter((q: unknown): q is string => typeof q === "string" && q.trim().length > 0)
        : [];
      const effectiveSearchQueries = metadataQueries.length > 0 ? metadataQueries : responseQueries;

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
        // Use res.retrieved_docs directly (same source as message.retrievedDocs)
        const docs: ReviewDoc[] = (res.retrieved_docs || []).map((doc, index) => ({
          docId: doc.id,
          rank: index + 1,
          content: doc.snippet,
          title: doc.title,
          page: doc.page ?? null,
          page_image_url: doc.page_image_url ?? null,
          score: doc.score ?? null,
          metadata: doc.metadata ?? null,
        }));

        // Extract devices/doc types for device_selection interrupt
        const devices: DeviceInfo[] = Array.isArray(payload?.devices)
          ? payload.devices.map((device) => {
              const source = isRecord(device) ? device : null;
              return {
                name: typeof source?.name === "string" ? source.name : "",
                doc_count: typeof source?.doc_count === "number" ? source.doc_count : 0,
              };
            }).filter((d: DeviceInfo) => d.name)
          : [];
        const docTypes: DocTypeInfo[] = Array.isArray(payload?.doc_types)
          ? payload.doc_types.map((docType) => {
              const source = isRecord(docType) ? docType : null;
              return {
                name: typeof source?.name === "string" ? source.name : "",
                doc_count: typeof source?.doc_count === "number" ? source.doc_count : 0,
              };
            }).filter((d: DocTypeInfo) => d.name)
          : [];

        if (threadId) {
          setPendingInterrupt({
            threadId,
            question,
            instruction,
            docs,
            devices: kind === "device_selection" ? devices : undefined,
            docTypes: kind === "device_selection" ? docTypes : undefined,
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
          sessionId,
          searchQueries: effectiveSearchQueries.length > 0 ? effectiveSearchQueries : null,
        }));
        setIsStreaming(false);
        return;
      }

      // Conversation completed - clear logs and set retrieved docs
      setPendingInterrupt(null);
      clearLogs();

      // Set completed retrieved docs in context if available
      // Fallback to all_retrieved_docs if retrieved_docs is empty (e.g., regeneration with no matching docs)
      const docsToShow = (res.retrieved_docs && res.retrieved_docs.length > 0)
        ? res.retrieved_docs
        : (res.all_retrieved_docs && res.all_retrieved_docs.length > 0)
          ? res.all_retrieved_docs
          : null;

      if (docsToShow) {
        setCompletedRetrievedDocs(docsToShow);
      } else {
        setCompletedRetrievedDocs(null);
      }

      updateMessage(assistantId, (m) => ({
        ...m,
        content: res.answer || "",
        retrievedDocs: res.retrieved_docs || [],
        allRetrievedDocs: res.all_retrieved_docs || [],  // 재생성용 전체 문서 (최대 retrieval_top_k개)
        rawAnswer: JSON.stringify(res, null, 2),
        currentNode: null,
        sessionId,
        // Store auto_parse and filter info for regeneration
        autoParse: (() => {
          const existing = m.autoParse ?? null;
          const incoming = res.auto_parse ?? null;
          if (!incoming) return existing;
          return {
            ...existing,
            ...incoming,
            message: incoming.message ?? existing?.message ?? null,
          };
        })(),
        selectedDevices: res.selected_devices ?? null,
        selectedDocTypes: res.selected_doc_types ?? null,
        searchQueries: effectiveSearchQueries.length > 0 ? effectiveSearchQueries : null,
      }));

      // If auto-parse could not detect a device, proactively open
      // the regeneration panel so user can run an additional device-filtered search.
      const streamedAutoParse = streamedAutoParseRef.current[assistantId];
      const autoParseInfo = (res.auto_parse ?? streamedAutoParse) as (Record<string, unknown> & {
        device?: string | null;
        devices?: string[] | null;
        equip_id?: string | null;
        equip_ids?: string[] | null;
      }) | null | undefined;
      const responseDetectedLanguage = typeof res.detected_language === "string";
      const hasAutoParseResult = Boolean(autoParseInfo) || responseDetectedLanguage;
      const parsedDevices = Array.isArray(autoParseInfo?.devices)
        ? autoParseInfo.devices.map((d) => String(d).trim()).filter((d) => d.length > 0)
        : [];
      const parsedDevice = typeof autoParseInfo?.device === "string" ? autoParseInfo.device.trim() : "";
      const effectiveParsedDevices = parsedDevices.filter((d) => isEffectiveParsedDevice(d));
      const hasEffectiveParsedDevice = parsedDevice.length > 0 && isEffectiveParsedDevice(parsedDevice);
      const hasParsedDeviceSignal =
        effectiveParsedDevices.length > 0 ||
        hasEffectiveParsedDevice;
      const docsForRegeneration = (res.all_retrieved_docs && res.all_retrieved_docs.length > 0)
        ? res.all_retrieved_docs
        : (res.retrieved_docs || []);

      const fallbackSuggest = hasAutoParseResult && !hasParsedDeviceSignal;
      const shouldSuggestAdditionalDeviceSearch = typeof res.suggest_additional_device_search === "boolean"
        ? (res.suggest_additional_device_search || fallbackSuggest)
        : fallbackSuggest;

      updateMessage(assistantId, (m) => ({
        ...m,
        suggestAdditionalDeviceSearch: shouldSuggestAdditionalDeviceSearch,
      }));

      if (shouldSuggestAdditionalDeviceSearch) {
        setPendingRegeneration({
          messageId: assistantId,
          originalQuery: res.query || fallbackQuestion,
          docs: docsForRegeneration,
          searchQueries: effectiveSearchQueries.length > 0
            ? effectiveSearchQueries
            : [res.query || fallbackQuestion].filter((q) => q && q.trim().length > 0),
          selectedDevices: [],
          selectedDocTypes: Array.isArray(res.selected_doc_types)
            ? res.selected_doc_types.filter((d): d is string => typeof d === "string" && d.trim().length > 0)
            : [],
          reason: "missing_device_parse",
        });
      }

      // Save turn to backend
      const userText = currentUserTextRef.current;
      const assistantText = res.answer || "";
      const docRefs = toDocRefs(res.retrieved_docs || []);

      // Determine title (only for first turn)
      turnCountRef.current += 1;
      const title = turnCountRef.current === 1
        ? (userText.length > 50 ? userText.slice(0, 50) + "..." : userText)
        : null;
      if (title) {
        sessionTitleRef.current = title;
      }

      saveTurn(sessionId, {
        user_text: userText,
        assistant_text: assistantText,
        doc_refs: docRefs,
        title,
      }).then((turn) => {
        updateMessage(assistantId, (m) => ({
          ...m,
          sessionId,
          turnId: turn.turn_id,
          feedback: extractFeedback(turn),
        }));
        onTurnSavedRef.current?.();
      }).catch((err) => {
        console.error("Failed to save turn:", err);
      });
    },
    [sessionId, updateMessage, setIsStreaming, clearLogs, setCompletedRetrievedDocs, setPendingRegeneration]
  );

  const send = useCallback(
    async ({ text, decisionOverride, overrides }: SendOptions) => {
      stop();
      setError(null);
      setPendingRegeneration(null);
      const pending = pendingInterrupt;
      // Only update user text if not resuming (keep original question for saves)
      if (!pending) {
        currentUserTextRef.current = text;
      }
      const isResume = Boolean(pending);
      if (isResume && !pending?.threadId) {
        setError("thread_id가 없어 검색 결과 확인을 이어갈 수 없습니다.");
        return;
      }

      // Clear logs when starting a new conversation (not resuming)
      if (!isResume && isFirstMessageRef.current) {
        clearLogs();
      }

      // Notify on first message of this session
      if (isFirstMessageRef.current && !isResume) {
        isFirstMessageRef.current = false;
        const title = text.length > 50 ? text.slice(0, 50) + "..." : text;
        onSessionChangeRef.current?.({ sessionId, title, isNew: true });
      }

      const userId = nanoid();
      const assistantId = nanoid();

      const requestMessage = isResume && pending ? pending.question : text;
      const decision = isResume ? (decisionOverride ?? resolveDecision(text)) : undefined;

      appendMessage({
        id: userId,
        role: "user",
        content: text,
        sessionId,
      });

      // Assistant placeholder so the UI shows progress immediately.
      // Note: logs are stored in context, not in message object
      appendMessage({
        id: assistantId,
        role: "assistant",
        content: "처리 중...",
        currentNode: null,
        sessionId,
        originalQuery: requestMessage,
      });
      setIsStreaming(true);

      try {
        // Extract previous turn for chat_history (only when not resuming)
        let chatHistory: { user_text: string; assistant_text: string; doc_ids: string[] }[] | undefined;
        if (!isResume && messages.length >= 2) {
          const lastAssistant = [...messages].reverse().find(
            (m) => m.role === "assistant" && m.content && m.content !== "처리 중..."
          );
          const lastAssistantIdx = lastAssistant ? messages.indexOf(lastAssistant) : -1;
          const lastUser = lastAssistantIdx > 0
            ? messages.slice(0, lastAssistantIdx).reverse().find((m) => m.role === "user")
            : undefined;
          if (lastUser && lastAssistant) {
            chatHistory = [{
              user_text: lastUser.content,
              assistant_text: lastAssistant.content,
              doc_ids: (lastAssistant.retrievedDocs || []).map((d) => d.id).filter(Boolean),
            }];
          }
        }

        const autoParseEnabled = overrides?.autoParse ?? !Boolean(overrides);
        const payload: AgentRequest = {
          message: requestMessage,
          auto_parse: autoParseEnabled,  // 자동 파싱 모드 활성화 (기본값)
          ask_user_after_retrieve: false,  // 문서 선택 UI 비활성화
          ...(chatHistory ? { chat_history: chatHistory } : {}),
          ...(overrides ? {
            filter_devices: overrides.filterDevices,
            filter_doc_types: overrides.filterDocTypes,
            search_queries: overrides.searchQueries,
            selected_doc_ids: overrides.selectedDocIds,
          } : {}),
          ...(isResume && pending
            ? {
                thread_id: pending.threadId,
                resume_decision: decision,
                auto_parse: false,  // resume 시에는 auto_parse 비활성화
                ask_user_after_retrieve: true,  // resume은 HIL 모드
              }
            : {}),
        };
        const { canStream, streamPath } = resolveChatPaths(env.chatPath);
        if (!canStream) {
          const res = await sendChatMessage(payload);
          handleAgentResponse(res, assistantId, requestMessage);
          return;
        }

        const controller = new AbortController();
        abortRef.current = controller;

        await connectSse(
          {
            path: streamPath,
            body: payload,
            signal: controller.signal,
          },
          {
            onMessage: (data) => {
              let evt: Record<string, unknown> | null = null;
              try {
                const parsed = JSON.parse(data) as unknown;
                if (isRecord(parsed)) {
                  evt = parsed;
                }
              } catch {
                return;
              }

              if (!evt) return;

              if (evt?.type === "log") {
                const logMessage = typeof evt?.message === "string" ? evt.message : "";
                const logNode = typeof evt?.node === "string" ? evt.node : null;

                // Add log to context (for right sidebar display)
                if (logMessage) {
                  addLog(assistantId, logMessage, logNode);
                }

                // Update only currentNode for message (logs are shown in right sidebar only)
                updateMessage(assistantId, (m) => {
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
                    currentNode,
                  };
                });
                return;
              }

              // Handle auto_parse event (display parsing result)
              if (evt?.type === "auto_parse") {
                const parseMessage = typeof evt?.message === "string" ? evt.message : null;
                const parseLanguage = typeof evt?.language === "string" ? evt.language : null;
                const parsedDevice = typeof evt?.device === "string" ? evt.device : null;
                const parsedDocType = typeof evt?.doc_type === "string" ? evt.doc_type : null;
                const parsedDevices = Array.isArray(evt?.devices)
                  ? evt.devices.map((device) => String(device)).filter((d: string) => d.trim())
                  : (parsedDevice ? [parsedDevice] : []);
                const parsedDocTypes = Array.isArray(evt?.doc_types)
                  ? evt.doc_types.map((docType) => String(docType)).filter((d: string) => d.trim())
                  : (parsedDocType ? [parsedDocType] : []);

                if (!parseMessage && parsedDevices.length === 0 && parsedDocTypes.length === 0 && !parseLanguage) {
                  return;
                }

                const messageText = parseMessage ?? `파싱 결과 - ${[
                  parsedDevices.length > 0 ? `장비: ${parsedDevices.join(", ")}` : null,
                  parsedDocTypes.length > 0 ? `문서: ${parsedDocTypes.join(", ")}` : null,
                  parseLanguage ? `언어: ${parseLanguage}` : null,
                ].filter(Boolean).join(", ")}`;

                addLog(assistantId, `🔍 ${messageText}`, "auto_parse");

                updateMessage(assistantId, (m) => ({
                  ...m,
                  content: `🔍 ${messageText}\n\n처리 중...`,
                  autoParse: {
                    device: parsedDevice,
                    doc_type: parsedDocType,
                    devices: parsedDevices,
                    doc_types: parsedDocTypes,
                    language: parseLanguage,
                    message: messageText,
                  },
                  selectedDevices: parsedDevices.length > 0 ? parsedDevices : m.selectedDevices ?? null,
                  selectedDocTypes: parsedDocTypes.length > 0 ? parsedDocTypes : m.selectedDocTypes ?? null,
                }));
                streamedAutoParseRef.current[assistantId] = {
                  device: parsedDevice,
                  devices: parsedDevices,
                  doc_type: parsedDocType,
                  doc_types: parsedDocTypes,
                  language: parseLanguage,
                  message: messageText,
                };
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
                if (isAgentResponseLike(res)) {
                  handleAgentResponse(res, assistantId, requestMessage);
                } else {
                  setError("응답 형식이 올바르지 않습니다.");
                }
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
    [appendMessage, stop, updateMessage, handleAgentResponse, pendingInterrupt, sessionId, addLog, clearLogs, setPendingRegeneration]
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

  const submitSearchQueries = useCallback(
    (modifiedQueries: string[]) => {
      if (!pendingInterrupt || pendingInterrupt.kind !== "retrieval_review") return;

      const validQueries = modifiedQueries.map((q) => q.trim()).filter((q) => q.length > 0);

      if (validQueries.length === 0) {
        setError("최소 1개 이상의 검색어를 입력해야 합니다.");
        return;
      }

      const summary = `검색어 수정: ${validQueries.join(", ")}`;
      setPendingInterrupt(null);

      send({
        text: summary,
        decisionOverride: {
          type: "modify_search_queries",
          search_queries: validQueries,
        },
      });
    },
    [pendingInterrupt, send]
  );

  const submitDeviceSelection = useCallback(
    (selectedDevices: string[], selectedDocTypes: string[]) => {
      if (!pendingInterrupt || pendingInterrupt.kind !== "device_selection") return;

      const hasDevices = selectedDevices.length > 0;
      const hasDocTypes = selectedDocTypes.length > 0;

      if (!hasDevices || !hasDocTypes) {
        setError("기기와 문서 종류를 각각 1개 이상 선택해야 합니다.");
        return;
      }

      const allDevicesSelected = pendingInterrupt.devices
        ? selectedDevices.length === pendingInterrupt.devices.length
        : false;
      const allDocTypesSelected = pendingInterrupt.docTypes
        ? selectedDocTypes.length === pendingInterrupt.docTypes.length
        : false;

      const summaryParts: string[] = [];
      if (allDevicesSelected) {
        summaryParts.push("기기: 전체");
      } else if (selectedDevices.length > 10) {
        summaryParts.push("기기: 다수 선택");
      } else {
        summaryParts.push(`기기: ${selectedDevices.join(", ")}`);
      }

      summaryParts.push(
        allDocTypesSelected
          ? "문서: 전체"
          : `문서: ${selectedDocTypes.join(", ")}`
      );

      setPendingInterrupt(null);

      send({
        text: summaryParts.length > 0 ? `선택: ${summaryParts.join(" / ")}` : "선택 조건 검색",
        decisionOverride: {
          type: "device_selection",
          selected_devices: selectedDevices,
          selected_doc_types: selectedDocTypes,
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
    streamedAutoParseRef.current = {};
    clearLogs();
    setCompletedRetrievedDocs(null);
    // Generate new session ID for next chat
    const newSessionId = nanoid();
    setSessionId(newSessionId);
    isFirstMessageRef.current = true;
    currentUserTextRef.current = "";
    sessionTitleRef.current = null;
    turnCountRef.current = 0;
  }, [stop, clearLogs, setCompletedRetrievedDocs]);

  const submitFeedback = useCallback(
    async ({ messageId, sessionId: msgSessionId, turnId, rating, reason }: FeedbackPayload) => {
      const targetSessionId = msgSessionId || sessionId;
      if (!targetSessionId || !turnId) {
        setError("만족도를 저장하려면 turn 정보가 필요합니다.");
        return;
      }

      const feedback: MessageFeedback = {
        rating,
        reason: reason ?? null,
        ts: new Date().toISOString(),
      };
      updateMessage(messageId, (m) => ({
        ...m,
        feedback,
      }));

      try {
        const updated = await saveFeedback(targetSessionId, turnId, {
          rating,
          reason,
        });
        updateMessage(messageId, (m) => ({
          ...m,
          feedback: extractFeedback(updated),
        }));
      } catch (err) {
        console.error("Failed to save feedback:", err);
        setError(err instanceof Error ? err.message : "만족도 저장에 실패했습니다.");
      }
    },
    [sessionId, updateMessage]
  );

  const submitDetailedFeedback = useCallback(
    async ({
      messageId,
      sessionId: msgSessionId,
      turnId,
      accuracy,
      completeness,
      relevance,
      comment,
      reviewerName,
      logs,
    }: DetailedFeedbackPayload) => {
      const targetSessionId = msgSessionId || sessionId;
      if (!targetSessionId || !turnId) {
        setError("피드백을 저장하려면 turn 정보가 필요합니다.");
        return;
      }

      // Find message to get user_text and assistant_text
      const msg = messages.find((m) => m.id === messageId);
      const userMsg = messages.find(
        (m) => m.role === "user" && messages.indexOf(m) === messages.indexOf(msg!) - 1
      );

      // Calculate avg score and rating
      const avgScore = (accuracy + completeness + relevance) / 3;
      const rating: FeedbackRating = avgScore >= 3 ? "up" : "down";

      // Optimistic update
      const feedback: MessageFeedback = {
        rating,
        accuracy,
        completeness,
        relevance,
        avgScore,
        comment: comment ?? null,
        ts: new Date().toISOString(),
      };
      updateMessage(messageId, (m) => ({
        ...m,
        feedback,
      }));

      try {
        // Save to feedback index
        const saved = await saveDetailedFeedback(targetSessionId, turnId, {
          accuracy,
          completeness,
          relevance,
          comment,
          reviewer_name: reviewerName,
          logs,
          user_text: userMsg?.content ?? "",
          assistant_text: msg?.content ?? "",
        });

        // Update with server response
        updateMessage(messageId, (m) => ({
          ...m,
          feedback: {
            rating: saved.rating as FeedbackRating,
            accuracy: saved.accuracy,
            completeness: saved.completeness,
            relevance: saved.relevance,
            avgScore: saved.avg_score,
            comment: saved.comment ?? null,
            ts: saved.ts,
          },
        }));

        // Also update the legacy feedback in chat_turns for backwards compatibility
        await saveFeedback(targetSessionId, turnId, {
          rating,
          reason: comment,
        });
      } catch (err) {
        console.error("Failed to save detailed feedback:", err);
        setError(err instanceof Error ? err.message : "피드백 저장에 실패했습니다.");
      }
    },
    [sessionId, messages, updateMessage]
  );

  // Load an existing session from the backend
  const loadSession = useCallback(
    async (targetSessionId: string) => {
      stop();
      setError(null);
      setIsLoadingSession(true);

      try {
        const session = await fetchSession(targetSessionId);

        // Convert turns to messages
        const loadedMessages: Message[] = [];
        for (const turn of session.turns) {
          // User message
          loadedMessages.push({
            id: nanoid(),
            role: "user",
            content: turn.user_text,
            createdAt: turn.ts,
            sessionId: session.session_id,
          });
          // Assistant message
          loadedMessages.push({
            id: nanoid(),
            role: "assistant",
            content: turn.assistant_text,
            createdAt: turn.ts,
            sessionId: session.session_id,
            turnId: turn.turn_id,
            feedback: extractFeedback(turn),
            retrievedDocs: turn.doc_refs.map((ref) => ({
              id: ref.doc_id,
              title: ref.title,
              snippet: ref.snippet,
              page: ref.page,
              expanded_pages: Array.isArray(ref.pages) && ref.pages.length > 0
                ? ref.pages
                : ref.page !== null && ref.page !== undefined
                  ? [ref.page]
                  : null,
              expanded_page_urls: Array.isArray(ref.pages) && ref.pages.length > 0
                ? ref.pages.map((p) => `/api/assets/docs/${ref.doc_id}/pages/${p}`)
                : ref.page !== null && ref.page !== undefined
                  ? [`/api/assets/docs/${ref.doc_id}/pages/${ref.page}`]
                  : null,
              page_image_url: ref.page !== null && ref.page !== undefined
                ? `/api/assets/docs/${ref.doc_id}/pages/${ref.page}`
                : null,
              score: ref.score,
            })),
          });
        }

        // Update state
        setSessionId(targetSessionId);
        setMessages(loadedMessages);
        setPendingInterrupt(null);
        clearLogs();
        setCompletedRetrievedDocs(null);

        // Update refs
        isFirstMessageRef.current = false;
        sessionTitleRef.current = session.title;
        turnCountRef.current = session.turn_count;
        currentUserTextRef.current = "";
      } catch (err) {
        console.error("[loadSession] Error:", err);
        setError(err instanceof Error ? err.message : "세션을 불러오는데 실패했습니다.");
      } finally {
        setIsLoadingSession(false);
      }
    },
    [stop, clearLogs, setCompletedRetrievedDocs]
  );

  return useMemo(
    () => ({
      sessionId,
      messages,
      isStreaming,
      isLoadingSession,
      error,
      send,
      stop,
      pendingReview: pendingInterrupt?.kind === "retrieval_review" ? pendingInterrupt : null,
      pendingDeviceSelection: pendingInterrupt?.kind === "device_selection" ? pendingInterrupt : null,
      submitReview,
      submitSearchQueries,
      submitDeviceSelection,
      submitFeedback,
      submitDetailedFeedback,
      inputPlaceholder: pendingInterrupt
        ? pendingInterrupt.kind === "device_selection"
          ? "기기를 선택하거나 건너뛰기를 클릭하세요..."
          : pendingInterrupt.kind === "retrieval_review"
            ? "검색 결과 승인/거절 또는 추가 키워드를 입력하세요..."
            : "승인/거절 또는 수정 답변을 입력하세요..."
        : "메시지를 입력하세요...",
      reset,
      loadSession,
    }),
    [
      sessionId,
      messages,
      isStreaming,
      isLoadingSession,
      error,
      send,
      stop,
      pendingInterrupt,
      submitReview,
      submitSearchQueries,
      submitDeviceSelection,
      submitFeedback,
      submitDetailedFeedback,
      reset,
      loadSession,
    ]
  );
}
