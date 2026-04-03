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
  suppressUserMessage?: boolean;
  overrides?: {
    filterDevices?: string[];
    filterDocTypes?: string[];
    searchQueries?: string[];
    selectedDocIds?: string[];
    autoParse?: boolean;
    contextChunkIds?: string[];
  };
};

type InterruptKind =
  | "auto_parse_confirm"
  | "device_selection"
  | "retrieval_review"
  | "human_review"
  | "abbreviation_resolve"
  | "issue_confirm"
  | "issue_case_selection"
  | "issue_sop_confirm"
  | "unknown";

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

type PendingGuidedSelection = {
  threadId: string;
  question: string;
  instruction: string;
  payload: Record<string, unknown>;
  stepIndex?: number;
  draftDecision?: {
    type: "auto_parse_confirm";
    target_language?: "ko" | "en" | "zh" | "ja";
    selected_device?: string | null;
    selected_equip_id?: string | null;
    task_mode?: "sop" | "issue" | "all";
  };
};

type IssueCase = {
  doc_id: string;
  title: string;
  summary: string;
};

type PendingIssueConfirm = {
  threadId: string;
  question: string;
  instruction: string;
  payload: {
    type: "issue_confirm";
    nonce: string;
    stage: "post_summary" | "post_detail";
    prompt: string;
  };
};

type PendingIssueCaseSelection = {
  threadId: string;
  question: string;
  instruction: string;
  payload: {
    type: "issue_case_selection";
    nonce: string;
    cases: IssueCase[];
  };
};

type PendingIssueSopConfirm = {
  threadId: string;
  question: string;
  instruction: string;
  payload: {
    type: "issue_sop_confirm";
    nonce: string;
    prompt: string;
    has_sop_ref: boolean;
    sop_hint?: string | null;
  };
};

type AbbreviationOption = {
  value: string;
  label: string;
  eng?: string;
  kr?: string | null;
};

type AbbreviationItem = {
  token: string;
  abbr_key: string;
  options: AbbreviationOption[];
};

type AbbreviationResolvePayload = {
  type: "abbreviation_resolve";
  abbreviations: AbbreviationItem[];
};

type PendingAbbreviationResolve = {
  threadId: string;
  question: string;
  instruction: string;
  payload: AbbreviationResolvePayload;
};

type GuidedOption = {
  value: string;
  recommended?: boolean;
};

const toGuidedOptions = (value: unknown): GuidedOption[] => {
  if (!Array.isArray(value)) return [];
  const options: GuidedOption[] = [];
  for (const item of value) {
    const src = isRecord(item) ? item : null;
    const v = typeof src?.value === "string" ? src.value.trim() : "";
    if (!v) continue;
    options.push({
      value: v,
      ...(src?.recommended === true ? { recommended: true } : {}),
    });
  }
  return options;
};

const normalizeTaskMode = (value: unknown): "sop" | "issue" | "all" => {
  const v = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (v === "sop" || v === "issue") return v;
  return "all";
};

const normalizeLanguage = (value: unknown): "ko" | "en" | "zh" | "ja" => {
  const v = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (v === "en" || v === "zh" || v === "ja") return v;
  return "ko";
};

type GuidedStep = "language" | "device" | "equip_id" | "task";

const DEFAULT_GUIDED_STEPS: GuidedStep[] = ["device", "task"];

const isGuidedStep = (value: unknown): value is GuidedStep => {
  return value === "language" || value === "device" || value === "equip_id" || value === "task";
};

const resolveGuidedSteps = (payload: Record<string, unknown>): GuidedStep[] => {
  const raw = payload.steps;
  if (Array.isArray(raw)) {
    const normalized = raw
      .map((step) => (typeof step === "string" ? step.trim() : ""))
      .filter((step): step is GuidedStep => isGuidedStep(step));
    if (normalized.length > 0) return normalized;
  }
  return DEFAULT_GUIDED_STEPS;
};

const buildGuidedSummaryText = ({
  selectedDevice,
  selectedEquipId,
  taskMode,
  includeEquipStep,
}: {
  selectedDevice: string | null;
  selectedEquipId: string | null;
  taskMode: "sop" | "issue" | "all";
  includeEquipStep: boolean;
}): string => {
  const parts = [selectedDevice ?? "(skip)"];
  if (includeEquipStep) {
    parts.push(selectedEquipId ?? "(skip)");
  }
  parts.push(taskMode);
  return `가이드 확인: ${parts.join(" / ")}`;
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
  if (payload?.type === "auto_parse_confirm") return "auto_parse_confirm";
  if (payload?.type === "device_selection") return "device_selection";
  if (payload?.type === "retrieval_review") return "retrieval_review";
  if (payload?.type === "human_review") return "human_review";
  if (payload?.type === "abbreviation_resolve") return "abbreviation_resolve";
  if (payload?.type === "issue_confirm") return "issue_confirm";
  if (payload?.type === "issue_case_selection") return "issue_case_selection";
  if (payload?.type === "issue_sop_confirm") return "issue_sop_confirm";
  return "unknown";
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const toIssueCases = (value: unknown): IssueCase[] => {
  if (!Array.isArray(value)) return [];
  const result: IssueCase[] = [];
  for (const item of value) {
    const src = isRecord(item) ? item : null;
    const doc_id = typeof src?.doc_id === "string" ? src.doc_id.trim() : "";
    if (!doc_id) continue;
    const title = typeof src?.title === "string" && src.title.trim() ? src.title.trim() : doc_id;
    const summary = typeof src?.summary === "string" ? src.summary : "";
    result.push({ doc_id, title, summary });
  }
  return result;
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
  if (kind === "abbreviation_resolve") {
    if (instruction && instruction.trim()) return instruction.trim();
    return "약어 의미를 선택해 주세요.";
  }
  if (instruction && instruction.trim()) return instruction.trim();
  return "추가 입력이 필요합니다. 승인/거절 또는 수정 답변을 입력해 주세요.";
};

const toAbbreviationOptions = (value: unknown): AbbreviationOption[] => {
  if (!Array.isArray(value)) return [];
  const options: AbbreviationOption[] = [];
  for (const item of value) {
    const src = isRecord(item) ? item : null;
    const optionValue = typeof src?.value === "string" ? src.value.trim() : "";
    if (!optionValue) continue;
    options.push({
      value: optionValue,
      label: typeof src?.label === "string" && src.label.trim() ? src.label.trim() : optionValue,
      eng: typeof src?.eng === "string" ? src.eng : undefined,
      kr: typeof src?.kr === "string" ? src.kr : null,
    });
  }
  return options;
};

const toAbbreviationResolvePayload = (
  payload?: Record<string, unknown> | null
): AbbreviationResolvePayload | null => {
  if (!payload || payload.type !== "abbreviation_resolve") return null;
  const rawItems = Array.isArray(payload.abbreviations) ? payload.abbreviations : [];
  const abbreviations: AbbreviationItem[] = [];

  for (const item of rawItems) {
    const src = isRecord(item) ? item : null;
    const token = typeof src?.token === "string" ? src.token.trim() : "";
    const abbrKey = typeof src?.abbr_key === "string" ? src.abbr_key.trim() : "";
    if (!abbrKey) continue;
    const options = toAbbreviationOptions(src?.options);
    if (options.length === 0) continue;
    abbreviations.push({ token, abbr_key: abbrKey, options });
  }

  if (abbreviations.length === 0) return null;
  return {
    type: "abbreviation_resolve",
    abbreviations,
  };
};

type IssueResumeContext = {
  isIssueResume: boolean;
  decisionType: "issue_confirm" | "issue_case_selection" | "issue_sop_confirm" | null;
  confirm: boolean | null;
};

const DEFAULT_ISSUE_RESUME_CONTEXT: IssueResumeContext = {
  isIssueResume: false,
  decisionType: null,
  confirm: null,
};

const hasPriorAssistantContentMatch = (messages: Message[], targetContent: string): boolean => {
  const normalized = targetContent.trim();
  if (!normalized) return false;
  const latestAssistantContent = [...messages]
    .reverse()
    .find(
      (m) =>
        m.role === "assistant" &&
        m.content !== "처리 중..." &&
        m.content.trim().length > 0
    )
    ?.content
    ?.trim();
  return Boolean(latestAssistantContent && latestAssistantContent === normalized);
};

const buildIssueResumeDuplicateAck = (context: IssueResumeContext): string => {
  if (context.decisionType === "issue_confirm" && context.confirm === false) {
    return "추가 이슈 탐색을 생략하고 현재 답변을 유지합니다.";
  }
  if (context.decisionType === "issue_sop_confirm" && context.confirm === false) {
    return "SOP 확인을 생략하고 현재 답변을 유지합니다.";
  }
  return "요청을 반영했습니다.";
};

const resolveInterruptDisplayContent = ({
  kind,
  instruction,
  answer,
  messages,
}: {
  kind: InterruptKind;
  instruction?: string;
  answer?: string | null;
  messages: Message[];
}): string => {
  const fallback = buildInterruptPrompt(kind, instruction);
  const answerText = typeof answer === "string" ? answer.trim() : "";
  if (!answerText) return fallback;

  if (hasPriorAssistantContentMatch(messages, answerText)) {
    return fallback;
  }
  return answerText;
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
  const [pendingGuidedSelection, setPendingGuidedSelection] = useState<PendingGuidedSelection | null>(null);
  const [pendingIssueConfirm, setPendingIssueConfirm] = useState<PendingIssueConfirm | null>(null);
  const [pendingIssueCaseSelection, setPendingIssueCaseSelection] = useState<PendingIssueCaseSelection | null>(null);
  const [pendingIssueSopConfirm, setPendingIssueSopConfirm] = useState<PendingIssueSopConfirm | null>(null);
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
  const guidedThreadIdRef = useRef<string | null>(null);
  const guidedQuestionRef = useRef<string | null>(null);
  const issueDecisionSubmittingRef = useRef(false);
  const consumedIssueDecisionKeysRef = useRef<Set<string>>(new Set());
  const threadIdRef = useRef<string | null>(null);
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
    (
      res: AgentResponse,
      assistantId: string,
      fallbackQuestion: string,
      issueResumeContext: IssueResumeContext = DEFAULT_ISSUE_RESUME_CONTEXT
    ) => {
      const metadataQueries = toStringArray(res.metadata?.search_queries);
      const responseQueries = Array.isArray(res.search_queries)
        ? res.search_queries.filter((q: unknown): q is string => typeof q === "string" && q.trim().length > 0)
        : [];
      const effectiveSearchQueries = metadataQueries.length > 0 ? metadataQueries : responseQueries;

      // Always capture thread_id from every response (interrupted or not)
      const tid = typeof res.thread_id === "string" ? res.thread_id.trim() : "";
      if (tid) {
        threadIdRef.current = tid;
      }

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
            : kind === "auto_parse_confirm"
              ? ""
              : "검색 결과를 확인한 뒤 승인/거절/키워드를 입력하세요.";

        if (kind === "auto_parse_confirm") {
          if (threadId && payload) {
            guidedThreadIdRef.current = threadId;
            guidedQuestionRef.current = question;

            const defaultsRaw = isRecord(payload.defaults) ? payload.defaults : {};
            const guidedSteps = resolveGuidedSteps(payload);
            const includeEquipStep = guidedSteps.includes("equip_id");
            const initialDecision = {
              type: "auto_parse_confirm" as const,
              target_language: normalizeLanguage(defaultsRaw.target_language),
              selected_device: typeof defaultsRaw.device === "string" ? defaultsRaw.device : null,
              selected_equip_id:
                includeEquipStep && typeof defaultsRaw.equip_id === "string"
                  ? defaultsRaw.equip_id
                  : null,
              task_mode: normalizeTaskMode(defaultsRaw.task_mode),
            };
            setPendingGuidedSelection({
              threadId,
              question,
              instruction,
              payload,
              stepIndex: 0,
              draftDecision: initialDecision,
            });
          }
          setPendingInterrupt(null);
          setPendingIssueConfirm(null);
          setPendingIssueCaseSelection(null);
          setPendingIssueSopConfirm(null);

          updateMessage(assistantId, (m) => ({
            ...m,
            content: instruction,
            retrievedDocs: res.retrieved_docs || [],
            rawAnswer: JSON.stringify(res, null, 2),
            currentNode: null,
            sessionId,
            searchQueries: effectiveSearchQueries.length > 0 ? effectiveSearchQueries : null,
          }));
          setIsStreaming(false);
          return;
        }

        if (kind === "issue_confirm") {
          const nonce = typeof payload?.nonce === "string" ? payload.nonce : "";
          if (nonce) {
            consumedIssueDecisionKeysRef.current.delete(`issue_confirm:${nonce}`);
          }
          const stage = payload?.stage === "post_detail" ? "post_detail" : "post_summary";
          const prompt = typeof payload?.prompt === "string" && payload.prompt.trim()
            ? payload.prompt
            : "상세히 보고싶은 문서가 있습니까?";
          if (threadId && nonce) {
            setPendingIssueConfirm({
              threadId,
              question,
              instruction,
              payload: {
                type: "issue_confirm",
                nonce,
                stage,
                prompt,
              },
            });
          }
          setPendingIssueCaseSelection(null);
          setPendingIssueSopConfirm(null);
          setPendingInterrupt(null);
          setPendingGuidedSelection(null);
          issueDecisionSubmittingRef.current = false;
          updateMessage(assistantId, (m) => ({
            ...m,
            content: resolveInterruptDisplayContent({
              kind,
              instruction,
              answer: res.answer,
              messages,
            }),
            retrievedDocs: res.retrieved_docs || [],
            rawAnswer: JSON.stringify(res, null, 2),
            currentNode: null,
            sessionId,
            searchQueries: effectiveSearchQueries.length > 0 ? effectiveSearchQueries : null,
          }));
          setIsStreaming(false);
          return;
        }

        if (kind === "issue_case_selection") {
          const nonce = typeof payload?.nonce === "string" ? payload.nonce : "";
          if (nonce) {
            consumedIssueDecisionKeysRef.current.delete(`issue_case_selection:${nonce}`);
          }
          const cases = toIssueCases(payload?.cases);
          if (threadId && nonce) {
            setPendingIssueCaseSelection({
              threadId,
              question,
              instruction,
              payload: {
                type: "issue_case_selection",
                nonce,
                cases,
              },
            });
          }
          setPendingIssueConfirm(null);
          setPendingIssueSopConfirm(null);
          setPendingInterrupt(null);
          setPendingGuidedSelection(null);
          issueDecisionSubmittingRef.current = false;
          updateMessage(assistantId, (m) => ({
            ...m,
            content: resolveInterruptDisplayContent({
              kind,
              instruction,
              answer: res.answer,
              messages,
            }),
            retrievedDocs: res.retrieved_docs || [],
            rawAnswer: JSON.stringify(res, null, 2),
            currentNode: null,
            sessionId,
            searchQueries: effectiveSearchQueries.length > 0 ? effectiveSearchQueries : null,
          }));
          setIsStreaming(false);
          return;
        }

        if (kind === "issue_sop_confirm") {
          const nonce = typeof payload?.nonce === "string" ? payload.nonce : "";
          if (nonce) {
            consumedIssueDecisionKeysRef.current.delete(`issue_sop_confirm:${nonce}`);
          }
          const prompt = typeof payload?.prompt === "string" && payload.prompt.trim()
            ? payload.prompt
            : "SOP 확인으로 이어가시겠습니까?";
          const hasSopRef = Boolean(payload?.has_sop_ref);
          const sopHint = typeof payload?.sop_hint === "string" ? payload.sop_hint : null;
          if (threadId && nonce) {
            setPendingIssueSopConfirm({
              threadId,
              question,
              instruction,
              payload: {
                type: "issue_sop_confirm",
                nonce,
                prompt,
                has_sop_ref: hasSopRef,
                sop_hint: sopHint,
              },
            });
          }
          setPendingIssueConfirm(null);
          setPendingIssueCaseSelection(null);
          setPendingInterrupt(null);
          setPendingGuidedSelection(null);
          issueDecisionSubmittingRef.current = false;
          updateMessage(assistantId, (m) => ({
            ...m,
            content: resolveInterruptDisplayContent({
              kind,
              instruction,
              answer: res.answer,
              messages,
            }),
            retrievedDocs: res.retrieved_docs || [],
            rawAnswer: JSON.stringify(res, null, 2),
            currentNode: null,
            sessionId,
            searchQueries: effectiveSearchQueries.length > 0 ? effectiveSearchQueries : null,
          }));
          setIsStreaming(false);
          return;
        }
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
      setPendingGuidedSelection(null);
      setPendingIssueConfirm(null);
      setPendingIssueCaseSelection(null);
      setPendingIssueSopConfirm(null);
      issueDecisionSubmittingRef.current = false;
      consumedIssueDecisionKeysRef.current.clear();
      guidedThreadIdRef.current = null;
      guidedQuestionRef.current = null;
      clearLogs();

      // Set completed retrieved docs in context if available
      // Fallback to all_retrieved_docs if retrieved_docs is empty (e.g., regeneration with no matching docs)
      const docsToShow = (res.retrieved_docs && res.retrieved_docs.length > 0)
        ? res.retrieved_docs
        : (res.all_retrieved_docs && res.all_retrieved_docs.length > 0)
          ? res.all_retrieved_docs
          : null;

      // SOP 절차조회: 답변에 사용된 문서 1개의 전체 페이지만 오른쪽 pane에 표시
      const isSopResponse =
        (typeof res.metadata?.selected_task_mode === "string" &&
          res.metadata.selected_task_mode.toLowerCase() === "sop") ||
        (Array.isArray(res.selected_doc_types) &&
          res.selected_doc_types.some(
            (dt) => typeof dt === "string" && dt.toLowerCase() === "sop"
          ));

      const sopExpandedDoc = isSopResponse && Array.isArray(res.expanded_docs) && res.expanded_docs.length > 0
        ? res.expanded_docs[0]
        : null;
      const sopFinalDocId = sopExpandedDoc?.doc_id ?? null;

      if (docsToShow && sopFinalDocId) {
        // 답변에 사용된 문서 1개의 전체 페이지를 오른쪽 pane에 표시
        const allDocs = (res.all_retrieved_docs && res.all_retrieved_docs.length > 0)
          ? res.all_retrieved_docs
          : docsToShow;
        const sameDocChunks = allDocs.filter((d) => d.id === sopFinalDocId);

        if (sameDocChunks.length > 0) {
          const base = sameDocChunks[0];
          // 우선 base chunk 표시 후, 비동기로 전체 페이지 수를 조회하여 업데이트
          setCompletedRetrievedDocs([base]);

          // 확장 섹션의 시작 페이지 우선, 없으면 검색 히트 페이지
          const sectionStartPage = typeof sopExpandedDoc?.start_page === "number"
            ? sopExpandedDoc.start_page
            : null;
          const answerPage = sectionStartPage ?? (typeof base.page === "number" ? base.page : 1);

          fetch(`/api/assets/docs/${encodeURIComponent(sopFinalDocId)}/info`)
            .then((r) => r.ok ? r.json() : null)
            .then((info: { total_pages?: number } | null) => {
              const totalPages = info?.total_pages;
              if (typeof totalPages === "number" && totalPages > 0) {
                const allPages = Array.from({ length: totalPages }, (_, i) => i + 1);
                const allPageUrls = allPages.map((p) => `/api/assets/docs/${sopFinalDocId}/pages/${p}`);
                const fullDoc: RetrievedDoc = {
                  ...base,
                  expanded_pages: allPages,
                  expanded_page_urls: allPageUrls,
                  page: answerPage,
                  page_image_url: allPageUrls[0],
                };
                setCompletedRetrievedDocs([fullDoc]);
              }
            })
            .catch(() => { /* 실패 시 base chunk 유지 */ });
        } else {
          setCompletedRetrievedDocs(docsToShow);
        }
      } else if (docsToShow) {
        setCompletedRetrievedDocs(docsToShow);
      } else {
        setCompletedRetrievedDocs(null);
      }

      const assistantAnswer = typeof res.answer === "string" ? res.answer : "";
      const hasDuplicateIssueAnswer =
        issueResumeContext.isIssueResume &&
        hasPriorAssistantContentMatch(messages, assistantAnswer);
      const finalAssistantContent = hasDuplicateIssueAnswer
        ? buildIssueResumeDuplicateAck(issueResumeContext)
        : assistantAnswer;

      updateMessage(assistantId, (m) => ({
        ...m,
        content: finalAssistantContent,
        retrievedDocs: res.retrieved_docs || [],
        allRetrievedDocs: res.all_retrieved_docs || [],  // 재생성용 전체 문서 (최대 retrieval_top_k개)
        expandedDocs: res.expanded_docs ?? null,
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
        relatedDocTypes: res.related_doc_types ?? m.relatedDocTypes ?? null,
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
      const hasCompletedAnswer = typeof res.answer === "string" && res.answer.trim().length > 0;
      const shouldShowMissingDevicePrompt =
        shouldSuggestAdditionalDeviceSearch && !hasCompletedAnswer;

      updateMessage(assistantId, (m) => ({
        ...m,
        suggestAdditionalDeviceSearch: shouldShowMissingDevicePrompt,
      }));

      if (shouldShowMissingDevicePrompt) {
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
      } else {
        setPendingRegeneration(null);
      }

      // Save turn to backend
      const userText = currentUserTextRef.current;
      const assistantText = finalAssistantContent;
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
        ...(res.metadata ? {
          retrieval_meta: {
            mq_mode: res.metadata.mq_mode,
            mq_used: res.metadata.mq_used,
            mq_reason: res.metadata.mq_reason,
            route: res.metadata.route,
            st_gate: res.metadata.st_gate,
            attempts: res.metadata.attempts,
            retry_strategy: res.metadata.retry_strategy,
            search_queries: effectiveSearchQueries,
          },
        } : {}),
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
    [
      sessionId,
      updateMessage,
      setIsStreaming,
      clearLogs,
      setCompletedRetrievedDocs,
      setPendingRegeneration,
      messages,
    ]
  );

  const send = useCallback(
    async ({ text, decisionOverride, suppressUserMessage, overrides }: SendOptions) => {
      stop();
      setError(null);
      setPendingRegeneration(null);
      const hilPending = pendingInterrupt;
      const decisionRecord = isRecord(decisionOverride) ? decisionOverride : null;
      const decisionType = typeof decisionRecord?.type === "string" ? decisionRecord.type : "";
      const isGuidedResume = decisionType === "auto_parse_confirm" && Boolean(guidedThreadIdRef.current);
      const issueThreadId = pendingIssueConfirm?.threadId ?? pendingIssueCaseSelection?.threadId ?? pendingIssueSopConfirm?.threadId ?? null;
      const isIssueResume = (
        decisionType === "issue_confirm" ||
        decisionType === "issue_case_selection" ||
        decisionType === "issue_sop_confirm"
      ) && Boolean(issueThreadId);
      const isHilResume = Boolean(hilPending);
      const isResume = isHilResume || isGuidedResume || isIssueResume;

      if (!isResume) {
        setPendingIssueConfirm(null);
        setPendingIssueCaseSelection(null);
        setPendingIssueSopConfirm(null);
        issueDecisionSubmittingRef.current = false;
      }

      const resumeThreadId = isHilResume
        ? (hilPending?.threadId ?? "")
        : isGuidedResume
          ? (guidedThreadIdRef.current ?? "")
          : isIssueResume
            ? (issueThreadId ?? "")
          : "";
      if (isResume && !resumeThreadId) {
        setError("thread_id가 없어 검색 결과 확인을 이어갈 수 없습니다.");
        return;
      }

      // Only update user text if not resuming (keep original question for saves)
      if (!isResume) {
        currentUserTextRef.current = text;
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

      const requestMessage = isHilResume
        ? (hilPending?.question ?? text)
        : isGuidedResume
          ? (guidedQuestionRef.current ?? text)
          : isIssueResume
            ? (pendingIssueConfirm?.question ?? pendingIssueCaseSelection?.question ?? pendingIssueSopConfirm?.question ?? text)
          : text;

      const decision = isHilResume
        ? (decisionOverride ?? resolveDecision(text))
        : isGuidedResume
          ? decisionRecord
          : isIssueResume
            ? decisionRecord
          : undefined;

      const issueResumeContext: IssueResumeContext = isIssueResume
        ? {
            isIssueResume: true,
            decisionType:
              decisionType === "issue_confirm" ||
              decisionType === "issue_case_selection" ||
              decisionType === "issue_sop_confirm"
                ? decisionType
                : null,
            confirm: typeof decisionRecord?.confirm === "boolean" ? decisionRecord.confirm : null,
          }
        : DEFAULT_ISSUE_RESUME_CONTEXT;

      if (!suppressUserMessage) {
        appendMessage({
          id: userId,
          role: "user",
          content: text,
          sessionId,
        });
      }

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

        const hasUserFilters = Boolean(overrides?.filterDevices?.length || overrides?.filterDocTypes?.length);
        const autoParseEnabled = hasUserFilters ? false : (overrides?.autoParse ?? !Boolean(overrides));
        const guidedConfirmEnabled = !isResume && autoParseEnabled && !hasUserFilters;
        const resumePayload: Partial<AgentRequest> = isHilResume
          ? {
              thread_id: resumeThreadId,
              resume_decision: decision,
              auto_parse: false,
              ask_user_after_retrieve: true,
            }
          : isGuidedResume
            ? {
                thread_id: resumeThreadId,
                resume_decision: decision,
                auto_parse: false,
                ask_user_after_retrieve: false,
              }
            : isIssueResume
              ? {
                  thread_id: resumeThreadId,
                  resume_decision: decision,
                  auto_parse: false,
                  ask_user_after_retrieve: false,
                }
            : {};

        const payload: AgentRequest = {
          message: requestMessage,
          ...(!isResume
            ? {
                auto_parse: autoParseEnabled,  // 자동 파싱 모드 활성화 (기본값)
                guided_confirm: guidedConfirmEnabled,
                ask_user_after_retrieve: false,  // 문서 선택 UI 비활성화
              }
            : {}),
          ...(chatHistory ? { chat_history: chatHistory } : {}),
          ...(overrides ? {
            filter_devices: overrides.filterDevices,
            filter_doc_types: overrides.filterDocTypes,
            search_queries: overrides.searchQueries,
            selected_doc_ids: overrides.selectedDocIds,
            ...(overrides.contextChunkIds?.length ? { context_chunk_ids: overrides.contextChunkIds } : {}),
          } : {}),
          ...resumePayload,
          ...(!isResume && threadIdRef.current
            ? { thread_id: threadIdRef.current }
            : {}),
        };
        const { canStream, streamPath } = resolveChatPaths(env.chatPath);
        if (!canStream) {
          const res = await sendChatMessage(payload);
          handleAgentResponse(res, assistantId, requestMessage, issueResumeContext);
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
                  handleAgentResponse(res, assistantId, requestMessage, issueResumeContext);
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
    [
      appendMessage,
      stop,
      updateMessage,
      handleAgentResponse,
      pendingInterrupt,
      pendingGuidedSelection,
      pendingIssueConfirm,
      pendingIssueCaseSelection,
      pendingIssueSopConfirm,
      sessionId,
      addLog,
      clearLogs,
      setPendingRegeneration,
    ]
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

  const submitGuidedSelectionNumber = useCallback(
    (token: string) => {
      const trimmed = token.trim();
      if (!pendingGuidedSelection) return;
      if (!/^[0-9]+$/.test(trimmed)) return;

      const num = parseInt(trimmed, 10);
      if (!Number.isFinite(num)) return;

      const steps = resolveGuidedSteps(pendingGuidedSelection.payload);
      const stepIndex = pendingGuidedSelection.stepIndex ?? 0;
      const step = steps[Math.min(stepIndex, steps.length - 1)] ?? "device";

      const optionsRaw = isRecord(pendingGuidedSelection.payload.options)
        ? pendingGuidedSelection.payload.options
        : {};

      const stepOptions =
        step === "language"
          ? toGuidedOptions(optionsRaw.language)
          : step === "device"
            ? toGuidedOptions(optionsRaw.device)
            : step === "equip_id"
              ? toGuidedOptions(optionsRaw.equip_id)
              : toGuidedOptions(optionsRaw.task);
      if (stepOptions.length === 0) return;

      const pickIndex = (n: number): number | null => {
        if (n === 0) {
          const recIdx = stepOptions.findIndex((o) => Boolean(o.recommended));
          if (recIdx >= 0) return recIdx;
          const skipIdx = stepOptions.findIndex((o) => o.value === "__skip__");
          if (skipIdx >= 0) return skipIdx;
          return 0;
        }
        if (n < 1 || n > stepOptions.length) return null;
        return n - 1;
      };

      const idx = pickIndex(num);
      if (idx === null) return;
      const chosen = stepOptions[idx];
      if (!chosen) return;

      const prevDecision = pendingGuidedSelection.draftDecision ?? { type: "auto_parse_confirm" as const };
      const nextDecision: NonNullable<PendingGuidedSelection["draftDecision"]> = {
        ...prevDecision,
        type: "auto_parse_confirm" as const,
      };

      if (step === "language") {
        nextDecision.target_language = normalizeLanguage(chosen.value);
      } else if (step === "device") {
        nextDecision.selected_device = chosen.value === "__skip__" ? null : chosen.value;
      } else if (step === "equip_id") {
        if (chosen.value === "__skip__" || chosen.value === "__manual__") {
          nextDecision.selected_equip_id = null;
        } else {
          nextDecision.selected_equip_id = chosen.value;
        }
      } else {
        nextDecision.task_mode = normalizeTaskMode(chosen.value);
      }

      const isLastStep = stepIndex >= steps.length - 1;
      if (isLastStep) {
        const includeEquipStep = steps.includes("equip_id");
        const finalDecision = {
          type: "auto_parse_confirm" as const,
          target_language: normalizeLanguage(nextDecision.target_language),
          selected_device: nextDecision.selected_device ?? null,
          selected_equip_id: includeEquipStep ? (nextDecision.selected_equip_id ?? null) : null,
          task_mode: normalizeTaskMode(nextDecision.task_mode),
        };
        setPendingGuidedSelection(null);
        send({
          text: buildGuidedSummaryText({
            selectedDevice: finalDecision.selected_device,
            selectedEquipId: finalDecision.selected_equip_id,
            taskMode: finalDecision.task_mode,
            includeEquipStep,
          }),
          decisionOverride: finalDecision,
        });
        return;
      }

      setPendingGuidedSelection((prev) => {
        if (!prev) return prev;
        const currentSteps = resolveGuidedSteps(prev.payload);
        const nextStepIndex = Math.min((prev.stepIndex ?? 0) + 1, currentSteps.length - 1);
        return {
          ...prev,
          stepIndex: nextStepIndex,
          draftDecision: nextDecision,
        };
      });
    },
    [pendingGuidedSelection, send]
  );

  const submitGuidedSelectionFinal = useCallback(
    (decision: {
      type: "auto_parse_confirm";
      target_language: "ko" | "en" | "zh" | "ja";
      selected_device?: string | null;
      selected_equip_id?: string | null;
      task_mode: "sop" | "issue" | "all";
    }) => {
      if (!pendingGuidedSelection) return;
      const currentSteps = resolveGuidedSteps(pendingGuidedSelection.payload);
      const includeEquipStep = currentSteps.includes("equip_id");
      const finalDecision = {
        ...decision,
        selected_equip_id: includeEquipStep ? (decision.selected_equip_id ?? null) : null,
      };
      setPendingGuidedSelection(null);
      send({
        text: buildGuidedSummaryText({
          selectedDevice: finalDecision.selected_device ?? null,
          selectedEquipId: finalDecision.selected_equip_id,
          taskMode: finalDecision.task_mode,
          includeEquipStep,
        }),
        decisionOverride: finalDecision,
      });
    },
    [pendingGuidedSelection, send]
  );

  const submitIssueConfirm = useCallback(
    (confirm: boolean, options?: { silent?: boolean }) => {
      if (!pendingIssueConfirm) return;
      if (issueDecisionSubmittingRef.current) return;
      const dedupeKey = `issue_confirm:${pendingIssueConfirm.payload.nonce}`;
      if (consumedIssueDecisionKeysRef.current.has(dedupeKey)) return;
      consumedIssueDecisionKeysRef.current.add(dedupeKey);
      issueDecisionSubmittingRef.current = true;
      const decision = {
        type: "issue_confirm" as const,
        nonce: pendingIssueConfirm.payload.nonce,
        stage: pendingIssueConfirm.payload.stage,
        confirm,
      };
      setPendingIssueConfirm(null);
      void send({
        text: options?.silent ? "" : `이슈 확인: ${confirm ? "예" : "아니오"}`,
        suppressUserMessage: Boolean(options?.silent),
        decisionOverride: decision,
      }).finally(() => {
        issueDecisionSubmittingRef.current = false;
      });
    },
    [pendingIssueConfirm, send]
  );

  const submitIssueCaseSelection = useCallback(
    (selectedDocId: string) => {
      if (!pendingIssueCaseSelection) return;
      if (issueDecisionSubmittingRef.current) return;
      const trimmed = selectedDocId.trim();
      if (!trimmed) return;
      const dedupeKey = `issue_case_selection:${pendingIssueCaseSelection.payload.nonce}`;
      if (consumedIssueDecisionKeysRef.current.has(dedupeKey)) return;
      consumedIssueDecisionKeysRef.current.add(dedupeKey);
      issueDecisionSubmittingRef.current = true;
      const decision = {
        type: "issue_case_selection" as const,
        nonce: pendingIssueCaseSelection.payload.nonce,
        selected_doc_id: trimmed,
      };
      setPendingIssueCaseSelection(null);
      void send({
        text: `이슈 사례 선택: ${trimmed}`,
        decisionOverride: decision,
      }).finally(() => {
        issueDecisionSubmittingRef.current = false;
      });
    },
    [pendingIssueCaseSelection, send]
  );

  const submitIssueSopConfirm = useCallback(
    (confirm: boolean, options?: { silent?: boolean }) => {
      if (!pendingIssueSopConfirm) return;
      if (issueDecisionSubmittingRef.current) return;
      const dedupeKey = `issue_sop_confirm:${pendingIssueSopConfirm.payload.nonce}`;
      if (consumedIssueDecisionKeysRef.current.has(dedupeKey)) return;
      consumedIssueDecisionKeysRef.current.add(dedupeKey);
      issueDecisionSubmittingRef.current = true;
      const decision = {
        type: "issue_sop_confirm" as const,
        nonce: pendingIssueSopConfirm.payload.nonce,
        confirm,
      };
      setPendingIssueSopConfirm(null);
      void send({
        text: options?.silent ? "" : `SOP 확인: ${confirm ? "예" : "아니오"}`,
        suppressUserMessage: Boolean(options?.silent),
        decisionOverride: decision,
      }).finally(() => {
        issueDecisionSubmittingRef.current = false;
      });
    },
    [pendingIssueSopConfirm, send]
  );

  const submitAbbreviationResolve = useCallback(
    (selections: Record<string, string>) => {
      if (!pendingInterrupt || pendingInterrupt.kind !== "abbreviation_resolve") return;

      const payload = toAbbreviationResolvePayload(pendingInterrupt.payload);
      if (!payload) {
        setError("약어 선택 정보를 확인할 수 없습니다.");
        return;
      }

      const normalized: Record<string, string> = {};
      for (const item of payload.abbreviations) {
        const raw = selections[item.abbr_key];
        const value = typeof raw === "string" ? raw.trim() : "";
        if (!value) {
          setError("모든 약어의 의미를 선택해야 합니다.");
          return;
        }
        normalized[item.abbr_key] = value;
      }

      setPendingInterrupt(null);
      void send({
        text: "약어 의미를 선택했습니다.",
        suppressUserMessage: true,
        decisionOverride: {
          type: "abbreviation_resolve",
          selections: normalized,
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
    setPendingGuidedSelection(null);
    setPendingIssueConfirm(null);
    setPendingIssueCaseSelection(null);
    setPendingIssueSopConfirm(null);
    issueDecisionSubmittingRef.current = false;
    consumedIssueDecisionKeysRef.current.clear();
    guidedThreadIdRef.current = null;
    guidedQuestionRef.current = null;
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
    threadIdRef.current = null;
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
        setPendingIssueConfirm(null);
        setPendingIssueCaseSelection(null);
        setPendingIssueSopConfirm(null);
        issueDecisionSubmittingRef.current = false;
        consumedIssueDecisionKeysRef.current.clear();
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
      pendingGuidedSelection,
      pendingIssueConfirm,
      pendingIssueCaseSelection,
      pendingIssueSopConfirm,
      pendingAbbreviationResolve:
        pendingInterrupt?.kind === "abbreviation_resolve"
          ? (() => {
              const payload = toAbbreviationResolvePayload(pendingInterrupt.payload);
              if (!payload) return null;
              return {
                threadId: pendingInterrupt.threadId,
                question: pendingInterrupt.question,
                instruction: pendingInterrupt.instruction,
                payload,
              } satisfies PendingAbbreviationResolve;
            })()
          : null,
      pendingReview: pendingInterrupt?.kind === "retrieval_review" ? pendingInterrupt : null,
      pendingDeviceSelection: pendingInterrupt?.kind === "device_selection" ? pendingInterrupt : null,
      submitGuidedSelectionNumber,
      submitGuidedSelectionFinal,
      submitIssueConfirm,
      submitIssueCaseSelection,
      submitIssueSopConfirm,
      submitAbbreviationResolve,
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
            : pendingInterrupt.kind === "abbreviation_resolve"
              ? "약어 의미를 선택해 주세요..."
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
      pendingGuidedSelection,
      pendingIssueConfirm,
      pendingIssueCaseSelection,
      pendingIssueSopConfirm,
      pendingInterrupt,
      submitGuidedSelectionNumber,
      submitGuidedSelectionFinal,
      submitIssueConfirm,
      submitIssueCaseSelection,
      submitIssueSopConfirm,
      submitAbbreviationResolve,
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
