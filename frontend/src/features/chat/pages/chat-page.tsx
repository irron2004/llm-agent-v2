import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useChatSession } from "../hooks/use-chat-session";
import { useChatHistoryContext } from "../context/chat-history-context";
import { useChatReview } from "../context/chat-review-context";
import {
  ChatContainer,
  MessageList,
  InputArea,
  MessageItem,
  ChatInput,
  DeviceSelectionPanel,
  AbbreviationResolvePanel,
  GuidedSelectionPanel,
} from "../components";
import type { RegeneratePayload } from "../components/message-item";
import { fetchDeviceCatalog } from "../api";
import type { DeviceCatalogResponse } from "../types";
import { Alert, Spin } from "antd";

const PRESET_SOP_TYPES = ["SOP", "set_up_manual"];
const PRESET_ISSUE_TYPES = ["myservice", "gcb", "trouble_shooting_guide"];

export default function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const sessionParam = searchParams.get("session");

  const { refresh: refreshHistory } = useChatHistoryContext();
  const {
    setPendingReview,
    pendingRegeneration,
    setIsStreaming,
    registerSubmitHandlers,
    setPendingRegeneration,
    registerRegenerationHandlers,
  } = useChatReview();
  const [suggestedDevices, setSuggestedDevices] = useState<Array<{ name: string; doc_count: number }>>([]);
  const [isLoadingSuggestedDevices, setIsLoadingSuggestedDevices] = useState(false);
  const [suggestedDeviceError, setSuggestedDeviceError] = useState<string | null>(null);
  const [showSuggestedDevicePanel, setShowSuggestedDevicePanel] = useState(false);
  const [lastSelectedDevice, setLastSelectedDevice] = useState<string | null>(null);

  // Filter bar state
  const [catalog, setCatalog] = useState<DeviceCatalogResponse | null>(null);
  const [selectedDocTypes, setSelectedDocTypes] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedEquip, setSelectedEquip] = useState<string | null>(null);

  // Load device catalog on mount for filter bar
  useEffect(() => {
    const p = fetchDeviceCatalog();
    if (p && typeof p.then === "function") {
      p.then((res) => { if (res) setCatalog(res); }).catch(() => {});
    }
  }, []);

  const docTypeOptions = useMemo(() => {
    const presets = [
      { label: "작업절차검색 (SOP + Setup)", value: "__preset_sop", isPreset: true },
      { label: "이슈검색 (myservice + gcb + TS)", value: "__preset_issue", isPreset: true },
    ];
    const individual = [
      { label: "myservice", value: "myservice" },
      { label: "SOP", value: "SOP" },
      { label: "gcb", value: "gcb" },
      { label: "set_up_manual", value: "set_up_manual" },
      { label: "trouble_shooting_guide", value: "trouble_shooting_guide" },
    ];
    return [...presets, ...individual];
  }, []);

  const modelOptions = useMemo(() =>
    (catalog?.devices ?? []).map((d) => ({ label: d.name, value: d.name })),
  [catalog]);

  const handleDocTypesChange = useCallback((types: string[]) => {
    const expanded = types.flatMap((t) => {
      if (t === "__preset_sop") return PRESET_SOP_TYPES;
      if (t === "__preset_issue") return PRESET_ISSUE_TYPES;
      return [t];
    });
    setSelectedDocTypes([...new Set(expanded)]);
  }, []);

  const handleTurnSaved = useCallback(() => {
    refreshHistory();
  }, [refreshHistory]);

  const {
    sessionId,
    messages,
    send,
    stop,
    isStreaming,
    isLoadingSession,
    error,
    reset,
    loadSession,
    inputPlaceholder,
    pendingReview,
    pendingDeviceSelection,
    pendingGuidedSelection,
    pendingAbbreviationResolve,
    pendingIssueConfirm,
    pendingIssueCaseSelection,
    pendingIssueSopConfirm,
    submitGuidedSelectionNumber,
    submitGuidedSelectionFinal,
    submitAbbreviationResolve,
    submitIssueConfirm,
    submitIssueCaseSelection,
    submitIssueSopConfirm,
    submitReview,
    submitSearchQueries,
    submitDeviceSelection,
    submitFeedback,
    submitDetailedFeedback,
  } = useChatSession({ onTurnSaved: handleTurnSaved });

  void sessionId;

  useEffect(() => {
    if (sessionParam) {
      setSearchParams({}, { replace: true });
      loadSession(sessionParam);
    }
  }, [sessionParam, loadSession, setSearchParams]);

  useEffect(() => {
    registerSubmitHandlers({ submitReview, submitSearchQueries });
  }, [submitReview, submitSearchQueries, registerSubmitHandlers]);

  useEffect(() => {
    if (!pendingIssueConfirm) return;
    submitIssueConfirm(true, { silent: true });
  }, [pendingIssueConfirm, submitIssueConfirm]);

  // issue_sop_confirm은 자동 확인하지 않고, 인라인 버튼으로 사용자가 선택

  useEffect(() => {
    if (!pendingIssueCaseSelection || pendingIssueCaseSelection.payload.cases.length === 0) return;

    const handleKeydown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.isComposing) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;

      if (!/^[1-9]$/.test(event.key)) return;
      const selectedIndex = Number.parseInt(event.key, 10) - 1;
      const selected = pendingIssueCaseSelection.payload.cases[selectedIndex];
      if (!selected) return;
      event.preventDefault();
      submitIssueCaseSelection(selected.doc_id);
    };

    window.addEventListener("keydown", handleKeydown);
    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  }, [pendingIssueCaseSelection, submitIssueCaseSelection]);

  useEffect(() => {
    setShowSuggestedDevicePanel(false);
  }, [pendingRegeneration?.messageId, pendingRegeneration?.reason]);

  useEffect(() => {
    if (
      pendingRegeneration?.reason === "missing_device_parse" &&
      lastSelectedDevice
    ) {
      setPendingRegeneration(null);
    }
  }, [pendingRegeneration?.messageId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let active = true;
    const shouldLoadCatalog =
      pendingRegeneration?.reason === "missing_device_parse" &&
      showSuggestedDevicePanel;
    if (!shouldLoadCatalog) {
      setSuggestedDevices([]);
      setSuggestedDeviceError(null);
      setIsLoadingSuggestedDevices(false);
      return () => { active = false; };
    }

    setIsLoadingSuggestedDevices(true);
    setSuggestedDeviceError(null);
    fetchDeviceCatalog()
      .then((res) => {
        if (!active) return;
        setSuggestedDevices(Array.isArray(res.devices) ? res.devices : []);
      })
      .catch(() => {
        if (!active) return;
        setSuggestedDevices([]);
        setSuggestedDeviceError("장비 목록을 불러오지 못했습니다.");
      })
      .finally(() => {
        if (!active) return;
        setIsLoadingSuggestedDevices(false);
      });

    return () => { active = false; };
  }, [pendingRegeneration?.reason, pendingRegeneration?.messageId, showSuggestedDevicePanel]);

  const sanitizeRegenerationQuery = useCallback((query: string): string => {
    let normalized = query.trim();
    if (!normalized) return "";
    normalized = normalized.replace(/^(?:\[\s*regenerate with[^\]]*\]\s*)+/gi, "").trim();
    normalized = normalized.replace(/^(?:regenerate with\b[^:\n]*[:\-]?\s*)+/gi, "").trim();
    normalized = normalized.replace(/^(?:재검색\s*(?:조건|필터)?\s*[:\-]?\s*)+/g, "").trim();
    if (/^[.\s…·•\-_~=]+$/.test(normalized)) return "";
    return normalized;
  }, []);

  const submitRegeneration = useCallback((payload: {
    originalQuery: string;
    searchQueries: string[];
    selectedDevices: string[];
    selectedDocTypes: string[];
    selectedDocIds: string[];
  }) => {
    const normalizedOriginalQuery = sanitizeRegenerationQuery(payload.originalQuery);
    const normalizedQueries = payload.searchQueries
      .map((query) => sanitizeRegenerationQuery(query))
      .filter((query) => query.length > 0);

    setPendingRegeneration(null);
    send({
      text: normalizedOriginalQuery || payload.originalQuery,
      overrides: {
        filterDevices: payload.selectedDevices,
        filterDocTypes: payload.selectedDocTypes,
        searchQueries: normalizedQueries.length > 0
          ? normalizedQueries
          : (normalizedOriginalQuery ? [normalizedOriginalQuery] : []),
        selectedDocIds: payload.selectedDocIds,
        autoParse: false,
      },
    });
  }, [send, setPendingRegeneration, sanitizeRegenerationQuery]);

  useEffect(() => {
    registerRegenerationHandlers({ submitRegeneration });
  }, [submitRegeneration, registerRegenerationHandlers]);

  useEffect(() => {
    setIsStreaming(isStreaming);
  }, [isStreaming, setIsStreaming]);

  useEffect(() => {
    if (pendingReview) {
      const queries = pendingReview.payload?.search_queries;
      const searchQueries = Array.isArray(queries)
        ? queries.map((q) => String(q))
        : [pendingReview.question];
      setPendingReview({
        threadId: pendingReview.threadId,
        question: pendingReview.question,
        instruction: pendingReview.instruction,
        docs: pendingReview.docs,
        searchQueries,
      });
    } else {
      setPendingReview(null);
    }
  }, [pendingReview, setPendingReview]);

  useEffect(() => {
    const handleNewChat = () => {
      reset();
      setPendingRegeneration(null);
    };
    window.addEventListener("pe-agent:new-chat", handleNewChat);
    return () => {
      window.removeEventListener("pe-agent:new-chat", handleNewChat);
    };
  }, [reset, setPendingRegeneration]);

  const handleDeviceSelect = (deviceIndex: number) => {
    if (!pendingRegeneration) return;
    const deviceName = suggestedDevices[deviceIndex].name;
    const originalQuery = pendingRegeneration.originalQuery;
    const displayText = `[${deviceName}] ${originalQuery}`;
    const searchQueries =
      pendingRegeneration.searchQueries?.length > 0
        ? pendingRegeneration.searchQueries
        : [originalQuery].filter((q) => q?.trim());

    setLastSelectedDevice(deviceName);
    setPendingRegeneration(null);
    send({
      text: displayText,
      overrides: {
        filterDevices: [deviceName],
        searchQueries,
        autoParse: false,
      },
    });
  };

  const handleSend = async (text: string) => {
    if (pendingGuidedSelection) {
      const trimmed = text.trim();
      if (trimmed === "0" || /^[1-9][0-9]*$/.test(trimmed)) {
        submitGuidedSelectionNumber(trimmed);
      }
      return;
    }

    if (
      pendingRegeneration?.reason === "missing_device_parse" &&
      !lastSelectedDevice
    ) {
      const trimmed = text.trim();
      if (!showSuggestedDevicePanel) {
        if (trimmed === "1") {
          setShowSuggestedDevicePanel(true);
          return;
        }
        if (trimmed === "2") {
          setPendingRegeneration(null);
          return;
        }
        return;
      }
      if (suggestedDevices.length > 0 && !isLoadingSuggestedDevices) {
        const num = parseInt(trimmed, 10);
        if (num === 0) {
          setPendingRegeneration(null);
          return;
        }
        if (!isNaN(num) && num >= 1 && num <= suggestedDevices.length) {
          handleDeviceSelect(num - 1);
          return;
        }
      }
      return;
    }

    // Build filter overrides from filter bar
    const filterOverrides: Record<string, unknown> = {};
    if (selectedModel) filterOverrides.filterDevices = [selectedModel];
    if (selectedDocTypes.length > 0) filterOverrides.filterDocTypes = selectedDocTypes;

    if (Object.keys(filterOverrides).length > 0) {
      await send({ text, overrides: filterOverrides as Parameters<typeof send>[0]["overrides"] });
    } else {
      await send({ text });
    }
  };

  const handleRegenerate = useCallback((payload: RegeneratePayload) => {
    const fallbackQueries = payload.originalQuery ? [payload.originalQuery] : [];
    setPendingRegeneration({
      messageId: payload.messageId,
      originalQuery: payload.originalQuery,
      docs: payload.retrievedDocs ?? [],
      searchQueries: payload.searchQueries && payload.searchQueries.length > 0
        ? payload.searchQueries
        : fallbackQueries,
      selectedDevices: payload.selectedDevices ?? [],
      selectedDocTypes: payload.selectedDocTypes ?? [],
      reason: "manual",
    });
  }, [setPendingRegeneration]);

  const getOriginalQuery = useMemo(() => {
    const queryMap = new Map<string, string>();
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === "user") {
        if (i + 1 < messages.length && messages[i + 1].role === "assistant") {
          queryMap.set(messages[i + 1].id, messages[i].content);
        }
      }
    }
    return (assistantId: string) => queryMap.get(assistantId) || "";
  }, [messages]);

  const pendingRegenerationMessage = useMemo(
    () => messages.find((msg) => msg.id === pendingRegeneration?.messageId && msg.role === "assistant"),
    [messages, pendingRegeneration?.messageId],
  );

  const shouldShowMissingDevicePrompt =
    pendingRegeneration?.reason === "missing_device_parse" &&
    !lastSelectedDevice &&
    !pendingRegenerationMessage?.content?.trim();

  return (
    <div className="chat-layout">
      <main className="chat-main">
        <ChatContainer>
          {isLoadingSession ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--color-text-secondary)" }}>
              <Spin size="large" tip="대화를 불러오는 중..." />
            </div>
          ) : (
            <MessageList autoScrollToBottom>
              {messages.length === 0 ? (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--color-text-secondary)", gap: 16, padding: 48 }}>
                  <div style={{ width: 64, height: 64, borderRadius: "50%", backgroundColor: "var(--color-accent-primary)", display: "flex", alignItems: "center", justifyContent: "center", color: "white", fontSize: 24, fontWeight: 600 }}>
                    PE
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <h2 style={{ margin: "0 0 8px", color: "var(--color-text-primary)" }}>PE Agent에 오신 것을 환영합니다</h2>
                    <p style={{ margin: 0 }}>질문을 입력하면 AI가 답변해 드립니다</p>
                  </div>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <MessageItem
                    key={msg.id}
                    message={msg}
                    isStreaming={isStreaming && idx === messages.length - 1 && msg.role === "assistant"}
                    onFeedback={submitFeedback}
                    onDetailedFeedback={submitDetailedFeedback}
                    onRegenerate={msg.role === "assistant" ? handleRegenerate : undefined}
                    onEdit={msg.role === "user" && !isStreaming ? (editedText) => send({ text: editedText }) : undefined}
                    issueCases={
                      msg.role === "assistant" &&
                      idx === messages.length - 1 &&
                      pendingIssueCaseSelection
                        ? pendingIssueCaseSelection.payload.cases
                        : undefined
                    }
                    onIssueCaseSelect={submitIssueCaseSelection}
                    showIssueSopButtons={
                      msg.role === "assistant" &&
                      idx === messages.length - 1 &&
                      !!pendingIssueSopConfirm
                    }
                    onIssueSopConfirm={(confirm) => submitIssueSopConfirm(confirm, { silent: true })}
                    originalQuery={msg.role === "assistant" ? getOriginalQuery(msg.id) : undefined}
                  />
                ))
              )}
            </MessageList>
          )}

          {shouldShowMissingDevicePrompt && (
            <div style={{ margin: "16px 0 8px", padding: "12px 16px", borderRadius: 8, border: "1px solid var(--color-border)", background: "var(--color-bg-secondary)", maxWidth: 480 }}>
              {!showSuggestedDevicePanel && (
                <>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>장비를 자동으로 파싱하지 못했습니다. 기기를 검색하시겠습니까?</div>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button className="action-button regenerate-button" onClick={() => setShowSuggestedDevicePanel(true)} type="button">1. 예</button>
                    <button className="action-button" onClick={() => setPendingRegeneration(null)} type="button">2. 아니오</button>
                  </div>
                </>
              )}
              {showSuggestedDevicePanel && isLoadingSuggestedDevices && (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Spin size="small" />
                  <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>장비 목록을 불러오는 중입니다...</span>
                </div>
              )}
              {showSuggestedDevicePanel && !isLoadingSuggestedDevices && suggestedDeviceError && (
                <Alert type="warning" showIcon message={suggestedDeviceError} />
              )}
              {showSuggestedDevicePanel && !isLoadingSuggestedDevices && !suggestedDeviceError && suggestedDevices.length > 0 && (
                <>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>아래에서 장비 번호를 입력해 주세요.</div>
                  <div style={{ maxHeight: 300, overflowY: "auto", display: "flex", flexDirection: "column", gap: 4 }}>
                    {suggestedDevices.map((device, idx) => (
                      <div
                        key={device.name}
                        style={{ fontSize: 13, padding: "4px 8px", borderRadius: 4, cursor: "pointer", transition: "background-color 0.15s ease" }}
                        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--color-action-bg)")}
                        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                        onClick={() => handleDeviceSelect(idx)}
                      >
                        <span style={{ fontWeight: 600, color: "var(--color-accent-primary)" }}>{idx + 1}.</span>{" "}
                        {device.name}
                        <span style={{ color: "var(--color-text-secondary)", marginLeft: 8, fontSize: 12 }}>({device.doc_count.toLocaleString()} 문서)</span>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: 8, fontSize: 12, color: "var(--color-text-secondary)" }}>번호를 입력하거나 클릭하여 선택하세요. (0: 취소)</div>
                </>
              )}
            </div>
          )}

          {pendingGuidedSelection && (
            <GuidedSelectionPanel
              question={pendingGuidedSelection.question}
              instruction={pendingGuidedSelection.instruction}
              payload={pendingGuidedSelection.payload}
              onComplete={submitGuidedSelectionFinal}
            />
          )}

          {pendingAbbreviationResolve && (
            <AbbreviationResolvePanel
              question={pendingAbbreviationResolve.question}
              instruction={pendingAbbreviationResolve.instruction}
              abbreviations={pendingAbbreviationResolve.payload.abbreviations}
              onSubmit={submitAbbreviationResolve}
            />
          )}

          {pendingDeviceSelection && (
            (pendingDeviceSelection.devices && pendingDeviceSelection.devices.length > 0) ||
            (pendingDeviceSelection.docTypes && pendingDeviceSelection.docTypes.length > 0)
          ) && (
            <DeviceSelectionPanel
              question={pendingDeviceSelection.question}
              devices={pendingDeviceSelection.devices ?? []}
              docTypes={pendingDeviceSelection.docTypes ?? []}
              instruction={pendingDeviceSelection.instruction}
              onSelect={submitDeviceSelection}
            />
          )}

          {error && (
            <Alert type="error" message={error} showIcon closable style={{ margin: "0 0 16px" }} />
          )}

          <InputArea>
            <ChatInput
              onSend={handleSend}
              onStop={stop}
              isStreaming={isStreaming}
              placeholder={inputPlaceholder}
              disabled={isLoadingSession || !!pendingGuidedSelection || !!pendingAbbreviationResolve}
              docTypeOptions={docTypeOptions}
              selectedDocTypes={selectedDocTypes}
              onDocTypesChange={handleDocTypesChange}
              modelOptions={modelOptions}
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
              equipOptions={[]}
              selectedEquip={selectedEquip}
              onEquipChange={setSelectedEquip}
            />
          </InputArea>
        </ChatContainer>
      </main>
    </div>
  );
}
