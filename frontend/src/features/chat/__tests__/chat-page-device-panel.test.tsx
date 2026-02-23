import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  renderChatPage,
  resetAllMocks,
  mockSend,
  mockSetPendingRegeneration,
  mockFetchDeviceCatalog,
} from "./helpers/render-chat-page";
import { makePendingRegeneration, makeDeviceCatalogResponse } from "./helpers/mock-data";

beforeEach(() => {
  resetAllMocks();
});

// ── Req 1: 확인 다이얼로그 표시 ──

describe("Req 1 — confirmation dialog", () => {
  it('shows "1. 예" and "2. 아니오" buttons when pendingRegeneration is set', async () => {
    await renderChatPage({
      pendingRegeneration: makePendingRegeneration(),
    });

    expect(screen.getByText("1. 예")).toBeInTheDocument();
    expect(screen.getByText("2. 아니오")).toBeInTheDocument();
  });
});

// ── Req 2: "1" 입력 → 기기 목록 로드 ──

describe("Req 2 — type '1' to load device list", () => {
  it("calls fetchDeviceCatalog after typing '1' + Enter", async () => {
    const catalogResponse = makeDeviceCatalogResponse();
    mockFetchDeviceCatalog.mockResolvedValue(catalogResponse);

    await renderChatPage({
      pendingRegeneration: makePendingRegeneration(),
    });

    const user = userEvent.setup();
    const textarea = screen.getByRole("textbox");
    await user.type(textarea, "1");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(mockFetchDeviceCatalog).toHaveBeenCalled();
    });
  });
});

// ── Req 3: "2" 입력 → 패널 닫힘 ──

describe("Req 3 — type '2' to dismiss panel", () => {
  it("calls setPendingRegeneration(null) after typing '2' + Enter", async () => {
    await renderChatPage({
      pendingRegeneration: makePendingRegeneration(),
    });

    const user = userEvent.setup();
    const textarea = screen.getByRole("textbox");
    await user.type(textarea, "2");
    await user.keyboard("{Enter}");

    expect(mockSetPendingRegeneration).toHaveBeenCalledWith(null);
  });
});

// ── Req 4: 번호 매겨진 기기 목록 + maxWidth 480 ──

describe("Req 4 — numbered device list with maxWidth 480", () => {
  it("renders device names and container has maxWidth 480", async () => {
    const catalog = makeDeviceCatalogResponse();
    mockFetchDeviceCatalog.mockResolvedValue(catalog);

    await renderChatPage({
      pendingRegeneration: makePendingRegeneration(),
    });

    // Click "1. 예" to advance to device list stage
    const user = userEvent.setup();
    const yesButton = screen.getByText("1. 예");
    await user.click(yesButton);

    // Wait for devices to render
    await waitFor(() => {
      expect(screen.getByText("Pump-A100")).toBeInTheDocument();
    });

    expect(screen.getByText("Compressor-B200")).toBeInTheDocument();
    expect(screen.getByText("Turbine-C300")).toBeInTheDocument();

    // Check numbered labels
    expect(screen.getByText("1.")).toBeInTheDocument();
    expect(screen.getByText("2.")).toBeInTheDocument();
    expect(screen.getByText("3.")).toBeInTheDocument();

    // Check container maxWidth — find the div with maxWidth: 480px
    const panelDiv = Array.from(document.querySelectorAll("div")).find(
      (div) => div.style.maxWidth === "480px",
    );
    expect(panelDiv).toBeTruthy();
  });
});

// ── Req 5: 기기 번호 입력 → send() 호출 ──

describe("Req 5 — type device number to send", () => {
  it('sends message with "[Pump-A100] ..." and overrides after typing device number', async () => {
    const catalog = makeDeviceCatalogResponse();
    mockFetchDeviceCatalog.mockResolvedValue(catalog);
    const pending = makePendingRegeneration();

    await renderChatPage({
      pendingRegeneration: pending,
    });

    // Step 1: click "1. 예" to show device list
    const user = userEvent.setup();
    await user.click(screen.getByText("1. 예"));

    await waitFor(() => {
      expect(screen.getByText("Pump-A100")).toBeInTheDocument();
    });

    // Step 2: type device number "1" + Enter
    const textarea = screen.getByRole("textbox");
    await user.type(textarea, "1");
    await user.keyboard("{Enter}");

    expect(mockSetPendingRegeneration).toHaveBeenCalledWith(null);
    expect(mockSend).toHaveBeenCalledWith({
      text: `[Pump-A100] ${pending.originalQuery}`,
      overrides: {
        filterDevices: ["Pump-A100"],
        searchQueries: pending.searchQueries,
        autoParse: false,
      },
    });
  });
});

// ── Req 6: 기기 클릭 → 동일 동작 ──

describe("Req 6 — click device item triggers same send()", () => {
  it("clicking a device item calls send() with the same arguments as typing the number", async () => {
    const catalog = makeDeviceCatalogResponse();
    mockFetchDeviceCatalog.mockResolvedValue(catalog);
    const pending = makePendingRegeneration();

    await renderChatPage({
      pendingRegeneration: pending,
    });

    const user = userEvent.setup();
    await user.click(screen.getByText("1. 예"));

    await waitFor(() => {
      expect(screen.getByText("Pump-A100")).toBeInTheDocument();
    });

    // Click on the first device item
    await user.click(screen.getByText("Pump-A100"));

    expect(mockSetPendingRegeneration).toHaveBeenCalledWith(null);
    expect(mockSend).toHaveBeenCalledWith({
      text: `[Pump-A100] ${pending.originalQuery}`,
      overrides: {
        filterDevices: ["Pump-A100"],
        searchQueries: pending.searchQueries,
        autoParse: false,
      },
    });
  });
});

// ── Req 7: "0" 입력 → 취소 ──

describe("Req 7 — type '0' to cancel", () => {
  it("calls setPendingRegeneration(null) and does NOT call send()", async () => {
    const catalog = makeDeviceCatalogResponse();
    mockFetchDeviceCatalog.mockResolvedValue(catalog);

    await renderChatPage({
      pendingRegeneration: makePendingRegeneration(),
    });

    const user = userEvent.setup();
    await user.click(screen.getByText("1. 예"));

    await waitFor(() => {
      expect(screen.getByText("Pump-A100")).toBeInTheDocument();
    });

    const textarea = screen.getByRole("textbox");
    await user.type(textarea, "0");
    await user.keyboard("{Enter}");

    expect(mockSetPendingRegeneration).toHaveBeenCalledWith(null);
    expect(mockSend).not.toHaveBeenCalled();
  });
});

// ── Req 8: 유효하지 않은 입력 → 무시 ──

describe("Req 8 — invalid input is ignored", () => {
  it("does not call send or setPendingRegeneration on invalid input", async () => {
    const catalog = makeDeviceCatalogResponse();
    mockFetchDeviceCatalog.mockResolvedValue(catalog);

    await renderChatPage({
      pendingRegeneration: makePendingRegeneration(),
    });

    const user = userEvent.setup();
    await user.click(screen.getByText("1. 예"));

    await waitFor(() => {
      expect(screen.getByText("Pump-A100")).toBeInTheDocument();
    });

    // Reset mocks after the initial setup interactions
    mockSetPendingRegeneration.mockClear();
    mockSend.mockClear();

    const textarea = screen.getByRole("textbox");
    await user.type(textarea, "hello");
    await user.keyboard("{Enter}");

    expect(mockSend).not.toHaveBeenCalled();
    expect(mockSetPendingRegeneration).not.toHaveBeenCalled();
  });
});

// ── Req 9: 한번 선택 후 재발생 시 자동 해제 ──

describe("Req 9 — auto-dismiss on subsequent missing_device_parse", () => {
  it("auto-calls setPendingRegeneration(null) when a device was previously selected", async () => {
    const catalog = makeDeviceCatalogResponse();
    mockFetchDeviceCatalog.mockResolvedValue(catalog);
    const pending = makePendingRegeneration();

    const { rerender } = await renderChatPage({
      pendingRegeneration: pending,
    });

    const user = userEvent.setup();

    // Select device (click "1. 예" then click a device)
    await user.click(screen.getByText("1. 예"));
    await waitFor(() => {
      expect(screen.getByText("Pump-A100")).toBeInTheDocument();
    });
    await user.click(screen.getByText("Pump-A100"));

    // After selecting, lastSelectedDevice is set internally.
    // Now simulate re-render with a NEW pendingRegeneration (different messageId)
    mockSetPendingRegeneration.mockClear();

    // We need to re-import and re-render with new pendingRegeneration having a different messageId
    // Since ChatPage uses internal state (lastSelectedDevice), we need to re-render the same component instance
    // The useEffect watching pendingRegeneration?.messageId should auto-dismiss

    // Unfortunately, with full module mocking, we can't easily change the useChatReview return
    // mid-render. Let's verify the logic differently:
    // After selecting a device, the component sets lastSelectedDevice.
    // When a new pendingRegeneration with a different messageId arrives,
    // the useEffect auto-calls setPendingRegeneration(null).

    // We need to trigger a re-render with a new pendingRegeneration.
    // The way to do this is to update the mock and re-render.
    const { useChatReview } = await import("../context/chat-review-context");
    const newPending = makePendingRegeneration({ messageId: "msg-2" });

    vi.mocked(useChatReview).mockReturnValue({
      pendingReview: null,
      pendingRegeneration: newPending,
      completedRetrievedDocs: null,
      selectedRanks: [],
      editableQueries: [],
      isEditingQueries: false,
      isStreaming: false,
      setPendingReview: vi.fn(),
      setPendingRegeneration: mockSetPendingRegeneration,
      setCompletedRetrievedDocs: vi.fn(),
      setSelectedRanks: vi.fn(),
      setEditableQueries: vi.fn(),
      setIsEditingQueries: vi.fn(),
      setIsStreaming: vi.fn(),
      submitReview: vi.fn(),
      submitSearchQueries: vi.fn(),
      submitRegeneration: vi.fn(),
      registerSubmitHandlers: vi.fn(),
      registerRegenerationHandlers: vi.fn(),
    });

    // Dynamic import ChatPage again for rerender
    const { default: ChatPage } = await import("../pages/chat-page");
    rerender(<ChatPage />);

    // The effect should auto-dismiss because lastSelectedDevice is set
    await waitFor(() => {
      expect(mockSetPendingRegeneration).toHaveBeenCalledWith(null);
    });
  });
});

// ── Req 10: 자동 해제가 send()를 호출하지 않음 ──

describe("Req 10 — auto-dismiss does NOT call send()", () => {
  it("auto-dismiss only calls setPendingRegeneration(null), not send()", async () => {
    const catalog = makeDeviceCatalogResponse();
    mockFetchDeviceCatalog.mockResolvedValue(catalog);
    const pending = makePendingRegeneration();

    const { rerender } = await renderChatPage({
      pendingRegeneration: pending,
    });

    const user = userEvent.setup();

    // Select a device first
    await user.click(screen.getByText("1. 예"));
    await waitFor(() => {
      expect(screen.getByText("Pump-A100")).toBeInTheDocument();
    });
    await user.click(screen.getByText("Pump-A100"));

    // Clear mocks after initial selection
    mockSend.mockClear();
    mockSetPendingRegeneration.mockClear();

    // Re-render with new pendingRegeneration
    const { useChatReview } = await import("../context/chat-review-context");
    const newPending = makePendingRegeneration({ messageId: "msg-3" });

    vi.mocked(useChatReview).mockReturnValue({
      pendingReview: null,
      pendingRegeneration: newPending,
      completedRetrievedDocs: null,
      selectedRanks: [],
      editableQueries: [],
      isEditingQueries: false,
      isStreaming: false,
      setPendingReview: vi.fn(),
      setPendingRegeneration: mockSetPendingRegeneration,
      setCompletedRetrievedDocs: vi.fn(),
      setSelectedRanks: vi.fn(),
      setEditableQueries: vi.fn(),
      setIsEditingQueries: vi.fn(),
      setIsStreaming: vi.fn(),
      submitReview: vi.fn(),
      submitSearchQueries: vi.fn(),
      submitRegeneration: vi.fn(),
      registerSubmitHandlers: vi.fn(),
      registerRegenerationHandlers: vi.fn(),
    });

    const { default: ChatPage } = await import("../pages/chat-page");
    rerender(<ChatPage />);

    await waitFor(() => {
      expect(mockSetPendingRegeneration).toHaveBeenCalledWith(null);
    });

    // send() should NOT have been called during auto-dismiss
    expect(mockSend).not.toHaveBeenCalled();
  });
});

// ── Req 12: HIL DeviceSelectionPanel 정상 렌더 ──

describe("Req 12 — HIL DeviceSelectionPanel renders", () => {
  it("renders DeviceSelectionPanel when pendingDeviceSelection is set", async () => {
    await renderChatPage({
      chatSession: {
        pendingDeviceSelection: {
          threadId: "thread-1",
          question: "검색 범위 선택",
          instruction: "기기를 선택하세요.",
          docs: [],
          devices: [
            { name: "Pump-X", doc_count: 50 },
            { name: "Motor-Y", doc_count: 30 },
          ],
          docTypes: [
            { name: "Manual", doc_count: 100 },
          ],
          kind: "device_selection" as const,
        },
      },
    });

    // DeviceSelectionPanel renders the question and device names
    expect(screen.getByText("검색 범위 선택")).toBeInTheDocument();
    expect(screen.getByText("Pump-X")).toBeInTheDocument();
    expect(screen.getByText("Motor-Y")).toBeInTheDocument();
  });
});
