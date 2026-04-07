/**
 * ChatPage rendering helper with module-level mocks.
 *
 * All external dependencies of ChatPage are mocked at the module level.
 * Use `renderChatPage(overrides)` to customise per-test mock values.
 */
import { render } from "@testing-library/react";
import type { PendingRegeneration } from "../../context/chat-review-context";
import type { DeviceCatalogResponse } from "../../types";

// ── Mock handles (exported so tests can assert on them) ──

export const mockSend = vi.fn();
export const mockStop = vi.fn();
export const mockReset = vi.fn();
export const mockLoadSession = vi.fn();
export const mockSubmitReview = vi.fn();
export const mockSubmitSearchQueries = vi.fn();
export const mockSubmitDeviceSelection = vi.fn();
export const mockSubmitFeedback = vi.fn();
export const mockSubmitDetailedFeedback = vi.fn();
export const mockSubmitAbbreviationResolve = vi.fn();

export const mockSetPendingReview = vi.fn();
export const mockSetPendingRegeneration = vi.fn();
export const mockSetIsStreaming = vi.fn();
export const mockRegisterSubmitHandlers = vi.fn();
export const mockRegisterRegenerationHandlers = vi.fn();

export const mockRefreshHistory = vi.fn();
export const mockSetSearchParams = vi.fn();

export const mockFetchDeviceCatalog = vi.fn<[], Promise<DeviceCatalogResponse>>();

let currentSearchParams = new URLSearchParams();

export function setMockSearchParams(value: string | URLSearchParams) {
  currentSearchParams =
    typeof value === "string" ? new URLSearchParams(value) : new URLSearchParams(value);
}

// ── Module-level mocks ──

vi.mock("../../hooks/use-chat-session", () => ({
  useChatSession: vi.fn(),
}));

vi.mock("../../context/chat-review-context", () => ({
  useChatReview: vi.fn(),
}));

vi.mock("../../context/chat-history-context", () => ({
  useChatHistoryContext: vi.fn(),
}));

vi.mock("../../api", () => ({
  fetchDeviceCatalog: (...args: unknown[]) => mockFetchDeviceCatalog(...(args as [])),
}));

vi.mock("react-router-dom", () => ({
  useSearchParams: vi.fn(() => [currentSearchParams, mockSetSearchParams]),
}));

// ── Lazy imports (after mocks are registered) ──

import { useChatSession } from "../../hooks/use-chat-session";
import { useChatReview } from "../../context/chat-review-context";
import { useChatHistoryContext } from "../../context/chat-history-context";

// ── Default return values ──

type ChatSessionReturn = ReturnType<typeof useChatSession>;
type ChatReviewReturn = ReturnType<typeof useChatReview>;

const defaultChatSession: ChatSessionReturn = {
  sessionId: "test-session",
  messages: [],
  send: mockSend,
  stop: mockStop,
  isStreaming: false,
  isLoadingSession: false,
  error: null,
  reset: mockReset,
  loadSession: mockLoadSession,
  inputPlaceholder: "메시지를 입력하세요...",
  pendingReview: null,
  pendingDeviceSelection: null,
  pendingAbbreviationResolve: null,
  submitAbbreviationResolve: mockSubmitAbbreviationResolve,
  submitReview: mockSubmitReview,
  submitSearchQueries: mockSubmitSearchQueries,
  submitDeviceSelection: mockSubmitDeviceSelection,
  submitFeedback: mockSubmitFeedback,
  submitDetailedFeedback: mockSubmitDetailedFeedback,
};

const defaultChatReview: ChatReviewReturn = {
  pendingReview: null,
  pendingRegeneration: null,
  completedRetrievedDocs: null,
  selectedRanks: [],
  editableQueries: [],
  isEditingQueries: false,
  isStreaming: false,
  setPendingReview: mockSetPendingReview,
  setPendingRegeneration: mockSetPendingRegeneration,
  setCompletedRetrievedDocs: vi.fn(),
  setSelectedRanks: vi.fn(),
  setEditableQueries: vi.fn(),
  setIsEditingQueries: vi.fn(),
  setIsStreaming: mockSetIsStreaming,
  submitReview: vi.fn(),
  submitSearchQueries: vi.fn(),
  submitRegeneration: vi.fn(),
  registerSubmitHandlers: mockRegisterSubmitHandlers,
  registerRegenerationHandlers: mockRegisterRegenerationHandlers,
};

// ── Override types ──

export interface RenderOverrides {
  chatSession?: Partial<ChatSessionReturn>;
  chatReview?: Partial<ChatReviewReturn>;
  pendingRegeneration?: PendingRegeneration | null;
}

/**
 * Render ChatPage with all dependencies mocked.
 *
 * Call `resetAllMocks()` in `beforeEach` to start fresh.
 */
export async function renderChatPage(overrides: RenderOverrides = {}) {
  // Build merged return values
  const sessionValue: ChatSessionReturn = {
    ...defaultChatSession,
    ...overrides.chatSession,
  };

  const reviewValue: ChatReviewReturn = {
    ...defaultChatReview,
    ...overrides.chatReview,
    ...(overrides.pendingRegeneration !== undefined
      ? { pendingRegeneration: overrides.pendingRegeneration }
      : {}),
  };

  vi.mocked(useChatSession).mockReturnValue(sessionValue);
  vi.mocked(useChatReview).mockReturnValue(reviewValue);
  vi.mocked(useChatHistoryContext).mockReturnValue({
    history: [],
    isLoading: false,
    error: null,
    deleteChat: vi.fn(),
    hideChat: vi.fn(),
    getChat: vi.fn().mockReturnValue(null),
    refresh: mockRefreshHistory,
  });

  // Default: fetchDeviceCatalog never resolves unless test overrides
  if (mockFetchDeviceCatalog.getMockImplementation() === undefined) {
    mockFetchDeviceCatalog.mockResolvedValue({
      devices: [],
      doc_types: [],
    });
  }

  // Dynamic import to pick up module-level mocks
  const { default: ChatPage } = await import("../../pages/chat-page");

  const result = render(<ChatPage />);
  return result;
}

/**
 * Reset every mock handle to their initial state.
 * Call in `beforeEach`.
 */
export function resetAllMocks() {
  mockSend.mockReset();
  mockStop.mockReset();
  mockReset.mockReset();
  mockLoadSession.mockReset();
  mockSubmitReview.mockReset();
  mockSubmitSearchQueries.mockReset();
  mockSubmitDeviceSelection.mockReset();
  mockSubmitFeedback.mockReset();
  mockSubmitDetailedFeedback.mockReset();
  mockSubmitAbbreviationResolve.mockReset();
  mockSetPendingReview.mockReset();
  mockSetPendingRegeneration.mockReset();
  mockSetIsStreaming.mockReset();
  mockRegisterSubmitHandlers.mockReset();
  mockRegisterRegenerationHandlers.mockReset();
  mockRefreshHistory.mockReset();
  mockSetSearchParams.mockReset();
  mockFetchDeviceCatalog.mockReset();
  currentSearchParams = new URLSearchParams();
}
