import { renderHook, act, waitFor } from "@testing-library/react";
import { ReactNode } from "react";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { useChatSession } from "../hooks/use-chat-session";
import { ChatLogsProvider } from "../context/chat-logs-context";
import { ChatReviewProvider } from "../context/chat-review-context";
import { env } from "../../../config/env";
import * as chatApi from "../api";
import { connectSse } from "../../../lib/sse";
import { apiClient } from "../../../lib/api-client";

vi.mock("../../../lib/sse", () => ({
  connectSse: vi.fn(),
}));

const wrapper = ({ children }: { children: ReactNode }) => (
  <ChatLogsProvider>
    <ChatReviewProvider>{children}</ChatReviewProvider>
  </ChatLogsProvider>
);

const mockAgentResponse = {
  query: "question",
  answer: "answer",
  retrieved_docs: [],
  all_retrieved_docs: [],
};

const mockTurnResponse = {
  session_id: "session-1",
  turn_id: 1,
  user_text: "question",
  assistant_text: "answer",
  doc_refs: [],
  ts: new Date().toISOString(),
};

describe("chat request payload", () => {
  const originalChatPath = env.chatPath;
  let postSpy: any;

  beforeEach(() => {
    vi.clearAllMocks();
    postSpy = vi.spyOn(apiClient, "post").mockResolvedValue(mockAgentResponse as any);
    vi.spyOn(chatApi, "saveTurn").mockResolvedValue(mockTurnResponse as any);
  });

  afterEach(() => {
    env.chatPath = originalChatPath;
  });

  it("sends normal request payload correctly", async () => {
    env.chatPath = "/api/agent";
    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "normal request" });
    });

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalledWith(
        "/api/agent",
        expect.objectContaining({
          message: "normal request",
        })
      );
    });
  });

  it("includes guided_confirm=true on normal send", async () => {
    env.chatPath = "/api/agent";
    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "guided confirm request" });
    });

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalledWith(
        "/api/agent",
        expect.objectContaining({
          message: "guided confirm request",
          guided_confirm: true,
        })
      );
    });
  });

  it("sends guided resume payload with thread_id and auto_parse_confirm decision", async () => {
    env.chatPath = "/api/agent";
    postSpy.mockResolvedValueOnce({
      ...mockAgentResponse,
      interrupted: true,
      thread_id: "t-1",
      interrupt_payload: {
        type: "auto_parse_confirm",
        question: "질문",
        instruction: "안내",
        options: {
          language: [{ value: "en", label: "English" }],
          device: [{ value: "__skip__", label: "건너뛰기" }],
          equip_id: [{ value: "__skip__", label: "건너뛰기" }],
          task: [{ value: "issue", label: "Issue" }],
        },
        defaults: {
          target_language: "en",
          device: null,
          equip_id: null,
          task_mode: "issue",
        },
      },
    } as any);

    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "first guided request" });
    });

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalledTimes(1);
      expect(result.current.pendingGuidedSelection).not.toBeNull();
    });

    await act(async () => {
      result.current.submitGuidedSelectionFinal({
        type: "auto_parse_confirm",
        target_language: "en",
        selected_device: null,
        selected_equip_id: null,
        task_mode: "issue",
      });
    });

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalledTimes(2);
      expect(postSpy).toHaveBeenNthCalledWith(
        2,
        "/api/agent",
        expect.objectContaining({
          thread_id: "t-1",
          resume_decision: expect.objectContaining({
            type: "auto_parse_confirm",
          }),
        })
      );
    });
  });

  it("progresses guided steps via numeric input and sends correct resume payload", async () => {
    env.chatPath = "/api/agent";
    postSpy.mockResolvedValueOnce({
      ...mockAgentResponse,
      interrupted: true,
      thread_id: "t-num",
      interrupt_payload: {
        type: "auto_parse_confirm",
        question: "질문",
        instruction: "안내",
        steps: ["language", "device", "equip_id", "task"],
        options: {
          language: [
            { value: "ko", label: "Korean", recommended: true },
            { value: "en", label: "English" },
          ],
          device: [
            { value: "ETCH-01", label: "ETCH-01" },
            { value: "__skip__", label: "건너뛰기" },
          ],
          equip_id: [
            { value: "EQ-100", label: "EQ-100" },
            { value: "__skip__", label: "건너뛰기" },
          ],
          task: [
            { value: "sop", label: "SOP" },
            { value: "issue", label: "Issue" },
            { value: "all", label: "All" },
          ],
        },
        defaults: {
          target_language: "ko",
          device: null,
          equip_id: null,
          task_mode: "all",
        },
      },
    } as any);

    const { result } = renderHook(() => useChatSession(), { wrapper });

    // Trigger guided interrupt
    await act(async () => {
      await result.current.send({ text: "numeric guided request" });
    });
    await waitFor(() => {
      expect(result.current.pendingGuidedSelection).not.toBeNull();
    });

    // Step 1 (language): "2" = English
    await act(async () => {
      result.current.submitGuidedSelectionNumber("2");
    });
    // Step 2 (device): "1" = ETCH-01
    await act(async () => {
      result.current.submitGuidedSelectionNumber("1");
    });
    // Step 3 (equip_id): "0" = recommended/skip
    await act(async () => {
      result.current.submitGuidedSelectionNumber("0");
    });
    // Step 4 (task): "1" = SOP — this completes the flow
    await act(async () => {
      result.current.submitGuidedSelectionNumber("1");
    });

    await waitFor(() => {
      expect(result.current.pendingGuidedSelection).toBeNull();
      expect(postSpy).toHaveBeenCalledTimes(2);
      expect(postSpy).toHaveBeenNthCalledWith(
        2,
        "/api/agent",
        expect.objectContaining({
          thread_id: "t-num",
          resume_decision: {
            type: "auto_parse_confirm",
            target_language: "en",
            selected_device: "ETCH-01",
            selected_equip_id: null,
            task_mode: "sop",
          },
        })
      );
    });
  });

  it("ignores out-of-range numeric input during guided flow", async () => {
    env.chatPath = "/api/agent";
    postSpy.mockResolvedValueOnce({
      ...mockAgentResponse,
      interrupted: true,
      thread_id: "t-oob",
      interrupt_payload: {
        type: "auto_parse_confirm",
        question: "질문",
        instruction: "안내",
        steps: ["language", "device", "equip_id", "task"],
        options: {
          language: [{ value: "ko", label: "Korean" }],
          device: [{ value: "__skip__", label: "건너뛰기" }],
          equip_id: [{ value: "__skip__", label: "건너뛰기" }],
          task: [{ value: "all", label: "All" }],
        },
        defaults: {
          target_language: "ko",
          device: null,
          equip_id: null,
          task_mode: "all",
        },
      },
    } as any);

    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "oob test" });
    });
    await waitFor(() => {
      expect(result.current.pendingGuidedSelection).not.toBeNull();
    });

    // Out-of-range: only 1 option, so "5" should be ignored
    await act(async () => {
      result.current.submitGuidedSelectionNumber("5");
    });

    // Still on step 0 (language)
    expect(result.current.pendingGuidedSelection).not.toBeNull();
    expect(result.current.pendingGuidedSelection?.stepIndex ?? 0).toBe(0);
  });

  it("sends regeneration payload with overrides correctly", async () => {
    env.chatPath = "/api/agent";
    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({
        text: "regen request",
        overrides: {
          filterDevices: ["Pump-A100"],
          searchQueries: ["pump alarm"],
          autoParse: false,
        },
      });
    });

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalledWith(
        "/api/agent",
        expect.objectContaining({
          message: "regen request",
          auto_parse: false,
          filter_devices: ["Pump-A100"],
          search_queries: ["pump alarm"],
        })
      );
    });
  });

  it("sends SSE request payload correctly", async () => {
    env.chatPath = "/api/agent/run";
    vi.mocked(connectSse).mockResolvedValue({ close: () => {} });
    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "stream request" });
    });

    expect(connectSse).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/agent/run/stream",
        body: expect.objectContaining({
          message: "stream request",
        }),
      }),
      expect.any(Object)
    );
  });

  it("uses explicit stream endpoint as-is when env.chatPath already ends with /stream", async () => {
    env.chatPath = "/api/agent/run/stream";
    vi.mocked(connectSse).mockResolvedValue({ close: () => {} });
    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "stream request explicit" });
    });

    expect(connectSse).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/agent/run/stream",
        body: expect.objectContaining({
          message: "stream request explicit",
        }),
      }),
      expect.any(Object)
    );
  });

  it("sendChatMessage posts JSON to /run when env.chatPath is /run/stream", async () => {
    env.chatPath = "/api/agent/run/stream";

    await chatApi.sendChatMessage({ message: "json request" } as any);

    expect(postSpy).toHaveBeenCalledWith(
      "/api/agent/run",
      expect.objectContaining({
        message: "json request",
      })
    );
  });

  it("persists thread_id across turns", async () => {
    env.chatPath = "/api/agent";

    // First request: response includes thread_id
    const responseWithThreadId = {
      ...mockAgentResponse,
      thread_id: "thread-123",
    };
    postSpy.mockResolvedValueOnce(responseWithThreadId as any);

    const { result } = renderHook(() => useChatSession(), { wrapper });

    // First send
    await act(async () => {
      await result.current.send({ text: "first request" });
    });

    await waitFor(() => {
      expect(postSpy).toHaveBeenCalledWith(
        "/api/agent",
        expect.objectContaining({
          message: "first request",
        })
      );
    });

    // Second request: should include thread_id from previous response
    await act(async () => {
      await result.current.send({ text: "second request" });
    });

    await waitFor(() => {
      expect(postSpy).toHaveBeenLastCalledWith(
        "/api/agent",
        expect.objectContaining({
          message: "second request",
          thread_id: "thread-123",
        })
      );
    });
  });

  it("saves turn with retrieval_meta when agent response has metadata", async () => {
    env.chatPath = "/api/agent";

    // Mock response that includes retrieval metadata
    const responseWithMetadata = {
      ...mockAgentResponse,
      metadata: {
        mq_mode: "fallback",
        mq_used: true,
        mq_reason: "initial_query",
        route: "search->rerank->generate",
        st_gate: "passed",
        attempts: 1,
        retry_strategy: "none",
      },
      search_queries: ["pump alarm", "chamber pressure"],
    };
    postSpy.mockResolvedValueOnce(responseWithMetadata as any);

    const saveTurnSpy = vi.spyOn(chatApi, "saveTurn").mockResolvedValue(mockTurnResponse as any);

    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "test request with metadata" });
    });

    await waitFor(() => {
      expect(saveTurnSpy).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          user_text: "test request with metadata",
          assistant_text: "answer",
          retrieval_meta: {
            mq_mode: "fallback",
            mq_used: true,
            mq_reason: "initial_query",
            route: "search->rerank->generate",
            st_gate: "passed",
            attempts: 1,
            retry_strategy: "none",
            search_queries: ["pump alarm", "chamber pressure"],
          },
        })
      );
    });
  });

  it("omits retrieval_meta when agent response has no metadata", async () => {
    env.chatPath = "/api/agent";

    // Mock response without metadata
    postSpy.mockResolvedValueOnce(mockAgentResponse as any);

    const saveTurnSpy = vi.spyOn(chatApi, "saveTurn").mockResolvedValue(mockTurnResponse as any);

    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "test request without metadata" });
    });

    await waitFor(() => {
      expect(saveTurnSpy).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          user_text: "test request without metadata",
          assistant_text: "answer",
        })
      );
    });

    // Verify retrieval_meta is not present in the call
    const saveTurnCall = saveTurnSpy.mock.calls[0];
    const payload = saveTurnCall[1];
    expect(payload).not.toHaveProperty("retrieval_meta");
  });
});
