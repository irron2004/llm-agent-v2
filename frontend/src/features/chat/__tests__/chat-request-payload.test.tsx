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
