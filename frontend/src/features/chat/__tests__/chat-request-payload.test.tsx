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
});
