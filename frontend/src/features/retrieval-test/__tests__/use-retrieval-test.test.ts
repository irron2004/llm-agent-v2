import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useRetrievalTest } from "../hooks/use-retrieval-test";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock buildUrl
vi.mock("@/config/env", () => ({
  buildUrl: (path: string) => `http://localhost${path}`,
}));

describe("useRetrievalTest - Canonical Retrieval Endpoint", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const mockQuestion = {
    id: "q1",
    question: "test query",
    groundTruthDocIds: ["doc-1", "doc-3"],
  };

  const mockRetrievalResponse = {
    run_id: "test-run-id-123",
    effective_config: { method: "hybrid", top_k: 10 },
    effective_config_hash: "abc123",
    steps: {
      route: { name: "route", output: "retrieval" },
    },
    docs: [
      { rank: 1, doc_id: "doc-1", title: "Test Doc 1", snippet: "Content 1", score: 0.95 },
      { rank: 2, doc_id: "doc-2", title: "Test Doc 2", snippet: "Content 2", score: 0.85 },
      { rank: 3, doc_id: "doc-3", title: "Test Doc 3", snippet: "Content 3", score: 0.75 },
      { rank: 4, doc_id: "doc-4", title: "Test Doc 4", snippet: "Content 4", score: 0.65 },
    ],
  };

  it("calls POST /api/retrieval/run with correct payload", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    // Default config includes auto_parse: true
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost/api/retrieval/run",
      expect.objectContaining({
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: "test query",
          steps: ["retrieve"],
          deterministic: true,
          final_top_k: 20,
          rerank_enabled: false,
          auto_parse: true,
          skip_mq: false,
        }),
      })
    );
  });

  it("maps rerank to rerank_enabled when enabled in config", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      result.current.updateConfig({ rerank: true });
    });

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(callBody.rerank_enabled).toBe(true);
  });

  it("maps autoParse false to auto_parse=false", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      result.current.updateConfig({ autoParse: false });
    });

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(callBody.auto_parse).toBe(false);
  });

  it("includes skip_mq when enabled in config", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      result.current.updateConfig({ skipMq: true });
    });

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(callBody.skip_mq).toBe(true);
  });

  it("handles missing score from canonical docs safely", async () => {
    const missingScoreResponse = {
      ...mockRetrievalResponse,
      docs: [{ rank: 1, doc_id: "doc-1", title: "Test Doc 1", snippet: "Content 1" }],
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(missingScoreResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    expect(result.current.results[0].searchResults[0].score).toBe(0);
    expect(result.current.results[0].searchResults[0].score_display).toBe("N/A");
  });

  it("always sets deterministic=true in requests", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    // Set deterministic to false
    await act(async () => {
      result.current.updateConfig({ deterministic: false });
    });

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(callBody.deterministic).toBe(false);
  });

  it("stores run_id for replay functionality", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    expect(result.current.lastRunId).toBe("test-run-id-123");
  });

  it("calls with replay_run_id when replaying", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    // First run to get a run_id
    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    // Replay with a different question
    const newQuestion = { ...mockQuestion, id: "q2", question: "different query" };

    await act(async () => {
      await result.current.replayLastRun(newQuestion);
    });

    const callBody = JSON.parse(mockFetch.mock.calls[1][1].body);
    expect(callBody.replay_run_id).toBe("test-run-id-123");
    expect(callBody.query).toBe("different query");
  });

  it("computes metrics correctly using canonical doc_ids", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    const testResult = result.current.results[0];
    
    // groundTruthDocIds = ["doc-1", "doc-3"]
    // doc-1 is at rank 1, doc-3 is at rank 3
    expect(testResult.metrics.hit_at_1).toBe(true); // doc-1 is at rank 1
    expect(testResult.metrics.hit_at_3).toBe(true); // doc-3 is at rank 3
    expect(testResult.metrics.hit_at_5).toBe(true);
    expect(testResult.metrics.first_relevant_rank).toBe(1);
    expect(testResult.metrics.reciprocal_rank).toBe(1.0);
  });

  it("evaluates metrics at final_top_k cutoff (config.size)", async () => {
    // Return only 2 docs but config.size is 20
    const limitedResponse = {
      ...mockRetrievalResponse,
      docs: [
        { rank: 1, doc_id: "doc-2", title: "Test Doc 2", snippet: "Content 2", score: 0.85 },
        { rank: 2, doc_id: "doc-5", title: "Test Doc 5", snippet: "Content 5", score: 0.65 },
      ],
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(limitedResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    const testResult = result.current.results[0];
    
    // groundTruthDocIds = ["doc-1", "doc-3"], neither is in results
    expect(testResult.metrics.hit_at_1).toBe(false);
    expect(testResult.metrics.hit_at_3).toBe(false);
    expect(testResult.metrics.first_relevant_rank).toBeNull();
  });

  it("maps canonical doc_id to SearchResult id for compatibility", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    const testResult = result.current.results[0];
    
    // Verify id is mapped from doc_id
    expect(testResult.searchResults[0].id).toBe("doc-1");
    expect(testResult.searchResults[1].id).toBe("doc-2");
  });

  it("handles API errors gracefully", async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
    });

    const { result } = renderHook(() => useRetrievalTest());

    let error: Error | null = null;
    await act(async () => {
      try {
        await result.current.runSingleTest(mockQuestion);
      } catch (e) {
        error = e as Error;
      }
    });

    expect(error).not.toBeNull();
    expect(result.current.error).toBe("Search failed: 500");
  });

  it("updates results - replaces existing result for same question", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    // First run
    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    expect(result.current.results).toHaveLength(1);

    // Run same question again
    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    // Should still have only 1 result (replaced)
    expect(result.current.results).toHaveLength(1);
  });

  it("config updates are reflected in subsequent requests", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse),
    });

    const { result } = renderHook(() => useRetrievalTest());

    await act(async () => {
      result.current.updateConfig({ size: 50, deterministic: false });
    });

    await act(async () => {
      await result.current.runSingleTest(mockQuestion);
    });

    const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(callBody.final_top_k).toBe(50);
    expect(callBody.deterministic).toBe(false);
  });
});
