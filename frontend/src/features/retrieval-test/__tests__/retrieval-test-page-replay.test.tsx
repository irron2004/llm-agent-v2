import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock fetch globally using vi.stubGlobal for TypeScript safety
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// Mock buildUrl
vi.mock("@/config/env", () => ({
  buildUrl: (path: string) => `http://localhost${path}`,
}));

// Mock the metrics calculation hook
vi.mock("../hooks/use-metrics-calculation", () => ({
  calculateMetrics: vi.fn().mockReturnValue({
    hit_at_1: true,
    hit_at_3: true,
    hit_at_5: true,
    hit_at_10: true,
    reciprocal_rank: 1.0,
    first_relevant_rank: 1,
  }),
  aggregateMetrics: vi.fn().mockReturnValue({
    total_queries: 1,
    hit_at_1: 1,
    hit_at_3: 1,
    hit_at_5: 1,
    hit_at_10: 1,
    mrr: 1,
    avg_first_relevant_rank: 1,
  }),
  useMetricsCalculation: () => ({
    aggregated: {
      hit_at_1: 0,
      hit_at_3: 0,
      hit_at_5: 0,
      mrr: 0,
      ndcg: 0,
    },
  }),
}));

import { render } from "@testing-library/react";
import RetrievalTestPage from "../pages/retrieval-test-page";

describe("RetrievalTestPage - Replay Last Run", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const mockRetrievalResponse = (runId: string) => ({
    run_id: runId,
    effective_config: { method: "hybrid", top_k: 10 },
    effective_config_hash: "abc123",
    steps: {
      route: { name: "route", output: "retrieval" },
    },
    docs: [
      { rank: 1, doc_id: "doc-1", title: "Test Doc 1", snippet: "Content 1", score: 0.95 },
      { rank: 2, doc_id: "doc-2", title: "Test Doc 2", snippet: "Content 2", score: 0.85 },
    ],
  });

  it("sends replay_run_id when clicking Replay last run button", async () => {
    // First call returns a run_id, second call is the replay
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockRetrievalResponse("run-id-12345")),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockRetrievalResponse("run-id-12345-replay")),
      });

    render(<RetrievalTestPage />);

    const user = userEvent.setup();

    // Find the select dropdown and select a question
    const select = screen.getByRole("combobox");
    await user.click(select);

    // Select the first question option - use getAllByText and pick first
    const options = await screen.findAllByText(/\[/);
    await user.click(options[0]);

    // Click "단일 실행" (single run) button
    const runButton = screen.getByText("단일 실행");
    await user.click(runButton);

    // Wait for the first run to complete
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    // Now click the Replay button - get by text content of button
    const replayButton = screen.getByRole("button", { name: /Replay last run/i });
    await user.click(replayButton);

    // Verify replay call includes replay_run_id
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    const secondCallBody = JSON.parse(mockFetch.mock.calls[1][1].body);
    expect(secondCallBody.replay_run_id).toBe("run-id-12345");
    // Query should be preserved from the selected question
    expect(secondCallBody.query).toBeDefined();
  });

  it("enables Replay button after a successful run", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse("test-run-id-abc")),
    });

    render(<RetrievalTestPage />);

    const user = userEvent.setup();

    // Initially replay button should be disabled (no lastRunId)
    const replayButton = screen.getByRole("button", { name: /Replay last run/i });
    expect(replayButton).toBeDisabled();

    // Select a question and run
    const select = screen.getByRole("combobox");
    await user.click(select);
    const options = await screen.findAllByText(/\[/);
    await user.click(options[0]);

    const runButton = screen.getByText("단일 실행");
    await user.click(runButton);

    // Wait for run to complete
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    // Now replay button should be enabled
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Replay last run/i })).toBeEnabled();
    });
  });

  it("displays last run_id when available", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockRetrievalResponse("unique-run-id-xyz")),
    });

    render(<RetrievalTestPage />);

    const user = userEvent.setup();

    // Select and run
    const select = screen.getByRole("combobox");
    await user.click(select);
    const options = await screen.findAllByText(/\[/);
    await user.click(options[0]);

    const runButton = screen.getByText("단일 실행");
    await user.click(runButton);

    // Wait for the run and verify lastRunId is displayed
    await waitFor(() => {
      expect(screen.getByText(/Last run_id:/)).toBeInTheDocument();
    });

    // Should show truncated run_id (first 12 chars)
    expect(screen.getByText(/unique-run/)).toBeInTheDocument();
  });
});
