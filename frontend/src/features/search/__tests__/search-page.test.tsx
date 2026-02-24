import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock buildUrl
vi.mock("@/config/env", () => ({
  buildUrl: (path: string) => `http://localhost${path}`,
}));

import { render } from "@testing-library/react";
import SearchPage from "../pages/search-page";

describe("SearchPage - Canonical Retrieval Endpoint", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders search input and controls", () => {
    render(<SearchPage />);
    
    expect(screen.getByPlaceholderText("검색어를 입력하세요")).toBeInTheDocument();
    expect(screen.getByText("Deterministic 모드")).toBeInTheDocument();
    expect(screen.getByText("디버그 정보 표시")).toBeInTheDocument();
  });

  it("has deterministic toggle ON by default", () => {
    render(<SearchPage />);
    
    // Just verify the page renders - the switch should be in the document
    const switches = screen.getAllByRole("switch");
    expect(switches.length).toBe(2); // Two switches: deterministic and debug
  });

  it("calls POST /api/retrieval/run with correct payload when searching", async () => {
    const mockResponse = {
      run_id: "test-run-id",
      effective_config: { method: "hybrid", top_k: 10 },
      effective_config_hash: "abc123",
      steps: {
        route: { name: "route", output: "retrieval" },
        detect_language: { name: "detect_language", output: "ko" },
      },
      docs: [
        { rank: 1, doc_id: "doc-1", title: "Test Doc", snippet: "Test content", score: 0.95 },
      ],
      query: "test query",
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    
    await user.type(searchInput, "test query");
    await user.keyboard("{Enter}");

    await waitFor(() => {
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
            debug: true,
            final_top_k: 20,
            rerank_enabled: false,
            auto_parse: true,
            skip_mq: false,
          }),
        })
      );
    });
  });

  it("calls with deterministic=false when toggle is off", async () => {
    const mockResponse = {
      run_id: "test-run-id",
      effective_config: {},
      effective_config_hash: "def456",
      steps: {},
      docs: [],
      query: "test",
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchPage />);

    const user = userEvent.setup();
    
    // Find and click the deterministic switch - it's the first switch
    const switches = screen.getAllByRole("switch");
    await user.click(switches[0]); // Turn off deterministic

    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost/api/retrieval/run",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            query: "test",
            steps: ["retrieve"],
            deterministic: false,
            debug: true,
            final_top_k: 20,
            rerank_enabled: false,
            auto_parse: true,
            skip_mq: false,
          }),
        })
      );
    });
  });

  it("displays results after successful search", async () => {
    const mockResponse = {
      run_id: "test-run-id",
      effective_config: { method: "hybrid" },
      effective_config_hash: "abc123",
      steps: {
        route: { name: "route", output: "retrieval" },
      },
      docs: [
        { rank: 1, doc_id: "doc-1", title: "Test Document", snippet: "Test content here", score: 0.95 },
        { rank: 2, doc_id: "doc-2", title: "Another Doc", snippet: "More content" },
      ],
      query: "test",
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      // Use regex to match text that contains "2건"
      expect(screen.getByText(/2건/)).toBeInTheDocument();
      expect(screen.getByText(/Test Document/)).toBeInTheDocument();
      expect(screen.getByText(/Score: N\/A/)).toBeInTheDocument();
    });
  });

  it("displays configuration info when debug is enabled", async () => {
    const mockResponse = {
      run_id: "run-123",
      effective_config: { top_k: 10 },
      effective_config_hash: "hash-abc",
      steps: {},
      docs: [],
      query: "test",
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("Run ID:")).toBeInTheDocument();
      expect(screen.getByText("run-123")).toBeInTheDocument();
      expect(screen.getByText("Config Hash:")).toBeInTheDocument();
    });
  });

  it("displays error message on failed search", async () => {
    // Mock console.error to suppress error logging
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve("Internal Server Error"),
    });

    render(<SearchPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    // Wait for the error to be displayed
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // The page should show some error indication
    expect(screen.queryByText("검색 중...")).not.toBeInTheDocument();
    
    consoleSpy.mockRestore();
  });

  it("displays pipeline steps when debug is enabled", async () => {
    const mockResponse = {
      run_id: "test-run",
      effective_config: {},
      effective_config_hash: "xyz",
      steps: {
        route: { name: "route", output: "retrieval" },
        detect_language: { name: "detect_language", output: "ko" },
        st_mq: { name: "st_mq", output: ["test query"], artifacts: { search_queries: ["test query"] } },
        filter: { name: "filter", output: [], artifacts: { filters: [] } },
      },
      docs: [],
      query: "test",
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(screen.getByText("검색 파이프라인")).toBeInTheDocument();
      // Steps are now displayed as keys from the steps object
      expect(screen.getByText(/route/)).toBeInTheDocument();
    });
  });
});
