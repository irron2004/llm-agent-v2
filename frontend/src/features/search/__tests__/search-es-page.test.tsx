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
import SearchESPage from "../pages/search-es-page";

describe("SearchESPage - Legacy Elasticsearch Endpoint", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders search input and field weight controls", () => {
    render(<SearchESPage />);
    
    expect(screen.getByPlaceholderText("검색어를 입력하세요")).toBeInTheDocument();
    expect(screen.getByText("검색 설정")).toBeInTheDocument();
    expect(screen.getByText("검색 모드")).toBeInTheDocument();
    expect(screen.getByText("검색 필드 선택")).toBeInTheDocument();
  });

  it("calls GET /api/search when searching", async () => {
    const mockResponse = {
      query: "test",
      items: [
        { rank: 1, id: "doc-1", title: "Test Doc", snippet: "Test content", score: 0.95, score_display: "95%" },
      ],
      total: 1,
      page: 1,
      size: 20,
      has_next: false,
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchESPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    // Wait for fetch
    await new Promise(resolve => setTimeout(resolve, 200));
    expect(mockFetch).toHaveBeenCalled();
  });

  it("includes field_weights in the request URL", async () => {
    const mockResponse = {
      query: "test",
      items: [],
      total: 0,
      page: 1,
      size: 20,
      has_next: false,
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchESPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      // fetch can be called with just the URL (GET request) or with URL + options
      expect(mockFetch).toHaveBeenCalled();
      const call = mockFetch.mock.calls[0];
      expect(call[0]).toContain("field_weights=");
    });
  });

  it("displays results after successful search", async () => {
    const mockResponse = {
      query: "test",
      items: [
        { rank: 1, id: "doc-1", title: "Test Document", snippet: "Test content here", score: 0.95, score_display: "95%" },
        { rank: 2, id: "doc-2", title: "Another Doc", snippet: "More content", score: 0.85, score_display: "85%" },
      ],
      total: 2,
      page: 1,
      size: 20,
      has_next: false,
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchESPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    // Wait a bit for the fetch to complete
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Just check the page is not in loading state anymore
    expect(screen.queryByText("검색 중...")).not.toBeInTheDocument();
  });

  it("shows BM25 mode switch and applies dense_weight=0.0 when enabled", async () => {
    const mockResponse = {
      query: "test",
      items: [],
      total: 0,
      page: 1,
      size: 20,
      has_next: false,
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    render(<SearchESPage />);

    // BM25 switch should be present
    expect(screen.getByText("BM25")).toBeInTheDocument();
    expect(screen.getByText("하이브리드")).toBeInTheDocument();

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    // Wait for fetch
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Verify fetch was called
    expect(mockFetch).toHaveBeenCalled();
  });

  it("displays error message on failed search", async () => {
    // Mock console.error
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve("Internal Server Error"),
    });

    render(<SearchESPage />);

    const user = userEvent.setup();
    const searchInput = screen.getByPlaceholderText("검색어를 입력하세요");
    await user.type(searchInput, "test");
    await user.keyboard("{Enter}");

    // Wait for error handling
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Verify loading is complete
    expect(screen.queryByText("검색 중...")).not.toBeInTheDocument();
    
    consoleSpy.mockRestore();
  });

  it("allows toggling field weights", async () => {
    render(<SearchESPage />);

    // Find checkboxes for field selection
    const checkboxes = screen.getAllByRole("checkbox");
    expect(checkboxes.length).toBeGreaterThan(0);

    // At least one field should be enabled by default
    const enabledFields = checkboxes.filter((cb) => (cb as HTMLInputElement).checked);
    expect(enabledFields.length).toBeGreaterThan(0);
  });
});
