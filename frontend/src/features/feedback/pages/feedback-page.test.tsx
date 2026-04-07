import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import FeedbackPage, {
  buildChatSessionHref,
  toSafeInternalHref,
} from "./feedback-page";

const mockListFeedback = vi.fn();
const mockGetFeedbackStatistics = vi.fn();
const mockExportFeedbackJson = vi.fn();
const mockExportFeedbackCsv = vi.fn();
const mockUpdateFeedbackResolution = vi.fn();

vi.mock("../../chat/api", () => ({
  listFeedback: (...args: unknown[]) => mockListFeedback(...args),
  getFeedbackStatistics: (...args: unknown[]) => mockGetFeedbackStatistics(...args),
  exportFeedbackJson: (...args: unknown[]) => mockExportFeedbackJson(...args),
  exportFeedbackCsv: (...args: unknown[]) => mockExportFeedbackCsv(...args),
  updateFeedbackResolution: (...args: unknown[]) => mockUpdateFeedbackResolution(...args),
}));

describe("feedback session link helpers", () => {
  it("builds an internal chat href from session id", () => {
    expect(buildChatSessionHref("session 123")).toBe("/?session=session%20123");
  });

  it("allows only internal resolved links", () => {
    expect(toSafeInternalHref("/?session=resolved-1")).toBe("/?session=resolved-1");
    expect(toSafeInternalHref("javascript:alert(1)")).toBeNull();
    expect(toSafeInternalHref("https://evil.example/")).toBeNull();
    expect(toSafeInternalHref("//evil.example/")).toBeNull();
  });
});

describe("FeedbackPage", () => {
  beforeEach(() => {
    mockListFeedback.mockReset();
    mockGetFeedbackStatistics.mockReset();
    mockExportFeedbackJson.mockReset();
    mockExportFeedbackCsv.mockReset();
    mockUpdateFeedbackResolution.mockReset();

    mockListFeedback.mockResolvedValue({
      items: [
        {
          session_id: "original-session",
          turn_id: 7,
          user_text: "question about previous answer",
          assistant_text: "assistant answer",
          accuracy: 1,
          completeness: 2,
          relevance: 1,
          avg_score: 1.3,
          rating: "down",
          comment: "not solved",
          reviewer_name: "qa",
          logs: [],
          resolved: true,
          resolved_link: "javascript:alert(1)",
          resolved_at: "2026-04-07T00:00:00",
          ts: "2026-04-07T00:00:00",
        },
      ],
      total: 1,
    });
    mockGetFeedbackStatistics.mockResolvedValue({
      total_count: 1,
      avg_accuracy: 1,
      avg_completeness: 2,
      avg_relevance: 1,
      avg_score: 1.3,
      rating_distribution: { down: 1 },
    });
  });

  it("opens the original chat from session_id and hides unsafe resolved links", async () => {
    render(<FeedbackPage />);

    await waitFor(() => {
      expect(mockListFeedback).toHaveBeenCalled();
    });

    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "상세" }));

    const originalChatLink = await screen.findByRole("link", { name: /원본 채팅 열기/i });
    expect(originalChatLink).toHaveAttribute("href", "/?session=original-session");

    expect(screen.queryByRole("link", { name: "열기" })).not.toBeInTheDocument();
    expect(
      screen.getByText("앱 내부 경로만 허용됩니다. 예: /?session=resolved-session-id"),
    ).toBeInTheDocument();
  });
});
