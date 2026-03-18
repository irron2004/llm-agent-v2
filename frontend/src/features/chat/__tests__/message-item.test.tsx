import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { MessageItem } from "../components/message-item";
import type { Message } from "../types";


describe("MessageItem", () => {
  it("renders re-search action and calls handler with original query", async () => {
    const user = userEvent.setup();
    const onRegenerate = vi.fn();

    const message: Message = {
      id: "assistant-1",
      role: "assistant",
      content: "답변입니다.",
      retrievedDocs: [
        {
          id: "doc-1",
          title: "Doc 1",
          snippet: "snippet",
        },
      ],
    };

    render(
      <MessageItem
        message={message}
        isStreaming={false}
        onRegenerate={onRegenerate}
        originalQuery="원본 질문"
      />,
    );

    const button = screen.getByRole("button", { name: /재생성|재검색/ });
    expect(button).toBeInTheDocument();

    await user.click(button);

    expect(onRegenerate).toHaveBeenCalledTimes(1);
    expect(onRegenerate).toHaveBeenCalledWith(
      expect.objectContaining({
        messageId: "assistant-1",
        originalQuery: "원본 질문",
      }),
    );
  });

  it("renders SOP flow chart first and shows single final source doc label", () => {
    const message: Message = {
      id: "assistant-sop-1",
      role: "assistant",
      content: "절차 안내 본문입니다.",
      retrievedDocs: [
        {
          id: "sop-doc-1",
          title: "SOP Doc",
          snippet: "work procedure snippet",
          metadata: { doc_type: "sop", section_chapter: "flow chart" },
          page_image_url: "/api/assets/docs/sop-doc-1/pages/1",
        },
      ],
      allRetrievedDocs: [
        {
          id: "sop-doc-1",
          title: "SOP Doc",
          snippet: "work procedure snippet",
          metadata: { doc_type: "sop", section_chapter: "flow chart" },
          page_image_url: "/api/assets/docs/sop-doc-1/pages/1",
        },
      ],
      selectedDocTypes: ["sop"],
      expandedDocs: [
        {
          rank: 1,
          doc_id: "sop-doc-1",
          content: "expanded content",
          content_length: 16,
        },
      ],
    };

    render(<MessageItem message={message} isStreaming={false} />);

    expect(screen.getByAltText("SOP flow chart")).toBeInTheDocument();
    expect(screen.getByText("답변에 사용된 문서 (1)")).toBeInTheDocument();
  });

  it("uses expandedDocs answer source instead of retrievedDocs order", () => {
    const message: Message = {
      id: "assistant-sop-2",
      role: "assistant",
      content: "절차 안내 본문입니다.",
      selectedDocTypes: ["sop"],
      expandedDocs: [
        {
          rank: 1,
          doc_id: "sop-doc-1",
          content: "expanded content",
          content_length: 16,
        },
      ],
      retrievedDocs: [
        {
          id: "sop-doc-2",
          title: "SOP Doc 2",
          snippet: "work procedure snippet",
          metadata: { doc_type: "sop", section_chapter: "flow chart" },
          page_image_url: "/api/assets/docs/sop-doc-2/pages/2",
        },
        {
          id: "sop-doc-1",
          title: "SOP Doc 1",
          snippet: "work procedure snippet",
          metadata: { doc_type: "sop", section_chapter: "flow chart" },
          page_image_url: "/api/assets/docs/sop-doc-1/pages/1",
        },
      ],
      allRetrievedDocs: [
        {
          id: "sop-doc-1",
          title: "SOP Doc 1",
          snippet: "work procedure snippet",
          metadata: { doc_type: "sop", section_chapter: "flow chart" },
          page_image_url: "/api/assets/docs/sop-doc-1/pages/1",
        },
      ],
    };

    render(<MessageItem message={message} isStreaming={false} />);

    const flowChart = screen.getByAltText("SOP flow chart") as HTMLImageElement;
    expect(flowChart.src).toContain("/api/assets/docs/sop-doc-1/pages/1");
  });

  it("enables SOP presentation when raw metadata selected_task_mode is sop", () => {
    const message: Message = {
      id: "assistant-sop-3",
      role: "assistant",
      content: "절차 안내 본문입니다.",
      rawAnswer: JSON.stringify({
        metadata: { selected_task_mode: "sop" },
        expanded_docs: [{ rank: 1, doc_id: "sop-doc-3" }],
      }),
      retrievedDocs: [
        {
          id: "sop-doc-3",
          title: "SOP Doc 3",
          snippet: "work procedure snippet",
          metadata: { doc_type: "sop", section_chapter: "flow chart" },
          page_image_url: "/api/assets/docs/sop-doc-3/pages/1",
        },
      ],
      allRetrievedDocs: [
        {
          id: "sop-doc-3",
          title: "SOP Doc 3",
          snippet: "work procedure snippet",
          metadata: { doc_type: "sop", section_chapter: "flow chart" },
          page_image_url: "/api/assets/docs/sop-doc-3/pages/1",
        },
      ],
    };

    render(<MessageItem message={message} isStreaming={false} />);

    expect(screen.getByAltText("SOP flow chart")).toBeInTheDocument();
    expect(screen.getByText("답변에 사용된 문서 (1)")).toBeInTheDocument();
  });

  it("renders inline issue case buttons and handles selection", async () => {
    const user = userEvent.setup();
    const onIssueCaseSelect = vi.fn();
    const message: Message = {
      id: "assistant-issue-1",
      role: "assistant",
      content: "이슈 결과를 정리했습니다.",
    };

    render(
      <MessageItem
        message={message}
        isStreaming={false}
        issueCases={[
          { doc_id: "doc-1", title: "Door Open Alarm", summary: "..." },
          { doc_id: "doc-2", title: "Vacuum Error", summary: "..." },
        ]}
        onIssueCaseSelect={onIssueCaseSelect}
      />,
    );

    expect(screen.getByRole("button", { name: "1. Door Open Alarm" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "2. Vacuum Error" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "2. Vacuum Error" }));
    expect(onIssueCaseSelect).toHaveBeenCalledWith("doc-2");
  });
});
