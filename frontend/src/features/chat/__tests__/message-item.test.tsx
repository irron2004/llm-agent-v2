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

    const button = screen.getByRole("button", { name: /재검색/ });
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
});
