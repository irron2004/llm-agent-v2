import { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { ChatLogsProvider } from "../context/chat-logs-context";
import { ChatReviewProvider } from "../context/chat-review-context";
import { useChatSession } from "../hooks/use-chat-session";
import { connectSse } from "../../../lib/sse";
import { env } from "../../../config/env";

vi.mock("../../../lib/sse", () => ({
  connectSse: vi.fn(),
}));

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ...actual,
    saveTurn: vi.fn().mockResolvedValue({
      session_id: "issue-thread-1",
      turn_id: 1,
      user_text: "initial question",
      assistant_text: "final answer",
      doc_refs: [],
      title: "initial question",
      ts: "2026-03-10T00:00:00Z",
      feedback_rating: null,
      feedback_reason: null,
      feedback_ts: null,
      retrieval_meta: null,
    }),
  };
});

const wrapper = ({ children }: { children: ReactNode }) => (
  <ChatLogsProvider>
    <ChatReviewProvider>{children}</ChatReviewProvider>
  </ChatLogsProvider>
);

describe("issue flow end-to-end", () => {
  const originalChatPath = env.chatPath;

  beforeEach(() => {
    vi.clearAllMocks();
    env.chatPath = "/api/agent/run";
  });

  afterEach(() => {
    env.chatPath = originalChatPath;
  });

  it("progresses summary confirm -> issue case -> sop confirm -> post-detail confirm loop", async () => {
    let call = 0;
    vi.mocked(connectSse).mockImplementation(async (req, handlers) => {
      call += 1;
      if (call === 1) {
        handlers.onMessage(
          JSON.stringify({
            type: "final",
            result: {
              query: "initial question",
              answer: "",
              interrupted: true,
              thread_id: "issue-thread-1",
              retrieved_docs: [],
              all_retrieved_docs: [],
              interrupt_payload: {
                type: "auto_parse_confirm",
                question: "initial question",
                instruction: "guided",
                options: {
                  language: [{ value: "ko", label: "Korean" }],
                  device: [{ value: "__skip__", label: "건너뛰기" }],
                  equip_id: [{ value: "__skip__", label: "건너뛰기" }],
                  task: [{ value: "issue", label: "Issue" }],
                },
                defaults: {
                  target_language: "ko",
                  device: null,
                  equip_id: null,
                  task_mode: "issue",
                },
              },
            },
          })
        );
      } else if (call === 2) {
        handlers.onMessage(
          JSON.stringify({
            type: "final",
            result: {
              query: "initial question",
              answer: "summary answer",
              interrupted: true,
              thread_id: "issue-thread-1",
              retrieved_docs: [],
              all_retrieved_docs: [],
              interrupt_payload: {
                type: "issue_confirm",
                nonce: "nonce-summary-1",
                stage: "post_summary",
                question: "initial question",
                instruction: "summary confirm",
                prompt: "더 확인하고 싶은 이슈가 있나요?",
              },
            },
          })
        );
      } else if (call === 3) {
        handlers.onMessage(
          JSON.stringify({
            type: "final",
            result: {
              query: "initial question",
              answer: "",
              interrupted: true,
              thread_id: "issue-thread-1",
              retrieved_docs: [],
              all_retrieved_docs: [],
              interrupt_payload: {
                type: "issue_case_selection",
                nonce: "nonce-case-1",
                question: "initial question",
                instruction: "case pick",
                cases: [
                  { doc_id: "doc-1", title: "Case 1", summary: "summary 1" },
                  { doc_id: "doc-2", title: "Case 2", summary: "summary 2" },
                ],
              },
            },
          })
        );
      } else if (call === 4) {
        handlers.onMessage(
          JSON.stringify({
            type: "final",
            result: {
              query: "initial question",
              answer: "detail answer",
              interrupted: true,
              thread_id: "issue-thread-1",
              retrieved_docs: [],
              all_retrieved_docs: [],
              interrupt_payload: {
                type: "issue_sop_confirm",
                nonce: "nonce-sop-1",
                question: "initial question",
                instruction: "sop confirm",
                prompt: "SOP 확인?",
                has_sop_ref: true,
                sop_hint: "SOP-ABC-001",
              },
            },
          })
        );
      } else if (call === 5) {
        handlers.onMessage(
          JSON.stringify({
            type: "final",
            result: {
              query: "initial question",
              answer: "detail answer",
              interrupted: true,
              thread_id: "issue-thread-1",
              retrieved_docs: [],
              all_retrieved_docs: [],
              interrupt_payload: {
                type: "issue_confirm",
                nonce: "nonce-post-detail-1",
                stage: "post_detail",
                question: "initial question",
                instruction: "other confirm",
                prompt: "다른 이슈도 볼까요?",
              },
            },
          })
        );
      } else {
        handlers.onMessage(
          JSON.stringify({
            type: "final",
            result: {
              query: "initial question",
              answer: "",
              interrupted: true,
              thread_id: "issue-thread-1",
              retrieved_docs: [],
              all_retrieved_docs: [],
              interrupt_payload: {
                type: "issue_case_selection",
                nonce: "nonce-case-2",
                question: "initial question",
                instruction: "case pick again",
                cases: [{ doc_id: "doc-2", title: "Case 2", summary: "summary 2" }],
              },
            },
          })
        );
      }
      handlers.onClose();
      return { close: () => {} };
    });

    const { result } = renderHook(() => useChatSession(), { wrapper });

    await act(async () => {
      await result.current.send({ text: "initial question" });
    });
    await waitFor(() => expect(result.current.pendingGuidedSelection).not.toBeNull());

    await act(async () => {
      result.current.submitGuidedSelectionFinal({
        type: "auto_parse_confirm",
        target_language: "ko",
        selected_device: null,
        selected_equip_id: null,
        task_mode: "issue",
      });
    });
    await waitFor(() => expect(result.current.pendingIssueConfirm).not.toBeNull());
    expect(result.current.messages[result.current.messages.length - 1]?.content).toBe("summary answer");

    await act(async () => {
      result.current.submitIssueConfirm(true);
    });
    await waitFor(() => expect(result.current.pendingIssueCaseSelection).not.toBeNull());

    await act(async () => {
      result.current.submitIssueCaseSelection("doc-1");
    });
    await waitFor(() => expect(result.current.pendingIssueSopConfirm).not.toBeNull());
    expect(result.current.messages[result.current.messages.length - 1]?.content).toBe("detail answer");

    await act(async () => {
      result.current.submitIssueSopConfirm(false);
    });
    await waitFor(() => expect(result.current.pendingIssueConfirm).not.toBeNull());

    await act(async () => {
      result.current.submitIssueConfirm(true);
    });
    await waitFor(() => expect(result.current.pendingIssueCaseSelection).not.toBeNull());

    expect(vi.mocked(connectSse)).toHaveBeenCalledTimes(6);

    const req2 = vi.mocked(connectSse).mock.calls[1][0];
    expect(req2.body).toMatchObject({
      thread_id: "issue-thread-1",
      ask_user_after_retrieve: false,
      resume_decision: { type: "auto_parse_confirm" },
    });

    const req3 = vi.mocked(connectSse).mock.calls[2][0];
    expect(req3.body).toMatchObject({
      thread_id: "issue-thread-1",
      ask_user_after_retrieve: false,
      resume_decision: {
        type: "issue_confirm",
        nonce: "nonce-summary-1",
        stage: "post_summary",
        confirm: true,
      },
    });

    const req4 = vi.mocked(connectSse).mock.calls[3][0];
    expect(req4.body).toMatchObject({
      thread_id: "issue-thread-1",
      ask_user_after_retrieve: false,
      resume_decision: {
        type: "issue_case_selection",
        nonce: "nonce-case-1",
        selected_doc_id: "doc-1",
      },
    });

    const req5 = vi.mocked(connectSse).mock.calls[4][0];
    expect(req5.body).toMatchObject({
      thread_id: "issue-thread-1",
      ask_user_after_retrieve: false,
      resume_decision: {
        type: "issue_sop_confirm",
        nonce: "nonce-sop-1",
        confirm: false,
      },
    });

    const req6 = vi.mocked(connectSse).mock.calls[5][0];
    expect(req6.body).toMatchObject({
      thread_id: "issue-thread-1",
      ask_user_after_retrieve: false,
      resume_decision: {
        type: "issue_confirm",
        nonce: "nonce-post-detail-1",
        stage: "post_detail",
        confirm: true,
      },
    });
  });
});
