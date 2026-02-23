import { renderHook, act } from "@testing-library/react";
import { ReactNode } from "react";
import { ChatReviewProvider, useChatReview } from "../context/chat-review-context";
import type { RetrievedDoc } from "../types";

const wrapper = ({ children }: { children: ReactNode }) => (
  <ChatReviewProvider>{children}</ChatReviewProvider>
);

const sampleDocs: RetrievedDoc[] = [
  { id: "doc-1", title: "Manual A", snippet: "test snippet", score: 0.9 },
];

describe("ChatReviewContext — setPendingRegeneration", () => {
  it("missing_device_parse does NOT clear completedRetrievedDocs", () => {
    const { result } = renderHook(() => useChatReview(), { wrapper });

    // Set completedRetrievedDocs first
    act(() => {
      result.current.setCompletedRetrievedDocs(sampleDocs);
    });
    expect(result.current.completedRetrievedDocs).toEqual(sampleDocs);

    // Set pendingRegeneration with missing_device_parse reason
    act(() => {
      result.current.setPendingRegeneration({
        messageId: "msg-1",
        originalQuery: "pump issue",
        docs: [],
        searchQueries: ["pump issue"],
        selectedDevices: [],
        selectedDocTypes: [],
        reason: "missing_device_parse",
      });
    });

    // completedRetrievedDocs should be preserved
    expect(result.current.completedRetrievedDocs).toEqual(sampleDocs);
    expect(result.current.pendingRegeneration).not.toBeNull();
  });

  it("manual reason clears completedRetrievedDocs", () => {
    const { result } = renderHook(() => useChatReview(), { wrapper });

    // Set completedRetrievedDocs first
    act(() => {
      result.current.setCompletedRetrievedDocs(sampleDocs);
    });
    expect(result.current.completedRetrievedDocs).toEqual(sampleDocs);

    // Set pendingRegeneration with manual reason
    act(() => {
      result.current.setPendingRegeneration({
        messageId: "msg-2",
        originalQuery: "pump issue",
        docs: [],
        searchQueries: ["pump issue"],
        selectedDevices: [],
        selectedDocTypes: [],
        reason: "manual",
      });
    });

    // completedRetrievedDocs should be cleared
    expect(result.current.completedRetrievedDocs).toBeNull();
    expect(result.current.pendingRegeneration).not.toBeNull();
  });
});
