import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { IssueConfirmPanel } from "../components/issue-confirm-panel";
import { IssueCaseSelectionPanel } from "../components/issue-case-selection-panel";
import { IssueSopConfirmPanel } from "../components/issue-sop-confirm-panel";

describe("Issue flow panels", () => {
  it("IssueCaseSelectionPanel submits selected doc_id", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();

    render(
      <IssueCaseSelectionPanel
        question="질문"
        instruction="선택하세요"
        cases={[
          { doc_id: "doc-1", title: "Case 1", summary: "first" },
          { doc_id: "doc-2", title: "Case 2", summary: "second" },
        ]}
        onSelect={onSelect}
      />
    );

    await user.click(screen.getByRole("button", { name: /1\. Case 1/i }));
    expect(onSelect).toHaveBeenCalledWith("doc-1");

    await user.type(screen.getByPlaceholderText(/번호로 선택/i), "2");
    await user.click(screen.getByRole("button", { name: "선택" }));
    expect(onSelect).toHaveBeenLastCalledWith("doc-2");
  });

  it("IssueSopConfirmPanel submits yes/no", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();

    render(
      <IssueSopConfirmPanel
        question="질문"
        instruction="확인하세요"
        prompt="SOP 확인할까요?"
        hasSopRef
        sopHint="SOP-ABC-001"
        onConfirm={onConfirm}
      />
    );

    await user.click(screen.getByRole("button", { name: "예" }));
    expect(onConfirm).toHaveBeenCalledWith(true);

    await user.click(screen.getByRole("button", { name: "아니오" }));
    expect(onConfirm).toHaveBeenLastCalledWith(false);
  });

  it("IssueConfirmPanel submits yes/no", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();

    render(
      <IssueConfirmPanel
        question="질문"
        instruction="확인하세요"
        prompt="다른 이슈도 볼까요?"
        stage="post_detail"
        onConfirm={onConfirm}
      />
    );

    await user.click(screen.getByRole("button", { name: "예" }));
    expect(onConfirm).toHaveBeenCalledWith(true);

    await user.click(screen.getByRole("button", { name: "아니오" }));
    expect(onConfirm).toHaveBeenLastCalledWith(false);
  });
});
