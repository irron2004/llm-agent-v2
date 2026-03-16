import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GuidedSelectionPanel } from "../components/guided-selection-panel";

describe("GuidedSelectionPanel", () => {
  it("completes 2-step click flow and calls onComplete once with expected decision", async () => {
    const user = userEvent.setup();
    const onComplete = vi.fn();

    render(
      <GuidedSelectionPanel
        question="질문"
        instruction="단계별로 선택하세요"
        payload={{
          steps: ["device", "task"],
          options: {
            device: [{ value: "ETCH-01", label: "ETCH-01" }],
            task: [
              { value: "sop", label: "SOP" },
              { value: "issue", label: "Issue" },
            ],
          },
          defaults: {
            target_language: "en",
            device: null,
            equip_id: null,
            task_mode: "all",
          },
        }}
        onComplete={onComplete}
      />,
    );

    await user.click(screen.getByRole("button", { name: /ETCH-01/ }));
    await user.click(screen.getByRole("button", { name: /Issue/ }));

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete).toHaveBeenCalledWith({
      type: "auto_parse_confirm",
      target_language: "en",
      selected_device: "ETCH-01",
      selected_equip_id: null,
      task_mode: "issue",
    });
  });

  it("uses manual equip path and submits typed selected_equip_id", async () => {
    const user = userEvent.setup();
    const onComplete = vi.fn();

    render(
      <GuidedSelectionPanel
        question="질문"
        instruction="단계별로 선택하세요"
        payload={{
          steps: ["device", "equip_id", "task"],
          options: {
            device: [{ value: "__skip__", label: "건너뛰기" }],
            equip_id: [{ value: "__manual__", label: "직접 입력" }],
            task: [{ value: "sop", label: "SOP" }],
          },
          defaults: {
            target_language: "ko",
            device: null,
            equip_id: null,
            task_mode: "all",
          },
        }}
        onComplete={onComplete}
      />,
    );

    await user.click(screen.getByRole("button", { name: /건너뛰기/ }));
    await user.click(screen.getByRole("button", { name: /직접 입력/ }));

    const equipInput = screen.getByPlaceholderText(/equip_id/i);
    await user.type(equipInput, "MANUAL-EQ-42");
    await user.click(screen.getByRole("button", { name: "확인" }));
    await user.click(screen.getByRole("button", { name: /SOP/ }));

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete).toHaveBeenCalledWith({
      type: "auto_parse_confirm",
      target_language: "ko",
      selected_device: null,
      selected_equip_id: "MANUAL-EQ-42",
      task_mode: "sop",
    });
  });

  it("reflects external stepIndex/draftDecision updates for numeric flow feedback", async () => {
    const onComplete = vi.fn();

    const { rerender } = render(
      <GuidedSelectionPanel
        question="질문"
        instruction="단계별로 선택하세요"
        payload={{
          steps: ["device", "task"],
          options: {
            device: [{ value: "ETCH-01", label: "ETCH-01" }],
            task: [{ value: "sop", label: "SOP" }],
          },
          defaults: {
            target_language: "ko",
            device: null,
            equip_id: null,
            task_mode: "all",
          },
        }}
        onComplete={onComplete}
      />,
    );

    rerender(
      <GuidedSelectionPanel
        question="질문"
        instruction="단계별로 선택하세요"
        payload={{
          steps: ["device", "task"],
          options: {
            device: [{ value: "ETCH-01", label: "ETCH-01" }],
            task: [{ value: "sop", label: "SOP" }],
          },
          defaults: {
            target_language: "ko",
            device: null,
            equip_id: null,
            task_mode: "all",
          },
        }}
        stepIndex={1}
        draftDecision={{
          type: "auto_parse_confirm",
          target_language: "en",
        }}
        onComplete={onComplete}
      />,
    );

    expect(screen.getByText("작업 선택")).toBeInTheDocument();
    expect(screen.getByText(/선택됨: 기기\(/)).toBeInTheDocument();
    expect(screen.queryByText(/선택됨: .*설비\(/)).not.toBeInTheDocument();
    expect(screen.queryByText(/선택됨: 언어\(/)).not.toBeInTheDocument();
  });
});
