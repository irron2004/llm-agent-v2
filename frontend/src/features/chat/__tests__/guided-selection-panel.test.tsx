import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GuidedSelectionPanel } from "../components/guided-selection-panel";

describe("GuidedSelectionPanel", () => {
  it("completes click flow and calls onComplete once with expected decision", async () => {
    const user = userEvent.setup();
    const onComplete = vi.fn();

    render(
      <GuidedSelectionPanel
        question="질문"
        instruction="단계별로 선택하세요"
        payload={{
          options: {
            language: [
              { value: "ko", label: "Korean" },
              { value: "en", label: "English" },
            ],
            device: [{ value: "ETCH-01", label: "ETCH-01" }],
            equip_id: [{ value: "EQ-777", label: "EQ-777" }],
            task: [
              { value: "sop", label: "SOP" },
              { value: "issue", label: "Issue" },
            ],
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

    await user.click(screen.getByRole("button", { name: "English" }));
    await user.click(screen.getByRole("button", { name: "ETCH-01" }));
    await user.click(screen.getByRole("button", { name: "EQ-777" }));
    await user.click(screen.getByRole("button", { name: "Issue" }));

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete).toHaveBeenCalledWith({
      type: "auto_parse_confirm",
      target_language: "en",
      selected_device: "ETCH-01",
      selected_equip_id: "EQ-777",
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
          options: {
            language: [{ value: "ko", label: "Korean" }],
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

    await user.click(screen.getByRole("button", { name: "Korean" }));
    await user.click(screen.getByRole("button", { name: "건너뛰기" }));
    await user.click(screen.getByRole("button", { name: "직접 입력" }));

    const equipInput = screen.getByPlaceholderText(/equip_id/i);
    await user.type(equipInput, "MANUAL-EQ-42");
    await user.click(screen.getByRole("button", { name: "확인" }));
    await user.click(screen.getByRole("button", { name: "SOP" }));

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete).toHaveBeenCalledWith({
      type: "auto_parse_confirm",
      target_language: "ko",
      selected_device: null,
      selected_equip_id: "MANUAL-EQ-42",
      task_mode: "sop",
    });
  });
});
