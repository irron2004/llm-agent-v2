import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ImagePreviewModal } from "../image-preview-modal";

describe("ImagePreviewModal", () => {
  it("clamps an out-of-range currentIndex and renders the nearest valid page", () => {
    render(
      <ImagePreviewModal
        visible
        images={[
          { url: "https://example.com/1.png", title: "Doc-1", page: 1 },
          { url: "https://example.com/2.png", title: "Doc-2", page: 2 },
        ]}
        currentIndex={99}
        onIndexChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText("2 / 2")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Doc-2 page 2/i })).toBeInTheDocument();
    expect(document.body.querySelector(".image-preview-modal-overlay")).toBeTruthy();
  });

  it("navigates from clamped index using previous button", async () => {
    const user = userEvent.setup();
    const onIndexChange = vi.fn();

    render(
      <ImagePreviewModal
        visible
        images={[
          { url: "https://example.com/1.png", title: "Doc-1", page: 1 },
          { url: "https://example.com/2.png", title: "Doc-2", page: 2 },
        ]}
        currentIndex={50}
        onIndexChange={onIndexChange}
        onClose={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "이전" }));
    expect(onIndexChange).toHaveBeenCalledWith(0);
  });
});
