import { waitFor } from "@testing-library/react";
import {
  renderChatPage,
  resetAllMocks,
  mockLoadSession,
  mockSetSearchParams,
  setMockSearchParams,
} from "./helpers/render-chat-page";

beforeEach(() => {
  resetAllMocks();
});

describe("ChatPage session link behavior", () => {
  it("loads the session from the URL query parameter on mount", async () => {
    setMockSearchParams("session=session-from-url");

    await renderChatPage({
      chatSession: {
        sessionId: "local-session",
      },
    });

    expect(mockLoadSession).toHaveBeenCalledWith("session-from-url");
  });

  it("syncs the current session into the URL when no query parameter is present", async () => {
    await renderChatPage({
      chatSession: {
        sessionId: "session-from-state",
      },
    });

    await waitFor(() => {
      expect(mockSetSearchParams).toHaveBeenCalledWith(
        { session: "session-from-state" },
        { replace: true },
      );
    });
  });

});
