import { createBrowserRouter } from "react-router-dom";
import Layout from "../components/layout";
import ChatPage from "../features/chat/pages/chat-page";
import FeedbackPage from "../features/feedback/pages/feedback-page";
import NotFoundPage from "../components/not-found-page";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      {
        index: true,
        element: <ChatPage />,
      },
      {
        path: "feedback",
        element: <FeedbackPage />,
      },
      {
        path: "*",
        element: <NotFoundPage />,
      },
    ],
  },
]);

export default router;
