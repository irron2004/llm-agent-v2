import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Layout from "../components/layout";
import ChatPage from "../features/chat/pages/chat-page";
import ParsingPage from "../features/parsing/pages/parsing-page";
import SearchPage from "../features/search/pages/search-page";
import RetrievalTestPage from "../features/retrieval-test/pages/retrieval-test-page";
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
        path: "parsing",
        element: <ParsingPage />,
      },
      {
        path: "search",
        element: <SearchPage />,
      },
      {
        path: "retrieval-test",
        element: <RetrievalTestPage />,
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

export default function AppRouter() {
  return <RouterProvider router={router} />;
}
