import { createBrowserRouter, RouterProvider } from "react-router-dom";
import ChatPage from "../features/chat/pages/chat-page";
import NotFoundPage from "../components/not-found-page";

const router = createBrowserRouter([
  {
    path: "/",
    element: <ChatPage />,
  },
  {
    path: "*",
    element: <NotFoundPage />,
  },
]);

export default function AppRouter() {
  return <RouterProvider router={router} />;
}
