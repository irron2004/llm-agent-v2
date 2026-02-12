import { RouterProvider } from "react-router-dom";
import prodRouter from "./router.prod";
import devRouter from "./router.dev";

// Dynamic import based on build target
// VITE_BUILD_TARGET=prod → only chat/feedback
// VITE_BUILD_TARGET=dev (or unset) → all pages
const isProd = import.meta.env.VITE_BUILD_TARGET === "prod";

const router = isProd ? prodRouter : devRouter;

export default function AppRouter() {
  return <RouterProvider router={router} />;
}
