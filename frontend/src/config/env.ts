type Env = {
  apiBase: string;
  chatPath: string;
  apiKey?: string;
};

const normalize = (value: string | undefined, fallback = "") =>
  (value ?? "").trim() || fallback;

export const env: Env = {
  apiBase: normalize(import.meta.env.VITE_API_BASE),
  chatPath: normalize(import.meta.env.VITE_CHAT_PATH, "/api/chat"),
  apiKey: normalize(import.meta.env.VITE_API_KEY) || undefined,
};

export const buildUrl = (path: string) => {
  const base = env.apiBase.replace(/\/$/, "");
  const rel = path.startsWith("/") ? path : `/${path}`;
  return base ? `${base}${rel}` : rel;
};
