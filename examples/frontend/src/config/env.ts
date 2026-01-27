type Env = {
  apiBase: string;
  chatPath: string;
  apiKey?: string;
  // Parsing 관련
  ingestionsBase: string;
  defaultIngestionRun: string;
};

const normalize = (value: string | undefined, fallback = "") =>
  (value ?? "").trim() || fallback;

export const env: Env = {
  apiBase: normalize(import.meta.env.VITE_API_BASE),
  chatPath: normalize(import.meta.env.VITE_CHAT_PATH, "/api/chat"),
  apiKey: normalize(import.meta.env.VITE_API_KEY) || undefined,
  // Parsing 관련
  ingestionsBase: normalize(import.meta.env.VITE_INGESTIONS_BASE, "/data/ingestions"),
  defaultIngestionRun: normalize(import.meta.env.VITE_DEFAULT_INGESTION_RUN, ""),
};

export const buildUrl = (path: string) => {
  const base = env.apiBase.replace(/\/$/, "");
  const rel = path.startsWith("/") ? path : `/${path}`;
  return base ? `${base}${rel}` : rel;
};
