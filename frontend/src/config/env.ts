type Env = {
  apiBase: string;
  apiPort?: string;
  chatPath: string;
  apiKey?: string;
  // Parsing 관련
  ingestionsBase: string;
  defaultIngestionRun: string;
};

const normalize = (value: string | undefined, fallback = "") =>
  (value ?? "").trim() || fallback;

const resolveApiBase = () => {
  // Explicit value wins (including empty string for proxy)
  const base = normalize(import.meta.env.VITE_API_BASE);
  if (base !== "") return base;

  // Default to same host/port (helps when running behind nginx)
  const proto = typeof window !== "undefined" ? window.location.protocol : "http:";
  const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
  const port = normalize(import.meta.env.VITE_API_PORT) || window.location.port || "8000";
  return `${proto}//${host}${port ? `:${port}` : ""}`;
};

export const env: Env = {
  apiBase: resolveApiBase(),
  apiPort: normalize(import.meta.env.VITE_API_PORT),
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
