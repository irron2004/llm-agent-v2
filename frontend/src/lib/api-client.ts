import { env, buildUrl } from "../config/env";

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

type RequestOptions = {
  method?: HttpMethod;
  path: string;
  body?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
};

async function request<T>(options: RequestOptions): Promise<T> {
  const { method = "GET", path, body, headers, signal } = options;
  const url = buildUrl(path);

  const res = await fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(env.apiKey ? { Authorization: env.apiKey } : {}),
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });

  if (!res.ok) {
    const message = await safeParseError(res);
    throw new Error(message);
  }

  if (res.status === 204) {
    return {} as T;
  }

  return (await res.json()) as T;
}

async function safeParseError(res: Response) {
  try {
    const payload = await res.json();
    // FastAPI uses "detail", others use "message"
    if (payload?.detail) {
      return payload.detail;
    }
    if (payload?.message) {
      return payload.message;
    }
  } catch (_) {
    // ignore
  }
  return `HTTP ${res.status}`;
}

export const apiClient = {
  get: <T>(path: string, opts?: Omit<RequestOptions, "path" | "method">) =>
    request<T>({ ...opts, path, method: "GET" }),
  post: <T>(path: string, body?: unknown, opts?: Omit<RequestOptions, "path" | "method" | "body">) =>
    request<T>({ ...opts, path, method: "POST", body }),
  patch: <T>(path: string, body?: unknown, opts?: Omit<RequestOptions, "path" | "method" | "body">) =>
    request<T>({ ...opts, path, method: "PATCH", body }),
  delete: <T>(path: string, opts?: Omit<RequestOptions, "path" | "method">) =>
    request<T>({ ...opts, path, method: "DELETE" }),
};
