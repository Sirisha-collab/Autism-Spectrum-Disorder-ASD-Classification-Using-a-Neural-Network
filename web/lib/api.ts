import type {
  MetricsPayload,
  ModelsPayload,
  PredictResponse,
  QuestionsPayload,
} from "./types";

/** FastAPI returns either a string detail or a list of validation errors. */
function readDetail(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const first = detail[0] as { msg?: string } | undefined;
    if (first?.msg) return first.msg;
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { cache: "no-store", ...init });
  } catch {
    throw new Error(
      "Can't reach the screening service. Start it with: uvicorn api.main:app --reload --port 8000",
    );
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(readDetail(body, `Request failed (${res.status})`));
  }
  return res.json();
}

export const getQuestions = () => request<QuestionsPayload>("/api/questions");
export const getModels = () => request<ModelsPayload>("/api/models");
export const getMetrics = () => request<MetricsPayload>("/api/metrics");

export const predict = (body: {
  answers: Record<string, number>;
  demographics: Record<string, string | number>;
  model?: string | null;
  consent: boolean;
}) =>
  request<PredictResponse>("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
