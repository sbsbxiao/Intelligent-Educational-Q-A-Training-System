import type {
  EducationAskRequest,
  EducationAskResponse,
  HealthResponse,
  IngestResponse,
  QaAskRequest,
  QaAskResponse,
  StatsResponse,
  UpdateRequest,
  UpdateResponse
} from "./types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8080";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers
    }
  });

  if (!response.ok) {
    const message = await readErrorMessage(response);
    throw new Error(message || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const data = await response.json();
    return data.detail || data.message || "";
  } catch {
    return "";
  }
}

export function createApiState<T>(): { loading: boolean; data: T | null; error: string } {
  return {
    loading: false,
    data: null,
    error: ""
  };
}

export const apiClient = {
  askEducation(payload: EducationAskRequest): Promise<EducationAskResponse> {
    return request<EducationAskResponse>("/api/education/ask", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  askQa(payload: QaAskRequest): Promise<QaAskResponse> {
    return request<QaAskResponse>("/api/qa/ask", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  uploadDocument(file: File): Promise<IngestResponse> {
    const formData = new FormData();
    formData.append("file", file);
    return request<IngestResponse>("/api/ingest/upload", {
      method: "POST",
      body: formData
    });
  },

  uploadBatch(files: File[]): Promise<IngestResponse[]> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    return request<IngestResponse[]>("/api/ingest/batch", {
      method: "POST",
      body: formData
    });
  },

  getStats(): Promise<StatsResponse> {
    return request<StatsResponse>("/api/admin/stats");
  },

  updateKnowledge(payload: UpdateRequest): Promise<UpdateResponse> {
    return request<UpdateResponse>("/api/admin/update", {
      method: "POST",
      body: JSON.stringify(payload)
    });
  },

  health(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/health");
  }
};
