export type ApiState<T> = {
  loading: boolean;
  data: T | null;
  error: string;
};

export type EducationAskRequest = {
  question: string;
};

export type TokenUsage = {
  task_id: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
  reasoning_tokens: number;
  llm_calls: number;
};

export type EducationAskResponse = {
  question: string;
  answer: string;
  skill: string;
  tools_used: string[];
  sources: Record<string, unknown>[];
  token_usage: TokenUsage;
};

export type QaAskRequest = {
  question: string;
};

export type QaAskResponse = {
  question: string;
  answer: string;
  confidence: number;
  intent: string;
  sources: Record<string, unknown>[];
  reasoning_steps: string[];
  token_usage: TokenUsage;
};

export type IngestResponse = {
  file_name: string;
  chunks_count: number;
  entities_count: number;
  relations_count: number;
  status: string;
};

export type StatsResponse = {
  vector_store: Record<string, unknown>;
  knowledge_graph: Record<string, unknown>;
};

export type UpdateRequest = {
  file_path: string;
  change_type: "created" | "modified" | "deleted";
};

export type UpdateResponse = {
  file_path: string;
  vectors_added: number;
  vectors_deleted: number;
  entities_added: number;
  relations_added: number;
  success: boolean;
  processing_time_ms: number;
};

export type HealthResponse = {
  status: string;
  service: string;
};
