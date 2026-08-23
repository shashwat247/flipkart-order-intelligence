/** Client for the real Python backend (backend/api.py).
 *
 * Every call here reaches the actual LangGraph agent or an actual saved model.
 * There is no local answer table, no keyword matching and no offline fallback
 * that invents a reply — if the backend is unreachable the UI says so.
 */

export interface AgentResponse {
  answer: string;
  source: "policy_kb" | "return_risk_tool" | "image_classifier_tool" | "product_catalog" | "conversational";
  confidence: number;
}

export interface Groundedness {
  grounded: boolean;
  best_score: number;
  threshold: number;
  n_chunks_considered?: number;
}

export interface DocHit {
  document_id: string;
  title: string;
  score: number;
  text?: string;
}

export interface ToolResult {
  status: string;
  // return-risk tool
  return_probability?: number;
  return_probability_percent?: number;
  risk_bucket?: "Low" | "Medium" | "High";
  threshold_rf?: number;
  bucket_cut_points?: Record<string, string>;
  model?: string;
  // image classifier tool
  predicted_class?: string;
  confidence?: number;
  confidence_percent?: number;
  top3?: { label: string; probability: number }[];
  [key: string]: unknown;
}

export interface Turn {
  conversation_id: string;
  response: AgentResponse;
  intent: string;
  fine_intent?: string;
  intent_evidence?: { matched_example?: string; similarity?: number };
  trace: string[];
  groundedness: Groundedness | null;
  doc_hits: DocHit[];
  product_hits: unknown[];
  tool_result: ToolResult | null;
  injection?: { blocked: boolean; patterns?: string[] };
  order_id: number | null;
  turn_index: number;
  similarity_threshold: number;
}

export interface ComponentStatus {
  name: string;
  ready: boolean;
  detail: string;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export function sendMessage(
  message: string,
  conversationId: string | null,
  image: string | null,
): Promise<Turn> {
  return post<Turn>("/api/chat", {
    message,
    conversation_id: conversationId,
    image: image || null,
  });
}

export function resetConversation(): Promise<{ conversation_id: string }> {
  return post<{ conversation_id: string }>("/api/conversations/reset", {});
}

export async function fetchStatus(): Promise<{
  components: ComponentStatus[];
  ready: number;
  total: number;
  mode: string;
}> {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error(`status ${res.status}`);
  return res.json();
}

export async function fetchSamples(): Promise<string[]> {
  const res = await fetch("/api/samples");
  if (!res.ok) throw new Error(`status ${res.status}`);
  const data = (await res.json()) as { images: string[] };
  return data.images;
}
