const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface SessionResponse {
  session_id: string;
  client_id: string;
  company: string;
  tier: string;
  sla_hours: number;
  message: string;
}

export interface MessageResponse {
  session_id: string;
  response: string;
  status: string;
  agent_statuses: AgentStatus[];
  ticket?: Record<string, unknown>;
  confidence_breakdown?: Record<string, number>;
}

export interface AgentStatus {
  agent: string;
  status: "pending" | "running" | "complete";
  confidence: number;
}

export const startSession = async (apiKey: string): Promise<SessionResponse> => {
  const res = await fetch(`${BASE_URL}/conversation/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
  if (!res.ok) throw new Error(`Failed to start session: ${res.status}`);
  return res.json();
};

export const sendMessage = async (sessionId: string, message: string): Promise<MessageResponse> => {
  const res = await fetch(`${BASE_URL}/conversation/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error(`Failed to send message: ${res.status}`);
  return res.json();
};

export const getAdminSessions = async () => {
  const res = await fetch(`${BASE_URL}/admin/sessions`);
  return res.json();
};

export const getAdminTickets = async () => {
  const res = await fetch(`${BASE_URL}/admin/tickets`);
  return res.json();
};

export const getAdminEscalations = async () => {
  const res = await fetch(`${BASE_URL}/admin/escalations`);
  return res.json();
};

export const getAdminMetrics = async () => {
  const res = await fetch(`${BASE_URL}/admin/metrics`);
  return res.json();
};

export const getAgentStatus = async (sessionId: string) => {
  const res = await fetch(`${BASE_URL}/admin/agent-status/${sessionId}`);
  return res.json();
};

export const getReasoningLog = async (sessionId: string) => {
  const res = await fetch(`${BASE_URL}/admin/reasoning/${sessionId}`);
  return res.json();
};

export const getSessionStatus = async (sessionId: string) => {
  const res = await fetch(`${BASE_URL}/conversation/status/${sessionId}`);
  if (!res.ok) throw new Error(`Failed to get session status: ${res.status}`);
  return res.json();
};

export const deleteSession = async (sessionId: string): Promise<void> => {
  await fetch(`${BASE_URL}/conversation/session/${sessionId}`, { method: "DELETE" });
};
