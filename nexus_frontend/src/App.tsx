import { useState } from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import ChatInterface from "./components/ChatInterface";
import AgentPipelinePanel from "./components/AgentPipelinePanel";
import LiveTicketPreview from "./components/LiveTicketPreview";
import ReasoningToggle from "./components/ReasoningToggle";
import AdminDashboard from "./pages/AdminDashboard";
import { startSession, sendMessage, getReasoningLog } from "./api/nexus";
import type { AgentStatus } from "./api/nexus";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

function SupportApp() {
  const [apiKey, setApiKey] = useState("nxa_acme_test_key_001");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [company, setCompany] = useState("");
  const [tier, setTier] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("active");
  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>([]);
  const [ticket, setTicket] = useState<Record<string, unknown> | null>(null);
  const [confidenceBreakdown, setConfidenceBreakdown] = useState<Record<string, number> | null>(null);
  const [reasoningLogs, setReasoningLogs] = useState<Record<string, any[]>>({});

  const handleEndSession = () => {
    setSessionId(null);
    setCompany("");
    setTier("");
    setMessages([]);
    setStatus("active");
    setAgentStatuses([]);
    setTicket(null);
    setConfidenceBreakdown(null);
    setReasoningLogs({});
  };

  const handleStartSession = async () => {
    try {
      setIsLoading(true);
      // Reset all previous session state before starting fresh
      setTicket(null);
      setAgentStatuses([]);
      setConfidenceBreakdown(null);
      setReasoningLogs({});
      setStatus("active");
      const session = await startSession(apiKey);
      setSessionId(session.session_id);
      setCompany(session.company);
      setTier(session.tier);
      setMessages([{ role: "assistant", content: session.message, timestamp: new Date() }]);
    } catch (err) {
      console.error("Failed to start session:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (msg: string) => {
    if (!sessionId) return;
    setMessages((prev) => [...prev, { role: "user", content: msg, timestamp: new Date() }]);
    setIsLoading(true);
    try {
      const result = await sendMessage(sessionId, msg);
      setMessages((prev) => [...prev, { role: "assistant", content: result.response, timestamp: new Date() }]);
      setStatus(result.status);
      setAgentStatuses(result.agent_statuses || []);
      if (result.ticket) setTicket(result.ticket);
      if (result.confidence_breakdown) setConfidenceBreakdown(result.confidence_breakdown);

      try {
        const logs = await getReasoningLog(sessionId);
        setReasoningLogs(logs.reasoning_logs || {});
      } catch {}
    } catch (err) {
      console.error("Failed to send message:", err);
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
        timestamp: new Date(),
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <div className="border-b border-slate-800 px-6 py-3 flex items-center justify-between bg-slate-900">
        <div className="flex items-center gap-3">
          <div className="text-blue-400 font-bold text-lg">Nexus</div>
          <div className="text-slate-600 text-sm">NexaCloud Support</div>
        </div>
        <div className="flex items-center gap-4">
          {sessionId && (
            <button
              onClick={handleEndSession}
              className="text-xs text-slate-400 hover:text-red-400 border border-slate-700 hover:border-red-700 px-3 py-1 rounded-lg transition-colors"
            >
              ← End Session
            </button>
          )}
          <Link to="/admin" className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
            Admin Dashboard →
          </Link>
        </div>
      </div>

      {!sessionId && (
        <div className="max-w-md mx-auto mt-16 p-6 bg-slate-900 rounded-xl border border-slate-700">
          <h2 className="text-lg font-semibold mb-4">Start Support Session</h2>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Client API Key</label>
              <select
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none"
              >
                <option value="nxa_acme_test_key_001">Acme Corp (Platinum) — NexaAuth/NexaPay</option>
                <option value="nxa_gretail_test_key_002">GlobalRetail SA (Gold) — NexaStore/NexaMsg</option>
                <option value="nxa_devstartup_test_key_003">DevStartup Ltd (Standard) — NexaAuth</option>
              </select>
            </div>
            <button
              onClick={handleStartSession}
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white py-2 rounded-lg text-sm font-medium transition-colors"
            >
              {isLoading ? "Connecting..." : "Start Session"}
            </button>
          </div>
        </div>
      )}

      {sessionId && (
        <div className="grid grid-cols-12 gap-4 p-4 h-[calc(100vh-57px)]">
          <div className="col-span-5 flex flex-col gap-3 overflow-hidden">
            <div className="flex-1 min-h-0">
              <ChatInterface
                sessionId={sessionId}
                company={company}
                tier={tier}
                onSendMessage={handleSendMessage}
                messages={messages}
                isLoading={isLoading}
                status={status}
                agentStatuses={agentStatuses}
              />
            </div>
            <ReasoningToggle sessionId={sessionId} reasoningLogs={reasoningLogs} />
          </div>

          <div className="col-span-2">
            <AgentPipelinePanel agentStatuses={agentStatuses} overallStatus={status} />
          </div>

          <div className="col-span-5 overflow-hidden">
            <LiveTicketPreview
              ticket={ticket}
              confidenceBreakdown={confidenceBreakdown}
              sessionId={sessionId}
              sessionStatus={status}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SupportApp />} />
        <Route path="/admin" element={<AdminDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}
