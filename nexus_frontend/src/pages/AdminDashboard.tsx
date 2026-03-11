import { useEffect, useState } from "react";
import { getAdminSessions, getAdminTickets, getAdminEscalations, getAdminMetrics } from "../api/nexus";

const TIER_COLORS: Record<string, string> = {
  platinum: "text-purple-400",
  gold: "text-yellow-400",
  standard: "text-slate-400",
};

const STATUS_COLORS: Record<string, string> = {
  self_resolve: "text-green-400",
  escalated: "text-red-400",
  pending: "text-yellow-400",
  active: "text-blue-400",
  complete: "text-green-400",
};

export default function AdminDashboard() {
  const [sessions, setSessions] = useState<any[]>([]);
  const [tickets, setTickets] = useState<any[]>([]);
  const [escalations, setEscalations] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<Record<string, any>>({});
  const [activeTab, setActiveTab] = useState<"sessions" | "tickets" | "escalations" | "metrics" | "observability">("sessions");

  const refresh = async () => {
    try {
      const [s, t, e, m] = await Promise.all([
        getAdminSessions(),
        getAdminTickets(),
        getAdminEscalations(),
        getAdminMetrics(),
      ]);
      setSessions(s.sessions || []);
      setTickets(t.tickets || []);
      setEscalations(e.escalations || []);
      setMetrics(m.metrics || {});
    } catch (err) {
      console.error("Admin fetch error:", err);
    }
  };

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-xl font-bold text-white">Nexus Admin</h1>
            <p className="text-sm text-slate-500">NexaCloud Support Operations Dashboard</p>
          </div>
          <button
            onClick={refresh}
            className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition-colors"
          >
            ↻ Refresh
          </button>
        </div>

        {/* Quick stats */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <StatCard label="Active Sessions" value={sessions.length} />
          <StatCard label="Total Tickets" value={tickets.length} />
          <StatCard label="Escalations" value={escalations.length} color="text-red-400" />
          <StatCard label="Self-Resolve Rate" value={metrics.self_resolve_rate || "—"} color="text-green-400" />
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-4 bg-slate-900 p-1 rounded-lg w-fit border border-slate-800">
          {(["sessions", "tickets", "escalations", "metrics", "observability"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`text-xs px-3 py-1.5 rounded-md capitalize transition-colors ${
                activeTab === tab
                  ? "bg-blue-600 text-white"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {tab}
              {tab === "escalations" && escalations.length > 0 && (
                <span className="ml-1.5 bg-red-500 text-white rounded-full px-1.5 text-xs">
                  {escalations.length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Sessions tab */}
        {activeTab === "sessions" && (
          <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
            <table className="w-full text-xs">
              <thead className="border-b border-slate-800">
                <tr className="text-slate-500">
                  <th className="px-4 py-2 text-left">Session</th>
                  <th className="px-4 py-2 text-left">Company</th>
                  <th className="px-4 py-2 text-left">Tier</th>
                  <th className="px-4 py-2 text-left">Status</th>
                  <th className="px-4 py-2 text-left">Sentiment</th>
                  <th className="px-4 py-2 text-right">Triage</th>
                  <th className="px-4 py-2 text-right">Diagnostic</th>
                  <th className="px-4 py-2 text-right">Resolution</th>
                </tr>
              </thead>
              <tbody>
                {sessions.length === 0 ? (
                  <tr><td colSpan={8} className="px-4 py-6 text-center text-slate-600">No active sessions</td></tr>
                ) : (
                  sessions.map((s, i) => (
                    <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                      <td className="px-4 py-2 font-mono text-blue-400">{s.session_id}</td>
                      <td className="px-4 py-2">{s.company}</td>
                      <td className={`px-4 py-2 font-medium ${TIER_COLORS[s.tier] || ""}`}>{s.tier?.toUpperCase()}</td>
                      <td className={`px-4 py-2 ${STATUS_COLORS[s.status] || "text-slate-400"}`}>{s.status}</td>
                      <td className="px-4 py-2 text-slate-400">{s.sentiment}</td>
                      <td className="px-4 py-2 text-right">{fmtConf(s.agent_confidence?.triage)}</td>
                      <td className="px-4 py-2 text-right">{fmtConf(s.agent_confidence?.diagnostic)}</td>
                      <td className="px-4 py-2 text-right">{fmtConf(s.agent_confidence?.resolution)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Tickets tab */}
        {activeTab === "tickets" && (
          <div className="space-y-3">
            {tickets.length === 0 ? (
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 text-center text-slate-600 text-sm">No tickets generated yet</div>
            ) : (
              tickets.map((t, i) => <TicketCard key={i} ticket={t} />)
            )}
          </div>
        )}

        {/* Escalations tab */}
        {activeTab === "escalations" && (
          <div className="space-y-3">
            {escalations.length === 0 ? (
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 text-center text-green-600 text-sm">No escalations — all resolved</div>
            ) : (
              escalations.map((t, i) => <TicketCard key={i} ticket={t} highlight />)
            )}
          </div>
        )}

        {/* Metrics tab */}
        {activeTab === "metrics" && (
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(metrics)
                .filter(([, val]) => typeof val !== "object")
                .map(([key, val]) => (
                  <div key={key} className="bg-slate-800 rounded-lg p-3">
                    <div className="text-xs text-slate-500 mb-1">{key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</div>
                    <div className="text-sm font-medium text-slate-200">{String(val)}</div>
                  </div>
                ))}
              {Object.keys(metrics).length === 0 && (
                <div className="col-span-2 text-center text-slate-600 text-sm">No metrics data yet</div>
              )}
            </div>
          </div>
        )}

        {/* Observability tab */}
        {activeTab === "observability" && <ObservabilityTab metrics={metrics} />}
      </div>
    </div>
  );
}

function StatCard({ label, value, color = "text-white" }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
    </div>
  );
}

function TicketCard({ ticket, highlight }: { ticket: any; highlight?: boolean }) {
  const issue = ticket.issue_summary || {};
  const client = ticket.client || {};
  return (
    <div className={`bg-slate-900 rounded-xl border p-4 ${highlight ? "border-red-500/40" : "border-slate-800"}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-blue-400 text-sm font-bold">{ticket.ticket_id}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">{client.tier?.toUpperCase()}</span>
          <span className={`text-xs font-medium ${STATUS_COLORS[ticket.status] || "text-slate-400"}`}>
            {ticket.status?.replace("_", " ").toUpperCase()}
          </span>
        </div>
      </div>
      <div className="text-xs text-slate-300">{client.company} · {issue.product} · {issue.environment}</div>
      <div className="text-xs text-slate-500 mt-1">{issue.error_message}</div>
      {ticket.nexus_summary && (
        <div className="text-xs text-slate-400 mt-2 italic">{ticket.nexus_summary}</div>
      )}
    </div>
  );
}

const fmtConf = (v: number | undefined) =>
  v != null ? <span className={v >= 0.8 ? "text-green-400" : v >= 0.6 ? "text-yellow-400" : "text-red-400"}>{(v * 100).toFixed(0)}%</span> : <span className="text-slate-600">—</span>;

function ObservabilityTab({ metrics }: { metrics: Record<string, any> }) {
  const latency: Record<string, string> = metrics.avg_latency_by_agent_ms || {};
  const confByAgent: Record<string, string> = metrics.avg_confidence_by_agent || {};
  const confDist: Record<string, number> = metrics.confidence_distribution || {};
  const totalCalls: number = metrics.total_agent_calls || 0;

  const DIST_BANDS = [
    { label: "≥ 85%",  key: "above_85pct",  bar: "bg-green-500",  dot: "bg-green-500" },
    { label: "70–85%", key: "70_to_85pct",   bar: "bg-yellow-500", dot: "bg-yellow-500" },
    { label: "50–70%", key: "50_to_70pct",   bar: "bg-orange-500", dot: "bg-orange-500" },
    { label: "< 50%",  key: "below_50pct",   bar: "bg-red-500",    dot: "bg-red-500" },
  ];

  const latencyEntries = Object.entries(latency).map(([agent, val]) => ({
    agent: agent.replace(/_agent$/, ""),
    ms: parseInt(val.replace("ms", "")),
  }));
  const maxLatency = Math.max(...latencyEntries.map((e) => e.ms), 1);
  const totalDistCalls = DIST_BANDS.reduce((s, b) => s + (confDist[b.key] || 0), 0);

  if (totalCalls === 0) {
    return (
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-10 text-center text-slate-600 text-sm">
        No observability data yet — run a session to populate metrics
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Row 1: Latency + Confidence distribution */}
      <div className="grid grid-cols-2 gap-4">
        {/* Latency bars */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-4">Avg Latency per Agent</div>
          <div className="space-y-3">
            {latencyEntries.length === 0 ? (
              <div className="text-slate-600 text-xs">No latency data</div>
            ) : (
              latencyEntries.map(({ agent, ms }) => (
                <div key={agent}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300 capitalize">{agent}</span>
                    <span className="text-slate-400 font-mono">{ms.toLocaleString()} ms</span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${(ms / maxLatency) * 100}%` }} />
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Confidence distribution */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-4">Confidence Distribution</div>
          {totalDistCalls > 0 && (
            <div className="flex h-5 rounded-md overflow-hidden mb-4 gap-px">
              {DIST_BANDS.map(({ key, bar }) => {
                const count = confDist[key] || 0;
                const pct = (count / totalDistCalls) * 100;
                return pct > 0 ? (
                  <div key={key} className={`${bar} transition-all`} style={{ width: `${pct}%` }} />
                ) : null;
              })}
            </div>
          )}
          <div className="space-y-2">
            {DIST_BANDS.map(({ label, key, dot }) => {
              const count = confDist[key] || 0;
              const pct = totalDistCalls > 0 ? ((count / totalDistCalls) * 100).toFixed(0) : "0";
              return (
                <div key={key} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-sm ${dot}`} />
                    <span className="text-slate-400">{label}</span>
                  </div>
                  <span className="text-slate-300 font-mono">{count} <span className="text-slate-600">({pct}%)</span></span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Row 2: Cost & usage */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-4">Cost & Usage</div>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Agent Calls",   value: totalCalls,                      color: "text-white" },
            { label: "Total Cost",    value: metrics.total_cost_usd || "—",   color: "text-green-400" },
            { label: "Avg / Session", value: metrics.avg_cost_per_session_usd || "—", color: "text-slate-200" },
            { label: "Total Tokens",  value: metrics.total_tokens_used != null ? Number(metrics.total_tokens_used).toLocaleString() : "—", color: "text-slate-200" },
          ].map(({ label, value, color }) => (
            <div key={label}>
              <div className="text-xs text-slate-500 mb-1">{label}</div>
              <div className={`text-2xl font-bold ${color}`}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Row 3: Per-agent confidence */}
      {Object.keys(confByAgent).length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-4">Avg Confidence per Agent</div>
          <div className="space-y-3">
            {Object.entries(confByAgent).map(([agent, pctStr]) => {
              const pct = parseInt(pctStr.replace("%", ""));
              return (
                <div key={agent}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300 capitalize">{agent.replace(/_agent$/, "")}</span>
                    <span className={`font-mono font-medium ${pct >= 85 ? "text-green-400" : pct >= 70 ? "text-yellow-400" : "text-red-400"}`}>{pctStr}</span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${pct >= 85 ? "bg-green-500" : pct >= 70 ? "bg-yellow-500" : "bg-red-500"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
