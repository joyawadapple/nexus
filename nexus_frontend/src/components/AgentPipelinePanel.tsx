import type { AgentStatus } from "../api/nexus";

interface Props {
  agentStatuses: AgentStatus[];
  overallStatus: string;
}

const AGENT_LABELS: Record<string, { label: string; step: number }> = {
  triage_agent: { label: "Triage", step: 1 },
  diagnostic_agent: { label: "Diagnosing", step: 2 },
  resolution_agent: { label: "Resolution", step: 3 },
  escalation_agent: { label: "Escalation", step: 4 },
};

const STATUS_ICON: Record<string, string> = {
  complete: "✓",
  running: "⟳",
  pending: "○",
};

const STATUS_COLOR: Record<string, string> = {
  complete: "text-green-400 bg-green-400/10 border-green-400/30",
  running: "text-blue-400 bg-blue-400/10 border-blue-400/30 animate-pulse",
  pending: "text-slate-500 bg-slate-800 border-slate-700",
};

const CONFIDENCE_BAR_COLOR = (conf: number) => {
  if (conf >= 0.85) return "bg-green-500";
  if (conf >= 0.70) return "bg-yellow-500";
  if (conf >= 0.50) return "bg-orange-500";
  return "bg-red-500";
};

export default function AgentPipelinePanel({ agentStatuses, overallStatus }: Props) {
  const agents = [
    "triage_agent",
    "diagnostic_agent",
    "resolution_agent",
    "escalation_agent",
  ];

  const getStatus = (agentId: string): AgentStatus => {
    return (
      agentStatuses.find((a) => a.agent === agentId) || {
        agent: agentId,
        status: "pending",
        confidence: 0,
      }
    );
  };

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-300">Agent Pipeline</h3>
        <span className="text-xs text-slate-500 uppercase tracking-wide">
          {overallStatus.replace("_", " ")}
        </span>
      </div>

      <div className="space-y-3">
        {agents.map((agentId, idx) => {
          const info = AGENT_LABELS[agentId];
          const agentStatus = getStatus(agentId);
          const conf = agentStatus.confidence;

          return (
            <div key={agentId}>
              {/* Connector line */}
              {idx > 0 && (
                <div className="flex justify-center mb-2">
                  <div className="w-px h-4 bg-slate-700" />
                </div>
              )}

              <div
                className={`rounded-lg border px-3 py-2.5 ${STATUS_COLOR[agentStatus.status]}`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm">
                      {STATUS_ICON[agentStatus.status]}
                    </span>
                    <span className="text-sm font-medium">{info.label}</span>
                  </div>
                  <span className="text-xs font-mono">
                    {agentStatus.status === "pending"
                      ? "—"
                      : `${(conf * 100).toFixed(0)}%`}
                  </span>
                </div>

                {/* Confidence bar */}
                {agentStatus.status !== "pending" && (
                  <div className="mt-1.5 h-1 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${CONFIDENCE_BAR_COLOR(conf)}`}
                      style={{ width: `${conf * 100}%` }}
                    />
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Parallel execution note */}
      <div className="mt-4 text-xs text-slate-600 text-center">
        Triage → Diagnostic ∥ Resolution → Escalation
      </div>
    </div>
  );
}
