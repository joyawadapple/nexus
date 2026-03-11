import { useState } from "react";

interface ReasoningEntry {
  step: string;
  decision: string;
  rationale: string;
  timestamp: string;
}

interface Props {
  sessionId: string | null;
  reasoningLogs: Record<string, ReasoningEntry[]>;
}

const STEP_COLORS: Record<string, string> = {
  LOAD: "text-blue-400",
  ANALYZE: "text-purple-400",
  REASON: "text-yellow-400",
  DECIDE: "text-orange-400",
  GENERATE: "text-green-400",
  RETURN: "text-teal-400",
};

export default function ReasoningToggle({ sessionId, reasoningLogs }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);

  const agents = Object.keys(reasoningLogs);
  const activeLog = activeAgent ? reasoningLogs[activeAgent] : null;

  if (!sessionId) return null;

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-slate-300 hover:text-white transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-slate-500">🔍</span>
          <span>Agent Reasoning</span>
          {agents.length > 0 && (
            <span className="text-xs bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded">
              {agents.length} agents
            </span>
          )}
        </div>
        <span className="text-slate-500">{isOpen ? "▲" : "▼"}</span>
      </button>

      {isOpen && (
        <div className="border-t border-slate-700 p-4">
          {agents.length === 0 ? (
            <div className="text-xs text-slate-500 text-center py-2">
              No reasoning data yet. Send a message to start.
            </div>
          ) : (
            <>
              {/* Agent selector */}
              <div className="flex gap-2 mb-4 flex-wrap">
                {agents.map((agent) => (
                  <button
                    key={agent}
                    onClick={() => setActiveAgent(activeAgent === agent ? null : agent)}
                    className={`text-xs px-2.5 py-1 rounded-lg border transition-colors ${
                      activeAgent === agent
                        ? "bg-blue-600 border-blue-500 text-white"
                        : "border-slate-600 text-slate-400 hover:border-slate-500"
                    }`}
                  >
                    {agent.replace("_agent", "").replace("_", " ")}
                  </button>
                ))}
              </div>

              {/* Reasoning entries */}
              {activeLog && (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {activeLog.map((entry, i) => (
                    <div key={i} className="text-xs bg-slate-800 rounded-lg p-3">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`font-mono font-bold ${STEP_COLORS[entry.step] || "text-slate-400"}`}>
                          {entry.step}
                        </span>
                        <span className="text-slate-500">
                          {new Date(entry.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="text-slate-300 mb-0.5">{entry.decision}</div>
                      <div className="text-slate-500">{entry.rationale}</div>
                    </div>
                  ))}
                </div>
              )}

              {!activeAgent && (
                <div className="text-xs text-slate-500 text-center">
                  Select an agent above to view its reasoning log
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
