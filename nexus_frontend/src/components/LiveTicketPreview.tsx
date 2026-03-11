import { useState, useEffect } from "react";
import { getSessionStatus } from "../api/nexus";

interface Props {
  ticket: Record<string, unknown> | null;
  confidenceBreakdown: Record<string, number> | null;
  sessionId: string | null;
  sessionStatus: string;
}

const STATUS_BADGE: Record<string, string> = {
  self_resolve: "bg-green-500/20 text-green-400 border border-green-500/30",
  escalated: "bg-red-500/20 text-red-400 border border-red-500/30",
  pending: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
  pending_review: "bg-orange-500/20 text-orange-400 border border-orange-500/30",
};

const PRIORITY_BADGE: Record<string, string> = {
  critical: "text-red-400",
  high: "text-orange-400",
  medium: "text-yellow-400",
  low: "text-green-400",
};

export default function LiveTicketPreview({ ticket, confidenceBreakdown, sessionId, sessionStatus }: Props) {
  const [partialTicket, setPartialTicket] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!sessionId || sessionStatus === "complete") {
      return;
    }
    const interval = setInterval(async () => {
      try {
        const data = await getSessionStatus(sessionId);
        if (data.partial_ticket && Object.keys(data.partial_ticket).length > 0) {
          setPartialTicket(data.partial_ticket);
        }
      } catch {
        // Silently ignore polling errors — ticket will still appear when message responds
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [sessionId, sessionStatus]);

  // When session completes, clear partial ticket (full ticket takes over)
  useEffect(() => {
    if (sessionStatus === "complete") {
      setPartialTicket(null);
    }
  }, [sessionStatus]);

  const displayTicket = ticket || partialTicket;

  if (!displayTicket) {
    return (
      <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 h-full flex flex-col items-center justify-center">
        <div className="text-slate-600 text-sm text-center">
          <div className="text-3xl mb-3">🎫</div>
          <div>Ticket will appear here</div>
          <div className="text-xs mt-1">Populated as agents complete</div>
        </div>
      </div>
    );
  }

  const issue = (displayTicket.issue_summary as Record<string, unknown>)
    || (displayTicket.issue as Record<string, unknown>) || {};
  const diagnosis = (displayTicket.diagnosis as Record<string, unknown>) || {};
  const resolution = (displayTicket.resolution as Record<string, unknown>) || {};
  const escalation = (displayTicket.escalation as Record<string, unknown>) || {};
  const client = (displayTicket.client as Record<string, unknown>) || {};
  const steps = (resolution.steps as Array<Record<string, unknown>>) || [];
  const isPartial = !ticket && Boolean(partialTicket);

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 flex flex-col h-full">
      {/* Ticket header */}
      <div className="px-4 py-3 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <span className="font-mono text-blue-400 text-sm font-bold">
            {(displayTicket.ticket_id as string) || (isPartial ? "Ticket in progress…" : "")}
          </span>
          <div className="flex items-center gap-2">
            {isPartial && (
              <span className="text-xs text-slate-500 italic">partial</span>
            )}
            <span className={`text-xs font-bold uppercase ${PRIORITY_BADGE[displayTicket.priority as string] || "text-slate-400"}`}>
              {displayTicket.priority as string}
            </span>
            <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_BADGE[displayTicket.status as string] || "text-slate-400"}`}>
              {(displayTicket.status as string)?.replace("_", " ")}
            </span>
          </div>
        </div>
        <div className="text-xs text-slate-500 mt-1">
          {client.company as string} · {(client.tier as string)?.toUpperCase()} · SLA: {client.sla_hours as number}h
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Issue */}
        <Section title="Issue">
          <Field label="Product" value={issue.product as string} />
          <Field label="Error" value={issue.error_message as string} />
          <Field label="Environment" value={issue.environment as string} />
          <Field label="Started" value={issue.started_at as string} />
          {Boolean(issue.known_incident) && <Badge label="Known Incident" color="red" />}
          {Boolean(issue.recurring) && <Badge label="Recurring" color="orange" />}
        </Section>

        {/* Diagnosis */}
        {Boolean(diagnosis.primary_cause) && (
          <Section title="Diagnosis">
            <Field label="Root Cause" value={diagnosis.primary_cause as string} />
            <ConfidenceBar label="Confidence" value={diagnosis.confidence as number} />
          </Section>
        )}

        {/* Resolution */}
        {steps.length > 0 && (
          <Section title="Resolution Steps">
            <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
              {steps.map((step, i) => (
                <div key={i} className="text-xs bg-slate-800 rounded-lg p-2">
                  <div className="font-medium text-slate-200">
                    {i + 1}. {step.action as string}
                  </div>
                  {Boolean(step.production_warning) && (
                    <div className="text-yellow-500 mt-1">⚠ {step.production_warning as string}</div>
                  )}
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* Escalation */}
        {Boolean(escalation.decision) && (
          <Section title="Escalation">
            <div className={`text-sm font-medium ${escalation.decision === "escalated" ? "text-red-400" : "text-green-400"}`}>
              {(escalation.decision as string)?.replace("_", " ").toUpperCase()}
            </div>
            <div className="text-xs text-slate-400 mt-1">{escalation.reason as string}</div>
          </Section>
        )}

        {/* Nexus summary */}
        {Boolean(displayTicket.nexus_summary) && (
          <Section title="Nexus Summary">
            <p className="text-xs text-slate-300 leading-relaxed">{displayTicket.nexus_summary as string}</p>
          </Section>
        )}

        {/* Confidence breakdown */}
        {confidenceBreakdown && (
          <Section title="Confidence">
            {Object.entries(confidenceBreakdown).map(([key, val]) => (
              <ConfidenceBar key={key} label={key} value={val} />
            ))}
          </Section>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">{title}</div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | undefined }) {
  if (!value) return null;
  return (
    <div className="flex gap-2 text-xs">
      <span className="text-slate-500 min-w-[70px]">{label}</span>
      <span className="text-slate-200">{value}</span>
    </div>
  );
}

function Badge({ label, color }: { label: string; color: "red" | "orange" }) {
  const colors = { red: "bg-red-500/20 text-red-400", orange: "bg-orange-500/20 text-orange-400" };
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${colors[color]}`}>{label}</span>
  );
}

function ConfidenceBar({ label, value }: { label: string; value: number | undefined }) {
  const pct = ((value || 0) * 100).toFixed(0);
  const color = (value || 0) >= 0.85 ? "bg-green-500" : (value || 0) >= 0.70 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-slate-500 min-w-[70px] capitalize">{label}</span>
      <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-slate-400 font-mono w-8 text-right">{pct}%</span>
    </div>
  );
}
