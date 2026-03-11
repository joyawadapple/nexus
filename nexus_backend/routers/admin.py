"""
Admin router — support team dashboard endpoints.

All endpoints require an ``X-Admin-Key`` header matching ``settings.ADMIN_API_KEY``
when that setting is non-empty. When ``ADMIN_API_KEY`` is not configured (the
default), the guard is disabled so dev workflows are unaffected.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException

import structlog

from agents.diagnostic_agent import DiagnosticAgent
from agents.escalation_agent import EscalationAgent
from agents.resolution_agent import ResolutionAgent
from agents.triage_agent import TriageAgent
from core.config import settings

log = structlog.get_logger("router.admin")

router = APIRouter(prefix="/admin", tags=["admin"])

_session_manager = None
_monitor = None
_evaluator = None


def set_services(session_manager, monitor, evaluator=None) -> None:
    global _session_manager, _monitor, _evaluator
    _session_manager = session_manager
    _monitor = monitor
    _evaluator = evaluator


async def _require_admin_key(x_admin_key: str | None = Header(None)) -> None:
    """Require X-Admin-Key header when ADMIN_API_KEY is configured."""
    if not settings.ADMIN_API_KEY:
        return  # guard disabled in dev when key is not set
    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/sessions", dependencies=[Depends(_require_admin_key)])
async def get_active_sessions():
    """All active support sessions with live confidence scores."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    sessions = _session_manager.list_sessions()
    result = []
    for s in sessions:
        triage = s.triage_finding or {}
        client = triage.get("client", {}) if isinstance(triage, dict) else {}
        result.append({
            "session_id": s.session_id,
            "client_id": s.client_id,
            "company": client.get("company", "Unknown"),
            "tier": client.get("tier", "unknown"),
            "status": s.status,
            "sentiment": s.current_sentiment,
            "created_at": s.created_at.isoformat(),
            "message_count": len(s.messages),
            "sla_deadline": client.get("sla_deadline"),
            "agent_confidence": {
                "triage": triage.get("confidence", 0.0) if isinstance(triage, dict) else 0.0,
                "diagnostic": (s.diagnostic_finding or {}).get("confidence", 0.0),
                "resolution": (s.resolution_finding or {}).get("confidence", 0.0),
            },
        })
    return {"sessions": result, "total": len(result)}


@router.get("/tickets", dependencies=[Depends(_require_admin_key)])
async def get_all_tickets():
    """All generated support tickets."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    sessions = _session_manager.list_sessions()
    tickets = [s.ticket for s in sessions if s.ticket]
    return {"tickets": tickets, "total": len(tickets)}


@router.get("/escalations", dependencies=[Depends(_require_admin_key)])
async def get_escalations():
    """Tickets flagged for human review."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    sessions = _session_manager.list_sessions()
    escalated = [
        s.ticket for s in sessions
        if s.ticket and s.ticket.get("status") in ("escalated", "pending_review", "pending")
    ]
    return {"escalations": escalated, "total": len(escalated)}


@router.get("/agent-status/{session_id}", dependencies=[Depends(_require_admin_key)])
async def get_agent_status(session_id: str):
    """Live agent confidence scores for a specific session."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "status": session.status,
        "agents": [
            {
                "agent": "triage_agent",
                "status": "complete" if session.triage_finding else "pending",
                "confidence": (session.triage_finding or {}).get("confidence", 0.0),
                "threshold": TriageAgent.confidence_threshold,
            },
            {
                "agent": "diagnostic_agent",
                "status": "complete" if session.diagnostic_finding else "pending",
                "confidence": (session.diagnostic_finding or {}).get("confidence", 0.0),
                "threshold": DiagnosticAgent.confidence_threshold,
            },
            {
                "agent": "resolution_agent",
                "status": "complete" if session.resolution_finding else "pending",
                "confidence": (session.resolution_finding or {}).get("confidence", 0.0),
                "threshold": ResolutionAgent.confidence_threshold,
            },
            {
                "agent": "escalation_agent",
                "status": "complete" if session.ticket else "pending",
                "confidence": (session.ticket or {}).get("confidence_breakdown", {}).get("escalation", 0.0),
                "threshold": EscalationAgent.confidence_threshold,
            },
        ],
        "sentiment": session.current_sentiment,
    }


@router.get("/reasoning/{session_id}", dependencies=[Depends(_require_admin_key)])
async def get_reasoning_log(session_id: str):
    """Full agent reasoning log for a session."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    reasoning_logs = {}
    for agent_id, finding_dict in [
        ("triage_agent", session.triage_finding),
        ("diagnostic_agent", session.diagnostic_finding),
        ("resolution_agent", session.resolution_finding),
        ("escalation_agent", session.escalation_finding),
    ]:
        if finding_dict:
            reasoning_logs[agent_id] = finding_dict.get("agent_reasoning_log", [])

    return {
        "session_id": session_id,
        "reasoning_logs": reasoning_logs,
    }


@router.get("/metrics", dependencies=[Depends(_require_admin_key)])
async def get_metrics():
    """System-wide metrics: resolution rate, avg confidence, SLA compliance."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    sessions = _session_manager.list_sessions()
    tickets = [s.ticket for s in sessions if s.ticket]

    total = len(tickets)
    if total == 0:
        return {"message": "No tickets generated yet", "metrics": {}}

    self_resolved = sum(1 for t in tickets if t.get("status") == "self_resolve")
    escalated = sum(1 for t in tickets if t.get("status") == "escalated")

    avg_overall_conf = sum(
        t.get("confidence_breakdown", {}).get("overall", 0.0) for t in tickets
    ) / total if total > 0 else 0.0

    # SLA compliance: tickets resolved before deadline
    sla_met = 0
    for t in tickets:
        deadline_str = t.get("sla_deadline", "")
        if deadline_str:
            try:
                deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                created_str = t.get("created_at", "")
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00")) if created_str else datetime.now(tz=timezone.utc)
                if created < deadline:
                    sla_met += 1
            except Exception:
                pass

    # Agent monitor metrics
    monitor_data = {}
    if _monitor:
        monitor_data = _monitor.get_aggregate_metrics()

    return {
        "metrics": {
            "total_tickets": total,
            "self_resolve_rate": f"{(self_resolved / total * 100):.1f}%" if total > 0 else "0%",
            "escalation_rate": f"{(escalated / total * 100):.1f}%" if total > 0 else "0%",
            "avg_overall_confidence": f"{avg_overall_conf:.0%}",
            "sla_compliance_rate": f"{(sla_met / total * 100):.1f}%" if total > 0 else "0%",
            **monitor_data,
        }
    }


@router.get("/evaluate/{session_id}", dependencies=[Depends(_require_admin_key)])
async def evaluate_session(session_id: str):
    """Score a completed session across 7 quality dimensions."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")
    if not _evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")

    session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    report = _evaluator.evaluate_session(session)
    return report.to_dict()


@router.get("/regression", dependencies=[Depends(_require_admin_key)])
async def run_regression():
    """Run golden case regression tests against tests/golden_cases.json."""
    if not _evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")

    report = _evaluator.run_regression()
    return report.to_dict()
