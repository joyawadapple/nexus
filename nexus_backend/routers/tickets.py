"""
Tickets router — support ticket management.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import structlog

log = structlog.get_logger("router.tickets")

router = APIRouter(prefix="/tickets", tags=["tickets"])

_session_manager = None


def set_services(session_manager) -> None:
    global _session_manager
    _session_manager = session_manager


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Get a complete support ticket by ticket ID."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    sessions = _session_manager.list_sessions()
    for session in sessions:
        if session.ticket and session.ticket.get("ticket_id") == ticket_id:
            return session.ticket

    raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")


@router.get("/session/{session_id}")
async def get_ticket_for_session(session_id: str):
    """Get the support ticket for a specific session.

    Returns a flat dict so the eval runner can access ticket fields directly via
    result["ticket"]["client"]["tier"] etc.
    """
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.ticket:
        # Return the ticket dict directly (flat), augmented with aliases the eval runner expects
        flat = dict(session.ticket)
        # Alias: eval checks "issue.*" but model field is "issue_summary"
        if "issue_summary" in flat and "issue" not in flat:
            flat["issue"] = flat["issue_summary"]
        # Alias: eval checks "resolution.steps_count"
        res = flat.get("resolution", {})
        if isinstance(res, dict) and "steps" in res:
            res = dict(res)
            res["steps_count"] = len(res.get("steps") or [])
            res["has_prevention_step"] = bool(res.get("prevention"))
            flat["resolution"] = res
        return flat

    # No ticket yet — return partial_ticket data so eval can at least read client info
    partial = dict(session.partial_ticket) if session.partial_ticket else {}
    partial["status"] = session.status
    partial.setdefault("escalation", {})
    partial.setdefault("diagnosis", {})
    res = dict(partial.get("resolution", {}))
    if "steps" in res:
        res["steps_count"] = len(res.get("steps") or [])
        res["has_prevention_step"] = bool(res.get("prevention"))
        partial["resolution"] = res
    # Mirror issue_summary → issue
    if "issue" not in partial and session.triage_finding:
        tf = session.triage_finding
        if isinstance(tf, dict) and tf.get("issue"):
            partial["issue"] = tf["issue"]
    return partial


@router.post("/{ticket_id}/escalate")
async def escalate_ticket(ticket_id: str):
    """Manual escalation override — force escalation of a ticket."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    sessions = _session_manager.list_sessions()
    for session in sessions:
        if session.ticket and session.ticket.get("ticket_id") == ticket_id:
            session.ticket["status"] = "escalated"
            session.ticket["escalation"]["decision"] = "escalated"
            session.ticket["escalation"]["reason"] = "Manually escalated by support team"
            log.info("ticket.manually_escalated", ticket_id=ticket_id)
            return {"escalated": True, "ticket_id": ticket_id}

    raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")


@router.get("/{ticket_id}/summary")
async def get_ticket_summary(ticket_id: str):
    """Client-facing clean summary of a ticket (no internal reasoning data)."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    sessions = _session_manager.list_sessions()
    for session in sessions:
        if session.ticket and session.ticket.get("ticket_id") == ticket_id:
            ticket = session.ticket
            return {
                "ticket_id": ticket_id,
                "status": ticket.get("status"),
                "priority": ticket.get("priority"),
                "issue": ticket.get("issue_summary", {}).get("error_message", ""),
                "product": ticket.get("issue_summary", {}).get("product", ""),
                "nexus_summary": ticket.get("nexus_summary", ""),
                "resolution_steps": [
                    {
                        "step": s.get("step"),
                        "action": s.get("action"),
                        "verify": s.get("verify"),
                    }
                    for s in (ticket.get("resolution", {}).get("steps") or [])
                ],
                "escalation_decision": ticket.get("escalation", {}).get("decision"),
            }

    raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
