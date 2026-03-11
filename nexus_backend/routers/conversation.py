"""
Conversation router — manages support sessions and message handling.
"""
from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

import structlog

log = structlog.get_logger("router.conversation")

router = APIRouter(prefix="/conversation", tags=["conversation"])

# These are injected at startup from main.py
_session_manager = None
_memory = None
_orchestrator = None
_graph_runner = None  # NexusGraphRunner — set only when LANGSMITH_TRACING=true
_storage = None


def set_services(session_manager, memory, orchestrator, graph_runner=None, storage=None) -> None:
    global _session_manager, _memory, _orchestrator, _graph_runner, _storage
    _session_manager = session_manager
    _memory = memory
    _orchestrator = orchestrator
    _graph_runner = graph_runner
    _storage = storage


# ── Request/Response models ────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    api_key: str | None = Field(default=None, max_length=128)
    client_id: str | None = Field(default=None, max_length=64)


class StartSessionResponse(BaseModel):
    session_id: str
    client_id: str
    company: str
    tier: str
    sla_hours: int
    message: str


class MessageRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)


class MessageResponse(BaseModel):
    session_id: str
    response: str
    status: str
    awaiting: str | None = None  # e.g. "resolution_confirmation" — frontend prompt hint
    agent_statuses: list[dict] = []
    ticket: dict | None = None
    confidence_breakdown: dict | None = None
    reasoning_logs: dict = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """Start a new support session. Authenticates client via API key."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    if request.client_id:
        client_id = request.client_id
    elif request.api_key:
        client_id = _session_manager.authenticate_client(request.api_key)
    else:
        client_id = None
    if not client_id:
        raise HTTPException(status_code=401, detail="Invalid API key")

    session = _session_manager.create_session(client_id)

    from db import database
    client_data = database.get_client(client_id) or {}

    # Populate partial_ticket immediately — client is known the moment the session starts
    session.partial_ticket["client"] = {
        "company": client_data.get("company", "Unknown"),
        "tier": client_data.get("tier", "standard"),
        "sla_hours": client_data.get("sla_hours", 24),
    }

    # Add welcome message to history
    tier = client_data.get("tier", "standard")
    sla_hours = client_data.get("sla_hours", 24)

    welcome = (
        f"Hello! I'm Nexus, NexaCloud's AI support system. "
        f"I'm here to help you resolve any issues with your NexaCloud APIs. "
        f"What's the issue you're experiencing today?"
    )
    session.add_message("assistant", welcome)

    log.info("session.started", session_id=session.session_id, client_id=client_id)

    return StartSessionResponse(
        session_id=session.session_id,
        client_id=client_id,
        company=client_data.get("company", ""),
        tier=tier,
        sla_hours=sla_hours,
        message=welcome,
    )


@router.post("/message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """Send a message and get Nexus response. Runs the full agent pipeline."""
    if not _session_manager or not _orchestrator:
        raise HTTPException(status_code=503, detail="Services not initialized")

    session = _session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Add user message
    session.add_message("user", request.message)

    # Run orchestrator — via LangGraph when tracing is enabled, directly otherwise
    _use_graph = bool(_graph_runner) and os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if _use_graph:
        result = await _graph_runner.invoke_graph(
            session_id=session.session_id,
            conversation_history=session.conversation_history(),
            session_state=session,
            client_id=session.client_id,
        )
    else:
        result = await _orchestrator.run(
            session_id=session.session_id,
            conversation_history=session.conversation_history(),
            session_state=session,
            client_id=session.client_id,
        )

    response_message = result.get("message", "I'm processing your request.")

    # Add assistant response to history
    session.add_message("assistant", response_message)

    # Persist escalated sessions (complete sessions are saved inside memory.set_ticket())
    if _storage and result.get("status") == "escalated" and session.status == "escalated":
        ticket = result.get("ticket")
        if ticket:
            session.ticket = ticket
        _storage.save_session(session)
        if ticket:
            ticket_id = ticket.get("ticket_id", session.session_id)
            _storage.save_ticket(ticket_id, ticket)

    return MessageResponse(
        session_id=session.session_id,
        response=response_message,
        status=result.get("status", "in_progress"),
        awaiting=result.get("awaiting"),
        agent_statuses=result.get("agent_statuses", []),
        ticket=result.get("ticket"),
        confidence_breakdown=result.get("confidence_breakdown"),
        reasoning_logs=result.get("reasoning_logs", {}),
    )


@router.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """Get current session status and partial ticket for live preview polling."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build agent_statuses from session state flags (not finding dicts) for accuracy.
    # Mirrors the same logic as _build_agent_statuses_from_state in the orchestrator.
    def _status(complete_flag: bool, has_finding: bool) -> str:
        if complete_flag:
            return "complete"
        if has_finding:
            return "running"
        return "pending"

    agent_statuses = [
        {
            "agent": "triage_agent",
            "status": _status(session.triage_complete, bool(session.triage_finding)),
            "confidence": session.triage_confidence,
        },
        {
            "agent": "diagnostic_agent",
            "status": _status(session.diagnostic_complete, bool(session.diagnostic_finding)),
            "confidence": session.diagnostic_confidence,
        },
        {
            "agent": "resolution_agent",
            "status": _status(session.resolution_complete, bool(session.resolution_finding)),
            "confidence": session.resolution_confidence,
        },
        {
            "agent": "escalation_agent",
            "status": "complete" if session.escalation_complete else (
                "running" if session.escalation_finding else "pending"
            ),
            "confidence": session.escalation_confidence,
        },
    ]

    return {
        "session_id": session_id,
        "agent_statuses": agent_statuses,
        "partial_ticket": session.partial_ticket,
        "asked_fields": session.asked_fields,
        "overall_status": session.status,
    }


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get full conversation history for a session."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "messages": session.conversation_history(),
        "status": session.status,
        "message_count": len(session.messages),
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a support session."""
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    deleted = _session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"deleted": True, "session_id": session_id}
