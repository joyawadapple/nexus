"""
Session manager — creates and manages support sessions.
Maps client API keys to client_ids; generates ticket sequence numbers.
"""
from __future__ import annotations

import uuid
from datetime import datetime

import structlog

from core.conversation_memory import ConversationMemory
from models.conversation import SessionState

log = structlog.get_logger("session_manager")

# Module-level ticket counter (reset on restart; acceptable for demo)
_ticket_counter: int = 4471  # Start from a realistic number

# API key → client_id mapping (simulated authentication)
# In production this would be a DB lookup
_API_KEY_MAP: dict[str, str] = {
    "nxa_acme_test_key_001": "client_001",
    "nxa_gretail_test_key_002": "client_002",
    "nxa_devstartup_test_key_003": "client_003",
    # Allow test key for any client (demo/testing)
    "nxa_test_key": "client_001",
}


class SessionManager:
    def __init__(self, memory: ConversationMemory) -> None:
        self._memory = memory

    def authenticate_client(self, api_key: str) -> str | None:
        """Map an API key to a client_id. Returns None if not found."""
        return _API_KEY_MAP.get(api_key)

    def create_session(self, client_id: str) -> SessionState:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session = self._memory.create_session(session_id, client_id)
        log.info("session_manager.created", session_id=session_id, client_id=client_id)
        return session

    def get_session(self, session_id: str) -> SessionState | None:
        return self._memory.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._memory.delete_session(session_id)

    def list_sessions(self) -> list[SessionState]:
        return self._memory.list_sessions()

    def generate_ticket_id(self) -> str:
        global _ticket_counter
        _ticket_counter += 1
        year = datetime.utcnow().year
        return f"NX-{year}-{_ticket_counter:04d}"

    def update_partial_ticket(self, session_id: str, agent_id: str, finding) -> None:
        """Update the session's partial_ticket incrementally as each agent completes."""
        session = self.get_session(session_id)
        if not session:
            return

        if agent_id == "triage_agent":
            if hasattr(finding, "client") and finding.client and hasattr(finding, "issue"):
                session.partial_ticket.update({
                    "client": {
                        "company": finding.client.company,
                        "tier": finding.client.tier,
                        "sla_hours": finding.client.sla_hours,
                    },
                    "issue": finding.issue.model_dump() if finding.issue else {},
                    "priority": finding.severity,
                    "sla_deadline": finding.client.sla_deadline,
                })

        elif agent_id == "diagnostic_agent":
            if hasattr(finding, "primary_hypothesis") and finding.primary_hypothesis:
                session.partial_ticket["diagnosis"] = {
                    "primary_cause": finding.primary_hypothesis.cause,
                    "confidence": finding.confidence,
                }

        elif agent_id == "resolution_agent":
            if hasattr(finding, "steps") and finding.steps:
                session.partial_ticket["resolution"] = {
                    "steps": [s.model_dump() for s in finding.steps],
                    "confidence": finding.confidence,
                }
