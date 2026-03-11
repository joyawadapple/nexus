"""
Conversation memory — in-memory session store keyed by session_id.
Manages message history, confirmed fields, sentiment, and agent findings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from models.conversation import SessionState

if TYPE_CHECKING:
    from core.storage import StorageManager

log = structlog.get_logger("conversation_memory")

# Approximate tokens per character (rough estimate for context window management)
_CHARS_PER_TOKEN = 4
_MAX_TOKENS = 6000


class ConversationMemory:
    """
    In-memory store for all active support sessions.
    Each session is a SessionState Pydantic model.
    """

    def __init__(self, storage: "StorageManager | None" = None) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._storage = storage

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def create_session(self, session_id: str, client_id: str) -> SessionState:
        session = SessionState(session_id=session_id, client_id=client_id)
        self._sessions[session_id] = session
        log.info("memory.session_created", session_id=session_id, client_id=client_id)
        return session

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            log.info("memory.session_deleted", session_id=session_id)
            return True
        return False

    def list_sessions(self) -> list[SessionState]:
        return list(self._sessions.values())

    # ── Message management ────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self._get_or_raise(session_id)
        session.add_message(role, content)
        self.compress_if_needed(session_id)

    def get_history(self, session_id: str) -> list[dict]:
        session = self._get_or_raise(session_id)
        return session.conversation_history()

    # ── Context compression ───────────────────────────────────────────────────

    def compress_if_needed(self, session_id: str) -> None:
        """
        If message history exceeds MAX_TOKENS, compress older turns.
        Always preserves: confirmed field values + last 4 message turns.
        Never loses any confirmed field value.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        total_chars = sum(len(m.content) for m in session.messages)
        estimated_tokens = total_chars // _CHARS_PER_TOKEN

        if estimated_tokens <= _MAX_TOKENS:
            return

        if len(session.messages) <= 4:
            return  # Nothing to compress

        log.info(
            "memory.compressing",
            session_id=session_id,
            estimated_tokens=estimated_tokens,
            message_count=len(session.messages),
        )

        # Keep last 4 messages verbatim
        messages_to_compress = session.messages[:-4]
        recent_messages = session.messages[-4:]

        # Build summary from older messages
        conversation_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages_to_compress
        )

        # Build confirmed fields summary (critical — never lose these)
        confirmed_summary = ""
        if session.confirmed_fields:
            fields_str = ", ".join(f"{k}={v}" for k, v in session.confirmed_fields.items())
            confirmed_summary = f"[Confirmed fields: {fields_str}] "

        summary_content = (
            f"{confirmed_summary}Summary of earlier conversation: "
            f"Client reported an issue. "
            f"Topics covered: {conversation_text[:300]}..."
        )

        from models.conversation import Message
        summary_message = Message(role="system", content=summary_content)

        session.messages = [summary_message] + list(recent_messages)
        log.info(
            "memory.compressed",
            session_id=session_id,
            new_message_count=len(session.messages),
        )

    # ── State helpers ─────────────────────────────────────────────────────────

    def update_findings(self, session_id: str, agent_id: str, findings: dict) -> None:
        session = self._get_or_raise(session_id)
        if agent_id == "triage_agent":
            session.triage_finding = findings
        elif agent_id == "diagnostic_agent":
            session.diagnostic_finding = findings
        elif agent_id == "resolution_agent":
            session.resolution_finding = findings
        elif agent_id == "escalation_agent":
            session.escalation_finding = findings

    def set_ticket(self, session_id: str, ticket: dict) -> None:
        session = self._get_or_raise(session_id)
        session.ticket = ticket
        session.status = "complete"
        if self._storage:
            ticket_id = ticket.get("ticket_id", session_id)
            self._storage.save_session(session)
            self._storage.save_ticket(ticket_id, ticket)

    def _get_or_raise(self, session_id: str) -> SessionState:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Session not found: {session_id}")
        return session
