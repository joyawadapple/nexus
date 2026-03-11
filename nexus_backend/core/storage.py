"""
StorageManager — writes completed sessions and tickets to disk as JSON.

On session completion or escalation, the full SessionState is serialized
to conversations/{session_id}.json. The final ticket (if present) is also
written to tickets/{ticket_id}.json.

Both directories are created automatically under base_dir (defaults to
the current working directory, i.e. the project root).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from models.conversation import SessionState

log = structlog.get_logger("storage")


class StorageManager:
    def __init__(self, base_dir: str = ".") -> None:
        self.conversations_dir = Path(base_dir) / "conversations"
        self.tickets_dir = Path(base_dir) / "tickets"
        self.conversations_dir.mkdir(exist_ok=True)
        self.tickets_dir.mkdir(exist_ok=True)
        log.info("storage.ready", conversations=str(self.conversations_dir), tickets=str(self.tickets_dir))

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_session(self, session: "SessionState") -> None:
        """Serialize and write the full session to conversations/{session_id}.json."""
        try:
            data = _serialize_session(session)
            path = self.conversations_dir / f"{session.session_id}.json"
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            log.info("storage.session_saved", session_id=session.session_id, path=str(path))
        except Exception as exc:
            log.error("storage.session_save_failed", session_id=session.session_id, error=str(exc))

    def save_ticket(self, ticket_id: str, ticket: dict) -> None:
        """Write ticket dict to tickets/{ticket_id}.json."""
        try:
            path = self.tickets_dir / f"{ticket_id}.json"
            path.write_text(json.dumps(ticket, indent=2, default=str), encoding="utf-8")
            log.info("storage.ticket_saved", ticket_id=ticket_id, path=str(path))
        except Exception as exc:
            log.error("storage.ticket_save_failed", ticket_id=ticket_id, error=str(exc))


# ── Serialization ──────────────────────────────────────────────────────────────

def _serialize_session(session: "SessionState") -> dict:
    """Build a plain dict from SessionState suitable for JSON serialization."""
    return {
        "session_id": session.session_id,
        "client_id": session.client_id,
        "created_at": str(session.created_at),
        "status": session.status,
        "detected_language": session.detected_language,
        "current_sentiment": session.current_sentiment,
        # Full conversation transcript
        "conversation": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": str(msg.timestamp),
            }
            for msg in session.messages
        ],
        # Sentiment history
        "sentiment_history": [
            {
                "turn": entry.turn,
                "label": entry.label,
                "compound": entry.compound,
                "text_snippet": entry.text_snippet,
            }
            for entry in session.sentiment_history
        ],
        # Fields collected during triage
        "extracted_data": session.partial_ticket,
        # Agent confidence scores
        "confidence_breakdown": {
            "triage": session.triage_confidence,
            "diagnostic": session.diagnostic_confidence,
            "resolution": session.resolution_confidence,
            "escalation": session.escalation_confidence,
        },
        # Agent findings
        "agent_findings": {
            "triage": session.triage_finding,
            "diagnostic": session.diagnostic_finding,
            "resolution": session.resolution_finding,
            "escalation": session.escalation_finding,
        },
        # Situation flags
        "flags": {
            "severity": session.severity,
            "known_incident": session.known_incident,
            "recurring": session.recurring,
        },
        # Final ticket (present when status is "complete" or "escalated")
        "ticket": session.ticket,
    }
