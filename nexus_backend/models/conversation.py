"""
Conversation and session state models.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class SentimentEntry(BaseModel):
    turn: int
    label: Literal["frustrated", "urgent", "calm", "positive", "confused", "anxious"]
    compound: float
    text_snippet: str


class ExtractedField(BaseModel):
    value: Any
    inferred: bool = False   # True = guessed/estimated, False = explicitly stated by client
    confidence: float = 1.0  # 0.0–1.0


class SessionState(BaseModel):
    session_id: str
    client_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: list[Message] = Field(default_factory=list)

    # Collected field tracking (deduplication of questions)
    asked_fields: list[str] = Field(default_factory=list)
    confirmed_fields: dict[str, Any] = Field(default_factory=dict)  # field → value

    # Sentiment history
    sentiment_history: list[SentimentEntry] = Field(default_factory=list)
    current_sentiment: Literal["frustrated", "urgent", "calm", "positive", "confused", "anxious"] = "calm"
    detected_language: str = "en"

    # Agent findings (populated as agents complete)
    triage_finding: dict | None = None
    diagnostic_finding: dict | None = None
    resolution_finding: dict | None = None
    escalation_finding: dict | None = None

    # Triage phase state
    triage_complete: bool = False
    triage_confidence: float = 0.0
    triage_run_count: int = 0

    # Diagnostic phase state
    diagnostic_complete: bool = False
    diagnostic_confidence: float = 0.0
    diagnostic_run_count: int = 0
    diagnostic_previous_confidence: float = 0.0

    # Resolution phase state
    resolution_complete: bool = False
    resolution_confidence: float = 0.0
    resolution_run_count: int = 0

    # Confidence history — used by ProgressTracker for velocity-based stuck detection
    triage_confidence_history: list[float] = Field(default_factory=list)
    diagnostic_confidence_history: list[float] = Field(default_factory=list)
    resolution_confidence_history: list[float] = Field(default_factory=list)

    # Escalation phase state
    escalation_complete: bool = False
    escalation_confidence: float = 0.0
    escalation_trigger: str | None = None  # reason passed to _handle_force_escalate

    # Situation flags set by triage
    known_incident: bool = False
    severity: str = "medium"
    recurring: bool = False

    # Client data snapshot (loaded at session start from DB)
    client: dict = Field(default_factory=dict)

    # Partial ticket — populated incrementally as agents complete
    partial_ticket: dict = Field(default_factory=dict)

    # Final ticket (populated when complete)
    ticket: dict | None = None
    final_ticket: dict | None = None
    status: Literal["active", "collecting", "in_progress", "complete", "escalated", "closed", "resolved"] = "active"

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))  # type: ignore[arg-type]

    def add_sentiment(self, label: str, compound: float, text: str) -> None:
        turn = len(self.messages)
        self.sentiment_history.append(SentimentEntry(
            turn=turn,
            label=label,  # type: ignore[arg-type]
            compound=compound,
            text_snippet=text[:100],
        ))
        self.current_sentiment = label  # type: ignore[assignment]

    def conversation_history(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]

    def mark_field_asked(self, field: str) -> None:
        if field not in self.asked_fields:
            self.asked_fields.append(field)

    def confirm_field(self, field: str, value: Any) -> None:
        self.confirmed_fields[field] = value
        self.mark_field_asked(field)
