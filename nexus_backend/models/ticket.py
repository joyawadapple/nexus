"""
Support ticket Pydantic models — the master output of the Nexus system.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


TicketStatus = Literal["self_resolve", "escalated", "pending", "pending_review", "closed"]
Priority = Literal["critical", "high", "medium", "low"]


class ClientSummary(BaseModel):
    company: str
    tier: Literal["platinum", "gold", "standard"]
    sla_hours: int
    sla_deadline: str
    csm: str
    csm_notified: bool = False
    recent_tickets: int = 0
    vip_flag: bool = False


class IssueSummary(BaseModel):
    product: str = "Unknown"
    error_message: str
    environment: Literal["production", "staging", "development"] = "production"
    started_at: str = "unknown"
    impact_scope: str = "unknown"
    known_incident: bool = False
    recurring: bool = False
    inferred_fields: dict[str, bool] = Field(default_factory=dict)
    unknown_product: bool = False  # True when product could not be confirmed from DB


class DiagnosisSummary(BaseModel):
    primary_cause: str
    confidence: float
    supporting_evidence: list[str] = Field(default_factory=list)
    alternative_causes: list[dict] = Field(default_factory=list)
    novel_issue: bool = False  # True when no KB or bug-db match found


class ResolutionStep(BaseModel):
    step: int
    action: str
    command: str | None = None
    why: str
    verify: str
    risk: Literal["none", "low", "medium", "high"] = "low"
    production_warning: str | None = None


class ResolutionPlan(BaseModel):
    estimated_resolution_time: str
    steps: list[ResolutionStep] = Field(default_factory=list)
    prevention: str | None = None
    rag_source: str | None = None
    confidence: float = 0.0


class EscalationDecision(BaseModel):
    decision: Literal["self_resolve", "escalated", "pending"]
    reason: str
    fallback: str | None = None
    escalation_path: str | None = None


class SentimentProfile(BaseModel):
    detected: Literal["frustrated", "urgent", "calm", "positive", "confused", "anxious"] = "calm"
    tone_adjustment_applied: bool = False
    note: str | None = None


class ConfidenceBreakdown(BaseModel):
    triage: float = 0.0
    diagnostic: float = 0.0
    resolution: float = 0.0
    escalation: float = 0.0
    overall: float = 0.0


class SupportTicket(BaseModel):
    ticket_id: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    sla_deadline: str
    status: TicketStatus
    priority: Priority

    client: ClientSummary
    issue_summary: IssueSummary
    diagnosis: DiagnosisSummary
    resolution: ResolutionPlan
    escalation: EscalationDecision
    sentiment_analysis: SentimentProfile

    nexus_summary: str
    data_provenance: dict = Field(default_factory=dict)
    agent_reasoning_logs: dict = Field(default_factory=dict)
    confidence_breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
