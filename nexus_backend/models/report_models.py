"""
Per-agent report models returned by each sub-agent to the orchestrator.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from models.agent_models import AgentFinding, HallucinationFlag, QuestionForClient


class ClientInfo(BaseModel):
    company: str
    tier: Literal["platinum", "gold", "standard"]
    sla_hours: int
    sla_deadline: str
    csm: str
    recent_tickets: int = 0
    vip_flag: bool = False


class IssueInfo(BaseModel):
    product: str | None = None
    error_message: str | None = None
    environment: Literal["production", "staging", "development"] | None = None
    started_at: str | None = None
    impact_scope: str | None = None
    known_incident: bool = False
    recurring: bool = False
    mentioned_products: list[str] = Field(default_factory=list)
    inferred_fields: dict[str, bool] = Field(default_factory=dict)


class TriageReport(AgentFinding):
    agent_id: str = "triage_agent"
    confidence_threshold: float = 0.90

    client: ClientInfo | None = None
    issue: IssueInfo = Field(default_factory=IssueInfo)
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    routing: str = "diagnostic_agent"


class RAGResult(BaseModel):
    source: str
    similarity: float
    excerpt_summary: str


class Hypothesis(BaseModel):
    cause: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    why_less_likely: str | None = None


class DiagnosticReport(AgentFinding):
    agent_id: str = "diagnostic_agent"
    confidence_threshold: float = 0.75

    primary_hypothesis: Hypothesis | None = None
    alternative_hypotheses: list[Hypothesis] = Field(default_factory=list)
    rag_results_used: list[RAGResult] = Field(default_factory=list)
    version_bugs_checked: list[str] = Field(default_factory=list)
    hypothesis_inferred: bool = False


class ResolutionStepOut(BaseModel):
    step: int
    action: str
    command: str | None = None
    why: str
    verify: str
    risk: Literal["none", "low", "medium", "high"] = "low"
    production_warning: str | None = None
    confidence_level: Literal["high", "medium", "low"] = "high"


class ResolutionReport(AgentFinding):
    agent_id: str = "resolution_agent"
    confidence_threshold: float = 0.80

    estimated_resolution_time: str = "unknown"
    steps: list[ResolutionStepOut] = Field(default_factory=list)
    prevention: str | None = None
    rag_source: str | None = None
    has_low_confidence_steps: bool = False


class EscalationReport(AgentFinding):
    agent_id: str = "escalation_agent"
    confidence_threshold: float = 0.95

    decision: Literal["self_resolve", "escalated", "pending"] = "pending"
    reason: str = ""
    fallback: str | None = None
    escalation_path: str | None = None
    csm_notified: bool = False


class MasterReport(BaseModel):
    session_id: str
    triage: TriageReport | None = None
    diagnostic: DiagnosticReport | None = None
    resolution: ResolutionReport | None = None
    escalation: EscalationReport | None = None
    overall_confidence: float = 0.0
    status: Literal["collecting", "in_progress", "complete", "known_incident"] = "collecting"
