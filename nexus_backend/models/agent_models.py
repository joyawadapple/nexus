"""
Agent-level Pydantic models shared across all Nexus sub-agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Reasoning log (mirrors Quorum's ReasoningEntry) ───────────────────────────

@dataclass
class ReasoningEntry:
    step: str          # "LOAD" | "ANALYZE" | "REASON" | "DECIDE" | "GENERATE" | "RETURN"
    decision: str
    rationale: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "decision": self.decision,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
        }


# ── Question model ─────────────────────────────────────────────────────────────

class QuestionForClient(BaseModel):
    field: str                          # e.g. "product", "error_message"
    question: str                       # Natural language question
    blocking: bool = False              # If True, cannot proceed without answer
    priority: Literal["high", "medium", "low"] = "medium"
    attempt_number: int = 1
    if_missing: str = "mark_unknown"    # "cannot_route" | "flag_for_human" | "assume_*" | "mark_unknown"


# ── Validation (HallucinationGuard output) ────────────────────────────────────

class HallucinationFlag(BaseModel):
    field: str
    agent_said: Any
    db_has: Any | None = None
    flag_type: Literal["invented_value", "confidence_inflation", "out_of_scope"] = "invented_value"
    severity: Literal["high", "medium", "low"] = "medium"


class ValidationResult(BaseModel):
    valid: bool
    flags: list[HallucinationFlag] = Field(default_factory=list)
    clamped_confidence: float | None = None    # Set if confidence was inflated


# ── Base finding (each agent produces one of these) ───────────────────────────

class AgentFinding(BaseModel):
    agent_id: str
    session_id: str
    confidence: float = 0.0
    confidence_threshold: float = 0.90
    questions_for_client: list[QuestionForClient] = Field(default_factory=list)
    agent_reasoning_log: list[dict] = Field(default_factory=list)
    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)
    completed: bool = False
