"""
BaseAgent — the 6-step loop all Nexus sub-agents inherit.

LOAD → ANALYZE → REASON → DECIDE → GENERATE → RETURN

Key invariants (identical to Quorum):
  - calculate_confidence() is ALWAYS mathematical (confirmed / total).
    It NEVER calls the LLM and NEVER guesses.
  - log_reasoning() produces a ReasoningEntry at every decision point.
  - Each agent runs as part of the orchestrator's sequential-then-parallel flow.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

log = structlog.get_logger("agent_base")

if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    from langsmith import traceable
else:
    def traceable(**_):  # type: ignore[misc]
        return lambda f: f


# ── Data containers for the 6-step pipeline ───────────────────────────────────

@dataclass
class ReasoningEntry:
    step: str        # "LOAD" | "ANALYZE" | "REASON" | "DECIDE" | "GENERATE" | "RETURN"
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


@dataclass
class LoadedData:
    session_id: str
    db_records: dict           # raw records from the relevant JSON DB
    conversation_history: list[dict]
    rag_context: list[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    total_fields: int          # number of items that need confirmation
    fields_in_db: list[str]    # fields we already have from the DB
    fields_missing: list[str]  # fields not in DB; must collect from client
    preliminary_data: dict = field(default_factory=dict)


@dataclass
class ReasoningResult:
    questions_to_ask: list[str]
    fields_confirmed_from_conversation: list[str]
    discrepancies: list[dict]           # DB value vs client-stated value
    field_values_extracted: dict = field(default_factory=dict)


@dataclass
class Decision:
    should_ask_questions: bool
    questions: list[str]         # already capped at ≤ 2 per agent per turn
    ready_to_finalize: bool      # True when confidence >= threshold


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all 4 Nexus domain agents.

    Subclasses must implement the 6 abstract methods below.
    The public entry point is run(), which executes all 6 steps in order.

    Class-level attributes that subclasses MUST define:
        agent_id: str
        confidence_threshold: float
    """

    agent_id: str
    confidence_threshold: float

    def __init__(self) -> None:
        self.reasoning_log: list[ReasoningEntry] = []

    # ── Confidence calculation ────────────────────────────────────────────────

    def calculate_confidence(
        self,
        confirmed_count: int,
        total_count: int,
    ) -> float:
        """
        Pure mathematical confidence calculation.

        confirmed_count / total_count

        Edge cases:
        - total_count == 0  → returns 1.0 (nothing to confirm, no gaps)
        - confirmed_count > total_count  → capped at 1.0

        NEVER calls the LLM. NEVER assigns a confidence score by estimation.
        """
        if total_count == 0:
            return 1.0
        raw = confirmed_count / total_count
        return min(raw, 1.0)

    # ── Reasoning log ─────────────────────────────────────────────────────────

    def log_reasoning(
        self,
        step: str,
        decision: str,
        rationale: str,
    ) -> ReasoningEntry:
        entry = ReasoningEntry(step=step, decision=decision, rationale=rationale)
        self.reasoning_log.append(entry)
        log.debug(
            "agent.reasoning",
            agent=self.agent_id,
            step=step,
            decision=decision[:120],
        )
        return entry

    # ── Abstract 6-step methods ───────────────────────────────────────────────

    @abstractmethod
    async def load(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list[dict],
    ) -> LoadedData:
        """
        STEP 1 — LOAD.
        Pull all relevant DB records for this session from the pre-loaded db_data dict.
        Never reads files directly; db_data is already loaded by the orchestrator.
        """
        ...

    @abstractmethod
    async def analyze(self, loaded_data: LoadedData) -> AnalysisResult:
        """
        STEP 2 — ANALYZE.
        Determine:
        - How many items need confirmation (total_fields)
        - Which are already in the DB (fields_in_db)
        - Which are genuinely missing (fields_missing)
        Pure data analysis — no LLM call, no I/O.
        """
        ...

    @abstractmethod
    async def reason(
        self,
        analysis: AnalysisResult,
        conversation_history: list[dict],
    ) -> ReasoningResult:
        """
        STEP 3 — REASON.
        Scan conversation history for confirmations already given by the client.
        Identify discrepancies between DB data and what the client stated.
        Return questions still needed + confirmed fields + discrepancies found.
        """
        ...

    @abstractmethod
    async def decide(self, reasoning: ReasoningResult) -> Decision:
        """
        STEP 4 — DECIDE.
        Determine whether to ask more questions and whether confidence threshold is met.
        Caps questions at ≤ 2 per agent per turn.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        decision: Decision,
        loaded_data: LoadedData,
        reasoning: ReasoningResult,
    ) -> Any:
        """
        STEP 5 — GENERATE.
        Build the domain-specific Pydantic findings object.
        Calls the LLM (claude_client.complete()) with the agent's system prompt.
        calculate_confidence() is called here with confirmed_count and total_count.
        """
        ...

    def return_findings(self, findings: Any) -> Any:
        """
        STEP 6 — RETURN.
        Attach the accumulated reasoning_log to the findings object and return.
        """
        if hasattr(findings, "agent_reasoning_log"):
            findings.agent_reasoning_log = [e.to_dict() for e in self.reasoning_log]
        self.log_reasoning(
            "RETURN",
            "Findings ready",
            f"confidence={getattr(findings, 'confidence', '?')}",
        )
        return findings

    # ── Public entry point ────────────────────────────────────────────────────

    @traceable(name="agent_6step_loop", run_type="chain")
    async def run(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list[dict],
        **kwargs: Any,
    ) -> Any:
        """
        Execute the full 6-step loop.
        Called by NexusOrchestrator (sequentially for triage, in parallel for diagnostic+resolution).
        Returns the domain-specific Pydantic findings object.
        kwargs allows passing triage_finding, prior_findings, etc. to sub-agents.
        """
        # Reset reasoning log for this run
        self.reasoning_log = []

        self.log_reasoning("LOAD", f"Loading data for session {session_id}", "Starting agent run")
        loaded = await self.load(session_id, db_data, conversation_history, **kwargs)

        analysis = await self.analyze(loaded)
        self.log_reasoning(
            "ANALYZE",
            f"total_fields={analysis.total_fields}, in_db={len(analysis.fields_in_db)}, missing={len(analysis.fields_missing)}",
            "Analysis complete",
        )

        reasoning = await self.reason(analysis, conversation_history)
        self.log_reasoning(
            "REASON",
            f"confirmed={len(reasoning.fields_confirmed_from_conversation)}, questions={len(reasoning.questions_to_ask)}, discrepancies={len(reasoning.discrepancies)}",
            "Reasoning complete",
        )

        decision = await self.decide(reasoning)
        self.log_reasoning(
            "DECIDE",
            f"should_ask={decision.should_ask_questions}, questions={len(decision.questions)}, ready={decision.ready_to_finalize}",
            "Decision made",
        )

        findings = await self.generate(decision, loaded, reasoning)
        self.log_reasoning(
            "GENERATE",
            f"agent={self.agent_id} confidence={getattr(findings, 'confidence', '?')}",
            "Findings generated",
        )

        return self.return_findings(findings)
