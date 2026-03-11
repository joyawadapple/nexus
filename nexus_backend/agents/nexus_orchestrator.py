"""
Nexus Orchestrator — signal-based routing with agent state caching.

Routing decision tree (decide_next_action):
  1. Client wants escalation           → force_escalate
  2. Known incident detected           → known_incident
  3. Client confirms issue resolved    → client_resolved
  4. Session already complete/escalated → complete
  5. Triage not done                   → collect_triage (with stuck detection)
  6. Diagnostic not done               → run_diagnostic (with stuck detection)
  7. Diagnostic confidence < threshold → run_escalation
  8. Resolution not done               → run_resolution
  9. Escalation not done               → run_escalation
  10. Everything done                  → complete

Agent findings are cached in SessionState — each agent runs at most until its
confidence threshold is reached, then never again for that session.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import structlog

if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    from langsmith import traceable
else:
    def traceable(**_):  # type: ignore[misc]
        return lambda f: f

from agents.agent_utils import (
    SENTIMENT_PROFILES,
    analyze_sentiment,
    bundle_questions,
    calculate_overall_confidence,
)
from agents.escalation_agent import EscalationAgent, assemble_ticket
from agents.triage_agent import TriageAgent
from core.claude_client import ClaudeClient
from core.rag_engine import RAGEngine
from models.conversation import SessionState
from prompts.orchestrator_prompt import build_orchestrator_prompt

log = structlog.get_logger("nexus_orchestrator")

# ── Module-level singletons ────────────────────────────────────────────────────
_claude: ClaudeClient | None = None
_db_data: dict = {}
_rag_engine: RAGEngine | None = None

# ── Intelligent routing components (injected at startup) ───────────────────────
_intent_classifier = None
_threshold_calculator = None
_progress_tracker = None
_monitor = None

# ── Agent instances ────────────────────────────────────────────────────────────
_escalation_agent = EscalationAgent()


def set_services(claude: ClaudeClient, db_data: dict, rag_engine: RAGEngine, monitor=None) -> None:
    """Inject dependencies at startup from main.py."""
    global _claude, _db_data, _rag_engine, _intent_classifier, _threshold_calculator, _progress_tracker, _monitor
    _claude = claude
    _db_data = db_data
    _rag_engine = rag_engine
    _monitor = monitor

    from agents.triage_agent import set_services as triage_set
    from agents.diagnostic_agent import set_services as diag_set
    from agents.resolution_agent import set_services as res_set
    from agents.escalation_agent import set_claude_client as esc_set
    from core.intent_classifier import IntentClassifier
    from agents.threshold_calculator import ThresholdCalculator
    from agents.progress_tracker import ProgressTracker

    triage_set(claude, rag_engine)
    diag_set(claude, rag_engine)
    res_set(claude, rag_engine)
    esc_set(claude)

    _threshold_calculator = ThresholdCalculator()
    _progress_tracker = ProgressTracker(_threshold_calculator)
    _intent_classifier = IntentClassifier(claude) if claude is not None else None

    if claude is not None:
        log.info("nexus_orchestrator.services_set", model=claude.model)


class NexusOrchestrator:
    """
    Main orchestrator for Nexus support sessions.
    Signal-based routing with cached agent findings per session.
    """

    # ── Signal lists ──────────────────────────────────────────────────────────

    ESCALATION_TRIGGERS = [
        "escalate", "speak to someone", "speak to a human",
        "talk to a person", "get me a human", "contact my csm",
        "get my csm", "i need a human", "transfer me",
        "real person", "supervisor", "manager",
        "this isn't working", "nothing is working",
        "your bot can't help",
    ]

    RESOLUTION_CONFIRMATIONS = [
        "working now", "fixed", "resolved", "that worked",
        "401s cleared", "authentication working", "all good",
        "issue resolved", "back up", "restored", "no more errors",
        "it's working", "problem solved", "sorted",
        "cleared now", "purge worked", "it worked",
    ]

    # ── Signal detection ──────────────────────────────────────────────────────

    # Minimum classifier confidence required to act on escalation or resolution intent.
    # Keyword fast-path bypasses this gate (exact matches are always trusted).
    _ESCALATION_CONFIDENCE_GATE = 0.75
    _RESOLUTION_CONFIDENCE_GATE = 0.70

    @traceable(name="intent_classification", run_type="chain")
    async def detect_client_intent(self, message: str, conversation_history: list) -> str:
        # Fast-path keyword check — exact matches are always trusted, no gate needed
        msg = message.lower()
        if any(t in msg for t in self.ESCALATION_TRIGGERS):
            return "wants_escalation"
        if any(t in msg for t in self.RESOLUTION_CONFIRMATIONS):
            return "issue_resolved"
        # Semantic classification for everything else
        if _intent_classifier:
            result = await _intent_classifier.classify(message, conversation_history)
            # Apply confidence gates to high-consequence intents.
            # A low-confidence "wants_escalation" from the classifier means the model
            # is unsure — treat the message as providing_information so triage continues.
            if result.intent == "wants_escalation" and result.confidence < self._ESCALATION_CONFIDENCE_GATE:
                return "providing_information"
            if result.intent == "issue_resolved" and result.confidence < self._RESOLUTION_CONFIDENCE_GATE:
                return "providing_information"
            return result.intent
        return "providing_information"

    def is_stuck(
        self,
        run_count: int,
        current_conf: float,
        previous_conf: float,
        tier: str,
        phase: str,
    ) -> bool:
        """Return True if an agent has hit its patience limit without improvement."""
        tier_limits = {
            "platinum": {"triage": 2, "diagnostic": 2, "resolution": 1},
            "gold":     {"triage": 3, "diagnostic": 3, "resolution": 2},
            "standard": {"triage": 4, "diagnostic": 4, "resolution": 3},
        }
        limit = tier_limits.get(tier, tier_limits["standard"]).get(phase, 4)
        not_improving = abs(current_conf - previous_conf) < 0.05
        return run_count >= limit and not_improving

    def get_escalation_threshold(self, session_state: SessionState) -> float:
        """Dynamic escalation threshold — uses ThresholdCalculator when available."""
        if _threshold_calculator:
            return _threshold_calculator.calculate("diagnostic", session_state)
        # Fallback: original logic
        base = 0.75
        sentiment_adj = {"frustrated": -0.10, "urgent": -0.15, "calm": 0.0}
        tier_adj = {"platinum": -0.05, "gold": 0.0, "standard": +0.05}
        tier = session_state.client.get("tier", "standard")
        sentiment = session_state.current_sentiment
        return base + sentiment_adj.get(sentiment, 0.0) + tier_adj.get(tier, 0.0)

    # ── Routing decision ──────────────────────────────────────────────────────

    @traceable(name="routing_decision", run_type="chain")
    async def decide_next_action(
        self, session_state: SessionState, last_message: str, conversation_history: list
    ) -> str:
        # Terminal state check FIRST — before intent classification.
        # A session that is complete/resolved/escalated must not re-trigger handlers
        # regardless of what the client says next (e.g. "no thanks" after resolution).
        if session_state.status in ("complete", "escalated", "resolved"):
            return "complete"

        intent = await self.detect_client_intent(last_message, conversation_history)
        try:
            from langdetect import detect as _detect_lang
            session_state.detected_language = _detect_lang(last_message)
        except Exception:
            session_state.detected_language = "en"
        tier = session_state.client.get("tier", "standard")

        # ── Overrides — always checked first ──
        if intent == "wants_escalation":
            session_state.escalation_trigger = "Client requested escalation"
            return "force_escalate"

        if intent == "contradicting_previous":
            # Invalidate stale findings — client has contradicted the diagnostic premise.
            # Reset confidence to 0 so escalation agent sees no false certainty.
            has_findings = bool(session_state.diagnostic_finding or session_state.resolution_finding)
            if session_state.diagnostic_finding:
                session_state.diagnostic_finding["hypothesis_invalidated"] = True
                session_state.diagnostic_finding["invalidation_reason"] = "Client contradicted premise"
                session_state.diagnostic_confidence = 0.0
            if session_state.resolution_finding:
                session_state.resolution_finding["hypothesis_invalidated"] = True
                session_state.resolution_confidence = 0.0
            if has_findings:
                session_state.escalation_trigger = "Client contradicted a confirmed diagnosis"
                return "force_escalate"
            # Still in triage — treat product correction as new information, re-run triage
            return "collect_triage"

        if session_state.known_incident:
            return "known_incident"

        if intent == "issue_resolved":
            return "client_resolved"

        # ── Triage phase ──
        if not session_state.triage_complete:
            if _progress_tracker:
                assessment = _progress_tracker.assess_progress("triage", session_state)
                log.info(
                    "DEBUG.stuck_detection",
                    agent="triage",
                    run_count=session_state.triage_run_count,
                    is_stuck=assessment.is_stuck,
                    reason=assessment.reason,
                    velocity=round(assessment.velocity, 4),
                    confidence=session_state.triage_confidence,
                    tier=session_state.client.get("tier"),
                )
                if assessment.is_stuck:
                    session_state.escalation_trigger = "Insufficient information after maximum attempts"
                    return "force_escalate"
            else:
                stuck = self.is_stuck(
                    session_state.triage_run_count,
                    session_state.triage_confidence,
                    0.0,
                    tier,
                    "triage",
                )
                if stuck:
                    session_state.escalation_trigger = "Insufficient information after maximum attempts"
                    return "force_escalate"
            return "collect_triage"

        # ── Diagnostic phase ──
        if not session_state.diagnostic_complete:
            if _progress_tracker:
                assessment = _progress_tracker.assess_progress("diagnostic", session_state)
                log.info(
                    "DEBUG.stuck_detection",
                    agent="diagnostic",
                    run_count=session_state.diagnostic_run_count,
                    is_stuck=assessment.is_stuck,
                    reason=assessment.reason,
                    velocity=round(assessment.velocity, 4),
                    confidence=session_state.diagnostic_confidence,
                    tier=session_state.client.get("tier"),
                )
                if assessment.is_stuck:
                    session_state.escalation_trigger = "Insufficient information after maximum attempts"
                    return "force_escalate"
            else:
                stuck = self.is_stuck(
                    session_state.diagnostic_run_count,
                    session_state.diagnostic_confidence,
                    session_state.diagnostic_previous_confidence,
                    tier,
                    "diagnostic",
                )
                if stuck:
                    session_state.escalation_trigger = "Insufficient information after maximum attempts"
                    return "force_escalate"
            return "run_diagnostic"

        # ── Check diagnostic confidence vs dynamic threshold ──
        threshold = self.get_escalation_threshold(session_state)
        if session_state.diagnostic_confidence < threshold:
            return "run_escalation"

        # ── Resolution phase ──
        if not session_state.resolution_complete:
            return "run_resolution"

        # ── Final escalation decision ──
        if not session_state.escalation_complete:
            return "run_escalation"

        return "complete"

    # ── Agent runner methods (with caching) ───────────────────────────────────

    async def _run_triage(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
    ):
        """Run triage agent, caching result on completion."""
        if session_state.triage_complete:
            from models.report_models import TriageReport
            return TriageReport.model_validate(session_state.triage_finding)

        triage_agent = TriageAgent()
        if _threshold_calculator:
            triage_agent.dynamic_threshold = _threshold_calculator.calculate("triage", session_state)

        session_state.triage_run_count += 1
        _t0 = time.monotonic()
        finding = await triage_agent.run(
            session_id=session_id,
            db_data=db_data,
            conversation_history=conversation_history,
        )
        if _monitor:
            from core.monitor import AgentEvent
            _usage = _claude.last_usage if _claude else {"input_tokens": 0, "output_tokens": 0}
            _monitor.track_agent_call(AgentEvent(
                agent_id="triage_agent",
                session_id=session_id,
                latency_ms=int((time.monotonic() - _t0) * 1000),
                input_tokens=_usage["input_tokens"],
                output_tokens=_usage["output_tokens"],
                confidence=finding.confidence,
            ))

        session_state.triage_finding = (
            finding.model_dump() if hasattr(finding, "model_dump") else {}
        )
        session_state.triage_confidence = finding.confidence
        session_state.triage_confidence_history.append(finding.confidence)

        log.debug(
            "orchestrator.triage_completion_check",
            completed=finding.completed,
            confidence=finding.confidence,
            threshold=triage_agent.dynamic_threshold or triage_agent.confidence_threshold,
            now_cached=session_state.triage_complete,
        )

        # Always upgrade severity — calculable from tier+environment alone, no full confirmation needed.
        # Upgrade-only: never downgrade mid-session (critical stays critical even if next run sees "high").
        _SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        new_severity = getattr(finding, "severity", None)
        if new_severity:
            current_rank = _SEVERITY_RANK.get(session_state.severity or "medium", 1)
            new_rank = _SEVERITY_RANK.get(new_severity, 1)
            if new_rank > current_rank:
                session_state.severity = new_severity

        if finding.completed:
            session_state.triage_complete = True
            if hasattr(finding, "issue") and finding.issue:
                session_state.known_incident = finding.issue.known_incident
                session_state.recurring = finding.issue.recurring
            session_state.severity = getattr(finding, "severity", "medium")

            # Update partial ticket
            if finding.client and finding.issue:
                _issue_dict = finding.issue.model_dump()
                _issue_dict["started_at"] = _issue_dict.get("started_at") or "unknown"
                _mentioned = getattr(finding.issue, "mentioned_products", [])
                if len(_mentioned) > 1:
                    session_state.partial_ticket["additional_products"] = [
                        p for p in _mentioned if p != finding.issue.product
                    ]
                    session_state.partial_ticket["multi_product_incident"] = True
                session_state.partial_ticket.update({
                    "client": {
                        "company": finding.client.company,
                        "tier": finding.client.tier,
                        "sla_hours": finding.client.sla_hours,
                    },
                    "issue": _issue_dict,
                    "priority": finding.severity,
                    "sla_deadline": finding.client.sla_deadline,
                })

        log.info(
            "orchestrator.triage_done",
            session_id=session_id,
            confidence=finding.confidence,
            completed=finding.completed,
            run_count=session_state.triage_run_count,
        )
        log.info(
            "orchestrator.session_state_snapshot",
            phase="after_triage",
            triage_complete=session_state.triage_complete,
            triage_confidence=round(session_state.triage_confidence, 4),
            triage_run_count=session_state.triage_run_count,
            diagnostic_complete=session_state.diagnostic_complete,
            resolution_complete=session_state.resolution_complete,
            escalation_complete=session_state.escalation_complete,
            known_incident=session_state.known_incident,
            severity=session_state.severity,
        )
        return finding

    async def _run_diagnostic(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
    ):
        """Run diagnostic agent, caching result on completion."""
        if session_state.diagnostic_complete:
            from models.report_models import DiagnosticReport
            return DiagnosticReport.model_validate(session_state.diagnostic_finding)

        session_state.diagnostic_previous_confidence = session_state.diagnostic_confidence
        session_state.diagnostic_run_count += 1

        from models.report_models import TriageReport
        from agents.diagnostic_agent import DiagnosticAgent, set_services as diag_set
        diag_set(_claude, _rag_engine)
        triage_finding = TriageReport.model_validate(session_state.triage_finding)
        diagnostic_agent = DiagnosticAgent(triage_finding=triage_finding)
        if _threshold_calculator:
            diagnostic_agent.dynamic_threshold = _threshold_calculator.calculate("diagnostic", session_state)

        _t0 = time.monotonic()
        try:
            finding = await diagnostic_agent.run(
                session_id=session_id,
                db_data=db_data,
                conversation_history=conversation_history,
                triage_finding=triage_finding,
            )
        except Exception as e:
            log.error("orchestrator.diagnostic_failed", error=str(e))
            from models.report_models import DiagnosticReport
            return DiagnosticReport(
                session_id=session_id, confidence=0.0, completed=False
            )
        if _monitor:
            from core.monitor import AgentEvent
            _usage = _claude.last_usage if _claude else {"input_tokens": 0, "output_tokens": 0}
            _monitor.track_agent_call(AgentEvent(
                agent_id="diagnostic_agent",
                session_id=session_id,
                latency_ms=int((time.monotonic() - _t0) * 1000),
                input_tokens=_usage["input_tokens"],
                output_tokens=_usage["output_tokens"],
                confidence=finding.confidence,
            ))

        session_state.diagnostic_finding = (
            finding.model_dump() if hasattr(finding, "model_dump") else {}
        )
        new_confidence = max(session_state.diagnostic_confidence, finding.confidence)
        session_state.diagnostic_confidence = new_confidence
        session_state.diagnostic_confidence_history.append(new_confidence)

        log.debug(
            "orchestrator.diagnostic_completion_check",
            completed=finding.completed,
            confidence=finding.confidence,
            threshold=diagnostic_agent.dynamic_threshold or diagnostic_agent.confidence_threshold,
            now_cached=session_state.diagnostic_complete,
        )

        if finding.completed:
            session_state.diagnostic_complete = True
            if finding.primary_hypothesis:
                session_state.partial_ticket["diagnosis"] = {
                    "primary_cause": finding.primary_hypothesis.cause,
                    "confidence": finding.confidence,
                }

        log.info(
            "orchestrator.diagnostic_done",
            confidence=finding.confidence,
            completed=finding.completed,
            run_count=session_state.diagnostic_run_count,
        )
        log.info(
            "orchestrator.session_state_snapshot",
            phase="after_diagnostic",
            triage_complete=session_state.triage_complete,
            triage_confidence=round(session_state.triage_confidence, 4),
            diagnostic_complete=session_state.diagnostic_complete,
            diagnostic_confidence=round(session_state.diagnostic_confidence, 4),
            diagnostic_run_count=session_state.diagnostic_run_count,
            diagnostic_confidence_history=session_state.diagnostic_confidence_history,
            resolution_complete=session_state.resolution_complete,
            escalation_complete=session_state.escalation_complete,
        )
        return finding

    async def _run_resolution(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
    ):
        """Run resolution agent, caching result on completion."""
        if session_state.resolution_complete:
            from models.report_models import ResolutionReport
            return ResolutionReport.model_validate(session_state.resolution_finding)

        session_state.resolution_run_count += 1

        from models.report_models import TriageReport
        from agents.resolution_agent import ResolutionAgent, set_services as res_set
        res_set(_claude, _rag_engine)
        triage_finding = TriageReport.model_validate(session_state.triage_finding)
        resolution_agent = ResolutionAgent(triage_finding=triage_finding)
        if _threshold_calculator:
            resolution_agent.dynamic_threshold = _threshold_calculator.calculate("resolution", session_state)

        _t0 = time.monotonic()
        try:
            finding = await resolution_agent.run(
                session_id=session_id,
                db_data=db_data,
                conversation_history=conversation_history,
                triage_finding=triage_finding,
            )
        except Exception as e:
            log.error("orchestrator.resolution_failed", error=str(e))
            from models.report_models import ResolutionReport
            return ResolutionReport(
                session_id=session_id, confidence=0.0, completed=False
            )
        if _monitor:
            from core.monitor import AgentEvent
            _usage = _claude.last_usage if _claude else {"input_tokens": 0, "output_tokens": 0}
            _monitor.track_agent_call(AgentEvent(
                agent_id="resolution_agent",
                session_id=session_id,
                latency_ms=int((time.monotonic() - _t0) * 1000),
                input_tokens=_usage["input_tokens"],
                output_tokens=_usage["output_tokens"],
                confidence=finding.confidence,
            ))

        session_state.resolution_finding = (
            finding.model_dump() if hasattr(finding, "model_dump") else {}
        )
        session_state.resolution_confidence = finding.confidence
        session_state.resolution_confidence_history.append(finding.confidence)

        log.debug(
            "orchestrator.resolution_completion_check",
            completed=finding.completed,
            confidence=finding.confidence,
            threshold=resolution_agent.dynamic_threshold or resolution_agent.confidence_threshold,
            now_cached=session_state.resolution_complete,
        )

        if finding.completed:
            session_state.resolution_complete = True
            if hasattr(finding, "steps") and finding.steps:
                session_state.partial_ticket["resolution"] = {
                    "steps": [s.model_dump() for s in finding.steps],
                    "confidence": finding.confidence,
                }

        log.info(
            "orchestrator.resolution_done",
            confidence=finding.confidence,
            completed=finding.completed,
        )
        log.info(
            "orchestrator.session_state_snapshot",
            phase="after_resolution",
            triage_complete=session_state.triage_complete,
            diagnostic_complete=session_state.diagnostic_complete,
            resolution_complete=session_state.resolution_complete,
            resolution_confidence=round(session_state.resolution_confidence, 4),
            resolution_run_count=session_state.resolution_run_count,
            resolution_confidence_history=session_state.resolution_confidence_history,
            escalation_complete=session_state.escalation_complete,
        )
        return finding

    async def _run_escalation(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
        force: bool = False,
        reason: str = "",
    ):
        """Run escalation agent. Use force=True for override decisions."""
        if not force and session_state.escalation_complete:
            from models.report_models import EscalationReport
            return EscalationReport.model_validate(session_state.escalation_finding)

        # Gather all completed findings
        findings = []
        if session_state.triage_finding:
            from models.report_models import TriageReport
            findings.append(TriageReport.model_validate(session_state.triage_finding))
        if session_state.diagnostic_finding:
            from models.report_models import DiagnosticReport
            findings.append(DiagnosticReport.model_validate(session_state.diagnostic_finding))
        if session_state.resolution_finding:
            from models.report_models import ResolutionReport
            findings.append(ResolutionReport.model_validate(session_state.resolution_finding))

        sentiment_label = session_state.current_sentiment
        sentiment_bias = SENTIMENT_PROFILES.get(sentiment_label, {}).get("escalation_bias", 0.0)

        _t0 = time.monotonic()
        finding = await _escalation_agent.run(
            session_id=session_id,
            db_data=db_data,
            conversation_history=conversation_history,
            findings=findings,
            sentiment_profile=sentiment_label,
            sentiment_bias=sentiment_bias,
        )
        if _monitor:
            from core.monitor import AgentEvent
            _usage = _claude.last_usage if _claude else {"input_tokens": 0, "output_tokens": 0}
            _monitor.track_agent_call(AgentEvent(
                agent_id="escalation_agent",
                session_id=session_id,
                latency_ms=int((time.monotonic() - _t0) * 1000),
                input_tokens=_usage["input_tokens"],
                output_tokens=_usage["output_tokens"],
                confidence=finding.confidence,
            ))

        if force:
            finding.decision = "escalated"
            if reason:
                finding.reason = reason

        session_state.escalation_finding = (
            finding.model_dump() if hasattr(finding, "model_dump") else {}
        )
        session_state.escalation_complete = True
        session_state.escalation_confidence = finding.confidence

        return finding

    def _build_ticket(
        self,
        session_id: str,
        session_state: SessionState,
        escalation_finding,
    ) -> dict:
        """Assemble and store the final ticket."""
        client_id = session_state.client_id
        client_data = _db_data.get("clients", {}).get(client_id, {})
        ticket_id = _generate_ticket_id()
        ticket = assemble_ticket(
            ticket_id=ticket_id,
            escalation_report=escalation_finding,
            session_state=session_state,
            client_data=client_data,
        )
        session_state.ticket = ticket.model_dump()
        return session_state.ticket

    # ── Action handlers ───────────────────────────────────────────────────────

    async def _handle_force_escalate(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
    ) -> dict:
        reason = session_state.escalation_trigger or "Insufficient information to resolve automatically"
        session_state.escalation_trigger = None
        escalation_finding = await self._run_escalation(
            session_id, db_data, conversation_history, session_state,
            force=True, reason=reason,
        )
        session_state.status = "escalated"
        ticket = self._build_ticket(session_id, session_state, escalation_finding)

        client = session_state.client
        csm_name = client.get("csm", "your CSM")
        tier = client.get("tier", "standard")
        time_frame = "15 minutes" if tier == "platinum" else "30 minutes"
        response = (
            f"Escalating to your CSM, {csm_name}, immediately. "
            f"Expected contact within {time_frame}. "
            f"Full incident context has been passed."
        )
        threshold = self.get_escalation_threshold(session_state)
        return {
            "status": "escalated",
            "ticket": ticket,
            "message": response,
            "agent_statuses": _build_agent_statuses_from_state(session_state),
            "reasoning_logs": {
                "escalation_agent": {
                    "confidence": escalation_finding.confidence,
                    "forced": True,
                    "threshold_used": threshold,
                },
            },
        }

    async def _handle_client_resolved(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
    ) -> dict:
        session_state.status = "resolved"
        escalation_finding = await self._run_escalation(
            session_id, db_data, conversation_history, session_state
        )
        # Client confirmed resolution — the escalation agent doesn't know this.
        # Override its decision regardless of what confidence numbers it saw.
        escalation_finding.decision = "self_resolve"
        escalation_finding.reason = "Client confirmed issue resolved"
        escalation_finding.csm_notified = False

        # Re-store the updated finding so the ticket reflects the override.
        session_state.escalation_finding = (
            escalation_finding.model_dump() if hasattr(escalation_finding, "model_dump") else {}
        )

        ticket = self._build_ticket(session_id, session_state, escalation_finding)
        response = (
            "Glad to hear it's resolved! I've closed this ticket and logged "
            "the full incident context. Is there anything you'd like noted "
            "for our records about what fixed it?"
        )
        return {
            "status": "resolved",
            "ticket": ticket,
            "message": response,
            "agent_statuses": _build_agent_statuses_from_state(session_state),
            "reasoning_logs": {},
        }

    async def _handle_known_incident(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
        triage_finding=None,
    ) -> dict:
        escalation_finding = await self._run_escalation(
            session_id, db_data, conversation_history, session_state,
            force=True, reason="Known platform incident",
        )
        session_state.status = "escalated"
        ticket = self._build_ticket(session_id, session_state, escalation_finding)

        if triage_finding is None and session_state.triage_finding:
            from models.report_models import TriageReport
            triage_finding = TriageReport.model_validate(session_state.triage_finding)

        incident_data: dict = {}
        if (triage_finding and hasattr(triage_finding, "issue")
                and triage_finding.issue and triage_finding.issue.product):
            incident_data = (
                _db_data.get("products", {})
                .get(triage_finding.issue.product, {})
                .get("active_incident") or {}
            )

        message = await self._generate_known_incident_response(
            session_state, incident_data, conversation_history
        )

        rl: dict = {}
        is_first_triage = session_state.triage_run_count == 1
        if is_first_triage and triage_finding:
            rl["triage_agent"] = {
                "confidence": getattr(triage_finding, "confidence", 0.0),
                "severity": getattr(triage_finding, "severity", "medium"),
                "recurring": (
                    triage_finding.issue.recurring
                    if hasattr(triage_finding, "issue") and triage_finding.issue
                    else False
                ),
                "cached": False,
            }
        rl["escalation_agent"] = {"confidence": escalation_finding.confidence}

        return {
            "status": "known_incident",
            "ticket": ticket,
            "message": message,
            "agent_statuses": _build_agent_statuses_from_state(session_state),
            "reasoning_logs": rl,
        }

    async def _handle_collect_triage(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
    ) -> dict:
        triage_finding = await self._run_triage(
            session_id, db_data, conversation_history, session_state
        )

        # If triage just completed, continue immediately to next step
        if session_state.triage_complete:
            if session_state.known_incident:
                return await self._handle_known_incident(
                    session_id, db_data, conversation_history, session_state, triage_finding
                )
            else:
                return await self._handle_run_diagnostic(
                    session_id, db_data, conversation_history, session_state,
                    session_state.current_sentiment, triage_finding,
                )

        # Still collecting triage fields — one question per turn for a natural conversation flow
        questions = bundle_questions(
            findings_list=[triage_finding] if triage_finding else [],
            asked_fields=session_state.asked_fields,
            max_questions=1,
        )
        for q in questions:
            session_state.mark_field_asked(q.field)

        message = await self._generate_conversational_response(
            session_state=session_state,
            agent_findings={"triage": session_state.triage_finding},
            sentiment=session_state.current_sentiment,
            conversation_history=conversation_history,
            asked_fields=session_state.asked_fields,
        )
        session_state.status = "collecting"

        rl: dict = {}
        if triage_finding:
            rl["triage_agent"] = {
                "confidence": getattr(triage_finding, "confidence", 0.0),
                "severity": getattr(triage_finding, "severity", "medium"),
                "known_incident": (
                    triage_finding.issue.known_incident
                    if hasattr(triage_finding, "issue") and triage_finding.issue
                    else False
                ),
                "recurring": (
                    triage_finding.issue.recurring
                    if hasattr(triage_finding, "issue") and triage_finding.issue
                    else False
                ),
                "cached": session_state.triage_complete,
                "run_count": session_state.triage_run_count,
            }

        return {
            "status": "collecting",
            "message": message,
            "agent_statuses": _build_agent_statuses(triage_finding, None, None, None),
            "reasoning_logs": rl,
        }

    async def _handle_run_diagnostic(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
        sentiment: str,
        triage_finding=None,
    ) -> dict:
        diagnostic_finding = await self._run_diagnostic(
            session_id, db_data, conversation_history, session_state
        )

        # Build reasoning log.
        # triage_finding is only passed when triage ran THIS turn (from _handle_collect_triage).
        # When triage_finding is None, triage was cached from a previous turn.
        rl: dict = {}
        if triage_finding:
            rl["triage_agent"] = {
                "confidence": getattr(triage_finding, "confidence", 0.0),
                "severity": getattr(triage_finding, "severity", "medium"),
                "known_incident": (
                    triage_finding.issue.known_incident
                    if hasattr(triage_finding, "issue") and triage_finding.issue
                    else False
                ),
                "recurring": (
                    triage_finding.issue.recurring
                    if hasattr(triage_finding, "issue") and triage_finding.issue
                    else False
                ),
                "cached": False,  # ran this turn — triage_finding is passed only when triage just ran
                "run_count": session_state.triage_run_count,
            }
        elif session_state.triage_confidence > 0:
            rl["triage_agent"] = {
                "confidence": session_state.triage_confidence,
                "cached": True,  # triage_finding is None — served from cache
                "run_count": session_state.triage_run_count,
            }
        rl["diagnostic_agent"] = {"confidence": diagnostic_finding.confidence}

        # If diagnostic just completed, continue immediately
        if session_state.diagnostic_complete:
            threshold = self.get_escalation_threshold(session_state)
            if session_state.diagnostic_confidence < threshold:
                return await self._escalate_low_confidence(
                    session_id, db_data, conversation_history, session_state, rl, threshold
                )
            else:
                return await self._handle_run_resolution(
                    session_id, db_data, conversation_history, session_state, sentiment, rl
                )

        # Diagnostic not yet complete — one question per turn
        questions = bundle_questions(
            findings_list=[diagnostic_finding],
            asked_fields=session_state.asked_fields,
            max_questions=1,
        )
        for q in questions:
            session_state.mark_field_asked(q.field)

        message = await self._generate_conversational_response(
            session_state=session_state,
            agent_findings={
                "triage": session_state.triage_finding,
                "diagnostic": session_state.diagnostic_finding,
            },
            sentiment=sentiment,
            conversation_history=conversation_history,
            asked_fields=session_state.asked_fields,
        )
        session_state.status = "in_progress"

        return {
            "status": "in_progress",
            "message": message,
            "agent_statuses": _build_agent_statuses_from_state(session_state),
            "reasoning_logs": rl,
        }

    async def _escalate_low_confidence(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
        base_rl: dict,
        threshold: float,
    ) -> dict:
        escalation_finding = await self._run_escalation(
            session_id, db_data, conversation_history, session_state
        )
        escalation_finding.reason = (
            f"Diagnostic confidence too low ({session_state.diagnostic_confidence:.0%}) "
            f"— below required threshold ({threshold:.0%})"
        )
        session_state.status = "escalated"
        ticket = self._build_ticket(session_id, session_state, escalation_finding)

        message = await self._generate_conversational_response(
            session_state=session_state,
            agent_findings={
                "triage": session_state.triage_finding,
                "diagnostic": session_state.diagnostic_finding,
                "escalation": session_state.escalation_finding,
            },
            sentiment=session_state.current_sentiment,
            conversation_history=conversation_history,
            asked_fields=session_state.asked_fields,
        )
        return {
            "status": "escalated",
            "ticket": ticket,
            "message": message,
            "agent_statuses": _build_agent_statuses_from_state(session_state),
            "reasoning_logs": {
                **base_rl,
                "escalation_agent": {"confidence": escalation_finding.confidence, "threshold_used": threshold},
            },
        }

    async def _handle_run_resolution(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
        sentiment: str,
        base_rl: dict | None = None,
    ) -> dict:
        if base_rl is None:
            base_rl = {}
            if session_state.triage_confidence > 0:
                base_rl["triage_agent"] = {"confidence": session_state.triage_confidence, "cached": True}
            if session_state.diagnostic_confidence > 0:
                base_rl["diagnostic_agent"] = {"confidence": session_state.diagnostic_confidence}

        resolution_finding = await self._run_resolution(
            session_id, db_data, conversation_history, session_state
        )
        base_rl["resolution_agent"] = {"confidence": resolution_finding.confidence}

        # Only finalize immediately if resolution confidence is high OR the client has
        # already had a turn to try the steps (run_count > 1).
        # Otherwise present the steps and wait for client feedback before escalating.
        #
        # Next turn routing (handled automatically by decide_next_action):
        # - Client says "it worked"  → detect_client_intent returns "issue_resolved"
        #                              → _handle_client_resolved → ticket closed
        # - Client says "still failing" → resolution_run_count > 1 on next _run_resolution call
        #                                  → HIGH_CONF branch fires → escalation runs
        # - Client provides more info → run_resolution reruns with updated conversation
        #                               → resolution_run_count increments to 2
        #                               → HIGH_CONF branch fires → escalation runs
        #
        # No special handling needed — the existing routing tree covers all three cases.
        _RESOLUTION_HIGH_CONF = 0.85
        if resolution_finding.confidence >= _RESOLUTION_HIGH_CONF or session_state.resolution_run_count > 1:
            # Finalize with escalation
            escalation_finding = await self._run_escalation(
                session_id, db_data, conversation_history, session_state
            )
            final_status = (
                "complete" if escalation_finding.decision == "self_resolve" else "escalated"
            )
            session_state.status = final_status
            ticket = self._build_ticket(session_id, session_state, escalation_finding)

            triage_conf = session_state.triage_confidence
            diag_conf = session_state.diagnostic_confidence
            res_conf = resolution_finding.confidence

            sla_remaining = _calculate_sla_remaining(session_state)
            message = await self._generate_conversational_response(
                session_state=session_state,
                agent_findings={
                    "triage": session_state.triage_finding,
                    "diagnostic": session_state.diagnostic_finding,
                    "resolution": session_state.resolution_finding,
                    "escalation": session_state.escalation_finding,
                },
                sentiment=sentiment,
                conversation_history=conversation_history,
                sla_remaining=sla_remaining,
                asked_fields=session_state.asked_fields,
            )

            threshold = self.get_escalation_threshold(session_state)
            return {
                "status": final_status,
                "ticket": ticket,
                "message": message,
                "agent_statuses": _build_agent_statuses_from_state(session_state),
                "confidence_breakdown": {
                    "triage": triage_conf,
                    "diagnostic": diag_conf,
                    "resolution": res_conf,
                    "escalation": escalation_finding.confidence,
                    "overall": calculate_overall_confidence([
                        type("F", (), {"confidence": triage_conf})(),
                        type("F", (), {"confidence": diag_conf})(),
                        type("F", (), {"confidence": res_conf})(),
                    ]),
                },
                "reasoning_logs": {
                    **base_rl,
                    "escalation_agent": {
                        "confidence": escalation_finding.confidence,
                        "threshold_used": threshold,
                    },
                },
            }
        else:
            # Present resolution steps, wait for client feedback.
            session_state.status = "in_progress"
            message = await self._generate_conversational_response(
                session_state=session_state,
                agent_findings={
                    "triage": session_state.triage_finding,
                    "diagnostic": session_state.diagnostic_finding,
                    "resolution": session_state.resolution_finding,
                },
                sentiment=sentiment,
                conversation_history=conversation_history,
                asked_fields=session_state.asked_fields,
            )
            return {
                "status": "in_progress",
                "awaiting": "resolution_confirmation",  # Frontend can use this to show "Did this work?" prompt
                "message": message,
                "agent_statuses": _build_agent_statuses_from_state(session_state),
                "reasoning_logs": base_rl,
            }

    async def _handle_run_escalation(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list,
        session_state: SessionState,
        sentiment: str,
    ) -> dict:
        """Called when routing decides escalation is needed on a fresh turn."""
        threshold = self.get_escalation_threshold(session_state)
        escalation_finding = await self._run_escalation(
            session_id, db_data, conversation_history, session_state
        )
        if session_state.diagnostic_confidence > 0 and session_state.diagnostic_confidence < threshold:
            escalation_finding.reason = (
                f"Diagnostic confidence too low ({session_state.diagnostic_confidence:.0%}) "
                f"— below required threshold ({threshold:.0%})"
            )

        session_state.status = "escalated"
        ticket = self._build_ticket(session_id, session_state, escalation_finding)

        message = await self._generate_conversational_response(
            session_state=session_state,
            agent_findings={
                "triage": session_state.triage_finding,
                "diagnostic": session_state.diagnostic_finding,
                "escalation": session_state.escalation_finding,
            },
            sentiment=sentiment,
            conversation_history=conversation_history,
            asked_fields=session_state.asked_fields,
        )
        return {
            "status": "escalated",
            "ticket": ticket,
            "message": message,
            "agent_statuses": _build_agent_statuses_from_state(session_state),
            "reasoning_logs": {
                "escalation_agent": {
                    "confidence": escalation_finding.confidence,
                    "threshold_used": threshold,
                }
            },
        }

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(
        self,
        session_id: str,
        conversation_history: list[dict],
        session_state: SessionState,
        client_id: str,
    ) -> dict:
        """Execute one conversation turn and return the response."""
        if not _claude:
            return {"status": "error", "message": "Nexus services not initialized."}

        db_data = {
            **_db_data,
            "_session_client_id": client_id,
            "_client_id": client_id,
            "clients": _db_data.get("clients", {}),
        }

        # Load client data into session state if not yet set
        if not session_state.client:
            session_state.client = _db_data.get("clients", {}).get(client_id, {})

        # Sentiment analysis
        last_message = next(
            (m["content"] for m in reversed(conversation_history) if m["role"] == "user"),
            "",
        )
        sentiment_result = analyze_sentiment(last_message)
        sentiment_label = sentiment_result["label"]
        session_state.add_sentiment(
            label=sentiment_label,
            compound=sentiment_result["compound"],
            text=last_message,
        )

        # Route
        action = await self.decide_next_action(session_state, last_message, conversation_history)
        log.info("orchestrator.routing", session_id=session_id, action=action)

        if action == "force_escalate":
            return await self._handle_force_escalate(
                session_id, db_data, conversation_history, session_state
            )
        elif action == "known_incident":
            return await self._handle_known_incident(
                session_id, db_data, conversation_history, session_state
            )
        elif action == "client_resolved":
            return await self._handle_client_resolved(
                session_id, db_data, conversation_history, session_state
            )
        elif action == "collect_triage":
            return await self._handle_collect_triage(
                session_id, db_data, conversation_history, session_state
            )
        elif action == "run_diagnostic":
            return await self._handle_run_diagnostic(
                session_id, db_data, conversation_history, session_state, sentiment_label
            )
        elif action == "run_resolution":
            return await self._handle_run_resolution(
                session_id, db_data, conversation_history, session_state, sentiment_label
            )
        elif action == "run_escalation":
            return await self._handle_run_escalation(
                session_id, db_data, conversation_history, session_state, sentiment_label
            )
        elif action == "complete":
            if session_state.status == "resolved":
                closing = "You're all set. The ticket has been logged for your records. Don't hesitate to reach out if the issue returns."
            elif session_state.status == "escalated":
                csm = session_state.client.get("csm", "your CSM")
                closing = f"This has been escalated and {csm} has the full context. You'll hear back shortly."
            else:
                closing = "Your issue has been addressed. Is there anything else I can help you with?"
            return {
                "status": session_state.status,
                "ticket": session_state.ticket,
                "message": closing,
                "agent_statuses": _build_agent_statuses_from_state(session_state),
                "reasoning_logs": {},
            }
        else:
            return {"status": "error", "message": f"Unknown routing action: {action}"}

    # ── Response generation ───────────────────────────────────────────────────

    @traceable(name="response_generation", run_type="llm")
    async def _generate_conversational_response(
        self,
        session_state: SessionState,
        agent_findings: dict,
        sentiment: str,
        conversation_history: list[dict],
        sla_remaining: float | None = None,
        asked_fields: list[str] | None = None,
    ) -> str:
        if not _claude:
            return "Thank you for the information. Our team is working on your issue."

        client_id = session_state.client_id
        client_data = _db_data.get("clients", {}).get(client_id, {})

        prompt = build_orchestrator_prompt(
            agent_findings=agent_findings,
            client_info=client_data,
            sentiment=sentiment,
            sla_remaining_hours=sla_remaining,
            conversation_history=conversation_history,
            session_status=session_state.status,
            detected_language=session_state.detected_language,
            asked_fields=asked_fields,
        )
        messages = [{"role": "user", "content": "Generate the Nexus response."}]

        try:
            return await _claude.complete(system=prompt, messages=messages, max_tokens=600)
        except Exception as e:
            log.error("orchestrator.response_generation_failed", error=str(e))
            return "I'm gathering information about your issue. Let me check the details."

    async def _generate_known_incident_response(
        self,
        session_state: SessionState,
        incident_data: dict,
        conversation_history: list[dict],
    ) -> str:
        title = incident_data.get("title", "an active incident")
        status = incident_data.get("status", "under investigation")
        workaround = incident_data.get("workaround", "")
        next_update = incident_data.get("next_update", "")

        csm_name = session_state.client.get("csm", "your CSM")

        lines = [
            f"We've identified that this is related to an active incident: {title}.",
            f"Status: {status.title()}.",
        ]
        if workaround:
            lines.append(f"Workaround available: {workaround}")
        if next_update:
            lines.append(f"Next status update: {next_update}")
        lines.append(
            f"Your CSM {csm_name} has been notified and is tracking this incident. "
            f"We'll update you as soon as it's resolved."
        )
        return " ".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

_ticket_counter = 4471


def _generate_ticket_id() -> str:
    global _ticket_counter
    _ticket_counter += 1
    year = datetime.now(tz=timezone.utc).year
    return f"NX-{year}-{_ticket_counter:04d}"


def _build_agent_statuses(triage, diagnostic, resolution, escalation) -> list[dict]:
    statuses = []
    agents = [
        ("triage_agent", triage),
        ("diagnostic_agent", diagnostic),
        ("resolution_agent", resolution),
        ("escalation_agent", escalation),
    ]
    for agent_id, finding in agents:
        statuses.append({
            "agent": agent_id,
            "status": (
                "complete" if finding and getattr(finding, "completed", False) else
                "running" if finding else "pending"
            ),
            "confidence": getattr(finding, "confidence", 0.0) if finding else 0.0,
        })
    return statuses


def _build_agent_statuses_from_state(session_state: SessionState) -> list[dict]:
    """Build agent status list from session state flags (not finding dicts)."""
    def _status(complete_flag: bool, has_finding: bool) -> str:
        if complete_flag:
            return "complete"
        if has_finding:
            return "running"
        return "pending"

    return [
        {
            "agent": "triage_agent",
            "status": _status(session_state.triage_complete, bool(session_state.triage_finding)),
            "confidence": session_state.triage_confidence,
        },
        {
            "agent": "diagnostic_agent",
            "status": _status(session_state.diagnostic_complete, bool(session_state.diagnostic_finding)),
            "confidence": session_state.diagnostic_confidence,
        },
        {
            "agent": "resolution_agent",
            "status": _status(session_state.resolution_complete, bool(session_state.resolution_finding)),
            "confidence": session_state.resolution_confidence,
        },
        {
            "agent": "escalation_agent",
            "status": "complete" if session_state.escalation_complete else (
                "running" if session_state.escalation_finding else "pending"
            ),
            "confidence": session_state.escalation_confidence,
        },
    ]


def _calculate_sla_remaining(session_state: SessionState) -> float | None:
    triage = session_state.triage_finding
    if not triage:
        return None
    client = triage.get("client") if isinstance(triage, dict) else {}
    if not client:
        return None
    sla_deadline_str = client.get("sla_deadline", "")
    if not sla_deadline_str:
        return None
    try:
        deadline = datetime.fromisoformat(sla_deadline_str.replace("Z", "+00:00"))
        now = datetime.now(tz=timezone.utc)
        remaining = (deadline - now).total_seconds() / 3600
        return max(remaining, 0.0)
    except Exception:
        return None
