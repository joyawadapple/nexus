"""
Escalation Agent — makes the human handoff decision and generates the complete support ticket.

Phase 3 of the Nexus orchestration pipeline. Always runs last.
Confidence threshold: 0.95
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from agents.agent_base import AnalysisResult, BaseAgent, Decision, LoadedData, ReasoningResult
from agents.agent_utils import calculate_overall_confidence, validate_agent_output
from models.agent_models import QuestionForClient
from models.report_models import EscalationReport
from models.ticket import (
    ClientSummary,
    ConfidenceBreakdown,
    DiagnosisSummary,
    EscalationDecision,
    IssueSummary,
    ResolutionPlan,
    ResolutionStep,
    SentimentProfile,
    SupportTicket,
)
from prompts.escalation_agent_prompt import build_escalation_prompt

log = structlog.get_logger("escalation_agent")

_claude = None


def set_claude_client(claude) -> None:
    global _claude
    _claude = claude


# ── Escalation rules ──────────────────────────────────────────────────────────

def evaluate_escalation_rules(
    triage: dict,
    diagnostic: dict | None,
    resolution: dict | None,
    sentiment_bias: float,
) -> tuple[str, str]:
    """
    Returns (decision, reason) where decision is "self_resolve" | "escalated" | "pending".
    sentiment_bias is added to confidence thresholds (lowers bar for escalation).
    """
    severity = triage.get("severity", "medium")
    tier = (triage.get("client") or {}).get("tier", "standard") if isinstance(triage.get("client"), dict) else "standard"
    environment = triage.get("environment", "production")
    known_incident = triage.get("known_incident", False)
    recurring = triage.get("recurring", False)

    diag_conf = (diagnostic or {}).get("confidence", 0.0)
    res_conf = (resolution or {}).get("confidence", 0.0)
    has_low_conf_steps = (resolution or {}).get("has_low_confidence_steps", False)

    # AUTO-ESCALATE rules
    if known_incident:
        return "escalated", "Active product incident detected — routing to engineering team"

    if severity == "critical" and res_conf < (0.70 + sentiment_bias):
        return "escalated", f"Critical severity with insufficient resolution confidence ({res_conf:.0%})"

    if tier == "platinum" and diag_conf < (0.75 + sentiment_bias):
        return "escalated", f"Platinum client requires higher diagnostic confidence — current: {diag_conf:.0%}"

    if recurring:
        return "escalated", "Recurring issue — root cause analysis required by engineering"

    if has_low_conf_steps:
        return "escalated", "Resolution plan contains low-confidence steps requiring human validation"

    error_auto_escalate = triage.get("_error_auto_escalate", False)
    if error_auto_escalate:
        return "escalated", "Error type requires mandatory engineering review"

    # RECOMMEND ESCALATE (but provide steps)
    est_time = (resolution or {}).get("estimated_resolution_time", "")
    if "60" in est_time or "hour" in est_time.lower():
        if tier in ("platinum", "gold") and environment == "production":
            return "escalated", f"Estimated resolution > 60 min for {tier.title()} production client"

    # SELF-RESOLVE
    if res_conf >= 0.85 and not has_low_conf_steps:
        return "self_resolve", f"High confidence diagnosis and resolution — known pattern ({res_conf:.0%})"

    if tier == "standard" and res_conf >= 0.70:
        return "self_resolve", f"Standard tier client with adequate resolution confidence ({res_conf:.0%})"

    # DEFAULT
    if res_conf >= 0.70:
        return "self_resolve", f"Adequate resolution confidence for self-service ({res_conf:.0%})"

    return "pending", "Insufficient confidence for self-resolve — human review recommended"


class EscalationAgent(BaseAgent):
    agent_id = "escalation_agent"
    confidence_threshold = 0.95

    def __init__(self) -> None:
        super().__init__()
        self.all_findings: list = []
        self.sentiment_profile: str = "calm"
        self.sentiment_bias: float = 0.0

    # ── Step 1: LOAD ──────────────────────────────────────────────────────────

    async def load(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list[dict],
        **kwargs,
    ) -> LoadedData:
        findings = kwargs.get("findings", [])
        self.all_findings = findings
        self.sentiment_profile = kwargs.get("sentiment_profile", "calm")
        self.sentiment_bias = kwargs.get("sentiment_bias", 0.0)

        client_id = db_data.get("_session_client_id", "")
        clients_db = db_data.get("clients", {})
        client_data = clients_db.get(client_id, {})

        products_db = db_data.get("products", {})

        return LoadedData(
            session_id=session_id,
            db_records={
                "client_data": client_data,
                "products_db": products_db,
                "findings": findings,
            },
            conversation_history=conversation_history,
        )

    # ── Step 2: ANALYZE ───────────────────────────────────────────────────────

    async def analyze(self, loaded_data: LoadedData) -> AnalysisResult:
        findings = loaded_data.db_records.get("findings", [])
        fields_in_db = [f"findings_count={len(findings)}", "client_data", "product_data"]
        return AnalysisResult(
            total_fields=len(findings),
            fields_in_db=fields_in_db,
            fields_missing=[],
            preliminary_data=loaded_data.db_records,
        )

    # ── Step 3: REASON ────────────────────────────────────────────────────────

    async def reason(
        self,
        analysis: AnalysisResult,
        conversation_history: list[dict],
    ) -> ReasoningResult:
        return ReasoningResult(
            questions_to_ask=[],
            fields_confirmed_from_conversation=[],
            discrepancies=[],
            field_values_extracted={},
        )

    # ── Step 4: DECIDE ────────────────────────────────────────────────────────

    async def decide(self, reasoning: ReasoningResult) -> Decision:
        return Decision(
            should_ask_questions=False,
            questions=[],
            ready_to_finalize=True,
        )

    # ── Step 5: GENERATE ──────────────────────────────────────────────────────

    async def generate(
        self,
        decision: Decision,
        loaded_data: LoadedData,
        reasoning: ReasoningResult,
    ) -> EscalationReport:
        db = loaded_data.db_records
        client_data = db.get("client_data", {})
        findings = db.get("findings", [])

        # Extract individual findings
        triage_finding = next((f for f in findings if getattr(f, "agent_id", "") == "triage_agent"), None)
        diagnostic_finding = next((f for f in findings if getattr(f, "agent_id", "") == "diagnostic_agent"), None)
        resolution_finding = next((f for f in findings if getattr(f, "agent_id", "") == "resolution_agent"), None)

        # Serialize findings for rule evaluation
        triage_dict = _serialize_finding(triage_finding)
        diagnostic_dict = _serialize_finding(diagnostic_finding)
        resolution_dict = _serialize_finding(resolution_finding)

        # Flatten issue sub-dict to top level so evaluate_escalation_rules and assemble_ticket
        # can read known_incident, product, environment, recurring, etc. directly
        if "issue" in triage_dict and isinstance(triage_dict["issue"], dict):
            for k, v in triage_dict["issue"].items():
                if k not in triage_dict:
                    triage_dict[k] = v

        # Add client tier to triage dict for rules
        if triage_finding and hasattr(triage_finding, "client") and triage_finding.client:
            triage_dict["client"] = triage_finding.client.model_dump()

        # Evaluate escalation rules
        decision_str, reason = evaluate_escalation_rules(
            triage=triage_dict,
            diagnostic=diagnostic_dict,
            resolution=resolution_dict,
            sentiment_bias=self.sentiment_bias,
        )

        # Generate fallback message
        csm = client_data.get("csm", "your account manager")
        fallback = f"If resolution steps do not resolve the issue within 30 minutes, contact {csm} for escalation."

        escalation_path = None
        csm_notified = False
        if decision_str == "escalated":
            product = ""
            if triage_finding and hasattr(triage_finding, "issue"):
                product = triage_finding.issue.product or ""
            product_eng = db.get("products_db", {}).get(product, {}).get("engineering_contact", "engineering@nexacloud.io")
            escalation_path = f"CSM: {client_data.get('csm_email', 'N/A')} → Engineering: {product_eng}"
            csm_notified = client_data.get("tier", "standard") in ("platinum", "gold")

        # Calculate escalation confidence (high since rules are deterministic)
        confidence = 0.97

        # LLM for nexus_summary
        nexus_summary = _generate_summary_fallback(
            triage_dict, diagnostic_dict, resolution_dict, client_data, decision_str
        )

        if _claude:
            try:
                prompt = build_escalation_prompt(
                    triage_report=triage_dict,
                    diagnostic_report=diagnostic_dict,
                    resolution_report=resolution_dict,
                    client_info=client_data,
                    sentiment_profile=self.sentiment_profile,
                    escalation_bias=self.sentiment_bias,
                    conversation_history=loaded_data.conversation_history,
                )
                messages = [{"role": "user", "content": "Generate the escalation decision and nexus summary."}]
                raw = await _claude.complete(system=prompt, messages=messages, max_tokens=800)
                parsed = _claude.safe_parse_json(raw)
                if parsed and parsed.get("nexus_summary"):
                    nexus_summary = parsed["nexus_summary"]
                    # Don't override rule-based decision with LLM decision
            except Exception as e:
                log.warning("escalation_agent.claude_failed", error=str(e))

        # Build escalation report
        report = EscalationReport(
            session_id=loaded_data.session_id,
            decision=decision_str,  # type: ignore[arg-type]
            reason=reason,
            fallback=fallback,
            escalation_path=escalation_path,
            csm_notified=csm_notified,
            confidence=confidence,
            questions_for_client=[],
            completed=True,
        )
        # Attach nexus_summary as metadata (stored externally on the ticket)
        report.__dict__["_nexus_summary"] = nexus_summary
        report.__dict__["_triage_dict"] = triage_dict
        report.__dict__["_diagnostic_dict"] = diagnostic_dict
        report.__dict__["_resolution_dict"] = resolution_dict

        self.log_reasoning(
            "GENERATE",
            f"decision={decision_str}, confidence={confidence:.2f}",
            f"reason={reason}",
        )
        return report


# ── Ticket assembly ────────────────────────────────────────────────────────────

def assemble_ticket(
    ticket_id: str,
    escalation_report: EscalationReport,
    session_state,
    client_data: dict,
) -> SupportTicket:
    """
    Assemble the complete SupportTicket from all agent findings.
    Called by the orchestrator after escalation_agent completes.
    """
    triage_dict = escalation_report.__dict__.get("_triage_dict", {})
    diagnostic_dict = escalation_report.__dict__.get("_diagnostic_dict", {})
    resolution_dict = escalation_report.__dict__.get("_resolution_dict", {})
    nexus_summary = escalation_report.__dict__.get("_nexus_summary", "Support ticket generated by Nexus.")

    # Client summary
    sla_deadline = ""
    if session_state and hasattr(session_state, "triage_finding") and session_state.triage_finding:
        tf = session_state.triage_finding
        if isinstance(tf, dict):
            sla_deadline = (tf.get("client") or {}).get("sla_deadline", "")

    client_summary = ClientSummary(
        company=client_data.get("company", "Unknown"),
        tier=client_data.get("tier", "standard"),  # type: ignore[arg-type]
        sla_hours=client_data.get("sla_hours", 24),
        sla_deadline=sla_deadline or _default_sla(client_data.get("sla_hours", 24)),
        csm=client_data.get("csm", ""),
        csm_notified=escalation_report.csm_notified,
        recent_tickets=len(client_data.get("recent_tickets", [])),
        vip_flag=client_data.get("vip_flag", False),
    )

    # Issue summary
    _VALID_PRODUCTS = {"NexaAuth", "NexaStore", "NexaMsg", "NexaPay"}
    product = triage_dict.get("product", "") or ""
    unknown_product = not product or product not in _VALID_PRODUCTS
    issue_summary = IssueSummary(
        product=product or "Unknown",
        error_message=triage_dict.get("error_message", ""),
        environment=triage_dict.get("environment", "production"),  # type: ignore[arg-type]
        started_at=triage_dict.get("started_at", "unknown"),
        impact_scope=triage_dict.get("impact_scope", "unknown"),
        known_incident=triage_dict.get("known_incident", False),
        recurring=triage_dict.get("recurring", False),
        inferred_fields=triage_dict.get("inferred_fields", {}),
        unknown_product=unknown_product,
    )

    # Diagnosis summary — detect novel issue (no KB or bug-db match)
    ph = (diagnostic_dict.get("primary_hypothesis") or {})
    rag_used = diagnostic_dict.get("rag_results_used") or []
    bugs_checked = diagnostic_dict.get("version_bugs_checked") or []
    _rag_no_match = len(rag_used) == 0 or all(
        (r.get("similarity", 0) if isinstance(r, dict) else getattr(r, "similarity", 0)) < 0.35
        for r in rag_used
    )
    _bugs_no_match = bool(bugs_checked) and all("no matching known bugs" in v for v in bugs_checked)
    novel_issue = _rag_no_match and (_bugs_no_match or not bugs_checked)
    diagnosis = DiagnosisSummary(
        primary_cause=ph.get("cause", "Under investigation") if isinstance(ph, dict) else str(ph),
        confidence=diagnostic_dict.get("confidence", 0.0),
        supporting_evidence=ph.get("evidence", []) if isinstance(ph, dict) else [],
        alternative_causes=[
            h if isinstance(h, dict) else h.model_dump()
            for h in (diagnostic_dict.get("alternative_hypotheses") or [])
        ],
        novel_issue=novel_issue,
    )

    # Resolution plan
    res_steps = []
    for s in (resolution_dict.get("steps") or []):
        if isinstance(s, dict):
            res_steps.append(ResolutionStep(**{k: v for k, v in s.items() if k in ResolutionStep.model_fields}))
        else:
            res_steps.append(ResolutionStep(
                step=getattr(s, "step", 1),
                action=getattr(s, "action", ""),
                command=getattr(s, "command", None),
                why=getattr(s, "why", ""),
                verify=getattr(s, "verify", ""),
                risk=getattr(s, "risk", "low"),
                production_warning=getattr(s, "production_warning", None),
            ))

    resolution = ResolutionPlan(
        estimated_resolution_time=resolution_dict.get("estimated_resolution_time", "unknown"),
        steps=res_steps,
        prevention=resolution_dict.get("prevention"),
        rag_source=resolution_dict.get("rag_source"),
        confidence=resolution_dict.get("confidence", 0.0),
    )

    # Escalation decision
    escalation_decision = EscalationDecision(
        decision=escalation_report.decision,
        reason=escalation_report.reason,
        fallback=escalation_report.fallback,
        escalation_path=escalation_report.escalation_path,
    )

    # Sentiment
    sentiment = SentimentProfile(
        detected=session_state.current_sentiment if session_state else "calm",  # type: ignore[arg-type]
        tone_adjustment_applied=session_state.current_sentiment in ("frustrated", "urgent") if session_state else False,
    )

    # Confidence breakdown
    breakdown = ConfidenceBreakdown(
        triage=triage_dict.get("confidence", 0.0),
        diagnostic=diagnostic_dict.get("confidence", 0.0),
        resolution=resolution_dict.get("confidence", 0.0),
        escalation=escalation_report.confidence,
        overall=calculate_overall_confidence([
            type("F", (), {"confidence": triage_dict.get("confidence", 0.0)})(),
            type("F", (), {"confidence": diagnostic_dict.get("confidence", 0.0)})(),
            type("F", (), {"confidence": resolution_dict.get("confidence", 0.0)})(),
        ]),
    )

    # Priority from severity
    severity = triage_dict.get("severity", "medium")
    priority_map = {"critical": "critical", "high": "high", "medium": "medium", "low": "low"}
    priority = priority_map.get(severity, "medium")

    # Status
    status = escalation_report.decision if escalation_report.decision in ("self_resolve", "escalated") else "pending"

    return SupportTicket(
        ticket_id=ticket_id,
        sla_deadline=client_summary.sla_deadline,
        status=status,  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        client=client_summary,
        issue_summary=issue_summary,
        diagnosis=diagnosis,
        resolution=resolution,
        escalation=escalation_decision,
        sentiment_analysis=sentiment,
        nexus_summary=nexus_summary,
        data_provenance={
            "triage_confidence": triage_dict.get("confidence", 0.0),
            "diagnostic_confidence": diagnostic_dict.get("confidence", 0.0),
            "resolution_confidence": resolution_dict.get("confidence", 0.0),
            "rag_source": resolution_dict.get("rag_source", ""),
            "hallucination_flags": len(
                [f for findings in [triage_dict, diagnostic_dict, resolution_dict]
                 for f in (findings.get("hallucination_flags") or [])]
            ),
        },
        confidence_breakdown=breakdown,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serialize_finding(finding) -> dict:
    if finding is None:
        return {}
    if isinstance(finding, dict):
        return finding
    try:
        return finding.model_dump()
    except Exception:
        return {}


def _generate_summary_fallback(
    triage: dict, diagnostic: dict, resolution: dict, client: dict, decision: str
) -> str:
    company = client.get("company", "The client")
    tier = client.get("tier", "standard").title()
    severity = triage.get("severity", "medium")
    est_time = (resolution or {}).get("estimated_resolution_time", "unknown")
    csm = client.get("csm", "your CSM")

    # Annotate inferred fields so reviewers know what was guessed vs. confirmed
    inferred = triage.get("inferred_fields", {})
    raw_product = triage.get("product", "a NexaCloud product")
    product = (
        f"{raw_product} (unconfirmed — client may have meant a different product)"
        if inferred.get("product") else raw_product
    )
    error = triage.get("error_message", "an error")

    cause = (diagnostic.get("primary_hypothesis") or {}).get("cause", "under investigation")
    if diagnostic.get("hypothesis_inferred"):
        cause = f"{cause} (low-confidence hypothesis — limited evidence)"

    if decision == "escalated":
        return (
            f"{company} ({tier}) is experiencing a {severity} issue on {product}: {error}. "
            f"Diagnosis: {cause}. This has been escalated to {csm} for immediate attention."
        )
    return (
        f"{company} ({tier}) is experiencing a {severity} issue on {product}: {error}. "
        f"Root cause identified: {cause}. Resolution steps provided — estimated fix time {est_time}. "
        f"Ticket generated by Nexus."
    )


def _default_sla(hours: int) -> str:
    from datetime import timedelta
    deadline = datetime.now(tz=timezone.utc) + timedelta(hours=hours)
    return deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
