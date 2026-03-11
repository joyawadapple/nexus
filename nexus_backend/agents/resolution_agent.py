"""
Resolution Agent — generates a concrete, ordered resolution plan.

Phase 2b of the Nexus orchestration pipeline (runs in parallel with diagnostic_agent).
RAG-powered, validated against knowledge base.
Confidence threshold: 0.80
"""
from __future__ import annotations

import structlog

from agents.agent_base import AnalysisResult, BaseAgent, Decision, LoadedData, ReasoningResult
from agents.agent_utils import validate_agent_output
from models.agent_models import QuestionForClient
from models.report_models import ResolutionReport, ResolutionStepOut
from prompts.resolution_agent_prompt import build_resolution_prompt

log = structlog.get_logger("resolution_agent")

_claude = None
_rag_engine = None


def set_services(claude, rag_engine) -> None:
    global _claude, _rag_engine
    _claude = claude
    _rag_engine = rag_engine


class ResolutionAgent(BaseAgent):
    agent_id = "resolution_agent"
    confidence_threshold = 0.80

    def __init__(self, triage_finding=None) -> None:
        super().__init__()
        self.triage_finding = triage_finding
        self.dynamic_threshold: float | None = None

    # ── Step 1: LOAD ──────────────────────────────────────────────────────────

    async def load(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list[dict],
        **kwargs,
    ) -> LoadedData:
        triage = kwargs.get("triage_finding") or self.triage_finding
        diagnostic = kwargs.get("diagnostic_finding")

        product = ""
        error_message = ""
        environment = "production"
        recurring = False
        if triage:
            if hasattr(triage, "issue"):
                product = triage.issue.product or ""
                error_message = triage.issue.error_message or ""
                environment = triage.issue.environment or "production"
                recurring = triage.issue.recurring

        primary_cause = ""
        if diagnostic and hasattr(diagnostic, "primary_hypothesis") and diagnostic.primary_hypothesis:
            primary_cause = diagnostic.primary_hypothesis.cause

        client_id = db_data.get("_session_client_id", "")
        clients_db = db_data.get("clients", {})
        products_db = db_data.get("products", {})

        client_data = clients_db.get(client_id, {})
        product_data = products_db.get(product, {})
        integration_type = client_data.get("integration_type", "rest")
        sdk_version = client_data.get("sdk_version", "")

        # RAG retrieval — resolution category
        rag_results = []
        if _rag_engine:
            query = f"{product} {primary_cause or error_message} resolution fix steps"
            rag_results = _rag_engine.query(
                query_text=query,
                category="resolution",
                product=product,
                top_k=3,
            )

        return LoadedData(
            session_id=session_id,
            db_records={
                "product": product,
                "error_message": error_message,
                "environment": environment,
                "recurring": recurring,
                "primary_cause": primary_cause,
                "integration_type": integration_type,
                "sdk_version": sdk_version,
                "client_data": client_data,
                "product_config": product_data.get("config_reference", {}),
            },
            conversation_history=conversation_history,
            rag_context=rag_results,
        )

    # ── Step 2: ANALYZE ───────────────────────────────────────────────────────

    async def analyze(self, loaded_data: LoadedData) -> AnalysisResult:
        rag_results = loaded_data.rag_context
        fields_in_db = ["product_config", "client_integration_type"]
        if rag_results:
            fields_in_db.append("rag_resolution_docs")

        return AnalysisResult(
            total_fields=3,
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
        # Resolution agent works entirely from prior findings — no new questions needed
        return ReasoningResult(
            questions_to_ask=[],
            fields_confirmed_from_conversation=analysis.fields_in_db,
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
    ) -> ResolutionReport:
        db = loaded_data.db_records
        rag_results = loaded_data.rag_context
        product = db.get("product", "")
        environment = db.get("environment", "production")
        recurring = db.get("recurring", False)
        integration_type = db.get("integration_type", "rest")
        client_data = db.get("client_data", {})

        # ── Mathematical confidence based on RAG quality ──────────────────────
        top_sim = rag_results[0].get("similarity", 0.0) if rag_results else 0.0
        if top_sim > 0.85:
            confidence = 0.89
        elif top_sim > 0.70:
            confidence = 0.75
        elif top_sim > 0.55:
            confidence = 0.60
        else:
            confidence = 0.45

        steps: list[ResolutionStepOut] = []
        prevention = None
        rag_source = rag_results[0].get("source", "") if rag_results else ""
        has_low_confidence_steps = confidence < 0.60
        estimated_time = "30-60 minutes"

        if _claude:
            triage_summary = {
                "product": product,
                "error_message": db.get("error_message", ""),
                "environment": environment,
                "recurring": recurring,
            }
            diagnostic_summary = {
                "primary_hypothesis": {"cause": db.get("primary_cause", "unknown")},
                "confidence": 0.80,
            }

            try:
                prompt = build_resolution_prompt(
                    triage_summary=triage_summary,
                    diagnostic_summary=diagnostic_summary,
                    client_info=client_data,
                    product_config=db.get("product_config", {}),
                    rag_results=rag_results,
                    conversation_history=loaded_data.conversation_history,
                )
                messages = [{"role": "user", "content": "Generate the resolution plan."}]
                raw = await _claude.complete(system=prompt, messages=messages, max_tokens=1500)
                parsed = _claude.safe_parse_json(raw)

                if parsed:
                    estimated_time = parsed.get("estimated_resolution_time", estimated_time)
                    rag_source = parsed.get("rag_source", rag_source)
                    prevention = parsed.get("prevention")
                    has_low_confidence_steps = parsed.get("has_low_confidence_steps", False)

                    for s in parsed.get("steps", []):
                        step = ResolutionStepOut(
                            step=s.get("step", len(steps) + 1),
                            action=s.get("action", ""),
                            command=s.get("command"),
                            why=s.get("why", ""),
                            verify=s.get("verify", ""),
                            risk=s.get("risk", "low"),
                            production_warning=_add_prod_warning(s, environment),
                            confidence_level=s.get("confidence_level", "high"),
                        )
                        steps.append(step)

                    if has_low_confidence_steps:
                        confidence = min(confidence, 0.65)

            except Exception as e:
                log.warning("resolution_agent.claude_failed", error=str(e))

        # Add production warnings to risky steps not already flagged
        steps = [_maybe_add_prod_warning(s, environment) for s in steps]

        # Add prevention step for recurring issues
        if recurring and prevention:
            log.info("resolution_agent.adding_prevention_step", recurring=recurring)

        # Validate
        agent_output = {"product": product, "confidence": confidence}
        validation = validate_agent_output("resolution_agent", agent_output, db, confidence)

        report = ResolutionReport(
            session_id=loaded_data.session_id,
            estimated_resolution_time=estimated_time,
            steps=steps,
            prevention=prevention,
            rag_source=rag_source,
            confidence=confidence,
            has_low_confidence_steps=has_low_confidence_steps,
            questions_for_client=[],
            hallucination_flags=validation.flags,
            completed=confidence >= (self.dynamic_threshold if self.dynamic_threshold is not None else self.confidence_threshold),
        )

        self.log_reasoning(
            "GENERATE",
            f"confidence={confidence:.2f}, steps={len(steps)}, has_low_conf={has_low_confidence_steps}",
            f"top_rag_sim={top_sim:.2f}, integration={integration_type}",
        )

        _threshold_used = self.dynamic_threshold if self.dynamic_threshold is not None else self.confidence_threshold
        log.info(
            "resolution_agent.generate_complete",
            completed=report.completed,
            confidence=round(confidence, 4),
            threshold_used=round(_threshold_used, 4),
            dynamic_threshold=self.dynamic_threshold,
            threshold_condition_met=confidence >= _threshold_used,
            top_rag_sim=round(top_sim, 4),
            steps_count=len(steps),
            has_low_confidence_steps=has_low_confidence_steps,
        )

        return report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_prod_warning(step_dict: dict, environment: str) -> str | None:
    if environment == "production" and step_dict.get("risk", "low") in ("medium", "high"):
        return step_dict.get("production_warning") or "Test in staging environment before applying to production"
    return step_dict.get("production_warning")


def _maybe_add_prod_warning(step: ResolutionStepOut, environment: str) -> ResolutionStepOut:
    if environment == "production" and step.risk in ("medium", "high") and not step.production_warning:
        step.production_warning = "Test in staging before applying to production"
    return step
