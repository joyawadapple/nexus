"""
Agent utilities shared across all Nexus agents.

Provides:
  - SentimentAnalyzer — VADER-based sentiment with NexaCloud-specific profiles
  - HallucinationGuard — validates agent output against DB data
  - bundle_questions — deduplication and prioritization of questions
  - calculate_overall_confidence — aggregate confidence across all agents
"""
from __future__ import annotations

import re
from typing import Any

import structlog
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from models.agent_models import HallucinationFlag, QuestionForClient, ValidationResult

log = structlog.get_logger("agent_utils")

# ── Sentiment profiles ─────────────────────────────────────────────────────────

SENTIMENT_PROFILES = {
    "frustrated": {
        "triggers": re.compile(
            r"\b(not working|broken|useless|losing money|urgent|critical|angry|unacceptable|"
            r"annoying|already|ridiculous|terrible|awful|waste|horrible|outage|down|failure)\b",
            re.IGNORECASE,
        ),
        "response_adjustment": "acknowledge_first",
        "sla_communication": "proactive",
        "escalation_bias": 0.10,   # Lower confidence threshold by this much
    },
    "urgent": {
        "triggers": re.compile(
            r"\b(down|outage|all users|production|revenue|cannot|can't|blocked|"
            r"emergency|immediately|asap|right now|critical|p0|p1|incident)\b",
            re.IGNORECASE,
        ),
        "response_adjustment": "concise_and_fast",
        "sla_communication": "immediate",
        "escalation_bias": 0.15,
    },
    "calm": {
        "triggers": None,  # Default — no trigger keywords
        "response_adjustment": "standard",
        "sla_communication": "on_request",
        "escalation_bias": 0.0,
    },
}


class SentimentAnalyzer:
    """
    Analyzes client message sentiment using VADER + NexaCloud keyword profiles.
    Returns: label, compound score, escalation_bias.
    """

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]

        # Check NexaCloud-specific profiles first (priority order: urgent > frustrated > calm)
        label = "calm"
        escalation_bias = 0.0

        for profile_name in ("urgent", "frustrated"):
            profile = SENTIMENT_PROFILES[profile_name]
            if profile["triggers"] and profile["triggers"].search(text):
                label = profile_name
                escalation_bias = profile["escalation_bias"]
                break

        # Fall back to VADER compound if no keyword match
        if label == "calm":
            if compound < -0.5:
                label = "frustrated"
                escalation_bias = SENTIMENT_PROFILES["frustrated"]["escalation_bias"]
            elif compound < -0.3:
                label = "frustrated"
                escalation_bias = SENTIMENT_PROFILES["frustrated"]["escalation_bias"] * 0.5

        return {
            "label": label,
            "compound": compound,
            "escalation_bias": escalation_bias,
            "positive": scores["pos"],
            "neutral": scores["neu"],
            "negative": scores["neg"],
        }


# Module-level singleton
_sentiment_analyzer = SentimentAnalyzer()


def analyze_sentiment(text: str) -> dict:
    return _sentiment_analyzer.analyze(text)


# ── HallucinationGuard ────────────────────────────────────────────────────────

class HallucinationGuard:
    """
    Validates agent output against loaded DB data to detect:
    1. Invented values — agent claims a product/error/cause not in DB
    2. Confidence inflation — agent confidence > calculated_confidence + 0.10
    """

    VALID_PRODUCTS = {"NexaAuth", "NexaStore", "NexaMsg", "NexaPay"}

    def validate(
        self,
        agent_id: str,
        agent_output: dict,
        db_data: dict,
        calculated_confidence: float,
    ) -> ValidationResult:
        flags: list[HallucinationFlag] = []

        # 1. Check product is real
        product = agent_output.get("product") or (
            agent_output.get("issue", {}).get("product") if isinstance(agent_output.get("issue"), dict) else None
        )
        if product and product not in self.VALID_PRODUCTS:
            flags.append(HallucinationFlag(
                field="product",
                agent_said=product,
                db_has=list(self.VALID_PRODUCTS),
                flag_type="invented_value",
                severity="high",
            ))

        # 2. Check error code exists in error_db for triage agent
        if agent_id == "triage_agent":
            issue = agent_output.get("issue") or {}
            if isinstance(issue, dict):
                error_code = issue.get("error_code") or ""
                if product and error_code:
                    product_errors = (db_data.get("error_db") or {}).get(product, {})
                    # Only flag snake_case codes like "401_invalid_token", not raw user messages
                    if product_errors and "_" in error_code and error_code not in product_errors:
                        flags.append(HallucinationFlag(
                            field="issue.error_code",
                            agent_said=error_code,
                            db_has=list(product_errors.keys()),
                            flag_type="invented_value",
                            severity="high",
                        ))

        # 3. Check primary cause exists in error_db for diagnostic agent
        if agent_id == "diagnostic_agent":
            primary_hyp = agent_output.get("primary_hypothesis", {})
            if isinstance(primary_hyp, dict):
                cause = primary_hyp.get("cause", "")
                # Soft check: flag if cause mentions a product that's not the triaged product
                if product and cause:
                    for other_product in self.VALID_PRODUCTS:
                        if other_product != product and other_product.lower() in cause.lower():
                            flags.append(HallucinationFlag(
                                field="primary_hypothesis.cause",
                                agent_said=cause[:100],
                                flag_type="out_of_scope",
                                severity="medium",
                            ))
                            break

                # Ground check: cause should appear in known_causes for some error in this product
                if cause and product:
                    product_errors = (db_data.get("error_db") or {}).get(product, {})
                    all_known_causes = [
                        c for entry in product_errors.values()
                        for c in (entry.get("known_causes") or [])
                    ]
                    if all_known_causes and not any(
                        cause.lower() in kc.lower() or kc.lower() in cause.lower()
                        for kc in all_known_causes
                    ):
                        flags.append(HallucinationFlag(
                            field="primary_hypothesis.cause",
                            agent_said=cause[:120],
                            db_has=all_known_causes[:5],
                            flag_type="invented_value",
                            severity="medium",
                        ))

        # 4. Check confidence inflation
        claimed_confidence = agent_output.get("confidence", 0.0)
        clamped: float | None = None
        if isinstance(claimed_confidence, (int, float)):
            if claimed_confidence > calculated_confidence + 0.10:
                clamped = min(calculated_confidence + 0.10, 1.0)
                flags.append(HallucinationFlag(
                    field="confidence",
                    agent_said=claimed_confidence,
                    db_has=f"calculated={calculated_confidence:.2f}, max_allowed={clamped:.2f}",
                    flag_type="confidence_inflation",
                    severity="medium",
                ))

        return ValidationResult(
            valid=len(flags) == 0,
            flags=flags,
            clamped_confidence=clamped,
        )


# Module-level singleton
_hallucination_guard = HallucinationGuard()


def validate_agent_output(
    agent_id: str,
    agent_output: dict,
    db_data: dict,
    calculated_confidence: float,
) -> ValidationResult:
    return _hallucination_guard.validate(agent_id, agent_output, db_data, calculated_confidence)


# ── Question bundling ─────────────────────────────────────────────────────────

def bundle_questions(
    findings_list: list,
    asked_fields: list[str],
    max_questions: int = 2,
) -> list[QuestionForClient]:
    """
    Collect questions from all agent findings, deduplicate, prioritize, and cap at max_questions.

    Priority order: blocking questions first, then high-priority.
    Skip any field already asked.
    """
    all_questions: list[QuestionForClient] = []
    seen_fields: set[str] = set(asked_fields)

    for finding in findings_list:
        if not finding:
            continue
        questions = getattr(finding, "questions_for_client", [])
        for q in questions:
            if isinstance(q, dict):
                q = QuestionForClient(**q)
            if q.field not in seen_fields:
                all_questions.append(q)
                seen_fields.add(q.field)

    # Sort: blocking questions first, then high priority — reverse=True so True (1) sorts first
    all_questions.sort(
        key=lambda q: (q.blocking, q.priority == "high"),
        reverse=True,
    )

    # HARD CAP — never exceed max_questions regardless of what agents request
    return all_questions[:max_questions]


# ── Overall confidence ────────────────────────────────────────────────────────

def calculate_overall_confidence(findings_list: list) -> float:
    """
    Geometric mean of all agent confidence scores.
    Returns 0.0 if no findings provided.
    """
    confidences = [
        getattr(f, "confidence", 0.0)
        for f in findings_list
        if f is not None and getattr(f, "confidence", 0.0) > 0
    ]
    if not confidences:
        return 0.0
    product = 1.0
    for c in confidences:
        product *= c
    return round(product ** (1 / len(confidences)), 4)
