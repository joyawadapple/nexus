"""
ThresholdCalculator — derives contextual confidence thresholds from session state.

Replaces hardcoded per-agent thresholds (0.90 / 0.75 / 0.80) with values that
adapt to severity, sentiment, tier, conversation length, and recurring flag.
"""
from __future__ import annotations

import os

from agents.agent_utils import SENTIMENT_PROFILES
from models.conversation import SessionState

if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    from langsmith import traceable
else:
    def traceable(**_):  # type: ignore[misc]
        return lambda f: f

_BASE_THRESHOLDS: dict[str, float] = {
    "triage": 0.85,
    "diagnostic": 0.70,
    "resolution": 0.75,
}

_SEVERITY_ADJ: dict[str, float] = {
    "critical": -0.10,
    "high": -0.05,
    "medium": 0.00,
    "low": +0.05,
}


_TIER_ADJ: dict[str, float] = {
    "platinum": -0.05,
    "gold": 0.00,
    "standard": +0.03,
}


class ThresholdCalculator:
    """
    Calculates per-agent confidence thresholds from session context.
    Higher stakes (critical, platinum) → threshold adjusts down for faster resolution.
    More information (longer conversation) → threshold relaxes slightly.
    All results clamped to [0.50, 0.95].
    """

    @traceable(name="threshold_calculation", run_type="chain")
    def calculate(self, agent: str, session_state: SessionState) -> float:
        base = _BASE_THRESHOLDS.get(agent, 0.75)
        total_adj = 0.0

        total_adj += _SEVERITY_ADJ.get(session_state.severity, 0.0)
        # Derive from SENTIMENT_PROFILES — single source of truth for escalation bias values.
        # escalation_bias is defined as "lower confidence threshold by this much", hence negation.
        total_adj -= SENTIMENT_PROFILES.get(
            session_state.current_sentiment, {}
        ).get("escalation_bias", 0.0)

        tier = session_state.client.get("tier", "standard")
        total_adj += _TIER_ADJ.get(tier, 0.0)

        if len(session_state.messages) > 10:
            total_adj -= 0.03

        if session_state.recurring:
            total_adj -= 0.05

        return round(max(0.50, min(0.95, base + total_adj)), 2)
