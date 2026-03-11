"""
ProgressTracker — velocity-based stuck detection for agents.

Replaces the fixed run_count + confidence_delta check in NexusOrchestrator.is_stuck().
Tracks how fast confidence is improving and escalates when progress stalls or regresses.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from agents.threshold_calculator import ThresholdCalculator
from models.conversation import SessionState

if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    from langsmith import traceable
else:
    def traceable(**_):  # type: ignore[misc]
        return lambda f: f

_TIER_LIMITS: dict[str, dict[str, int]] = {
    "platinum": {"triage": 2, "diagnostic": 2, "resolution": 1},
    "gold":     {"triage": 3, "diagnostic": 3, "resolution": 2},
    "standard": {"triage": 4, "diagnostic": 4, "resolution": 3},
}


@dataclass
class ProgressAssessment:
    is_stuck: bool
    reason: str
    velocity: float
    run_count: int


class ProgressTracker:
    """
    Assesses whether an agent is making sufficient progress toward its threshold.
    Uses confidence velocity (gain per turn) instead of a fixed run-count limit.
    """

    def __init__(self, threshold_calculator: ThresholdCalculator) -> None:
        self._threshold_calculator = threshold_calculator

    @traceable(name="stuck_detection", run_type="chain")
    def assess_progress(self, agent: str, session_state: SessionState) -> ProgressAssessment:
        run_count = getattr(session_state, f"{agent}_run_count", 0)

        # First run — never stuck, no history yet
        if run_count == 0:
            return ProgressAssessment(is_stuck=False, reason="first_run", velocity=0.0, run_count=0)

        history: list[float] = getattr(session_state, f"{agent}_confidence_history", [])
        tier = session_state.client.get("tier", "standard")
        limits = _TIER_LIMITS.get(tier, _TIER_LIMITS["standard"])
        limit = limits.get(agent, 4)

        # Velocity: confidence gained in the last step
        if len(history) >= 2:
            velocity = history[-1] - history[-2]
        elif len(history) == 1:
            velocity = history[0]
        else:
            velocity = 0.0

        # Stuck condition 1: hit run limit and not improving meaningfully
        if run_count >= limit and velocity < 0.05:
            return ProgressAssessment(
                is_stuck=True,
                reason=f"hit_limit: run_count={run_count} >= limit={limit}, velocity={velocity:.3f}",
                velocity=velocity,
                run_count=run_count,
            )

        # Stuck condition 2: confidence is regressing
        if run_count >= 2 and velocity < 0.0:
            return ProgressAssessment(
                is_stuck=True,
                reason=f"regression: velocity={velocity:.3f} on run {run_count}",
                velocity=velocity,
                run_count=run_count,
            )

        # Stuck condition 3: plateau over last 3 runs
        if len(history) >= 3 and (max(history[-3:]) - min(history[-3:])) < 0.03:
            return ProgressAssessment(
                is_stuck=True,
                reason=f"plateau: last 3 readings range < 0.03 ({history[-3:]})",
                velocity=velocity,
                run_count=run_count,
            )

        return ProgressAssessment(
            is_stuck=False,
            reason="making_progress",
            velocity=velocity,
            run_count=run_count,
        )
