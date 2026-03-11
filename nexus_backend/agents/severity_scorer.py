"""
SeverityScorer — derives issue severity from a weighted combination of context signals.

Replaces the hardcoded if/else in TriageAgent.apply_severity_rules().
Adding a new signal requires one line — no logic changes needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeverityResult:
    severity: str          # critical / high / medium / low
    score: float           # normalised 0-1
    fired_signals: list[str] = field(default_factory=list)
    reasoning: str = ""


# Each signal: weight (additive) + detect lambda that takes context dict.
# Context keys: tier, environment, impact_scope, error_message, recurring
SEVERITY_SIGNALS: dict[str, dict] = {
    "platinum_tier": {
        "weight": 0.30,
        "detect": lambda ctx: ctx.get("tier") == "platinum",
    },
    "production_env": {
        "weight": 0.25,
        "detect": lambda ctx: ctx.get("environment") in ("production", "unknown", None),
    },
    "all_users_impact": {
        "weight": 0.20,
        "detect": lambda ctx: ctx.get("impact_scope") == "all_users",
    },
    "server_error": {
        "weight": 0.15,
        "detect": lambda ctx: "500" in str(ctx.get("error_message", "")),
    },
    "auth_failure": {
        "weight": 0.10,
        "detect": lambda ctx: "401" in str(ctx.get("error_message", "")),
    },
    "recurring_issue": {
        "weight": 0.10,
        "detect": lambda ctx: ctx.get("recurring") is True,
    },
}

_SEVERITY_BANDS = [
    (0.55, "critical"),
    (0.40, "high"),
    (0.20, "medium"),
    (0.00, "low"),
]


class SeverityScorer:
    """
    Derives severity from a weighted combination of signals.
    Each signal contributes its weight when its detect() returns True.
    Score is normalised to 0-1 and mapped to a severity band.
    """

    def score(self, context: dict) -> SeverityResult:
        total = 0.0
        fired: list[str] = []

        for name, signal in SEVERITY_SIGNALS.items():
            try:
                if signal["detect"](context):
                    total += signal["weight"]
                    fired.append(name)
            except Exception:
                pass  # signal failure never breaks scoring

        normalised = min(total, 1.0)

        severity = "low"
        for threshold, label in _SEVERITY_BANDS:
            if normalised >= threshold:
                severity = label
                break

        reasoning = f"score={normalised:.2f} from [{', '.join(fired)}]" if fired else "score=0.00 no signals fired"

        return SeverityResult(
            severity=severity,
            score=normalised,
            fired_signals=fired,
            reasoning=reasoning,
        )
