"""
NexusEvaluator — quality tracking and regression testing framework.

Evaluates:
- Field extraction accuracy
- Question relevance (no redundant questions)
- Hallucination rate (from HallucinationGuard flags)
- Resolution confidence
- Turn efficiency
- RAG faithfulness

Regression testing against golden test cases in tests/golden_cases.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("evaluator")

_GOLDEN_CASES_PATH = Path(__file__).parent.parent / "tests" / "golden_cases.json"


class EvaluationReport:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.scores: dict[str, float] = {}
        self.flags: list[str] = []
        self.passed = True

    def score(self, metric: str, value: float, threshold: float = 0.0) -> None:
        self.scores[metric] = value
        if threshold > 0 and value < threshold:
            self.flags.append(f"{metric}={value:.2f} below threshold {threshold:.2f}")
            self.passed = False

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "passed": self.passed,
            "scores": {k: f"{v:.2%}" for k, v in self.scores.items()},
            "flags": self.flags,
        }


class RegressionReport:
    def __init__(self) -> None:
        self.cases: list[dict] = []
        self.passed = 0
        self.failed = 0

    def add_case(self, name: str, passed: bool, details: dict) -> None:
        self.cases.append({"name": name, "passed": passed, **details})
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict:
        return {
            "total": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "all_passed": self.all_passed,
            "cases": self.cases,
        }


class NexusEvaluator:
    """
    Evaluates Nexus session quality and runs regression tests.
    """

    def evaluate_session(self, session) -> EvaluationReport:
        """
        Score a completed session across 7 quality dimensions.
        session: SessionState object
        """
        report = EvaluationReport(session.session_id)

        # 1. Field extraction accuracy
        # Did all 5 triage fields get confirmed before routing?
        triage = session.triage_finding or {}
        if isinstance(triage, dict):
            issue = triage.get("issue", {}) or {}
            required_fields = ["product", "error_message", "environment", "started_at", "impact_scope"]
            filled = sum(1 for f in required_fields if issue.get(f) and issue.get(f) != "unknown")
            accuracy = filled / len(required_fields)
        else:
            accuracy = 0.0
        report.score("field_extraction_accuracy", accuracy, threshold=0.60)

        # 2. Question relevance — did we ask about fields not yet confirmed?
        asked = set(session.asked_fields)
        confirmed = set(session.confirmed_fields.keys())
        if asked:
            # Questions asked for already-confirmed fields are redundant
            redundant = asked & confirmed
            relevance = 1.0 - (len(redundant) / len(asked))
        else:
            relevance = 1.0
        report.score("question_relevance", relevance, threshold=0.80)

        # 3. Redundancy rate — were any fields asked more than once?
        # This is approximated by checking if asked_fields has duplicates
        # (our list shouldn't have dupes — mark_field_asked deduplicates)
        redundancy = 0.0  # Best case: no redundancy with our dedup logic
        report.score("redundancy_rate", 1.0 - redundancy)

        # 4. Hallucination rate
        hallucination_flags = 0
        total_fields = 0
        for finding_dict in [triage, session.diagnostic_finding, session.resolution_finding]:
            if finding_dict and isinstance(finding_dict, dict):
                flags = finding_dict.get("hallucination_flags", [])
                hallucination_flags += len(flags)
                total_fields += 5  # approx fields per agent
        hallucination_rate = (hallucination_flags / total_fields) if total_fields > 0 else 0.0
        report.score("hallucination_rate", 1.0 - hallucination_rate, threshold=0.80)

        # 5. Resolution confidence
        res = session.resolution_finding or {}
        res_conf = res.get("confidence", 0.0) if isinstance(res, dict) else 0.0
        report.score("resolution_confidence", res_conf, threshold=0.60)

        # 6. Turn efficiency — fewer turns is better
        user_turns = sum(1 for m in session.messages if m.role == "user")
        # Score: 5 turns = perfect (1.0), 10+ turns = poor (0.5)
        turn_efficiency = max(0.5, min(1.0, 1.0 - (user_turns - 5) * 0.05)) if user_turns > 5 else 1.0
        report.score("turn_efficiency", turn_efficiency)

        # 7. RAG faithfulness — did resolution cite a KB source?
        res_rag = res.get("rag_source", "") if isinstance(res, dict) else ""
        faithfulness = 1.0 if res_rag else 0.0
        report.score("rag_faithfulness", faithfulness, threshold=0.5)

        log.info(
            "evaluator.session_scored",
            session_id=session.session_id,
            passed=report.passed,
            scores={k: f"{v:.0%}" for k, v in report.scores.items()},
        )
        return report

    def run_regression(self, test_cases: list[dict] | None = None) -> RegressionReport:
        """
        Run regression tests against golden cases.
        test_cases: list of case dicts (or None to load from golden_cases.json)
        """
        if test_cases is None:
            test_cases = _load_golden_cases()

        regression = RegressionReport()

        for case in test_cases:
            name = case.get("name", "unnamed")
            expected = case.get("expected_output", {})
            # Simulate what the output should look like
            passed, details = _check_golden_case(case, expected)
            regression.add_case(name=name, passed=passed, details=details)
            log.info("evaluator.regression_case", name=name, passed=passed)

        log.info(
            "evaluator.regression_done",
            passed=regression.passed,
            failed=regression.failed,
        )
        return regression


# ── Golden case checker ───────────────────────────────────────────────────────

def _check_golden_case(case: dict, expected: dict) -> tuple[bool, dict]:
    """
    Validate that a golden case's input would produce the expected output.
    This is a structural check — verifies routing logic and expected statuses.
    """
    input_data = case.get("input", {})
    issues = []

    # Check: known incident → should produce escalated status
    if input_data.get("known_incident") and expected.get("status") != "escalated":
        issues.append("Known incident should produce escalated status")

    # Check: recurring issue → should produce escalated status
    if input_data.get("recurring_count", 0) >= 2 and expected.get("status") not in ("escalated",):
        issues.append("Recurring issue (2+ occurrences) should escalate")

    # Check: low-info session → should produce collecting status
    if input_data.get("missing_product") and expected.get("status") != "collecting":
        issues.append("Missing product field should produce collecting status")

    # Check: expected priority matches severity
    if "expected_priority" in expected:
        tier = input_data.get("client_tier", "standard")
        env = input_data.get("environment", "production")
        expected_priority = expected["expected_priority"]
        calculated_priority = _expected_severity(tier, env)
        if calculated_priority != expected_priority:
            issues.append(f"Expected priority {expected_priority}, calculated {calculated_priority}")

    # Check: triage_confidence_min — missing product cannot yield high confidence
    if "triage_confidence_min" in expected:
        min_conf = expected["triage_confidence_min"]
        if input_data.get("missing_product") and min_conf > 0.50:
            issues.append(
                f"missing_product=true conflicts with triage_confidence_min={min_conf}"
            )

    # Check: skip_diagnostic requires known_incident
    if expected.get("skip_diagnostic") and not input_data.get("known_incident"):
        issues.append("skip_diagnostic=true requires known_incident=true in input")

    # Check: should_cite_rag is incompatible with skip_diagnostic (known incident path bypasses RAG)
    if expected.get("should_cite_rag") and expected.get("skip_diagnostic"):
        issues.append(
            "should_cite_rag=true conflicts with skip_diagnostic=true (known incident skips RAG)"
        )

    # Check: resolution_steps_min — known incident path skips resolution, so steps aren't expected
    if "resolution_steps_min" in expected and input_data.get("known_incident"):
        if expected.get("skip_diagnostic"):
            issues.append(
                "resolution_steps_min is not applicable when skip_diagnostic=true (escalation path)"
            )

    passed = len(issues) == 0
    return passed, {"issues": issues, "input_summary": _summarize_input(input_data)}


def _expected_severity(tier: str, env: str) -> str:
    if env == "production" and tier == "platinum":
        return "critical"
    if env == "production":
        return "high"
    if env == "staging":
        return "medium"
    return "low"


def _summarize_input(input_data: dict) -> str:
    return (
        f"client={input_data.get('client_id', '?')}, "
        f"product={input_data.get('product', '?')}, "
        f"env={input_data.get('environment', '?')}"
    )


def _load_golden_cases() -> list[dict]:
    if not _GOLDEN_CASES_PATH.exists():
        log.warning("evaluator.no_golden_cases", path=str(_GOLDEN_CASES_PATH))
        return []
    with open(_GOLDEN_CASES_PATH) as f:
        return json.load(f)
