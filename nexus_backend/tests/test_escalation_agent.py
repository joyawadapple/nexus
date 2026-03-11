"""Tests for EscalationAgent."""
import pytest

from agents.escalation_agent import EscalationAgent, evaluate_escalation_rules, set_claude_client


@pytest.fixture(autouse=True)
def inject_claude(mock_claude_client):
    set_claude_client(mock_claude_client)


def test_auto_escalates_critical_severity_low_confidence():
    """Critical severity with low resolution confidence → escalated."""
    triage = {"severity": "critical", "known_incident": False, "recurring": False, "environment": "production"}
    diagnostic = {"confidence": 0.70}
    resolution = {"confidence": 0.55, "has_low_confidence_steps": False, "estimated_resolution_time": "20 min"}
    decision, reason = evaluate_escalation_rules(triage, diagnostic, resolution, sentiment_bias=0.0)
    assert decision == "escalated"


def test_auto_escalates_platinum_low_diagnostic_confidence():
    """Platinum client with diagnostic confidence < 0.75 → escalated."""
    triage = {
        "severity": "high",
        "known_incident": False,
        "recurring": False,
        "environment": "production",
        "client": {"tier": "platinum"},
    }
    diagnostic = {"confidence": 0.60}
    resolution = {"confidence": 0.85, "has_low_confidence_steps": False, "estimated_resolution_time": "15 min"}
    decision, reason = evaluate_escalation_rules(triage, diagnostic, resolution, sentiment_bias=0.0)
    assert decision == "escalated"


def test_self_resolves_high_confidence_standard_tier():
    """Standard tier with high confidence → self_resolve."""
    triage = {
        "severity": "high",
        "known_incident": False,
        "recurring": False,
        "environment": "production",
        "client": {"tier": "standard"},
    }
    diagnostic = {"confidence": 0.88}
    resolution = {"confidence": 0.92, "has_low_confidence_steps": False, "estimated_resolution_time": "15 min"}
    decision, reason = evaluate_escalation_rules(triage, diagnostic, resolution, sentiment_bias=0.0)
    assert decision == "self_resolve"


def test_escalates_known_incident():
    """Known incident always escalates regardless of confidence."""
    triage = {"severity": "high", "known_incident": True, "recurring": False, "environment": "production"}
    diagnostic = {"confidence": 0.95}
    resolution = {"confidence": 0.95, "has_low_confidence_steps": False, "estimated_resolution_time": "5 min"}
    decision, reason = evaluate_escalation_rules(triage, diagnostic, resolution, sentiment_bias=0.0)
    assert decision == "escalated"
    assert "incident" in reason.lower()


def test_escalates_recurring_issue():
    """Recurring issue → escalated for root cause analysis."""
    triage = {"severity": "high", "known_incident": False, "recurring": True, "environment": "production"}
    diagnostic = {"confidence": 0.88}
    resolution = {"confidence": 0.88, "has_low_confidence_steps": False, "estimated_resolution_time": "15 min"}
    decision, reason = evaluate_escalation_rules(triage, diagnostic, resolution, sentiment_bias=0.0)
    assert decision == "escalated"


@pytest.mark.asyncio
async def test_generates_complete_ticket(
    sample_db_data,
    triage_finding_fixture,
    diagnostic_finding_fixture,
    resolution_finding_fixture,
):
    """Escalation agent should always produce a complete EscalationReport."""
    agent = EscalationAgent()
    history = [{"role": "user", "content": "401 NexaAuth production critical"}]
    result = await agent.run(
        "sess_test",
        sample_db_data,
        history,
        findings=[triage_finding_fixture, diagnostic_finding_fixture, resolution_finding_fixture],
        sentiment_profile="calm",
        sentiment_bias=0.0,
    )
    assert result.agent_id == "escalation_agent"
    assert result.decision in ("self_resolve", "escalated", "pending")
    assert result.reason != ""
    assert result.confidence > 0.0


def test_sentiment_bias_affects_threshold():
    """Frustrated client bias should lower escalation threshold — catch borderline cases."""
    triage = {
        "severity": "high",
        "known_incident": False,
        "recurring": False,
        "environment": "production",
        "client": {"tier": "platinum"},
    }
    diagnostic = {"confidence": 0.78}  # Just above 0.75 normally...
    resolution = {"confidence": 0.85, "has_low_confidence_steps": False, "estimated_resolution_time": "15 min"}

    # Without bias: diag 0.78 >= threshold 0.75 → would self_resolve
    decision_no_bias, _ = evaluate_escalation_rules(triage, diagnostic, resolution, sentiment_bias=0.0)

    # With frustrated bias +0.10: threshold becomes 0.85, diag 0.78 < 0.85 → escalated
    decision_with_bias, _ = evaluate_escalation_rules(triage, diagnostic, resolution, sentiment_bias=0.10)
    assert decision_with_bias == "escalated"
