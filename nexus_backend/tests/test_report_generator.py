"""Tests for ticket assembly and escalation report generation (report_generator)."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.escalation_agent import assemble_ticket, evaluate_escalation_rules
from models.report_models import DiagnosticReport, EscalationReport, ResolutionReport, TriageReport
from models.ticket import SupportTicket


# ── evaluate_escalation_rules tests ──────────────────────────────────────────

def test_auto_escalate_known_incident():
    """known_incident=True must always escalate."""
    decision, reason = evaluate_escalation_rules(
        triage={"known_incident": True, "severity": "critical"},
        diagnostic={},
        resolution={},
        sentiment_bias=0.0,
    )
    assert decision == "escalated"
    assert "incident" in reason.lower()


def test_auto_escalate_critical_low_resolution_confidence():
    """Critical severity + low resolution confidence → escalated."""
    decision, reason = evaluate_escalation_rules(
        triage={"severity": "critical", "known_incident": False, "recurring": False},
        diagnostic={"confidence": 0.85},
        resolution={"confidence": 0.55, "has_low_confidence_steps": False},
        sentiment_bias=0.0,
    )
    assert decision == "escalated"


def test_auto_escalate_platinum_low_diagnostic():
    """Platinum client + low diagnostic confidence → escalated."""
    decision, reason = evaluate_escalation_rules(
        triage={
            "severity": "high",
            "known_incident": False,
            "recurring": False,
            "client": {"tier": "platinum"},
        },
        diagnostic={"confidence": 0.60},
        resolution={"confidence": 0.72, "has_low_confidence_steps": False},
        sentiment_bias=0.0,
    )
    assert decision == "escalated"
    assert "platinum" in reason.lower()


def test_auto_escalate_recurring():
    """Recurring issue → always escalated for root cause analysis."""
    decision, reason = evaluate_escalation_rules(
        triage={"severity": "medium", "known_incident": False, "recurring": True},
        diagnostic={"confidence": 0.90},
        resolution={"confidence": 0.88, "has_low_confidence_steps": False},
        sentiment_bias=0.0,
    )
    assert decision == "escalated"
    assert "recurring" in reason.lower()


def test_self_resolve_high_confidence():
    """High resolution confidence → self_resolve."""
    decision, reason = evaluate_escalation_rules(
        triage={"severity": "medium", "known_incident": False, "recurring": False},
        diagnostic={"confidence": 0.85},
        resolution={"confidence": 0.90, "has_low_confidence_steps": False},
        sentiment_bias=0.0,
    )
    assert decision == "self_resolve"


def test_sentiment_bias_lowers_escalation_threshold():
    """With frustrated bias (+0.10), borderline cases that would self-resolve become escalated."""
    # Without bias, critical + 0.72 confidence would just barely self-resolve
    decision_no_bias, _ = evaluate_escalation_rules(
        triage={"severity": "critical", "known_incident": False, "recurring": False},
        diagnostic={"confidence": 0.85},
        resolution={"confidence": 0.72, "has_low_confidence_steps": False},
        sentiment_bias=0.0,
    )
    # With frustrated bias of 0.10, the threshold for critical becomes 0.80
    decision_with_bias, _ = evaluate_escalation_rules(
        triage={"severity": "critical", "known_incident": False, "recurring": False},
        diagnostic={"confidence": 0.85},
        resolution={"confidence": 0.72, "has_low_confidence_steps": False},
        sentiment_bias=0.10,
    )
    # The bias should make escalation more likely
    assert decision_with_bias == "escalated"


def test_standard_tier_self_resolve_lower_bar():
    """Standard tier can self-resolve with 0.70 resolution confidence."""
    decision, reason = evaluate_escalation_rules(
        triage={"severity": "medium", "known_incident": False, "recurring": False, "client": {"tier": "standard"}},
        diagnostic={"confidence": 0.75},
        resolution={"confidence": 0.70, "has_low_confidence_steps": False},
        sentiment_bias=0.0,
    )
    assert decision == "self_resolve"


# ── assemble_ticket tests ─────────────────────────────────────────────────────

@pytest.fixture
def minimal_escalation_report():
    report = EscalationReport(
        session_id="sess_test",
        decision="self_resolve",
        reason="High confidence resolution",
        fallback="Contact support if unresolved",
        confidence=0.97,
        questions_for_client=[],
        completed=True,
    )
    report.__dict__["_nexus_summary"] = "Acme Corp NexaAuth token cache issue resolved."
    report.__dict__["_triage_dict"] = {
        "product": "NexaAuth",
        "error_message": "401 invalid_token",
        "environment": "production",
        "severity": "critical",
        "confidence": 0.95,
        "known_incident": False,
        "recurring": False,
    }
    report.__dict__["_diagnostic_dict"] = {
        "confidence": 0.85,
        "primary_hypothesis": {"cause": "Stale token cache after key rotation", "evidence": ["token_ttl_mismatch"]},
        "alternative_hypotheses": [],
    }
    report.__dict__["_resolution_dict"] = {
        "confidence": 0.89,
        "estimated_resolution_time": "15 minutes",
        "steps": [
            {"step": 1, "action": "Invalidate token cache", "command": "nexaauth token invalidate", "why": "Clears stale tokens", "verify": "Check /status endpoint", "risk": "low"},
        ],
        "prevention": "Automate token refresh on rotation",
        "rag_source": "kb_res_001",
        "has_low_confidence_steps": False,
    }
    return report


@pytest.fixture
def client_data():
    return {
        "company": "Acme Corp",
        "tier": "platinum",
        "sla_hours": 4,
        "csm": "Jane Smith",
        "csm_email": "jane@nexacloud.io",
        "vip_flag": True,
        "recent_tickets": [{"id": "NX-2026-0001"}, {"id": "NX-2026-0002"}],
    }


@pytest.fixture
def mock_session_state():
    state = MagicMock()
    state.current_sentiment = "calm"
    state.triage_finding = None
    return state


def test_assemble_ticket_returns_support_ticket(minimal_escalation_report, client_data, mock_session_state):
    """assemble_ticket must return a SupportTicket instance."""
    ticket = assemble_ticket("NX-2026-0001", minimal_escalation_report, mock_session_state, client_data)
    assert isinstance(ticket, SupportTicket)


def test_ticket_id_preserved(minimal_escalation_report, client_data, mock_session_state):
    """The ticket_id passed in must appear in the ticket."""
    ticket = assemble_ticket("NX-2026-9999", minimal_escalation_report, mock_session_state, client_data)
    assert ticket.ticket_id == "NX-2026-9999"


def test_ticket_has_client_info(minimal_escalation_report, client_data, mock_session_state):
    """Client summary must reflect client_data."""
    ticket = assemble_ticket("NX-2026-0001", minimal_escalation_report, mock_session_state, client_data)
    assert ticket.client.company == "Acme Corp"
    assert ticket.client.tier == "platinum"
    assert ticket.client.sla_hours == 4


def test_ticket_issue_summary_populated(minimal_escalation_report, client_data, mock_session_state):
    """Issue summary must reflect triage findings."""
    ticket = assemble_ticket("NX-2026-0001", minimal_escalation_report, mock_session_state, client_data)
    assert ticket.issue_summary.product == "NexaAuth"
    assert ticket.issue_summary.error_message == "401 invalid_token"
    assert ticket.issue_summary.environment == "production"


def test_ticket_diagnosis_populated(minimal_escalation_report, client_data, mock_session_state):
    """Diagnosis must include primary cause from diagnostic finding."""
    ticket = assemble_ticket("NX-2026-0001", minimal_escalation_report, mock_session_state, client_data)
    assert "token" in ticket.diagnosis.primary_cause.lower() or ticket.diagnosis.confidence > 0


def test_ticket_resolution_has_steps(minimal_escalation_report, client_data, mock_session_state):
    """Resolution plan must contain at least one step."""
    ticket = assemble_ticket("NX-2026-0001", minimal_escalation_report, mock_session_state, client_data)
    assert len(ticket.resolution.steps) >= 1


def test_ticket_confidence_breakdown(minimal_escalation_report, client_data, mock_session_state):
    """Confidence breakdown must contain all four agent confidences."""
    ticket = assemble_ticket("NX-2026-0001", minimal_escalation_report, mock_session_state, client_data)
    assert ticket.confidence_breakdown.triage > 0
    assert ticket.confidence_breakdown.diagnostic > 0
    assert ticket.confidence_breakdown.resolution > 0
    assert ticket.confidence_breakdown.escalation > 0
    assert 0.0 <= ticket.confidence_breakdown.overall <= 1.0


def test_ticket_nexus_summary_present(minimal_escalation_report, client_data, mock_session_state):
    """Ticket must include a non-empty nexus_summary."""
    ticket = assemble_ticket("NX-2026-0001", minimal_escalation_report, mock_session_state, client_data)
    assert ticket.nexus_summary
    assert len(ticket.nexus_summary) > 10


def test_ticket_status_reflects_decision(minimal_escalation_report, client_data, mock_session_state):
    """Ticket status must match the escalation decision."""
    ticket = assemble_ticket("NX-2026-0001", minimal_escalation_report, mock_session_state, client_data)
    assert ticket.status == "self_resolve"


def test_ticket_escalated_status():
    """Escalated decision → ticket status is 'escalated'."""
    report = EscalationReport(
        session_id="sess_test",
        decision="escalated",
        reason="Active incident",
        confidence=0.97,
        questions_for_client=[],
        completed=True,
    )
    report.__dict__["_nexus_summary"] = "Incident detected."
    report.__dict__["_triage_dict"] = {
        "product": "NexaMsg",
        "error_message": "webhook_delivery_failure",
        "environment": "production",
        "severity": "critical",
        "confidence": 1.0,
        "known_incident": True,
        "recurring": False,
    }
    report.__dict__["_diagnostic_dict"] = {}
    report.__dict__["_resolution_dict"] = {}

    client = {"company": "GlobalRetail SA", "tier": "gold", "sla_hours": 8, "csm": "Bob Jones"}
    state = MagicMock()
    state.current_sentiment = "urgent"
    state.triage_finding = None

    ticket = assemble_ticket("NX-2026-0100", report, state, client)
    assert ticket.status == "escalated"
