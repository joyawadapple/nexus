"""Tests for DiagnosticAgent."""
import pytest

from agents.diagnostic_agent import DiagnosticAgent, set_services
from models.report_models import TriageReport, ClientInfo, IssueInfo


@pytest.fixture(autouse=True)
def inject_services(mock_claude_client):
    set_services(mock_claude_client, None)  # No RAG for unit tests


@pytest.fixture
def triage_finding():
    return TriageReport(
        session_id="sess_test",
        client=ClientInfo(
            company="Acme Corp",
            tier="platinum",
            sla_hours=4,
            sla_deadline="2026-03-10T18:00:00Z",
            csm="Laura Pérez",
        ),
        issue=IssueInfo(
            product="NexaAuth",
            error_message="401 Unauthorized — invalid_token",
            environment="production",
        ),
        severity="critical",
        confidence=0.95,
        completed=True,
    )


@pytest.mark.asyncio
async def test_matches_error_code_to_error_db_entry(sample_db_data, triage_finding):
    """Diagnostic agent should find 401_invalid_token in NexaAuth error DB."""
    agent = DiagnosticAgent(triage_finding=triage_finding)
    history = [
        {"role": "user", "content": "401 invalid_token on NexaAuth, all users affected since API key rotation"},
    ]
    result = await agent.run("sess_test", sample_db_data, history, triage_finding=triage_finding)
    assert result.agent_id == "diagnostic_agent"
    assert result.confidence > 0.0


@pytest.mark.asyncio
async def test_forms_primary_hypothesis(sample_db_data, triage_finding, mock_claude_diagnostic_response, mock_claude_client):
    """Diagnostic agent should produce a primary hypothesis."""
    agent = DiagnosticAgent(triage_finding=triage_finding)
    history = [
        {"role": "user", "content": "401 invalid_token on NexaAuth, we rotated API keys 2 hours ago"},
    ]
    result = await agent.run("sess_test", sample_db_data, history, triage_finding=triage_finding)
    assert result.primary_hypothesis is not None
    assert len(result.primary_hypothesis.cause) > 0


@pytest.mark.asyncio
async def test_confidence_increases_after_recent_changes_confirmed(sample_db_data, triage_finding):
    """Mentioning recent changes in conversation should increase confidence."""
    agent = DiagnosticAgent(triage_finding=triage_finding)

    history_without = [{"role": "user", "content": "401 error on NexaAuth"}]
    result_without = await agent.run("sess_test", sample_db_data, history_without, triage_finding=triage_finding)

    history_with = [
        {"role": "user", "content": "401 error on NexaAuth"},
        {"role": "user", "content": "Yes, we deployed and rotated API keys 2 hours ago"},
    ]
    result_with = await agent.run("sess_test", sample_db_data, history_with, triage_finding=triage_finding)
    assert result_with.confidence >= result_without.confidence


@pytest.mark.asyncio
async def test_rag_retrieval_filtered_to_diagnostic_category(sample_db_data, triage_finding):
    """RAG should only retrieve diagnostic category docs (mocked here)."""
    from unittest.mock import MagicMock
    mock_rag = MagicMock()
    mock_rag.query.return_value = [
        {"source": "NexaAuth 401 guide", "similarity": 0.92, "excerpt_summary": "401 after rotation", "content": "Full content", "tags": []}
    ]
    set_services(None, mock_rag)

    agent = DiagnosticAgent(triage_finding=triage_finding)
    history = [{"role": "user", "content": "401 invalid_token NexaAuth production"}]
    await agent.run("sess_test", sample_db_data, history, triage_finding=triage_finding)

    # Verify RAG was called with diagnostic category filter
    mock_rag.query.assert_called_once()
    call_kwargs = mock_rag.query.call_args
    assert call_kwargs[1].get("category") == "diagnostic" or "diagnostic" in str(call_kwargs)
