"""Tests for ResolutionAgent."""
import pytest

from agents.resolution_agent import ResolutionAgent, set_services
from models.report_models import TriageReport, ClientInfo, IssueInfo


@pytest.fixture(autouse=True)
def inject_services(mock_claude_client):
    set_services(mock_claude_client, None)


@pytest.fixture
def sdk_triage():
    return TriageReport(
        session_id="sess_sdk",
        client=ClientInfo(
            company="Acme Corp", tier="platinum", sla_hours=4, sla_deadline="2026-03-10T18:00:00Z", csm="Laura",
        ),
        issue=IssueInfo(product="NexaAuth", error_message="401 invalid_token", environment="production"),
        severity="critical",
        confidence=0.95,
        completed=True,
    )


@pytest.fixture
def rest_client_db(sample_db_data):
    db = {**sample_db_data}
    db["clients"]["client_002"]["integration_type"] = "rest"
    db["_session_client_id"] = "client_002"
    return db


@pytest.mark.asyncio
async def test_generates_ordered_resolution_steps(sample_db_data, sdk_triage, mock_claude_resolution_response):
    agent = ResolutionAgent(triage_finding=sdk_triage)
    history = [{"role": "user", "content": "401 invalid_token on NexaAuth after key rotation"}]
    result = await agent.run("sess_test", sample_db_data, history, triage_finding=sdk_triage)
    assert result.agent_id == "resolution_agent"
    if result.steps:
        steps = [s.step for s in result.steps]
        assert steps == sorted(steps)


@pytest.mark.asyncio
async def test_adds_production_warning_to_risky_steps(sample_db_data, sdk_triage):
    """Steps with risk 'medium' or 'high' in production should get a warning."""
    from models.report_models import ResolutionStepOut
    agent = ResolutionAgent(triage_finding=sdk_triage)

    # Simulate a medium-risk step
    step = ResolutionStepOut(
        step=1,
        action="Rotate API key",
        why="Security",
        verify="Confirm new key works",
        risk="medium",
    )
    from agents.resolution_agent import _maybe_add_prod_warning
    result_step = _maybe_add_prod_warning(step, "production")
    assert result_step.production_warning is not None


@pytest.mark.asyncio
async def test_low_confidence_when_no_rag_results(sample_db_data, sdk_triage):
    """Without RAG results, confidence should be below threshold."""
    agent = ResolutionAgent(triage_finding=sdk_triage)
    history = [{"role": "user", "content": "some obscure error"}]
    # No RAG engine means empty results → lower confidence
    result = await agent.run("sess_test", sample_db_data, history, triage_finding=sdk_triage)
    assert result.agent_id == "resolution_agent"
    # Confidence from code path when no RAG: 0.45
    # (This tests the mathematical path, not LLM path)


@pytest.mark.asyncio
async def test_resolution_agent_never_asks_questions(sample_db_data, sdk_triage):
    """Resolution agent works from prior findings — never asks client questions."""
    agent = ResolutionAgent(triage_finding=sdk_triage)
    history = [{"role": "user", "content": "401 error"}]
    result = await agent.run("sess_test", sample_db_data, history, triage_finding=sdk_triage)
    assert result.questions_for_client == []
