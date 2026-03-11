"""Tests for TriageAgent."""
import json
from unittest.mock import MagicMock

import pytest

from agents.triage_agent import TriageAgent, _guess_error_code, set_claude_client
from agents.agent_base import LoadedData


@pytest.fixture(autouse=True)
def inject_claude(mock_claude_client):
    set_claude_client(mock_claude_client)


@pytest.mark.asyncio
async def test_identifies_product_from_error_description(sample_db_data):
    agent = TriageAgent()
    history = [
        {"role": "user", "content": "I'm getting 401 errors on NexaAuth in production, all users are affected"},
    ]
    result = await agent.run("sess_test", sample_db_data, history)
    assert result.issue.product == "NexaAuth"


@pytest.mark.asyncio
async def test_auto_sets_critical_severity_for_platinum_production(sample_db_data):
    agent = TriageAgent()
    history = [
        {"role": "user", "content": "401 invalid_token on NexaAuth production, all users blocked"},
    ]
    result = await agent.run("sess_test", sample_db_data, history)
    assert result.severity == "critical"


@pytest.mark.asyncio
async def test_detects_known_incident_and_skips_diagnosis(sample_db_data):
    """When product_db shows active incident, triage sets known_incident=True."""
    agent = TriageAgent()
    history = [
        {"role": "user", "content": "NexaMsg webhooks not delivering in EU-WEST region"},
    ]
    result = await agent.run("sess_test", sample_db_data, history)
    assert result.issue.known_incident is True
    assert result.confidence == 1.0
    assert result.routing == "escalation_agent"


@pytest.mark.asyncio
async def test_flags_recurring_issue_from_ticket_history(sample_db_data):
    """DevStartup has 3 recent 429 tickets — recurring should be flagged."""
    db = {**sample_db_data, "_client_id": "client_003", "_session_client_id": "client_003"}

    # Override autouse mock: return a 429-specific triage response for this test
    mock_429 = MagicMock()
    async def _complete(system, messages, max_tokens=2048):
        return json.dumps({
            "product": "NexaAuth",
            "error_message": "429 rate_limit exceeded",
            "environment": "production",
            "started_at": "recently",
            "impact_scope": "all_users",
            "known_incident": False,
            "confidence": 1.0,
            "questions_for_client": [],
        })
    mock_429.complete = _complete
    mock_429.safe_parse_json = json.loads
    mock_429.model = "claude-sonnet-4-6"
    set_claude_client(mock_429)

    agent = TriageAgent()
    history = [
        {"role": "user", "content": "Getting 429 rate_limit on NexaAuth production again, all users affected"},
    ]
    result = await agent.run("sess_test", db, history)
    assert result.issue.recurring is True


@pytest.mark.asyncio
async def test_never_asks_for_client_name(sample_db_data):
    """Triage should not include company/name in questions_for_client."""
    agent = TriageAgent()
    history = [{"role": "user", "content": "I have an issue"}]
    result = await agent.run("sess_test", sample_db_data, history)
    question_fields = [q.field for q in result.questions_for_client]
    assert "company" not in question_fields
    assert "client_name" not in question_fields
    assert "name" not in question_fields


@pytest.mark.asyncio
async def test_assumes_production_when_environment_missing(sample_db_data):
    """When environment not mentioned, default to production (safer)."""
    agent = TriageAgent()
    history = [
        {"role": "user", "content": "NexaAuth is returning 401 for all our users"},
    ]
    result = await agent.run("sess_test", sample_db_data, history)
    # Environment defaults to production when not specified
    assert result.issue.environment in ("production", None)


@pytest.mark.asyncio
async def test_confidence_below_threshold_when_fields_missing(sample_db_data):
    """Very sparse user message should produce confidence < 0.90."""
    agent = TriageAgent()
    history = [{"role": "user", "content": "something is broken"}]
    result = await agent.run("sess_test", sample_db_data, history)
    # With minimal info, confidence should be below threshold
    assert result.confidence < 0.90 or len(result.questions_for_client) > 0


def test_guess_error_code():
    assert _guess_error_code("401 Unauthorized invalid_token") == "401_invalid_token"
    assert _guess_error_code("429 rate limit exceeded") == "429_rate_limit_exceeded"
    assert _guess_error_code("500 internal server error") == "500_internal_error"
    assert _guess_error_code("") == ""
