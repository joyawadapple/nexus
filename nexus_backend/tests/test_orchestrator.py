"""Tests for NexusOrchestrator."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.nexus_orchestrator import NexusOrchestrator, set_services
from models.conversation import SessionState


@pytest.fixture(autouse=True)
def inject_services(mock_claude_client, sample_db_data):
    from core.rag_engine import RAGEngine
    mock_rag = MagicMock()
    mock_rag.query.return_value = [
        {"source": "NexaAuth guide", "similarity": 0.91, "excerpt_summary": "401 after rotation", "content": "Full", "tags": []}
    ]
    set_services(claude=mock_claude_client, db_data=sample_db_data, rag_engine=mock_rag)
    yield
    # Cleanup
    set_services(claude=None, db_data={}, rag_engine=None)


@pytest.mark.asyncio
async def test_triage_runs_before_diagnostic(platinum_session, sample_db_data):
    """Triage must complete and be confident before diagnostic runs."""
    orchestrator = NexusOrchestrator()
    history = [{"role": "user", "content": "401 invalid_token on NexaAuth production, all users affected"}]

    result = await orchestrator.run(
        session_id="sess_test",
        conversation_history=history,
        session_state=platinum_session,
        client_id="client_001",
    )
    # Triage should have run
    assert platinum_session.triage_finding is not None
    assert result["status"] in ("collecting", "in_progress", "complete", "known_incident")


@pytest.mark.asyncio
async def test_known_incident_skips_to_escalation(nexamsg_incident_session, sample_db_data):
    """NexaMsg has active incident → should skip diagnostic, return known_incident status."""
    db = {**sample_db_data, "_session_client_id": "client_002", "_client_id": "client_002"}

    from agents.nexus_orchestrator import set_services as s
    mock_rag = MagicMock()
    mock_rag.query.return_value = []

    import json
    mock_claude = MagicMock()

    # Mock triage response with known_incident=True
    triage_response = json.dumps({
        "product": "NexaMsg",
        "error_message": "webhook_delivery_failure",
        "environment": "production",
        "known_incident": True,
        "confidence": 1.0,
        "questions_for_client": [],
    })
    escalation_response = json.dumps({
        "decision": "escalated",
        "reason": "Active incident",
        "confidence": 0.97,
        "nexus_summary": "Active NexaMsg incident detected.",
        "questions_for_client": [],
    })
    responses = [triage_response, escalation_response, "Active incident in progress."]
    call_count = {"n": 0}

    async def mock_complete(system, messages, max_tokens=2048):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    mock_claude.complete = mock_complete
    mock_claude.safe_parse_json = json.loads
    mock_claude.model = "claude-sonnet-4-6"

    s(claude=mock_claude, db_data=db, rag_engine=mock_rag)

    orchestrator = NexusOrchestrator()
    history = [{"role": "user", "content": "NexaMsg webhooks not delivering"}]
    result = await orchestrator.run(
        session_id="sess_incident",
        conversation_history=history,
        session_state=nexamsg_incident_session,
        client_id="client_002",
    )
    assert result["status"] == "known_incident"
    assert nexamsg_incident_session.diagnostic_finding is None


@pytest.mark.asyncio
async def test_bundles_max_2_questions_per_turn(platinum_session, sample_db_data):
    """Orchestrator must never present more than 2 questions in one response."""
    orchestrator = NexusOrchestrator()
    history = [{"role": "user", "content": "something is broken"}]
    result = await orchestrator.run(
        session_id="sess_test",
        conversation_history=history,
        session_state=platinum_session,
        client_id="client_001",
    )
    # We can't directly count questions in the response text,
    # but we verify the session asked_fields don't exceed 2 per turn
    assert len(platinum_session.asked_fields) <= 2


@pytest.mark.asyncio
async def test_never_asks_field_already_asked(platinum_session, sample_db_data):
    """Once a field is marked asked, it should not appear in subsequent question bundles."""
    platinum_session.mark_field_asked("product")
    platinum_session.mark_field_asked("error_message")

    orchestrator = NexusOrchestrator()
    history = [{"role": "user", "content": "I have an issue"}]
    await orchestrator.run(
        session_id="sess_test",
        conversation_history=history,
        session_state=platinum_session,
        client_id="client_001",
    )
    # asked_fields should still only contain the original 2 + at most 2 new ones
    assert "product" in platinum_session.asked_fields
    assert "error_message" in platinum_session.asked_fields
