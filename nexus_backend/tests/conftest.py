"""
Pytest fixtures for Nexus test suite.
Mocks: claude_client.complete(), RAG engine, all database loading.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.conversation import SessionState
from models.report_models import (
    ClientInfo,
    DiagnosticReport,
    EscalationReport,
    Hypothesis,
    IssueInfo,
    RAGResult,
    ResolutionReport,
    ResolutionStepOut,
    TriageReport,
)

# ── Database fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_client_db():
    return {
        "client_001": {
            "company": "Acme Corp",
            "tier": "platinum",
            "sla_hours": 4,
            "csm": "Laura Pérez",
            "csm_email": "laura.perez@nexacloud.io",
            "integration_type": "sdk",
            "sdk_version": "nexacloud-python-sdk==3.2.1",
            "products_subscribed": ["NexaAuth", "NexaPay"],
            "recent_tickets": [
                {"ticket_id": "NX-2026-4410", "product": "NexaAuth", "error": "401_invalid_token", "resolved": True},
                {"ticket_id": "NX-2026-4388", "product": "NexaAuth", "error": "401_invalid_token", "resolved": True},
            ],
            "vip_flag": True,
            "region": "EU",
            "api_key_prefix": "nxa_acme_",
        },
        "client_002": {
            "company": "GlobalRetail SA",
            "tier": "gold",
            "sla_hours": 8,
            "csm": "Marco Bianchi",
            "csm_email": "marco.bianchi@nexacloud.io",
            "integration_type": "rest",
            "sdk_version": None,
            "products_subscribed": ["NexaStore", "NexaMsg"],
            "recent_tickets": [],
            "vip_flag": False,
        },
        "client_003": {
            "company": "DevStartup Ltd",
            "tier": "standard",
            "sla_hours": 24,
            "csm": "Sofia Torres",
            "csm_email": "sofia.torres@nexacloud.io",
            "integration_type": "sdk",
            "sdk_version": "nexacloud-python-sdk==3.1.0",
            "products_subscribed": ["NexaAuth"],
            "recent_tickets": [
                {"ticket_id": "NX-2026-4401", "product": "NexaAuth", "error": "429_rate_limit_exceeded", "resolved": True},
                {"ticket_id": "NX-2026-4375", "product": "NexaAuth", "error": "429_rate_limit_exceeded", "resolved": True},
                {"ticket_id": "NX-2026-4350", "product": "NexaAuth", "error": "429_rate_limit_exceeded", "resolved": True},
            ],
            "vip_flag": False,
        },
    }


@pytest.fixture
def sample_product_db():
    return {
        "NexaAuth": {
            "current_version": "3.2.1",
            "active_incident": None,
            "changelog": {"3.2.1": "Fixed clock skew; improved token cache invalidation"},
            "known_bugs": {
                "3.1.0": ["Token cache not invalidated on key rotation — fixed in 3.2.1"]
            },
            "engineering_contact": "nexaauth-eng@nexacloud.io",
            "config_reference": {"token_lifetime": "Default 3600s", "cache_ttl": "Matches token_lifetime"},
        },
        "NexaMsg": {
            "current_version": "1.8.2",
            "active_incident": {
                "incident_id": "INC-2026-0310",
                "title": "NexaMsg webhook delivery degraded — EU-WEST region",
                "status": "investigating",
                "started_at": "2026-03-10T11:30:00Z",
                "workaround": "POST /v2/msg/webhooks/{id}/retry",
            },
            "engineering_contact": "nexamsg-eng@nexacloud.io",
            "config_reference": {},
        },
        "NexaStore": {
            "current_version": "2.4.0",
            "active_incident": None,
            "changelog": {"2.4.0": "Added multipart upload"},
            "known_bugs": {},
            "engineering_contact": "nexastore-eng@nexacloud.io",
            "config_reference": {"cors_origins": "Set via PUT /v2/store/buckets/{id}/cors"},
        },
        "NexaPay": {
            "current_version": "4.1.0",
            "active_incident": None,
            "changelog": {},
            "known_bugs": {},
            "engineering_contact": "nexapay-eng@nexacloud.io",
            "config_reference": {"idempotency_key": "Required for all payments"},
        },
    }


@pytest.fixture
def sample_error_db():
    return {
        "NexaAuth": {
            "401_invalid_token": {
                "category": "auth",
                "known_causes": ["expired_token", "key_rotation_cache_not_cleared", "clock_skew"],
                "typical_resolution_minutes": 20,
                "auto_escalate": False,
            },
            "429_rate_limit_exceeded": {
                "category": "rate_limit",
                "known_causes": ["burst_traffic", "missing_backoff"],
                "typical_resolution_minutes": 5,
                "auto_escalate": False,
            },
            "500_internal_error": {
                "category": "server",
                "known_causes": ["nexa_infra_issue"],
                "typical_resolution_minutes": None,
                "auto_escalate": True,
            },
        },
        "NexaStore": {
            "403_access_denied": {
                "category": "auth",
                "known_causes": ["bucket_policy_mismatch", "cors_origin_not_allowed"],
                "typical_resolution_minutes": 25,
                "auto_escalate": False,
            },
        },
    }


@pytest.fixture
def sample_knowledge_base():
    return [
        {
            "id": "kb_diag_001",
            "title": "NexaAuth 401 errors — complete diagnostic guide",
            "category": "diagnostic",
            "product": "NexaAuth",
            "content": "NexaAuth 401 errors fall into four categories: token expiry, key rotation cache, clock skew, wrong environment key.",
            "tags": ["401", "auth", "invalid_token"],
        },
        {
            "id": "kb_res_001",
            "title": "Resolving NexaAuth token expiry issues",
            "category": "resolution",
            "product": "NexaAuth",
            "content": "Step 1: Invalidate token cache DELETE /v2/auth/tokens/cache. Step 2: Update token_lifetime. Step 3: Verify API key in all environments.",
            "tags": ["resolution", "401", "token_expiry"],
        },
    ]


@pytest.fixture
def sample_db_data(sample_client_db, sample_product_db, sample_error_db, sample_knowledge_base):
    return {
        "clients": sample_client_db,
        "products": sample_product_db,
        "errors": sample_error_db,
        "knowledge_base": sample_knowledge_base,
        "_session_client_id": "client_001",
        "_client_id": "client_001",
    }


# ── Session fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def platinum_session():
    session = SessionState(session_id="sess_test_platinum", client_id="client_001")
    session.add_message("assistant", "Hello, I'm Nexus. How can I help?")
    session.add_message("user", "I'm getting 401 invalid_token errors on NexaAuth in production. All users affected. Started 2 hours ago.")
    return session


@pytest.fixture
def gold_session():
    session = SessionState(session_id="sess_test_gold", client_id="client_002")
    session.add_message("user", "Our NexaStore bucket is returning 403 access denied errors in production.")
    return session


@pytest.fixture
def standard_session():
    session = SessionState(session_id="sess_test_standard", client_id="client_003")
    session.add_message("user", "Getting 429 rate limit errors on NexaAuth. This keeps happening.")
    return session


@pytest.fixture
def nexamsg_incident_session():
    """Session for a client using NexaMsg — should trigger known incident shortcut."""
    session = SessionState(session_id="sess_test_incident", client_id="client_002")
    session.add_message("user", "NexaMsg webhooks are not being delivered. EU-WEST region affected.")
    return session


# ── Mock Claude client ────────────────────────────────────────────────────────

@pytest.fixture
def mock_claude_triage_response():
    return json.dumps({
        "product": "NexaAuth",
        "error_message": "401 Unauthorized — invalid_token",
        "environment": "production",
        "started_at": "2 hours ago",
        "impact_scope": "all_users",
        "known_incident": False,
        "recurring": False,
        "severity": "critical",
        "confidence": 1.0,
        "questions_for_client": [],
    })


@pytest.fixture
def mock_claude_diagnostic_response():
    return json.dumps({
        "primary_hypothesis": {
            "cause": "Expired JWT token — token lifetime misconfiguration post API key rotation",
            "confidence": 0.87,
            "evidence": [
                "Error code 401 invalid_token matches NexaAuth token expiry pattern",
                "Client confirmed API key rotation 3 hours ago",
            ],
        },
        "alternative_hypotheses": [
            {"cause": "Clock skew between client and NexaAuth servers", "confidence": 0.31, "why_less_likely": "Consistent failures, not intermittent"},
        ],
        "rag_results_used": [
            {"source": "NexaAuth 401 errors guide", "similarity": 0.91, "excerpt_summary": "401 after key rotation requires cache invalidation"},
        ],
        "version_bugs_checked": ["NexaAuth 3.2.1 — no matching known bugs"],
        "confidence": 0.87,
        "questions_for_client": [],
    })


@pytest.fixture
def mock_claude_resolution_response():
    return json.dumps({
        "estimated_resolution_time": "15-30 minutes",
        "steps": [
            {
                "step": 1,
                "action": "Invalidate the token cache",
                "command": "DELETE /v2/auth/tokens/cache",
                "why": "API key rotation does not automatically invalidate existing tokens",
                "verify": "Re-authenticate and confirm new token issued",
                "risk": "low",
                "production_warning": None,
                "confidence_level": "high",
            },
            {
                "step": 2,
                "action": "Update token lifetime configuration",
                "command": "PATCH /v2/auth/config — {\"token_lifetime\": 3600}",
                "why": "Default lifetime may be shorter than expected",
                "verify": "Check token expiry timestamp in JWT payload",
                "risk": "low",
                "production_warning": "Test in staging before applying to production",
                "confidence_level": "high",
            },
        ],
        "prevention": "Set up token expiry monitoring alerts",
        "rag_source": "NexaAuth troubleshooting guide v2.3",
        "confidence": 0.89,
        "has_low_confidence_steps": False,
        "questions_for_client": [],
    })


@pytest.fixture
def mock_claude_escalation_response():
    return json.dumps({
        "decision": "self_resolve",
        "reason": "High confidence diagnosis and resolution — standard known pattern",
        "fallback": "If steps don't resolve within 30 minutes, escalate to CSM",
        "escalation_path": None,
        "csm_notified": False,
        "confidence": 0.97,
        "nexus_summary": "Acme Corp (Platinum) is experiencing a critical NexaAuth outage following API key rotation. Root cause: token cache not invalidated. Resolution steps provided with 89% confidence. Estimated fix time 15-30 minutes.",
        "sentiment_note": None,
        "questions_for_client": [],
    })


@pytest.fixture
def mock_claude_client(
    mock_claude_triage_response,
    mock_claude_diagnostic_response,
    mock_claude_resolution_response,
    mock_claude_escalation_response,
):
    """Mock ClaudeClient that returns appropriate responses per call."""
    client = MagicMock()
    call_count = {"count": 0}
    responses = [
        mock_claude_triage_response,
        mock_claude_diagnostic_response,
        mock_claude_resolution_response,
        mock_claude_escalation_response,
        "Thank you for the information. Here are the resolution steps...",  # orchestrator response
    ]

    async def mock_complete(system, messages, max_tokens=2048):
        idx = min(call_count["count"], len(responses) - 1)
        call_count["count"] += 1
        return responses[idx]

    client.complete = mock_complete
    client.safe_parse_json = staticmethod(json.loads)
    client.model = "claude-sonnet-4-6"
    return client


# ── Agent finding fixtures ────────────────────────────────────────────────────

@pytest.fixture
def triage_finding_fixture():
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
            started_at="2 hours ago",
            impact_scope="all_users",
        ),
        severity="critical",
        confidence=0.95,
        completed=True,
    )


@pytest.fixture
def diagnostic_finding_fixture():
    return DiagnosticReport(
        session_id="sess_test",
        primary_hypothesis=Hypothesis(
            cause="Expired JWT token — token lifetime misconfiguration post API key rotation",
            confidence=0.87,
            evidence=["Error code 401 matches token expiry pattern", "Key rotation 2 hours ago"],
        ),
        confidence=0.87,
        completed=True,
    )


@pytest.fixture
def resolution_finding_fixture():
    return ResolutionReport(
        session_id="sess_test",
        estimated_resolution_time="15-30 minutes",
        steps=[
            ResolutionStepOut(
                step=1,
                action="Invalidate the token cache",
                command="DELETE /v2/auth/tokens/cache",
                why="Key rotation does not invalidate cached tokens",
                verify="Re-authenticate successfully",
                risk="low",
            )
        ],
        rag_source="NexaAuth troubleshooting guide v2.3",
        confidence=0.89,
        completed=True,
    )
