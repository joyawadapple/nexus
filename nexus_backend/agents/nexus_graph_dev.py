"""
nexus_graph_dev.py — Standalone entry point for LangGraph Studio / langgraph dev.

Initialises all Nexus dependencies synchronously at module import time and
exports a compiled graph as the module-level name `graph`, which is what
langgraph.json's `graphs` key must point to.

Usage:  langgraph dev   (reads langgraph.json → agents/nexus_graph_dev.py:graph)
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# Clear langsmith lru_cache so env vars from .env are used, not defaults
try:
    from langsmith.utils import get_env_var as _ls_get_env_var
    _ls_get_env_var.cache_clear()
except Exception:
    pass

# ── 1. Databases ──────────────────────────────────────────────────────────────
from db import database
_db_data = database.load_all()

# ── 2. RAG engine ─────────────────────────────────────────────────────────────
from core.rag_engine import RAGEngine
_rag_engine = RAGEngine(knowledge_base=_db_data["knowledge_base"])

# ── 3. Claude client ──────────────────────────────────────────────────────────
from core.claude_client import ClaudeClient
_claude = ClaudeClient(
    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    model=os.getenv("MODEL", "claude-sonnet-4-6"),
)

# ── 4. Session infrastructure ─────────────────────────────────────────────────
from core.conversation_memory import ConversationMemory
from core.session_manager import SessionManager
_memory = ConversationMemory()
_session_manager = SessionManager(memory=_memory)

# ── 5. Orchestrator ───────────────────────────────────────────────────────────
from agents.nexus_orchestrator import NexusOrchestrator, set_services
set_services(claude=_claude, db_data=_db_data, rag_engine=_rag_engine)
_orchestrator = NexusOrchestrator()

# ── 6. Pre-create test sessions so Studio can invoke without /conversation/start
# Each client gets a stable session whose ID is printed at startup.
_DEFAULT_CLIENT_ID = "client_001"  # Acme Corp — platinum tier
_studio_session = _session_manager.create_session(_DEFAULT_CLIENT_ID)
_studio_session.partial_ticket["client"] = {
    "company": _db_data.get("clients", {}).get(_DEFAULT_CLIENT_ID, {}).get("company", "Acme Corp"),
    "tier": _db_data.get("clients", {}).get(_DEFAULT_CLIENT_ID, {}).get("tier", "platinum"),
    "sla_hours": _db_data.get("clients", {}).get(_DEFAULT_CLIENT_ID, {}).get("sla_hours", 4),
}
import structlog as _structlog
_structlog.get_logger("nexus_graph_dev").info(
    "studio.test_session_ready",
    session_id=_studio_session.session_id,
    client_id=_DEFAULT_CLIENT_ID,
    hint="Use this session_id in the Studio input form",
)

# ── 7. Compiled graph (exported for langgraph.json) ───────────────────────────
# Wraps build_nexus_graph with a session-autocreate shim for Studio invocations:
# if the session_id provided doesn't exist, a new session is created on-the-fly
# for client_001 so Studio runs don't require a prior /conversation/start call.
from langgraph.graph import StateGraph, START, END
from agents.nexus_graph import build_nexus_graph
from agents.nexus_graph_state import NexusGraphState
from typing_extensions import TypedDict


class StudioInput(TypedDict, total=False):
    """Minimal input schema shown in LangGraph Studio — only last_message is needed."""
    last_message: str       # required: the user's message to process
    session_id: str         # optional: omit to auto-create a session
    client_id: str          # optional: defaults to client_001 (Acme Corp, platinum)

_inner_graph = build_nexus_graph(
    orchestrator=_orchestrator,
    session_manager=_session_manager,
    db_data=_db_data,
)

_DEFAULT_SESSION_HISTORY = [
    {"role": "assistant", "content": "Hello! I'm Nexus. What issue are you experiencing today?"}
]


async def _studio_entry(state: NexusGraphState) -> NexusGraphState:
    """
    Pre-processing node that runs before route_turn.
    Ensures a valid session exists, filling in sensible Studio defaults
    for any missing or empty state fields.
    """
    session_id = state.get("session_id") or _studio_session.session_id
    client_id = state.get("client_id") or _DEFAULT_CLIENT_ID
    last_message = state.get("last_message") or ""
    conversation_history = state.get("conversation_history") or []

    # Auto-create session if it doesn't exist
    if not _session_manager.get_session(session_id):
        new_session = _session_manager.create_session(client_id)
        # Swap to the newly created session_id
        session_id = new_session.session_id
        client_data = _db_data.get("clients", {}).get(client_id, {})
        new_session.partial_ticket["client"] = {
            "company": client_data.get("company", ""),
            "tier": client_data.get("tier", "standard"),
            "sla_hours": client_data.get("sla_hours", 24),
        }

    # Add the user message to session history if provided
    session = _session_manager.get_session(session_id)
    if last_message and not any(
        m.get("content") == last_message and m.get("role") == "user"
        for m in session.messages
    ):
        session.add_message("user", last_message)

    if not conversation_history and last_message:
        conversation_history = _DEFAULT_SESSION_HISTORY + [
            {"role": "user", "content": last_message}
        ]

    return {
        **state,
        "session_id": session_id,
        "client_id": client_id,
        "last_message": last_message,
        "conversation_history": conversation_history,
        "routing_action": state.get("routing_action") or "",
        "intent": state.get("intent") or "",
        "sentiment": state.get("sentiment") or "",
        "triage_confidence": state.get("triage_confidence") or 0.0,
        "diagnostic_confidence": state.get("diagnostic_confidence") or 0.0,
        "resolution_confidence": state.get("resolution_confidence") or 0.0,
        "escalation_confidence": state.get("escalation_confidence") or 0.0,
        "threshold_used": state.get("threshold_used") or 0.0,
        "agent_ran": state.get("agent_ran") or [],
        "auto_advanced": state.get("auto_advanced") or False,
        "handler_result": state.get("handler_result") or {},
        "error": state.get("error"),
    }


_studio_graph_builder = StateGraph(NexusGraphState, input=StudioInput)
_studio_graph_builder.add_node("studio_entry", _studio_entry)
_studio_graph_builder.add_node("nexus", _inner_graph)
_studio_graph_builder.add_edge(START, "studio_entry")
_studio_graph_builder.add_edge("studio_entry", "nexus")
_studio_graph_builder.add_edge("nexus", END)
graph = _studio_graph_builder.compile()
