"""
nexus_graph.py — LangGraph StateGraph wrapping the Nexus orchestrator pipeline.

Each node wraps an existing orchestrator handler method — no logic is duplicated.
The graph models exactly one conversation turn (one POST to /conversation/message).

Graph topology:
    START → route_turn → (conditional) → [handler node] → END

Enabled only when LANGSMITH_TRACING=true. Built once at startup via build_nexus_graph().
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agents.agent_utils import analyze_sentiment
from agents.nexus_graph_state import NexusGraphState
from agents.nexus_orchestrator import _build_agent_statuses_from_state


def build_nexus_graph(orchestrator, session_manager, db_data):
    """
    Build and compile the Nexus LangGraph.

    All nodes are closures that capture orchestrator, session_manager, and db_data.
    No logic is reimplemented — every node delegates to the existing handler methods.

    Args:
        orchestrator: NexusOrchestrator instance
        session_manager: SessionManager instance (holds live SessionState objects)
        db_data: pre-loaded database dict from database.load_all()

    Returns:
        Compiled LangGraph (CompiledStateGraph)
    """

    # ── Shared helper ──────────────────────────────────────────────────────────

    def _db_data_for_turn(client_id: str) -> dict:
        """Build the per-turn db_data dict, same as orchestrator.run() does."""
        return {
            **db_data,
            "_session_client_id": client_id,
            "_client_id": client_id,
            "clients": db_data.get("clients", {}),
        }

    def _get_session(session_id: str):
        session_state = session_manager.get_session(session_id)
        if not session_state:
            raise ValueError(f"Session not found: {session_id}")
        return session_state

    # ── Nodes ──────────────────────────────────────────────────────────────────

    async def route_turn(state: NexusGraphState) -> dict:
        """
        Analyze sentiment, classify intent, and decide the routing action.
        This is the only node that modifies session_state before a handler runs.
        Returns only the keys it changes.
        """
        session_id = state["session_id"]
        client_id = state["client_id"]
        last_message = state["last_message"]
        conversation_history = state["conversation_history"]

        session_state = _get_session(session_id)

        # Load client data into session if not yet set (mirrors orchestrator.run())
        if not session_state.client:
            session_state.client = db_data.get("clients", {}).get(client_id, {})

        # Sentiment analysis
        sentiment_result = analyze_sentiment(last_message)
        sentiment_label = sentiment_result["label"]
        session_state.add_sentiment(
            label=sentiment_label,
            compound=sentiment_result["compound"],
            text=last_message,
        )

        # Intent classification — captured as a traceable child span when tracing is on.
        # decide_next_action also calls detect_client_intent internally, so it will show
        # twice in LangSmith: once here (explicit) and once inside routing_decision.
        # The explicit call here surfaces intent as a state field for easy filtering.
        intent = await orchestrator.detect_client_intent(last_message, conversation_history)

        # Routing decision — this is the @traceable span that shows the full 8-way decision
        routing_action = await orchestrator.decide_next_action(
            session_state, last_message, conversation_history
        )

        return {
            "routing_action": routing_action,
            "intent": intent,
            "sentiment": sentiment_label,
        }

    async def node_collect_triage(state: NexusGraphState) -> dict:
        """Wrap _handle_collect_triage. Auto-advance to diagnostic is internal."""
        session_id = state["session_id"]
        client_id = state["client_id"]
        session_state = _get_session(session_id)
        db = _db_data_for_turn(client_id)

        result = await orchestrator._handle_collect_triage(
            session_id, db, state["conversation_history"], session_state
        )

        # auto_advanced is True when triage completed and diagnostic also ran this turn
        auto_advanced = session_state.triage_complete and session_state.diagnostic_confidence > 0

        agents = ["triage_agent"]
        if auto_advanced:
            agents.append("diagnostic_agent")

        return {
            "handler_result": result,
            "triage_confidence": session_state.triage_confidence,
            "diagnostic_confidence": session_state.diagnostic_confidence,
            "auto_advanced": auto_advanced,
            "agent_ran": agents,
        }

    async def node_run_diagnostic(state: NexusGraphState) -> dict:
        """Wrap _handle_run_diagnostic."""
        session_id = state["session_id"]
        client_id = state["client_id"]
        session_state = _get_session(session_id)
        db = _db_data_for_turn(client_id)

        result = await orchestrator._handle_run_diagnostic(
            session_id, db, state["conversation_history"], session_state,
            state["sentiment"],
        )

        return {
            "handler_result": result,
            "triage_confidence": session_state.triage_confidence,
            "diagnostic_confidence": session_state.diagnostic_confidence,
            "agent_ran": ["diagnostic_agent"],
        }

    async def node_run_resolution(state: NexusGraphState) -> dict:
        """Wrap _handle_run_resolution."""
        session_id = state["session_id"]
        client_id = state["client_id"]
        session_state = _get_session(session_id)
        db = _db_data_for_turn(client_id)

        result = await orchestrator._handle_run_resolution(
            session_id, db, state["conversation_history"], session_state,
            state["sentiment"],
        )

        agents = ["resolution_agent"]
        if session_state.escalation_complete:
            agents.append("escalation_agent")

        return {
            "handler_result": result,
            "resolution_confidence": session_state.resolution_confidence,
            "escalation_confidence": session_state.escalation_confidence,
            "agent_ran": agents,
        }

    async def node_run_escalation(state: NexusGraphState) -> dict:
        """Wrap _handle_run_escalation (low-confidence diagnostic path)."""
        session_id = state["session_id"]
        client_id = state["client_id"]
        session_state = _get_session(session_id)
        db = _db_data_for_turn(client_id)

        threshold = orchestrator.get_escalation_threshold(session_state)
        result = await orchestrator._handle_run_escalation(
            session_id, db, state["conversation_history"], session_state,
            state["sentiment"],
        )

        return {
            "handler_result": result,
            "escalation_confidence": session_state.escalation_confidence,
            "threshold_used": threshold,
            "agent_ran": ["escalation_agent"],
        }

    async def node_force_escalate(state: NexusGraphState) -> dict:
        """Wrap _handle_force_escalate (client-requested or stuck)."""
        session_id = state["session_id"]
        client_id = state["client_id"]
        session_state = _get_session(session_id)
        db = _db_data_for_turn(client_id)

        result = await orchestrator._handle_force_escalate(
            session_id, db, state["conversation_history"], session_state
        )

        return {
            "handler_result": result,
            "escalation_confidence": session_state.escalation_confidence,
            "agent_ran": ["escalation_agent"],
        }

    async def node_known_incident(state: NexusGraphState) -> dict:
        """Wrap _handle_known_incident (platform-wide incident detected)."""
        session_id = state["session_id"]
        client_id = state["client_id"]
        session_state = _get_session(session_id)
        db = _db_data_for_turn(client_id)

        result = await orchestrator._handle_known_incident(
            session_id, db, state["conversation_history"], session_state
        )

        return {
            "handler_result": result,
            "escalation_confidence": session_state.escalation_confidence,
            "agent_ran": ["escalation_agent"],
        }

    async def node_client_resolved(state: NexusGraphState) -> dict:
        """Wrap _handle_client_resolved (client confirms issue is fixed)."""
        session_id = state["session_id"]
        client_id = state["client_id"]
        session_state = _get_session(session_id)
        db = _db_data_for_turn(client_id)

        result = await orchestrator._handle_client_resolved(
            session_id, db, state["conversation_history"], session_state
        )

        return {
            "handler_result": result,
            "escalation_confidence": session_state.escalation_confidence,
            "agent_ran": ["escalation_agent"],
        }

    async def node_complete(state: NexusGraphState) -> dict:
        """Session is already terminal — return closing message without running any agents."""
        session_state = _get_session(state["session_id"])

        if session_state.status == "resolved":
            closing = (
                "You're all set. The ticket has been logged for your records. "
                "Don't hesitate to reach out if the issue returns."
            )
        elif session_state.status == "escalated":
            csm = session_state.client.get("csm", "your CSM")
            closing = (
                f"This has been escalated and {csm} has the full context. "
                f"You'll hear back shortly."
            )
        else:
            closing = "Your issue has been addressed. Is there anything else I can help you with?"

        return {
            "handler_result": {
                "status": session_state.status,
                "ticket": session_state.ticket,
                "message": closing,
                "agent_statuses": _build_agent_statuses_from_state(session_state),
                "reasoning_logs": {},
            }
        }

    # ── Graph assembly ─────────────────────────────────────────────────────────

    graph = StateGraph(NexusGraphState)

    graph.add_node("route_turn", route_turn)
    graph.add_node("collect_triage", node_collect_triage)
    graph.add_node("run_diagnostic", node_run_diagnostic)
    graph.add_node("run_resolution", node_run_resolution)
    graph.add_node("run_escalation", node_run_escalation)
    graph.add_node("force_escalate", node_force_escalate)
    graph.add_node("known_incident", node_known_incident)
    graph.add_node("client_resolved", node_client_resolved)
    graph.add_node("complete", node_complete)

    # Entry point (set_entry_point is deprecated — use add_edge from START)
    graph.add_edge(START, "route_turn")

    # Conditional routing from route_turn based on routing_action
    graph.add_conditional_edges(
        "route_turn",
        lambda s: s["routing_action"],
        {
            "collect_triage": "collect_triage",
            "run_diagnostic": "run_diagnostic",
            "run_resolution": "run_resolution",
            "run_escalation": "run_escalation",
            "force_escalate": "force_escalate",
            "known_incident": "known_incident",
            "client_resolved": "client_resolved",
            "complete": "complete",
        },
    )

    # Every handler node terminates this turn
    for node_name in [
        "collect_triage", "run_diagnostic", "run_resolution",
        "run_escalation", "force_escalate", "known_incident",
        "client_resolved", "complete",
    ]:
        graph.add_edge(node_name, END)

    return graph.compile()
