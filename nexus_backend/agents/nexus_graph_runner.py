"""
nexus_graph_runner.py — Invocation wrapper for the compiled Nexus LangGraph.

NexusGraphRunner.invoke_graph() is a drop-in replacement for orchestrator.run().
The router calls this instead when LANGSMITH_TRACING=true.

LangSmith auto-traces every node as a child span. The RunnableConfig adds
session-level metadata so runs are filterable by session_id and client_id
in the LangSmith UI.
"""
from __future__ import annotations

from agents.nexus_graph_state import NexusGraphState


class NexusGraphRunner:
    """
    Wraps a compiled LangGraph and invokes it for a single conversation turn.

    Accepts the same arguments as orchestrator.run() and returns the same dict.
    """

    def __init__(self, compiled_graph) -> None:
        self._graph = compiled_graph

    async def invoke_graph(
        self,
        session_id: str,
        conversation_history: list[dict],
        session_state,      # SessionState — used to snapshot confidence at turn start
        client_id: str,
    ) -> dict:
        """
        Execute one conversation turn through the LangGraph and return the
        same dict that orchestrator.run() would return.

        The session_state passed in is used only to read the initial confidence
        values for the graph state snapshot. All mutation happens on the live
        SessionState object inside the session_manager.
        """
        last_message = next(
            (m["content"] for m in reversed(conversation_history) if m["role"] == "user"),
            "",
        )

        initial_state: NexusGraphState = {
            # Per-turn inputs
            "session_id": session_id,
            "client_id": client_id,
            "conversation_history": conversation_history,
            "last_message": last_message,
            # Routing metadata — populated by route_turn
            "routing_action": "",
            "intent": "",
            "sentiment": "",
            # Agent state snapshot at turn start
            "triage_confidence": session_state.triage_confidence,
            "diagnostic_confidence": session_state.diagnostic_confidence,
            "resolution_confidence": session_state.resolution_confidence,
            "escalation_confidence": session_state.escalation_confidence,
            "threshold_used": 0.0,
            "agent_ran": [],
            "auto_advanced": False,
            # Output — populated by handler nodes
            "handler_result": {},
            "error": None,
        }

        config = {
            "run_name": f"nexus_turn_{session_id[:8]}",
            "tags": [f"session:{session_id}", f"client:{client_id}"],
            "metadata": {
                "session_id": session_id,
                "client_id": client_id,
                "triage_complete": session_state.triage_complete,
                "diagnostic_complete": session_state.diagnostic_complete,
                "resolution_complete": session_state.resolution_complete,
                "severity": session_state.severity,
                "status": session_state.status,
            },
        }

        final_state = await self._graph.ainvoke(initial_state, config=config)
        return final_state["handler_result"]
