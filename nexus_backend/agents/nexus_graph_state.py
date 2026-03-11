"""
NexusGraphState — shared state TypedDict for the Nexus LangGraph.

One instance flows through a single conversation turn (one POST to /conversation/message).
Nodes return only the keys they change; LangGraph merges automatically.

SessionState is NOT included here — nodes retrieve it from SessionManager by session_id.
This keeps the graph state lean and avoids Pydantic serialization issues in LangSmith.
"""
from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


class NexusGraphState(TypedDict):
    # ── Per-turn inputs (set at graph entry, never mutated by nodes) ──────────
    session_id: str
    client_id: str
    conversation_history: list[dict]
    last_message: str                   # extracted from conversation_history once

    # ── Routing metadata (written by route_turn, read by conditional edge) ───
    routing_action: str                 # one of: collect_triage | run_diagnostic |
                                        #   run_resolution | run_escalation |
                                        #   force_escalate | known_incident |
                                        #   client_resolved | complete
    intent: str                         # from detect_client_intent — visible in trace
    sentiment: str                      # from analyze_sentiment — visible in trace

    # ── Agent state snapshot (written by handler nodes) ──────────────────────
    triage_confidence: float
    diagnostic_confidence: float
    resolution_confidence: float
    escalation_confidence: float
    threshold_used: float               # escalation threshold calculated this turn
    agent_ran: Annotated[list[str], operator.add]  # accumulates across nodes in one turn
    auto_advanced: bool                 # True when triage→diagnostic chained in one turn

    # ── Output (written by terminal handler nodes) ────────────────────────────
    handler_result: dict                # drop-in replacement for orchestrator.run() return
    error: str | None                   # non-None if a node raised
