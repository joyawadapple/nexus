"""
AgentMonitor — tracks per-agent-call metrics in memory.

Feeds GET /admin/metrics with real data:
- Latency per agent
- Token usage and cost
- Confidence distribution
- Session-level stats
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev
from typing import Literal

import structlog

log = structlog.get_logger("monitor")

# claude-sonnet-4-6 pricing (as of 2026)
_INPUT_COST_PER_1M = 3.00   # USD per 1M input tokens
_OUTPUT_COST_PER_1M = 15.00  # USD per 1M output tokens


@dataclass
class AgentEvent:
    agent_id: str
    session_id: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def cost_usd(self) -> float:
        input_cost = (self.input_tokens / 1_000_000) * _INPUT_COST_PER_1M
        output_cost = (self.output_tokens / 1_000_000) * _OUTPUT_COST_PER_1M
        return round(input_cost + output_cost, 6)


class AgentMonitor:
    """
    In-memory metrics store for the Nexus agent pipeline.
    Resets on restart — use persistent storage for production.
    """

    def __init__(self) -> None:
        self._events: list[AgentEvent] = []
        self._session_costs: dict[str, float] = {}

    def track_agent_call(self, event: AgentEvent) -> None:
        self._events.append(event)
        self._session_costs[event.session_id] = (
            self._session_costs.get(event.session_id, 0.0) + event.cost_usd
        )
        log.debug(
            "monitor.tracked",
            agent=event.agent_id,
            latency_ms=event.latency_ms,
            confidence=event.confidence,
            cost_usd=event.cost_usd,
        )

    def get_aggregate_metrics(self) -> dict:
        if not self._events:
            return {}

        # Per-agent latency averages
        agent_latencies: dict[str, list[int]] = {}
        agent_confidences: dict[str, list[float]] = {}
        for e in self._events:
            agent_latencies.setdefault(e.agent_id, []).append(e.latency_ms)
            agent_confidences.setdefault(e.agent_id, []).append(e.confidence)

        avg_latency_by_agent = {
            agent: f"{mean(lats):.0f}ms"
            for agent, lats in agent_latencies.items()
        }

        # Confidence distribution (buckets: <50%, 50-70%, 70-85%, 85%+)
        all_confidences = [e.confidence for e in self._events]
        confidence_dist = {
            "below_50pct": sum(1 for c in all_confidences if c < 0.50),
            "50_to_70pct": sum(1 for c in all_confidences if 0.50 <= c < 0.70),
            "70_to_85pct": sum(1 for c in all_confidences if 0.70 <= c < 0.85),
            "above_85pct": sum(1 for c in all_confidences if c >= 0.85),
        }

        # Cost stats
        total_cost = sum(e.cost_usd for e in self._events)
        avg_session_cost = mean(self._session_costs.values()) if self._session_costs else 0.0
        total_tokens = sum(e.input_tokens + e.output_tokens for e in self._events)

        return {
            "total_agent_calls": len(self._events),
            "avg_latency_by_agent_ms": avg_latency_by_agent,
            "confidence_distribution": confidence_dist,
            "avg_confidence_by_agent": {
                agent: f"{mean(confs):.0%}"
                for agent, confs in agent_confidences.items()
            },
            "total_cost_usd": f"${total_cost:.4f}",
            "avg_cost_per_session_usd": f"${avg_session_cost:.4f}",
            "total_tokens_used": total_tokens,
            "active_sessions": len(self._session_costs),
        }

    def get_session_cost(self, session_id: str) -> float:
        return self._session_costs.get(session_id, 0.0)
