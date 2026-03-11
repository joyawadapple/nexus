"""
probe_conversation.py — fire a 4-turn conversation at the running server and
pretty-print every raw API response so we can inspect:
  • agent_statuses (status + confidence per agent)
  • reasoning_logs  (what each agent reported)
  • awaiting field  (resolution gate)
  • completed values (via reasoning_logs, as they appear in the API response)
  • structlog output is in the *server terminal*, not here

Usage:
    python tests/probe_conversation.py
    python tests/probe_conversation.py client_002   # use a different client
"""

import asyncio
import json
import sys

import httpx

BASE_URL = "http://localhost:8000"


CONVERSATION = [
    # Turn 1 — full context upfront so triage+diagnostic+resolution all complete in one turn
    # Expected: status=in_progress, awaiting=resolution_confirmation (Bug 5 gate)
    "We're getting 401 Unauthorized errors on NexaAuth in production. "
    "All API calls started failing about 2 hours ago after we rotated our API key. "
    "Redis is involved — we cache tokens there. Every request fails, not intermittent.",

    # Turn 2 — client confirms resolution immediately after seeing steps
    # Expected: status=resolved, ticket.escalation.decision=self_resolve, csm_notified=False
    "That worked — authentication is back up now.",

    # Turn 3 — post-resolution message (Fix 2 verification)
    # Expected: status=resolved (complete action), same ticket_id, NO new ticket created
    "No thanks, all good.",
]


def pp(label: str, data: dict) -> None:
    """Pretty-print a labelled JSON block."""
    print(f"\n{'━'*70}")
    print(f"  {label}")
    print(f"{'━'*70}")
    print(json.dumps(data, indent=2, default=str))


async def main(client_id: str = "client_001") -> None:
    async with httpx.AsyncClient(timeout=120.0) as http:
        # ── Start session ──
        resp = await http.post(f"{BASE_URL}/conversation/start",
                               json={"client_id": client_id})
        resp.raise_for_status()
        session = resp.json()
        session_id = session["session_id"]
        print(f"\n✓ Session started  session_id={session_id}  client={client_id}")

        # ── Send each message ──
        for i, message in enumerate(CONVERSATION, start=1):
            print(f"\n\n{'█'*70}")
            print(f"  TURN {i}: {message[:80]}{'...' if len(message) > 80 else ''}")
            print(f"{'█'*70}")

            resp = await http.post(
                f"{BASE_URL}/conversation/message",
                json={"session_id": session_id, "message": message},
            )
            resp.raise_for_status()
            turn_data = resp.json()

            # ── Focused view ──
            print(f"\n  status        : {turn_data.get('status')}")
            print(f"  awaiting      : {turn_data.get('awaiting', '—')}")
            has_ticket = turn_data.get("ticket") is not None
            print(f"  ticket present: {has_ticket}")

            print("\n  agent_statuses:")
            for s in turn_data.get("agent_statuses", []):
                print(f"    {s['agent']:<22} status={s['status']:<10} confidence={s.get('confidence', 0.0):.4f}")

            print("\n  reasoning_logs:")
            for agent, rl in turn_data.get("reasoning_logs", {}).items():
                print(f"    {agent}:")
                for k, v in rl.items():
                    print(f"      {k}: {v}")

            if turn_data.get("confidence_breakdown"):
                print(f"\n  confidence_breakdown: {turn_data['confidence_breakdown']}")

            # ── Full raw JSON ──
            pp(f"RAW RESPONSE — TURN {i}", turn_data)


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "client_001"
    asyncio.run(main(cid))
