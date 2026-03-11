import asyncio
import json
import httpx
import re
from datetime import datetime
from pathlib import Path


BASE_URL = "http://localhost:8000"
RESULTS_FILE = Path("tests/eval_results.json")

EMPATHY_WORDS = [
    "understand", "sorry", "apologize", "frustrating", "appreciate",
    "concern", "important", "urgent", "critical", "help you",
    "here to help", "i hear", "i can see",
]


def get_nested(obj: dict, path: str, default=None):
    """Get nested dict/list value using dot notation: 'ticket.client.tier' or 'agent_statuses.0.confidence'"""
    keys = path.split(".")
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        elif isinstance(obj, list):
            try:
                obj = obj[int(key)]
            except (ValueError, IndexError):
                return default
        else:
            return default
        if obj is default:
            return default
    return obj


def assert_expected(result: dict, expected: dict) -> list[dict]:
    """Compare result against expected values. Returns list of failures."""
    failures = []

    for key, expected_value in expected.items():

        # Special assertions
        if key == "triage_runs_once":
            count = result.get("meta", {}).get("triage_run_count", 0)
            if count != 1:
                failures.append({
                    "field": key,
                    "expected": 1,
                    "actual": count,
                    "reason": f"Triage ran {count} times — should run exactly once"
                })

        elif key == "max_questions_per_turn":
            violations = result.get("meta", {}).get("question_count_violations", [])
            if violations:
                failures.append({
                    "field": key,
                    "expected": f"<= {expected_value} questions per turn",
                    "actual": violations,
                    "reason": "Question bundling exceeded max per turn"
                })

        elif key == "diagnostic_agent_ran":
            ran = result.get("meta", {}).get("diagnostic_ran", False)
            if ran != expected_value:
                failures.append({
                    "field": key,
                    "expected": expected_value,
                    "actual": ran,
                    "reason": "Diagnostic agent ran when it should not have (known incident path)"
                })

        elif key == "resolution_agent_ran":
            ran = result.get("meta", {}).get("resolution_ran", False)
            if ran != expected_value:
                failures.append({
                    "field": key,
                    "expected": expected_value,
                    "actual": ran,
                    "reason": f"Resolution agent ran={ran}, expected={expected_value}"
                })

        elif key == "escalation_agent_ran":
            ran = result.get("meta", {}).get("escalation_ran", False)
            if ran != expected_value:
                failures.append({
                    "field": key,
                    "expected": expected_value,
                    "actual": ran,
                    "reason": f"Escalation agent ran={ran}, expected={expected_value}"
                })

        elif key == "triage_cached_after_completion":
            cached = result.get("meta", {}).get("triage_cached", False)
            if cached != expected_value:
                failures.append({
                    "field": key,
                    "expected": expected_value,
                    "actual": cached,
                    "reason": "Triage was not served from cache after first completion"
                })

        elif key == "turns_to_ticket":
            actual = result.get("meta", {}).get("turns_to_ticket")
            if actual != expected_value:
                failures.append({
                    "field": key,
                    "expected": expected_value,
                    "actual": actual,
                    "reason": f"Ticket generated after {actual} turns, expected {expected_value}"
                })

        elif key in ("resolution_started_after_diagnostic_threshold",
                     "resolution_only_starts_after_diagnostic_threshold"):
            passed = result.get("meta", {}).get("resolution_gate_respected", False)
            if not passed:
                failures.append({
                    "field": key,
                    "expected": True,
                    "actual": False,
                    "reason": "Resolution started before diagnostic crossed confidence threshold"
                })

        elif key == "nexus_first_response_contains_empathy":
            first = result.get("meta", {}).get("first_response", "")
            has_empathy = any(word in first.lower() for word in EMPATHY_WORDS)
            if has_empathy != expected_value:
                failures.append({
                    "field": key,
                    "expected": expected_value,
                    "actual": has_empathy,
                    "reason": f"First response empathy check failed: '{first[:120]}'"
                })

        elif key == "escalation_threshold_lowered":
            threshold = result.get("meta", {}).get("escalation_threshold")
            lowered = threshold is not None and threshold < 0.75
            if lowered != expected_value:
                failures.append({
                    "field": key,
                    "expected": expected_value,
                    "actual": lowered,
                    "reason": f"Escalation threshold={threshold}, expected < 0.75"
                })

        elif key == "no_repeated_questions":
            repeated = result.get("meta", {}).get("repeated_questions_detected", False)
            if repeated:
                failures.append({
                    "field": key,
                    "expected": True,
                    "actual": False,
                    "reason": "Repeated questions were detected across turns"
                })

        elif key == "asked_fields_never_repeated":
            repeated = result.get("meta", {}).get("asked_fields_repeated", False)
            if repeated:
                failures.append({
                    "field": key,
                    "expected": True,
                    "actual": False,
                    "reason": "An asked field was repeated across turns"
                })

        elif key.endswith("_min"):
            field = key[:-4]
            actual = get_nested(result, field)
            if actual is None or actual < expected_value:
                failures.append({
                    "field": field,
                    "expected": f">= {expected_value}",
                    "actual": actual,
                    "reason": "Value below minimum threshold"
                })

        elif key.endswith("_max"):
            field = key[:-4]
            actual = get_nested(result, field)
            if actual is not None and actual > expected_value:
                failures.append({
                    "field": field,
                    "expected": f"<= {expected_value}",
                    "actual": actual,
                    "reason": "Value above maximum threshold"
                })

        elif key.endswith("_contains"):
            field = key[:-9]
            actual = get_nested(result, field, "")
            if expected_value.lower() not in str(actual).lower():
                failures.append({
                    "field": field,
                    "expected": f"contains '{expected_value}'",
                    "actual": actual,
                    "reason": "String does not contain expected value"
                })

        else:
            actual = get_nested(result, key)
            if actual != expected_value:
                failures.append({
                    "field": key,
                    "expected": expected_value,
                    "actual": actual,
                    "reason": "Value mismatch"
                })

    return failures


async def run_conversation(client: httpx.AsyncClient, client_id: str, messages: list[str]) -> dict:
    """Run a full conversation and return the session result."""

    # Start session
    resp = await client.post(f"{BASE_URL}/conversation/start",
                             json={"client_id": client_id})
    resp.raise_for_status()
    session_data = resp.json()
    session_id = session_data["session_id"]

    meta = {
        "triage_run_count": 0,
        "question_count_violations": [],
        "diagnostic_ran": False,
        "resolution_ran": False,
        "escalation_ran": False,
        "resolution_gate_respected": True,
        "diagnostic_confidence_per_turn": [],
        "triage_severity_turn_1": None,
        "triage_recurring": None,
        "triage_cached": False,
        "triage_confidence_final": None,
        "turns_to_ticket": None,
        "first_response": None,
        "escalation_threshold": None,
        "repeated_questions_detected": False,
        "asked_fields_repeated": False,
        "_seen_questions": [],
        "_seen_asked_fields": set(),
    }

    # Send each message
    for turn_idx, message in enumerate(messages):
        resp = await client.post(f"{BASE_URL}/conversation/message",
                                 json={"session_id": session_id, "message": message})
        resp.raise_for_status()
        turn_data = resp.json()

        response_text = turn_data.get("response", "")

        # Capture first response for empathy check
        if turn_idx == 0:
            meta["first_response"] = response_text

        # Count questions in response
        question_count = response_text.count("?")
        if question_count > 2:
            meta["question_count_violations"].append({
                "turn": turn_idx + 1,
                "question_count": question_count
            })

        # Detect repeated questions across turns
        new_questions = [s.strip() for s in response_text.split("?") if s.strip()]
        for q in new_questions:
            q_lower = q.lower()[-60:]  # Use tail of question as fingerprint
            if q_lower in meta["_seen_questions"]:
                meta["repeated_questions_detected"] = True
            else:
                meta["_seen_questions"].append(q_lower)

        # Track when ticket first appears
        if turn_data.get("ticket") is not None and meta["turns_to_ticket"] is None:
            meta["turns_to_ticket"] = turn_idx + 1

        # Track agent runs from reasoning logs
        reasoning = turn_data.get("reasoning_logs", {})

        if "triage_agent" in reasoning:
            triage_data = reasoning["triage_agent"]
            if not triage_data.get("cached", False):
                meta["triage_run_count"] += 1
            # Capture severity on first triage run
            if meta["triage_severity_turn_1"] is None and triage_data.get("severity"):
                meta["triage_severity_turn_1"] = triage_data["severity"]
            # Capture recurring flag
            if triage_data.get("recurring") is not None:
                meta["triage_recurring"] = triage_data["recurring"]
            # Track caching
            if triage_data.get("cached"):
                meta["triage_cached"] = True

        if "diagnostic_agent" in reasoning:
            meta["diagnostic_ran"] = True
            diag_conf = reasoning["diagnostic_agent"].get("confidence", 0)
            meta["diagnostic_confidence_per_turn"].append(diag_conf)
            # Store per-turn as individual keys (1-indexed)
            turn_num = len(meta["diagnostic_confidence_per_turn"])
            meta[f"diagnostic_confidence_turn_{turn_num}"] = diag_conf
            # Check resolution gate: if resolution already ran, was diagnostic ready?
            if meta["resolution_ran"] and diag_conf < 0.75:
                meta["resolution_gate_respected"] = False

        if "resolution_agent" in reasoning:
            meta["resolution_ran"] = True
            # Check gate: diagnostic confidence must be >= 0.75 before resolution
            if meta["diagnostic_confidence_per_turn"]:
                last_diag_conf = meta["diagnostic_confidence_per_turn"][-1]
                if last_diag_conf < 0.75:
                    meta["resolution_gate_respected"] = False

        if "escalation_agent" in reasoning:
            meta["escalation_ran"] = True
            esc_data = reasoning["escalation_agent"]
            if esc_data.get("threshold_used") is not None:
                meta["escalation_threshold"] = esc_data["threshold_used"]

        # Track asked_fields for repeat detection
        agent_statuses_turn = turn_data.get("agent_statuses", [])
        for status_item in agent_statuses_turn:
            fields = status_item.get("asked_fields", []) if isinstance(status_item, dict) else []
            for field in fields:
                if field in meta["_seen_asked_fields"]:
                    meta["asked_fields_repeated"] = True
                else:
                    meta["_seen_asked_fields"].add(field)

    # Get final ticket
    ticket_resp = await client.get(f"{BASE_URL}/tickets/session/{session_id}")
    ticket = ticket_resp.json() if ticket_resp.status_code == 200 else {}

    # Get agent status
    status_resp = await client.get(f"{BASE_URL}/conversation/status/{session_id}")
    status = status_resp.json() if status_resp.status_code == 200 else {}

    # Capture final triage confidence from agent_statuses
    agent_statuses = status.get("agent_statuses", [])
    for s in agent_statuses:
        if isinstance(s, dict) and s.get("agent") == "triage_agent":
            meta["triage_confidence_final"] = s.get("confidence", 0.0)
            break

    # Clean up internal tracking fields
    del meta["_seen_questions"]
    del meta["_seen_asked_fields"]

    return {
        "session_id": session_id,
        "ticket": ticket,
        "meta": meta,
        "agent_statuses": agent_statuses,
        "partial_ticket": status.get("partial_ticket", {})
    }


async def run_eval_case(client: httpx.AsyncClient, case: dict) -> dict:
    """Run a single eval case and return pass/fail with details."""
    print(f"\n{'='*60}")
    print(f"Running: [{case['id']}] {case['name']}")
    print(f"{'='*60}")

    try:
        # Handle session switch cases
        if "then_switch_to" in case:
            result_1 = await run_conversation(client, case["client_id"], case["conversation"])
            result_2 = await run_conversation(client, case["then_switch_to"], case["second_conversation"])
            result = {
                "session_1": result_1,
                "session_2": result_2,
                "session_ids_different": result_1["session_id"] != result_2["session_id"],
                "no_state_bleed": (
                    get_nested(result_1, "ticket.client.company") !=
                    get_nested(result_2, "ticket.client.company")
                )
            }
        else:
            result = await run_conversation(client, case["client_id"], case["conversation"])

        failures = assert_expected(result, case["expected"])

        status = "PASS" if not failures else "FAIL"
        print(f"Result: {status}")

        if failures:
            print(f"Failures ({len(failures)}):")
            for f in failures:
                print(f"  ✗ {f['field']}: expected {f['expected']}, got {f['actual']}")
                print(f"    → {f['reason']}")
        else:
            print("  ✓ All assertions passed")

        return {
            "id": case["id"],
            "name": case["name"],
            "status": status,
            "failures": failures,
            "result": result
        }

    except Exception as e:
        import traceback
        print(f"Result: ERROR — {e}")
        traceback.print_exc()
        return {
            "id": case["id"],
            "name": case["name"],
            "status": "ERROR",
            "error": str(e),
            "failures": []
        }


async def run_all_evals(case_ids: list[str] = None) -> dict:
    """Run all eval cases and produce a summary report."""

    with open("tests/eval_cases.json") as f:
        cases = json.load(f)

    # Filter to specific cases if requested
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]

    print(f"\nNexus Eval Runner")
    print(f"Running {len(cases)} cases against {BASE_URL}")
    print(f"Started: {datetime.utcnow().isoformat()}")

    results = []
    async with httpx.AsyncClient(timeout=90.0) as client:
        for case in cases:
            result = await run_eval_case(client, case)
            results.append(result)
            await asyncio.sleep(1)  # Brief pause between cases

    # Summary
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    errors = [r for r in results if r["status"] == "ERROR"]

    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total:  {len(results)}")
    print(f"Passed: {len(passed)} ✓")
    print(f"Failed: {len(failed)} ✗")
    print(f"Errors: {len(errors)} !")
    print(f"Score:  {len(passed)}/{len(results)} ({100*len(passed)//len(results)}%)")

    if failed:
        print(f"\nFailed cases:")
        for r in failed:
            print(f"  ✗ [{r['id']}] {r['name']}")
            for f in r["failures"]:
                print(f"      → {f['field']}: {f['reason']}")

    if errors:
        print(f"\nErrored cases:")
        for r in errors:
            print(f"  ! [{r['id']}] {r['name']}: {r.get('error', '?')}")

    # Save results
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "score": f"{len(passed)}/{len(results)}",
        "passed": len(passed),
        "failed": len(failed),
        "errors": len(errors),
        "results": results
    }

    RESULTS_FILE.write_text(json.dumps(report, indent=2))
    print(f"\nFull results saved to {RESULTS_FILE}")

    return report


if __name__ == "__main__":
    import sys
    case_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(run_all_evals(case_ids))
