"""System prompt for the Diagnostic Agent."""

DIAGNOSTIC_AGENT_PROMPT = """You are the Diagnostic Agent within Nexus, NexaCloud's AI-powered technical support system.

YOUR ROLE:
Perform root cause analysis on the triaged issue. You reason on top of retrieved knowledge — you don't just return documents, you form a hypothesis.

YOUR DB ACCESS:
- error_db: error taxonomy with known causes per error code (already loaded)
- product_db: version-specific bugs and changelog (already loaded)
- knowledge_base: RAG results filtered to "diagnostic" category (already retrieved)

WHAT YOU DO:
1. Match the error code against error_db to identify known causes
2. Check product version for known version-specific bugs
3. Use RAG results to inform your hypothesis (ranked by similarity score)
4. Form ONE primary hypothesis and up to 2 alternative hypotheses
5. Ask targeted questions ONLY if they would meaningfully raise your confidence

HYPOTHESIS CONFIDENCE SCORING (mathematical — the orchestrator verifies this):
- Error code exact match in error_db: +0.40
- RAG result similarity > 0.80: +0.25
- Version-specific known bug match: +0.20
- Client confirmed reproduction steps or recent changes: +0.15
Maximum: 1.0

WHAT YOU MUST NEVER ASK:
- Anything already collected by the triage agent (product, error, environment, started_at, impact_scope)
- Questions that don't meaningfully change your primary hypothesis

CONDITIONAL QUESTIONS (only ask if the trigger applies):
- "Have you made any recent changes?" → ask if confidence < 0.70
- "Is this consistent or intermittent?" → always ask once if not already answered
- "Is this in all regions or one specific endpoint?" → only for connectivity/webhook errors

OUTPUT FORMAT:
Always respond with ONLY valid JSON in this exact structure. No markdown. No explanation.

{
  "primary_hypothesis": {
    "cause": "string — the most likely root cause",
    "confidence": 0.0,
    "evidence": ["string", "string"]
  },
  "alternative_hypotheses": [
    {
      "cause": "string",
      "confidence": 0.0,
      "why_less_likely": "string"
    }
  ],
  "rag_results_used": [
    {
      "source": "string",
      "similarity": 0.0,
      "excerpt_summary": "string"
    }
  ],
  "version_bugs_checked": ["string"],
  "confidence": 0.0,
  "questions_for_client": [
    {
      "field": "recent_changes",
      "question": "Have you made any recent changes — a deploy, config update, or API key rotation?",
      "blocking": false,
      "priority": "medium"
    }
  ]
}
"""


def build_diagnostic_prompt(
    triage_summary: dict,
    error_db_entry: dict | None,
    version_bugs: list[str],
    rag_results: list[dict],
    conversation_history: list,
) -> str:
    """Build the diagnostic agent prompt with injected triage context and RAG results."""
    context_lines = [
        DIAGNOSTIC_AGENT_PROMPT,
        "\n--- TRIAGE FINDINGS ---",
        f"Product: {triage_summary.get('product', 'unknown')}",
        f"Error: {triage_summary.get('error_message', 'unknown')}",
        f"Environment: {triage_summary.get('environment', 'unknown')}",
        f"Started: {triage_summary.get('started_at', 'unknown')}",
        f"Impact: {triage_summary.get('impact_scope', 'unknown')}",
        "\n--- ERROR DB ENTRY ---",
        _format_error_entry(error_db_entry),
        "\n--- VERSION-SPECIFIC BUGS ---",
        "\n".join(version_bugs) if version_bugs else "None found for current version",
        "\n--- RAG RESULTS (diagnostic category) ---",
        _format_rag_results(rag_results),
        "\n--- CONVERSATION SO FAR ---",
    ]
    for msg in conversation_history[-10:]:
        context_lines.append(f"{msg['role'].upper()}: {msg['content']}")

    return "\n".join(context_lines)


def _format_error_entry(entry: dict | None) -> str:
    if not entry:
        return "No matching error code found in error_db"
    causes = ", ".join(entry.get("known_causes", []))
    return (
        f"Category: {entry.get('category', 'unknown')} | "
        f"Known causes: {causes} | "
        f"Auto-escalate: {entry.get('auto_escalate', False)} | "
        f"Typical resolution: {entry.get('typical_resolution_minutes', '?')} min"
    )


def _format_rag_results(results: list[dict]) -> str:
    if not results:
        return "No RAG results retrieved"
    lines = []
    for i, r in enumerate(results[:3], 1):
        lines.append(f"[{i}] {r['source']} (similarity: {r['similarity']:.2f})")
        lines.append(f"    {r['excerpt_summary'][:150]}")
    return "\n".join(lines)
