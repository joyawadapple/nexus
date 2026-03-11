"""System prompt for the Escalation Agent."""

ESCALATION_AGENT_PROMPT = """You are the Escalation Agent within Nexus, NexaCloud's AI-powered technical support system.

YOUR ROLE:
Make the human handoff decision and generate the complete, structured support ticket.
You run LAST — after triage, diagnostic, and resolution agents have all completed.

YOUR DB ACCESS:
- client_db: CSM contact, escalation path, VIP flag (already loaded)
- product_db: engineering team contact for this product (already loaded)

ESCALATION DECISION RULES (evaluate in order):

AUTO-ESCALATE (immediately route to human support):
- severity = "critical" AND resolution_confidence < 0.70
- client_tier = "platinum" AND diagnostic_confidence < 0.75
- recurring = true AND same_error_count >= 2
- resolution_agent has any steps with confidence_level = "low"
- known_incident = true (always escalate — engineering team handles)
- error auto_escalate = true in error_db
- hypothesis_invalidated = true in diagnostic findings (client contradicted premise — escalate for human re-diagnosis)
- multi_product_incident = true in triage (multiple products affected — likely shared infrastructure incident)

RECOMMEND ESCALATE (flag for human review, but provide resolution steps):
- estimated_resolution_time > 60 minutes
- client_tier in ["platinum", "gold"] AND environment = "production"
- alternative_hypothesis confidence > 0.50 (close call on diagnosis)

SELF-RESOLVE (send resolution steps to client without human involvement):
- resolution_confidence >= 0.85
- error is a known pattern with high-confidence resolution
- client tier = "standard"

SENTIMENT ADJUSTMENT:
- If sentiment = "frustrated": lower escalation confidence threshold by escalation_bias
- If sentiment = "urgent": lower escalation confidence threshold by escalation_bias

TICKET GENERATION:
Always generate a complete ticket regardless of the decision.
The ticket is the master output — it includes all agent findings.
Generate a nexus_summary paragraph (2-3 sentences, natural language).
Generate a ticket_id in format: NX-{year}-{sequence} — use a placeholder if unknown.

OUTPUT FORMAT:
Always respond with ONLY valid JSON. No markdown. No explanation.

{
  "decision": "self_resolve" | "escalated" | "pending",
  "reason": "string — one sentence explaining the decision",
  "fallback": "string or null — what to do if self-resolve steps don't work",
  "escalation_path": "string or null — who to contact if escalated",
  "csm_notified": false,
  "confidence": 0.0,
  "nexus_summary": "string — 2-3 sentence natural language summary for the ticket",
  "sentiment_note": "string or null — any tone adjustment applied",
  "questions_for_client": []
}
"""


def build_escalation_prompt(
    triage_report: dict,
    diagnostic_report: dict,
    resolution_report: dict,
    client_info: dict,
    sentiment_profile: str,
    escalation_bias: float,
    conversation_history: list,
) -> str:
    """Build the escalation agent prompt with all prior agent findings."""
    context_lines = [
        ESCALATION_AGENT_PROMPT,
        "\n--- CLIENT CONTEXT ---",
        f"Company: {client_info.get('company', 'Unknown')}",
        f"Tier: {client_info.get('tier', 'unknown')} | SLA: {client_info.get('sla_hours', '?')}h",
        f"CSM: {client_info.get('csm', 'N/A')} ({client_info.get('csm_email', 'N/A')})",
        "CRITICAL: Always use the exact CSM name above in nexus_summary. Never generate or infer a different name.",
        f"VIP: {client_info.get('vip_flag', False)}",
        "\n--- TRIAGE FINDINGS ---",
        f"Product: {triage_report.get('product', 'unknown')}",
        f"Error: {triage_report.get('error_message', 'unknown')}",
        f"Environment: {triage_report.get('environment', 'unknown')}",
        f"Severity: {triage_report.get('severity', 'unknown')}",
        f"Known incident: {triage_report.get('known_incident', False)}",
        f"Recurring: {triage_report.get('recurring', False)}",
        f"Triage confidence: {triage_report.get('confidence', 0.0):.0%}",
    ]
    _mentioned = triage_report.get("mentioned_products", [])
    if len(_mentioned) > 1:
        _primary = triage_report.get("product", "unknown")
        _additional = [p for p in _mentioned if p != _primary]
        context_lines += [
            "\n--- MULTI-PRODUCT INCIDENT ---",
            f"Primary product: {_primary}",
            f"Additional affected products: {', '.join(_additional)}",
            "INSTRUCTION: Note ALL affected products in nexus_summary. Flag as potential shared infrastructure incident.",
        ]
    if diagnostic_report and diagnostic_report.get("hypothesis_invalidated"):
        context_lines += [
            "\n--- DIAGNOSTIC FINDINGS ---",
            "Primary cause: Root cause under re-investigation — initial hypothesis invalidated by client",
            f"Invalidation reason: {diagnostic_report.get('invalidation_reason', 'Client contradicted premise')}",
            "Diagnostic confidence: 0%",
        ]
    else:
        context_lines += [
            "\n--- DIAGNOSTIC FINDINGS ---",
            f"Primary cause: {diagnostic_report.get('primary_hypothesis', {}).get('cause', 'unknown') if diagnostic_report else 'Not completed'}",
            f"Diagnostic confidence: {diagnostic_report.get('confidence', 0.0):.0%}" if diagnostic_report else "N/A",
        ]
    context_lines += [
        "\n--- RESOLUTION FINDINGS ---",
        f"Estimated time: {resolution_report.get('estimated_resolution_time', 'unknown') if resolution_report else 'Not completed'}",
        f"Steps count: {len(resolution_report.get('steps', [])) if resolution_report else 0}",
        f"Has low-confidence steps: {resolution_report.get('has_low_confidence_steps', False) if resolution_report else False}",
        f"Resolution confidence: {resolution_report.get('confidence', 0.0):.0%}" if resolution_report else "N/A",
        "\n--- SENTIMENT ---",
        f"Detected: {sentiment_profile}",
        f"Escalation bias (lower threshold by): {escalation_bias:.2f}",
        "\n--- CONVERSATION SO FAR ---",
    ]
    for msg in conversation_history[-10:]:
        context_lines.append(f"{msg['role'].upper()}: {msg['content']}")

    return "\n".join(context_lines)
