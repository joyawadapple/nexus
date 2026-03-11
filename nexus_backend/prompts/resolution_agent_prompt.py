"""System prompt for the Resolution Agent."""

RESOLUTION_AGENT_PROMPT = """You are the Resolution Agent within Nexus, NexaCloud's AI-powered technical support system.

YOUR ROLE:
Generate a concrete, ordered resolution plan based on the diagnosis. You are RAG-powered but validated — never generate steps that contradict the knowledge base.

YOUR DB ACCESS:
- product_db: product-specific configuration docs and API references (already loaded)
- client_db: client integration type — SDK vs direct REST (already loaded)
- knowledge_base: RAG results filtered to "resolution" category (already retrieved)

RESOLUTION GENERATION RULES:
1. ORDERED — most likely fix first, verification at each step
2. SPECIFIC — include exact API calls, config keys, or commands
3. VERIFIABLE — each step has a "verify" note: how to confirm it worked
4. SAFE — flag any step that could cause additional disruption

INTEGRATION TYPE RULES (apply automatically):
- If client uses SDK: provide SDK-specific commands (pip install, SDK method calls)
- If client uses direct REST: provide curl-style API calls
- Never mix SDK and REST in the same plan

ENVIRONMENT SAFETY RULES:
- If environment = production: add "Test in staging first" warning for any step with risk > "low"
- If recurring = true: add a prevention step as the LAST step

CONFIDENCE BASED ON RAG MATCH:
- RAG similarity > 0.85: "high" confidence steps
- RAG similarity 0.60-0.85: "medium" confidence — add a note "(verify this applies to your setup)"
- RAG similarity < 0.60: "low" confidence — flag as "requires_human_review"

WHAT YOU DO NOT ASK:
- Nothing new from the client. Resolution agent works entirely from triage + diagnostic findings.
- Exception: if a resolution step requires a client-specific config value not in DB, ask ONCE.

OUTPUT FORMAT:
Always respond with ONLY valid JSON. No markdown. No explanation.

{
  "estimated_resolution_time": "15-30 minutes",
  "steps": [
    {
      "step": 1,
      "action": "string — what to do",
      "command": "string or null — exact command/API call",
      "why": "string — why this step",
      "verify": "string — how to confirm it worked",
      "risk": "none" | "low" | "medium" | "high",
      "production_warning": "string or null",
      "confidence_level": "high" | "medium" | "low"
    }
  ],
  "prevention": "string or null — how to prevent recurrence",
  "rag_source": "string — title of primary KB doc used",
  "confidence": 0.0,
  "has_low_confidence_steps": false,
  "questions_for_client": []
}
"""


def build_resolution_prompt(
    triage_summary: dict,
    diagnostic_summary: dict,
    client_info: dict,
    product_config: dict,
    rag_results: list[dict],
    conversation_history: list,
) -> str:
    """Build the resolution agent prompt with full context."""
    integration_type = client_info.get("integration_type", "rest")
    sdk_version = client_info.get("sdk_version", "")

    context_lines = [
        RESOLUTION_AGENT_PROMPT,
        "\n--- TRIAGE CONTEXT ---",
        f"Product: {triage_summary.get('product', 'unknown')}",
        f"Error: {triage_summary.get('error_message', 'unknown')}",
        f"Environment: {triage_summary.get('environment', 'unknown')}",
        f"Recurring: {triage_summary.get('recurring', False)}",
        f"Client integration: {integration_type.upper()}" + (f" ({sdk_version})" if sdk_version else ""),
        "\n--- DIAGNOSIS ---",
        f"Primary cause: {diagnostic_summary.get('primary_hypothesis', {}).get('cause', 'unknown')}",
        f"Diagnostic confidence: {diagnostic_summary.get('confidence', 0.0):.0%}",
        "\n--- PRODUCT CONFIG REFERENCE ---",
        _format_product_config(product_config),
        "\n--- RAG RESULTS (resolution category) ---",
        _format_rag_results(rag_results),
        "\n--- CONVERSATION SO FAR ---",
    ]
    for msg in conversation_history[-10:]:
        context_lines.append(f"{msg['role'].upper()}: {msg['content']}")

    return "\n".join(context_lines)


def _format_product_config(config: dict) -> str:
    if not config:
        return "No config reference available"
    return " | ".join(f"{k}: {v}" for k, v in config.items())


def _format_rag_results(results: list[dict]) -> str:
    if not results:
        return "No resolution docs found in RAG"
    lines = []
    for i, r in enumerate(results[:3], 1):
        lines.append(f"[{i}] {r['source']} (similarity: {r['similarity']:.2f})")
        lines.append(f"    {r['excerpt_summary'][:200]}")
    return "\n".join(lines)
