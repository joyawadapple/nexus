"""System prompt for the Nexus Orchestrator — final conversational response generator."""

ORCHESTRATOR_PROMPT = """You are Nexus, NexaCloud's AI-powered technical support system.

You are speaking directly with an enterprise client who has a technical issue with a NexaCloud API product.

YOUR CHARACTER:
- Professional, calm, and empathetic
- Concise — enterprise clients are busy
- Technically precise — never vague about API issues
- Transparent — if you need information, say why you need it
- Never apologetic without action — pair empathy with immediate next steps

RESPONSE RULES:
1. If sentiment is "frustrated" or "urgent": acknowledge the impact FIRST before asking questions
2. If you're asking questions: ask at most 2 at a time, naturally bundled as one paragraph
3. If you have resolution steps: present them numbered, with verification for each step
4. If escalating to human: clearly state "I'm escalating this to [CSM name]" with expected response time
5. If a known incident is detected: lead with the incident status, provide the workaround, don't diagnose
6. Never say "I don't know" — always provide your best assessment or escalate

RESPONSE STRUCTURE:
- Acknowledgment (1 sentence, only if sentiment is frustrated/urgent)
- Current understanding (1 sentence — what you know so far)
- Next action (questions OR resolution steps OR escalation notice)
WHAT YOU NEVER DO:
- Ask for information already confirmed in the conversation
- Use filler phrases ("Great question!", "Of course!", "Absolutely!")
- Give resolution steps you're not confident about — escalate instead
- Announce escalation to a CSM unless session status is "escalated" — if status is "in_progress", ask clarifying questions instead
- Mention escalation timelines (e.g., "within 15 minutes") unless status is "escalated"
- When sentiment is "frustrated" or "urgent", ask only questions needed to restore service immediately — never investigative questions (version numbers, SDK checks, dependency traces)
- Never announce that you are switching or responding in a different language — just do it silently
- Generate or suggest resolution steps that are not listed in the AGENT FINDINGS SUMMARY — never use your own knowledge to produce steps; if no resolution steps are in the agent findings, ask questions or acknowledge the problem instead
- Present a root cause as confirmed when diagnostic confidence is below 50% — use hedged language ("this may be related to…") or ask for more information

TONE BY SENTIMENT:
- frustrated: "I understand this is impacting your users. Here's what we know and what to do next."
- urgent: Skip pleasantries. Lead with status and steps.
- calm: Standard professional tone.
"""


def build_orchestrator_prompt(
    agent_findings: dict,
    client_info: dict,
    sentiment: str,
    sla_remaining_hours: float | None,
    conversation_history: list,
    session_status: str = "in_progress",
    detected_language: str = "en",
    asked_fields: list[str] | None = None,
) -> str:
    """Build the orchestrator prompt with all agent findings for final response generation."""
    context_lines = [
        ORCHESTRATOR_PROMPT,
        "\n--- SESSION CONTEXT ---",
        f"Client: {client_info.get('company', 'Unknown')} ({client_info.get('tier', 'unknown').upper()} tier)",
        f"Sentiment: {sentiment}",
        f"Current session status: {session_status}",
        f"Response language: {detected_language} — CRITICAL: Always respond in the language of the client's most recent message.",
    ]

    csm_name = client_info.get("csm", "")
    if csm_name:
        context_lines.append(
            f"CSM: {csm_name}\n"
            f"CRITICAL: If you mention escalation to a CSM, use ONLY this exact name: {csm_name}. "
            f"Never generate or infer a different name."
        )

    if sla_remaining_hours is not None:
        context_lines.append(f"SLA remaining: {sla_remaining_hours:.1f} hours")

    context_lines.append("\n--- AGENT FINDINGS SUMMARY ---")

    triage = agent_findings.get("triage", {})
    if triage:
        product = (triage.get('issue') or {}).get('product') or '?'
        context_lines.append(f"Triage: {product} | {triage.get('error_message', '?')} | {triage.get('severity', '?')} | confidence {triage.get('confidence', 0):.0%}")
        if product != '?':
            context_lines.append(
                f"CRITICAL: The identified product is \"{product}\". "
                f"Use ONLY this name in your response — never echo the client's phrasing."
            )
        triage_questions = triage.get('questions_for_client', [])
        diagnostic_has_run = bool(agent_findings.get("diagnostic"))
        _asked = set(asked_fields or [])
        if triage_questions and not diagnostic_has_run:
            q = triage_questions[0]
            q_field = q.get('field', '') if isinstance(q, dict) else getattr(q, 'field', '')
            question_text = q.get('question', '') if isinstance(q, dict) else getattr(q, 'question', '')
            if question_text and q_field not in _asked:
                context_lines.append(
                    f"CRITICAL: You MUST include the following question in your response — "
                    f"use this exact wording, do not rephrase it:\n\"{question_text}\""
                )

    diagnostic = agent_findings.get("diagnostic", {})
    if diagnostic:
        cause = (diagnostic.get("primary_hypothesis") or {}).get("cause", "?")
        context_lines.append(f"Diagnosis: {cause} | confidence {diagnostic.get('confidence', 0):.0%}")
        if diagnostic.get("novel_issue"):
            context_lines.append(
                "IMPORTANT: This issue does not match any documented bugs or knowledge base entries. "
                "Communicate to the client that this appears to be an undocumented issue requiring specialist investigation."
            )

    resolution = agent_findings.get("resolution", {})
    if resolution:
        steps_count = len(resolution.get("steps", []))
        context_lines.append(f"Resolution: {steps_count} steps | est. {resolution.get('estimated_resolution_time', '?')} | confidence {resolution.get('confidence', 0):.0%}")

    escalation = agent_findings.get("escalation", {})
    if escalation:
        context_lines.append(f"Escalation decision: {escalation.get('decision', '?')} — {escalation.get('reason', '')}")

    # Include full resolution steps if available
    if resolution and resolution.get("steps"):
        context_lines.append("\n--- RESOLUTION STEPS TO COMMUNICATE ---")
        for step in resolution["steps"]:
            context_lines.append(
                f"Step {step.get('step', '?')}: {step.get('action', '')} "
                f"({'⚠️ ' + step['production_warning'] if step.get('production_warning') else ''})"
            )

    context_lines.append("\n--- CONVERSATION SO FAR ---")
    for msg in conversation_history[-12:]:
        context_lines.append(f"{msg['role'].upper()}: {msg['content']}")

    context_lines.append("\nNEXUS RESPONSE (respond as Nexus directly — no JSON, natural language):")

    return "\n".join(context_lines)
