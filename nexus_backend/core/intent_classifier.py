"""
IntentClassifier — uses a lightweight Claude call to understand client message intent.

Replaces string-matching in NexusOrchestrator.detect_client_intent().
Handles any phrasing, language, or indirect expression of intent.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import structlog

from core.claude_client import ClaudeClient

if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    from langsmith import traceable
else:
    def traceable(**_):  # type: ignore[misc]
        return lambda f: f

log = structlog.get_logger("intent_classifier")

_VALID_INTENTS = {
    "providing_information",
    "wants_escalation",
    "issue_resolved",
    "asking_question",
    "contradicting_previous",
    "expressing_frustration",
    "unclear",
}

_SYSTEM_PROMPT = """\
You classify the intent of a client message in a technical support conversation.

Respond with JSON only. No text outside the JSON object.

Intent must be exactly one of:
- providing_information: client is answering questions or sharing technical details (including brief or telegraphic answers like "401s", "production", "identity", "all users", "NexaAuth")
- wants_escalation: client EXPLICITLY requests a human, supervisor, CSM, escalation, or says they want to stop using the bot. Short technical answers are NEVER escalation requests.
- issue_resolved: client is confirming the issue is fixed or no longer present
- asking_question: client is asking something rather than reporting an issue
- contradicting_previous: client is correcting or contradicting something they said before
- expressing_frustration: client is venting, expressing anger, or dissatisfaction (but not explicitly requesting escalation)
- unclear: message is too vague, short, or ambiguous to classify

IMPORTANT: In technical support, short answers like "api", "identity", "all", "production", "401s" are ALWAYS providing_information — the client is answering your questions. Only classify as wants_escalation if the message contains a clear, unambiguous request to speak to a human or stop the automated flow.

JSON schema:
{
  "intent": "<one of the above>",
  "confidence": <float 0.0-1.0>,
  "signals": ["<phrase that led to this classification>"],
  "reasoning": "<one sentence>"
}"""


@dataclass
class IntentResult:
    intent: str
    confidence: float
    signals: list[str] = field(default_factory=list)
    reasoning: str = ""


class IntentClassifier:
    """
    Classifies client intent using a single lightweight Claude API call.
    max_tokens=150, no tools — optimised for speed.
    """

    def __init__(self, claude: ClaudeClient) -> None:
        self._claude = claude

    @traceable(name="intent_llm_call", run_type="llm")
    async def classify(self, message: str, conversation_history: list) -> IntentResult:
        context_lines = [
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in conversation_history[-4:]
        ]
        context_block = "\n".join(context_lines) if context_lines else "(start of conversation)"

        user_content = (
            f"Recent conversation:\n{context_block}\n\n"
            f"Classify this message: \"{message}\""
        )

        try:
            raw = await self._claude.complete(
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=150,
            )
            parsed = self._claude.safe_parse_json(raw)
            if not parsed:
                return _fallback("empty_response")

            intent = parsed.get("intent", "unclear")
            if intent not in _VALID_INTENTS:
                intent = "unclear"

            return IntentResult(
                intent=intent,
                confidence=float(parsed.get("confidence", 0.5)),
                signals=parsed.get("signals", []),
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception as exc:
            log.warning("intent_classifier.failed", error=str(exc))
            return _fallback(str(exc))


def _fallback(reason: str) -> IntentResult:
    return IntentResult(intent="unclear", confidence=0.0, signals=[], reasoning=f"classifier_error: {reason}")
