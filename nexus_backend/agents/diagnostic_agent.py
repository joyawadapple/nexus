"""
Diagnostic Agent — RAG-powered root cause analysis.

Phase 2a of the Nexus orchestration pipeline (runs in parallel with resolution_agent).
Requires triage_finding to be complete before running.
Confidence threshold: 0.75
"""
from __future__ import annotations

import re
from typing import Any

import structlog

from agents.agent_base import AnalysisResult, BaseAgent, Decision, LoadedData, ReasoningResult
from agents.agent_utils import validate_agent_output
from models.agent_models import QuestionForClient
from models.report_models import DiagnosticReport, Hypothesis, RAGResult
from prompts.diagnostic_agent_prompt import build_diagnostic_prompt

log = structlog.get_logger("diagnostic_agent")

RECENT_CHANGES_PATTERN = re.compile(
    r"\b(deployed|deploy|updated|rotated|changed|migrated|upgrade|rollback|config|key rotation|restart)\b",
    re.IGNORECASE,
)
INTERMITTENT_PATTERN = re.compile(
    r"\b(intermittent|sometimes|occasionally|random|not always|50%|partial)\b",
    re.IGNORECASE,
)
CONSISTENT_PATTERN = re.compile(
    r"\b(always|every request|100%|consistent|every time|all requests)\b",
    re.IGNORECASE,
)
REGION_PATTERN = re.compile(
    r"\b(eu-west|eu-east|us-east|us-west|apac|latam|all regions|specific region|one region)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_PATTERN = re.compile(
    r"^\s*(yes|yep|yeah|correct|right|confirmed|exactly|that'?s right|that is correct)\b",
    re.IGNORECASE,
)

_claude = None
_rag_engine = None


def set_services(claude, rag_engine) -> None:
    global _claude, _rag_engine
    _claude = claude
    _rag_engine = rag_engine


class DiagnosticAgent(BaseAgent):
    agent_id = "diagnostic_agent"
    confidence_threshold = 0.75

    def __init__(self, triage_finding=None) -> None:
        super().__init__()
        self.triage_finding = triage_finding
        self.dynamic_threshold: float | None = None

    # ── Step 1: LOAD ──────────────────────────────────────────────────────────

    async def load(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list[dict],
        **kwargs,
    ) -> LoadedData:
        triage = kwargs.get("triage_finding") or self.triage_finding
        product = ""
        error_message = ""
        if triage:
            if hasattr(triage, "issue"):
                product = triage.issue.product or ""
                error_message = triage.issue.error_message or ""
            elif isinstance(triage, dict):
                product = triage.get("product", "")
                error_message = triage.get("error_message", "")

        errors_db = db_data.get("errors", {})
        products_db = db_data.get("products", {})

        # Extract stable error code for DB lookup — handles stack traces and plain messages.
        # _guess_error_code is kept only for db_records["error_code"] (fallback label in generate()).
        error_code = _guess_error_code(error_message)
        normalized_code = extract_error_from_text(error_message)
        product_errors = errors_db.get(product, {})
        error_entry = _lookup_product_error(product_errors, normalized_code)

        log.debug(
            "error_db.lookup",
            product=product,
            raw_error=error_message,
            normalized_code=normalized_code,
            matched_category=(error_entry or {}).get("category"),
            match_hit=bool(error_entry),
        )

        # Get version bugs — collect from ALL historical versions as pattern indicators
        version_bugs = []
        if product:
            prod_data = products_db.get(product, {})
            known_bugs = prod_data.get("known_bugs", {})
            for ver_bugs in known_bugs.values():
                version_bugs.extend(ver_bugs)

        # RAG retrieval — include recent user messages for richer query
        rag_results = []
        if _rag_engine and product:
            recent_msgs = " ".join([
                m["content"] for m in conversation_history[-4:]
                if m["role"] == "user"
            ])
            query = f"{product} {error_message} {recent_msgs}".strip()
            rag_results = _rag_engine.query(
                query_text=query,
                category="diagnostic",
                product=product,
                top_k=3,
            )

        return LoadedData(
            session_id=session_id,
            db_records={
                "error_entry": error_entry,
                "version_bugs": version_bugs,
                "product": product,
                "error_message": error_message,
                "error_code": error_code,
            },
            conversation_history=conversation_history,
            rag_context=rag_results,
        )

    # ── Step 2: ANALYZE ───────────────────────────────────────────────────────

    async def analyze(self, loaded_data: LoadedData) -> AnalysisResult:
        error_entry = loaded_data.db_records.get("error_entry", {})
        version_bugs = loaded_data.db_records.get("version_bugs", [])
        rag_results = loaded_data.rag_context

        fields_in_db = []
        if error_entry:
            fields_in_db.append("error_taxonomy")
        if version_bugs:
            fields_in_db.append("version_bugs")
        if rag_results:
            fields_in_db.append("rag_context")

        # These are the confirmable fields that increase confidence
        fields_missing = [
            f for f in ("recent_changes", "reproducibility")
            if f not in fields_in_db
        ]

        return AnalysisResult(
            total_fields=4,  # error_match, rag_match, version_bug, client_confirmation
            fields_in_db=fields_in_db,
            fields_missing=fields_missing,
            preliminary_data=loaded_data.db_records,
        )

    # ── Step 3: REASON ────────────────────────────────────────────────────────

    async def reason(
        self,
        analysis: AnalysisResult,
        conversation_history: list[dict],
    ) -> ReasoningResult:
        full_text = " ".join(
            msg["content"] for msg in conversation_history if msg["role"] == "user"
        )
        full_text_lower = full_text.lower()

        confirmed = []
        extracted: dict[str, Any] = {}

        # Inherit environment from triage — it already confirmed this field
        triage = self.triage_finding
        if triage:
            env = None
            if hasattr(triage, "issue") and triage.issue and triage.issue.environment:
                env = triage.issue.environment
            elif isinstance(triage, dict):
                env = (triage.get("issue") or {}).get("environment")
            if env:
                confirmed.append("environment")
                extracted["environment"] = env

        # Check if client mentioned recent changes
        if RECENT_CHANGES_PATTERN.search(full_text):
            confirmed.append("recent_changes")
            extracted["recent_changes"] = True

        # Check reproducibility
        if CONSISTENT_PATTERN.search(full_text):
            confirmed.append("reproducibility")
            extracted["reproducibility"] = "consistent"
        elif INTERMITTENT_PATTERN.search(full_text):
            confirmed.append("reproducibility")
            extracted["reproducibility"] = "intermittent"

        # Check caching layer
        if "redis" in full_text_lower:
            confirmed.append("caching_layer")
            extracted["caching_layer"] = "redis"
        elif "memcache" in full_text_lower:
            confirmed.append("caching_layer")
            extracted["caching_layer"] = "memcached"
        elif "in-memory" in full_text_lower or "in memory" in full_text_lower:
            confirmed.append("caching_layer")
            extracted["caching_layer"] = "in_memory"

        # Check region (for connectivity errors)
        if REGION_PATTERN.search(full_text):
            confirmed.append("affected_region")
            match = REGION_PATTERN.search(full_text)
            extracted["affected_region"] = match.group(0) if match else "specific"

        # ── Q&A affirmative detection ─────────────────────────────────────────
        # If the user gave a short affirmative ("yes", "correct", etc.), look at the
        # immediately preceding assistant message to infer what field was confirmed.
        for idx, msg in enumerate(conversation_history):
            if msg["role"] == "user" and _AFFIRMATIVE_PATTERN.match(msg["content"]):
                for prev in reversed(conversation_history[:idx]):
                    if prev["role"] == "assistant":
                        q_lower = prev["content"].lower()
                        if "recent_changes" not in confirmed and any(
                            kw in q_lower for kw in ("rotat", "key", "deploy", "config", "restart", "migrat", "updated")
                        ):
                            confirmed.append("recent_changes")
                            extracted["recent_changes"] = True
                        if "reproducibility" not in confirmed:
                            if any(kw in q_lower for kw in ("consistent", "every request", "always")):
                                confirmed.append("reproducibility")
                                extracted["reproducibility"] = "consistent"
                            elif any(kw in q_lower for kw in ("intermittent", "sometimes", "random", "occasional")):
                                confirmed.append("reproducibility")
                                extracted["reproducibility"] = "intermittent"
                        break

        questions = []
        if "recent_changes" not in confirmed:
            questions.append("Have you made any recent changes — a deploy, config update, or API key rotation?")
        if "reproducibility" not in confirmed:
            questions.append("Is this happening consistently with every request, or intermittently?")

        return ReasoningResult(
            questions_to_ask=questions[:2],
            fields_confirmed_from_conversation=confirmed,
            discrepancies=[],
            field_values_extracted=extracted,
        )

    # ── Step 4: DECIDE ────────────────────────────────────────────────────────

    async def decide(self, reasoning: ReasoningResult) -> Decision:
        # Base confidence from DB matches (scored in generate step)
        confirmed_count = len(reasoning.fields_confirmed_from_conversation)
        # Start with pre-computed score, add client confirmations
        base_score = 0.0
        db = getattr(self, "_loaded_data", None)

        questions = reasoning.questions_to_ask[:2]
        # If we have high DB confidence, don't bother asking
        # This is a simplification — full scoring happens in generate()
        ready = confirmed_count >= 1  # At least one confirmation

        return Decision(
            should_ask_questions=bool(questions),
            questions=questions,
            ready_to_finalize=ready,
        )

    # ── Confidence calculation ─────────────────────────────────────────────────

    def _calculate_diagnostic_confidence(
        self,
        db_records: dict,
        rag_results: list,
        conversation_history: list,
        confirmed_fields: list[str] | None = None,
    ) -> float:
        """
        Accumulates evidence signals from the full conversation history.
        Static DB signals are the same every turn; RAG and conversation signals
        improve as more information is provided.
        """
        confidence = 0.0
        full_text = " ".join([
            m["content"].lower() for m in conversation_history
            if m["role"] == "user"
        ])

        # Static signal — error code matched in DB (same every turn)
        if db_records.get("error_entry"):
            confidence += 0.40

        # RAG signal — improves as query gets richer each turn
        if rag_results:
            top_sim = max(r.get("similarity", 0) for r in rag_results)
            if top_sim >= 0.85:
                confidence += 0.25
            elif top_sim >= 0.70:
                confidence += 0.15
            elif top_sim >= 0.55:
                confidence += 0.08

        # Cumulative evidence signals — scan entire conversation history
        evidence_signals = [
            (["rotat", "key rotation", "rotated", "new key"], 0.10),
            (["all requests", "every request", "consistent"], 0.05),
            (["redis", "memcache", "in-memory"], 0.05),
            (["deploy", "deployment", "release", "pushed"], 0.05),
            (["204", "purge succeeded", "cache cleared", "cache purge", "clearing"], 0.10),
            (["staging key", "wrong key", "wrong environment"], 0.10),
            (["jwt", "oauth", "api key", "service account"], 0.05),
            (["restarted", "redeployed", "rolled back"], 0.05),
        ]
        for signals, weight in evidence_signals:
            if any(s in full_text for s in signals):
                confidence += weight

        # Q&A-confirmed fields — client explicitly answered a direct question.
        # Only add the bonus when the direct keyword scan didn't already capture it
        # (avoids double-counting when client said "I rotated my key" AND later "yes").
        _confirmed = set(confirmed_fields or [])
        if "recent_changes" in _confirmed:
            if not any(s in full_text for s in ["rotat", "key rotation", "rotated", "new key"]):
                confidence += 0.10

        return min(round(confidence, 2), 1.0)

    # ── Step 5: GENERATE ──────────────────────────────────────────────────────

    async def generate(
        self,
        decision: Decision,
        loaded_data: LoadedData,
        reasoning: ReasoningResult,
    ) -> DiagnosticReport:
        db = loaded_data.db_records
        error_entry = db.get("error_entry") or {}
        version_bugs = db.get("version_bugs", [])
        product = db.get("product", "")
        error_message = db.get("error_message", "")
        rag_results = loaded_data.rag_context

        # ── Mathematical confidence scoring (accumulates across conversation) ─
        calculated_confidence = self._calculate_diagnostic_confidence(
            db_records=db,
            rag_results=rag_results,
            conversation_history=loaded_data.conversation_history,
            confirmed_fields=reasoning.fields_confirmed_from_conversation,
        )

        # Flag the hypothesis as inferred when evidence is weak (low RAG similarity + no DB match)
        rag_top_sim = max(r.get("similarity", 0) for r in rag_results) if rag_results else 0.0
        error_db_match = bool(db.get("error_entry"))
        hypothesis_inferred = rag_top_sim < 0.70 and not error_db_match

        # ── LLM hypothesis formation ──────────────────────────────────────────
        primary_hypothesis = None
        alternative_hypotheses = []
        rag_results_used = []

        if _claude:
            triage_summary = {}
            if self.triage_finding:
                tf = self.triage_finding
                if hasattr(tf, "issue"):
                    triage_summary = {
                        "product": tf.issue.product,
                        "error_message": tf.issue.error_message,
                        "environment": tf.issue.environment,
                        "started_at": tf.issue.started_at,
                        "impact_scope": tf.issue.impact_scope,
                    }

            try:
                prompt = build_diagnostic_prompt(
                    triage_summary=triage_summary,
                    error_db_entry=error_entry,
                    version_bugs=version_bugs,
                    rag_results=rag_results,
                    conversation_history=loaded_data.conversation_history,
                )
                messages = [{"role": "user", "content": "Analyze this issue and form your diagnostic hypothesis."}]
                raw = await _claude.complete(system=prompt, messages=messages, max_tokens=1200)
                parsed = _claude.safe_parse_json(raw)

                if parsed:
                    # Parse primary hypothesis
                    ph = parsed.get("primary_hypothesis", {})
                    if ph:
                        primary_hypothesis = Hypothesis(
                            cause=ph.get("cause", "Unknown cause"),
                            confidence=calculated_confidence,  # Use our calculated score
                            evidence=ph.get("evidence", []),
                        )

                    # Parse alternatives
                    for alt in parsed.get("alternative_hypotheses", []):
                        alternative_hypotheses.append(Hypothesis(
                            cause=alt.get("cause", ""),
                            confidence=alt.get("confidence", 0.0),
                            why_less_likely=alt.get("why_less_likely"),
                        ))

                    # Parse RAG results used
                    for r in parsed.get("rag_results_used", []):
                        rag_results_used.append(RAGResult(
                            source=r.get("source", ""),
                            similarity=r.get("similarity", 0.0),
                            excerpt_summary=r.get("excerpt_summary", ""),
                        ))

                    # Get questions from LLM
                    decision.questions = [
                        QuestionForClient(**q) if isinstance(q, dict) else q
                        for q in parsed.get("questions_for_client", [])
                    ]

            except Exception as e:
                log.warning("diagnostic_agent.claude_failed", error=str(e))

        # Fallback if LLM failed
        if not primary_hypothesis and error_entry:
            causes = error_entry.get("known_causes", [])
            primary_hypothesis = Hypothesis(
                cause=causes[0].replace("_", " ").title() if causes else "Unknown cause",
                confidence=calculated_confidence,
                evidence=[f"Error code matched {db.get('error_code', 'unknown')} in error taxonomy"],
            )
            if len(causes) > 1:
                alternative_hypotheses = [
                    Hypothesis(cause=c.replace("_", " ").title(), confidence=0.20)
                    for c in causes[1:3]
                ]

        # Format RAG results if none from LLM
        if not rag_results_used and rag_results:
            for r in rag_results[:3]:
                rag_results_used.append(RAGResult(
                    source=r.get("source", ""),
                    similarity=r.get("similarity", 0.0),
                    excerpt_summary=r.get("excerpt_summary", ""),
                ))

        # Validate
        agent_output = {
            "product": product,
            "primary_hypothesis": primary_hypothesis.model_dump() if primary_hypothesis else {},
            "confidence": calculated_confidence,
        }
        validation = validate_agent_output(
            "diagnostic_agent", agent_output, loaded_data.db_records, calculated_confidence
        )

        # Questions for client — only ask about fields not yet confirmed
        confirmed_set = set(reasoning.fields_confirmed_from_conversation)
        questions_for_client = [
            q for q in [
                QuestionForClient(
                    field="recent_changes",
                    question="Have you made any recent changes — a deploy, config update, or API key rotation?",
                    blocking=False,
                    priority="medium",
                ) if "recent_changes" not in confirmed_set else None,
                QuestionForClient(
                    field="reproducibility",
                    question="Is this happening consistently with every request, or intermittently?",
                    blocking=False,
                    priority="medium",
                ) if "reproducibility" not in confirmed_set else None,
            ]
            if q is not None
        ]

        # Detect novel issue — no RAG match and no known-bug match
        _rag_has_match = any(r.similarity >= 0.35 for r in rag_results_used) if rag_results_used else (
            bool(rag_results) and rag_results[0].get("similarity", 0) >= 0.35
        )
        _bugs_has_match = any("no matching known bugs" not in v for v in (version_bugs or []))
        if not _rag_has_match and not _bugs_has_match and "reproduction_steps" not in confirmed_set:
            questions_for_client.insert(0, QuestionForClient(
                field="reproduction_steps",
                question="This doesn't match any documented issues. Could you share the exact error message or response body, and describe step by step what triggers it?",
                blocking=False,
                priority="high",
            ))

        report = DiagnosticReport(
            session_id=loaded_data.session_id,
            primary_hypothesis=primary_hypothesis,
            alternative_hypotheses=alternative_hypotheses,
            rag_results_used=rag_results_used,
            version_bugs_checked=version_bugs or [f"{product} current version — no matching known bugs"],
            confidence=calculated_confidence,
            questions_for_client=questions_for_client[:1],
            hallucination_flags=validation.flags,
            hypothesis_inferred=hypothesis_inferred,
            completed=calculated_confidence >= (self.dynamic_threshold if self.dynamic_threshold is not None else self.confidence_threshold),
        )

        self.log_reasoning(
            "GENERATE",
            f"confidence={calculated_confidence:.2f}, hypothesis='{primary_hypothesis.cause if primary_hypothesis else 'none'}'",
            f"rag_top_sim={rag_results[0]['similarity'] if rag_results else 0:.2f}",
        )

        _threshold_used = self.dynamic_threshold if self.dynamic_threshold is not None else self.confidence_threshold
        log.info(
            "diagnostic_agent.generate_complete",
            completed=report.completed,
            confidence=round(calculated_confidence, 4),
            threshold_used=round(_threshold_used, 4),
            dynamic_threshold=self.dynamic_threshold,
            threshold_condition_met=calculated_confidence >= _threshold_used,
            rag_top_sim=round(rag_results[0]["similarity"], 4) if rag_results else None,
            hypothesis=primary_hypothesis.cause if primary_hypothesis else None,
            alt_count=len(alternative_hypotheses),
            questions_remaining=len(questions_for_client),
        )

        return report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _guess_error_code(error_message: str) -> str:
    msg = (error_message or "").lower()
    if "401" in msg or "invalid_token" in msg or "unauthorized" in msg:
        return "401_invalid_token"
    if "401" in msg and "insufficient_scope" in msg:
        return "401_insufficient_scope"
    if "429" in msg or "rate_limit" in msg or "too many" in msg:
        return "429_rate_limit_exceeded"
    if "403" in msg or "forbidden" in msg or "access_denied" in msg or "cors" in msg:
        return "403_access_denied"
    if "409" in msg or "duplicate" in msg or "idempotency" in msg:
        return "409_duplicate_payment"
    if "500" in msg or "internal error" in msg:
        return "500_internal_error"
    if "503" in msg or "unavailable" in msg:
        return "503_service_unavailable"
    if "webhook" in msg or "delivery" in msg:
        return "webhook_delivery_failure"
    return ""


def normalize_error_code(raw_error: str) -> str:
    """Extract the first whitespace-delimited token and strip trailing punctuation.

    Examples:
        "403 error on payment requests" -> "403"
        "401 invalid_token"             -> "401"
        "rate_limit_exceeded"           -> "rate_limit_exceeded"
        ""                              -> ""
    """
    if not raw_error:
        return ""
    first_token = raw_error.strip().split()[0]
    return first_token.rstrip(".,;:")


def extract_error_from_text(raw: str) -> str:
    """Extract a meaningful error code from raw text including stack traces.

    Tries in order:
    1. HTTP status code (4xx/5xx) anywhere in the text
    2. JSON-like 'error' field value
    3. Falls back to normalize_error_code (first token)
    """
    if not raw:
        return ""
    http_match = re.search(r"\b(4\d{2}|5\d{2})\b", raw)
    if http_match:
        return http_match.group(1)
    code_match = re.search(r"'error':\s*'([^']+)'", raw)
    if code_match:
        return code_match.group(1)
    return normalize_error_code(raw)


def _lookup_product_error(product_errors: dict, normalized_code: str) -> dict | None:
    """Look up an error entry by normalized code prefix.

    Tries exact match first, then finds the first key starting with
    normalized_code + "_". Handles both bare keys ("rate_limit_exceeded")
    and prefixed keys ("403_forbidden", "403_access_denied").
    """
    if not normalized_code:
        return None
    exact = product_errors.get(normalized_code)
    if exact is not None:
        return exact
    prefix = normalized_code + "_"
    for key, entry in product_errors.items():
        if key.startswith(prefix):
            return entry
    return None


def _find_closest_error(product_errors: dict, error_message: str) -> dict | None:
    """Find the best matching error entry by keyword matching."""
    msg = (error_message or "").lower()
    for code, entry in product_errors.items():
        code_lower = code.lower()
        if any(part in msg for part in code_lower.split("_") if len(part) > 2):
            return entry
    return None
