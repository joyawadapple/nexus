"""
Triage Agent — classifies issue before any diagnosis happens.

Phase 1 of the Nexus orchestration pipeline. Always runs first.
Collects: product, error_message, environment, started_at, impact_scope.
Determines: severity, known_incident, recurring.
Confidence threshold: 0.90
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import structlog

from agents.agent_base import AnalysisResult, BaseAgent, Decision, LoadedData, ReasoningResult
from agents.agent_utils import validate_agent_output
from models.agent_models import QuestionForClient
from models.report_models import ClientInfo, IssueInfo, TriageReport
from prompts.triage_agent_prompt import build_triage_prompt

log = structlog.get_logger("triage_agent")

# ── Field definitions ─────────────────────────────────────────────────────────

TRIAGE_FIELDS = {
    "product": {
        "blocking": True,
        "max_attempts": 2,
        "questions": [
            "Which NexaCloud product are you having trouble with?",
            "Is this NexaAuth, NexaStore, NexaMsg, or NexaPay?",
        ],
        "if_missing": "cannot_route",
    },
    "error_message": {
        "blocking": True,
        "max_attempts": 2,
        "questions": [
            "What error message or behaviour are you seeing?",
            "Can you share the exact error code or message from your logs?",
        ],
        "if_missing": "flag_for_human",
    },
    "environment": {
        "blocking": False,
        "max_attempts": 2,
        "questions": [
            "Is this happening in production or a non-production environment?",
            "Just to confirm — is this affecting live traffic?",
        ],
        "if_missing": "assume_production",
    },
    "started_at": {
        "blocking": False,
        "max_attempts": 1,
        "questions": ["When did this start — roughly how long ago?"],
        "if_missing": "mark_unknown",
    },
    "impact_scope": {
        "blocking": False,
        "max_attempts": 1,
        "questions": ["Is this affecting all your users, a subset, or just your internal testing?"],
        "if_missing": "assume_all_users",
    },
}

VALID_PRODUCTS = {"NexaAuth", "NexaStore", "NexaMsg", "NexaPay"}

# Patterns for extracting triage fields from conversation
PRODUCT_PATTERN = re.compile(
    r"\b(NexaAuth|NexaStore|NexaMsg|NexaPay)\b", re.IGNORECASE
)
ENV_PATTERNS = {
    "production": re.compile(r"\b(production|prod|live|live traffic|real users)\b", re.IGNORECASE),
    "staging": re.compile(r"\b(staging|stage|pre-prod|uat)\b", re.IGNORECASE),
    "development": re.compile(r"\b(development|dev|local|sandbox|test)\b", re.IGNORECASE),
}
SCOPE_PATTERNS = {
    "all_users": re.compile(r"\b(all users|everyone|all traffic|all requests|completely down)\b", re.IGNORECASE),
    "subset": re.compile(r"\b(some users|subset|specific|certain|partial)\b", re.IGNORECASE),
    "internal": re.compile(r"\b(internal|just me|our team|testing|not yet live)\b", re.IGNORECASE),
}
RECENCY_PATTERN = re.compile(r"\b(\d+)\s*(minute|hour|day|week)s?\s*ago\b|\b(today|yesterday|this morning|last night)\b", re.IGNORECASE)
ERROR_CODE_PATTERN = re.compile(r"\b(4\d{2}|5\d{2})\b|\b(invalid_token|rate_limit|unauthorized|forbidden|not_found|timeout|error)\b", re.IGNORECASE)
RECURRING_KEYWORDS = re.compile(
    r"\b(again|same|third|recurring|keep|keeps|another|already|previously|before)\b",
    re.IGNORECASE,
)


# Module-level singletons (injected at startup)
_claude = None
_product_identifier = None
_severity_scorer = None


def set_claude_client(claude) -> None:
    global _claude
    _claude = claude


def set_services(claude, rag_engine=None) -> None:
    """Inject all triage dependencies. Call instead of set_claude_client at startup."""
    global _claude, _product_identifier, _severity_scorer
    from agents.severity_scorer import SeverityScorer
    _claude = claude
    _severity_scorer = SeverityScorer()
    if rag_engine is not None:
        from core.product_identifier import ProductIdentifier
        _product_identifier = ProductIdentifier(rag_engine)
    else:
        _product_identifier = None


class TriageAgent(BaseAgent):
    agent_id = "triage_agent"
    confidence_threshold = 0.90

    def __init__(self) -> None:
        super().__init__()
        self.dynamic_threshold: float | None = None

    # ── Step 1: LOAD ──────────────────────────────────────────────────────────

    async def load(
        self,
        session_id: str,
        db_data: dict,
        conversation_history: list[dict],
        **kwargs,
    ) -> LoadedData:
        clients = db_data.get("clients", {})
        products = db_data.get("products", {})

        return LoadedData(
            session_id=session_id,
            db_records={
                "clients": clients,
                "products": products,
                "_client_id": db_data.get("_client_id", ""),
            },
            conversation_history=conversation_history,
        )

    # ── Step 2: ANALYZE ───────────────────────────────────────────────────────

    async def analyze(self, loaded_data: LoadedData) -> AnalysisResult:
        db = loaded_data.db_records
        client_id = db.get("_client_id", "")
        client_data = db.get("clients", {}).get(client_id, {})

        # Fields always loaded from DB
        fields_in_db = ["client_company", "client_tier", "client_sla"]

        # Product is inferable when client subscribes to exactly one product
        products_subscribed = client_data.get("products_subscribed", [])
        if len(products_subscribed) == 1:
            fields_in_db.append("product")

        # Environment defaults to production — safe assumption, downgrade only if client says otherwise
        fields_in_db.append("environment")

        # Only fields not pre-confirmable from DB genuinely need client input
        db_confirmed = {f for f in ("product", "environment") if f in fields_in_db}
        fields_missing = [f for f in TRIAGE_FIELDS if f not in db_confirmed]

        return AnalysisResult(
            total_fields=5,
            fields_in_db=fields_in_db,
            fields_missing=fields_missing,
            preliminary_data=db,
        )

    # ── Step 3: REASON ────────────────────────────────────────────────────────

    async def reason(
        self,
        analysis: AnalysisResult,
        conversation_history: list[dict],
    ) -> ReasoningResult:
        """Scan conversation history for triage field confirmations.

        Pre-populates product (single-subscription clients) and environment
        (safe production default) from DB so they count toward confidence
        even before the client mentions them.
        """
        db = analysis.preliminary_data
        client_id = db.get("_client_id", "")
        client_data = db.get("clients", {}).get(client_id, {})

        extracted: dict[str, Any] = {}
        inferred_field_names: set[str] = set()
        full_text = " ".join(
            msg["content"] for msg in conversation_history if msg["role"] == "user"
        )

        # ── Product ───────────────────────────────────────────────────────────
        _pm = None
        if _product_identifier:
            _pm = _product_identifier.identify(full_text)
            if _pm.product:
                extracted["product"] = _pm.product
                if _pm.inferred:
                    inferred_field_names.add("product")
            extracted["_product_needs_clarification"] = _pm.needs_clarification
        else:
            # Fallback: regex + single-subscription DB inference + webhook heuristic
            product_match = PRODUCT_PATTERN.search(full_text)
            if product_match:
                raw = product_match.group(1)
                for p in VALID_PRODUCTS:
                    if p.lower() == raw.lower():
                        extracted["product"] = p
                        break
            else:
                products_subscribed = client_data.get("products_subscribed", [])
                if len(products_subscribed) == 1:
                    raw = products_subscribed[0]
                    for p in VALID_PRODUCTS:
                        if p.lower() == raw.lower():
                            extracted["product"] = p
                            break
                if not extracted.get("product") and re.search(r"\bwebhook", full_text, re.IGNORECASE):
                    extracted["product"] = "NexaMsg"

        # Multi-product detection: flag if multiple NexaCloud products are mentioned.
        _mentioned = [p for p in VALID_PRODUCTS if p.lower() in full_text.lower()]
        if len(_mentioned) > 1:
            extracted["mentioned_products"] = _mentioned

        # ── Environment ───────────────────────────────────────────────────────
        env_from_conversation = False
        for env, pattern in ENV_PATTERNS.items():
            if pattern.search(full_text):
                extracted["environment"] = env
                env_from_conversation = True
                break
        if not env_from_conversation:
            # Safe default — production. Platinum stays critical until client says otherwise.
            extracted["environment"] = "production"
            inferred_field_names.add("environment")
        # Store flag so generate() won't let the LLM override this with a guess
        extracted["_env_from_conversation"] = env_from_conversation

        # ── Error message ─────────────────────────────────────────────────────
        if ERROR_CODE_PATTERN.search(full_text):
            # Prefer first message that contains a numeric HTTP error code (4xx/5xx)
            error_msg = None
            for msg in conversation_history:
                if msg["role"] == "user" and len(msg["content"]) > 5:
                    if re.search(r"\b(4\d{2}|5\d{2})\b", msg["content"]):
                        error_msg = msg["content"][:200]
                        break
            if not error_msg:
                # Fall back to last user message matching any error keyword
                for msg in reversed(conversation_history):
                    if msg["role"] == "user" and len(msg["content"]) > 5:
                        if ERROR_CODE_PATTERN.search(msg["content"]):
                            error_msg = msg["content"][:200]
                            break
            if error_msg:
                extracted["error_message"] = error_msg
                _noisy = "traceback" in error_msg.lower() or len(error_msg) > 100
                if _noisy:
                    inferred_field_names.add("error_message")

        # ── Start time ────────────────────────────────────────────────────────
        if RECENCY_PATTERN.search(full_text):
            match = RECENCY_PATTERN.search(full_text)
            extracted["started_at"] = match.group(0) if match else "recently"

        # ── Impact scope ──────────────────────────────────────────────────────
        for scope, pattern in SCOPE_PATTERNS.items():
            if pattern.search(full_text):
                extracted["impact_scope"] = scope
                break

        # confirmed = all fields we have values for (conversation OR DB-inferred)
        confirmed = [f for f in extracted if not f.startswith("_")]

        # Build as dicts so is_confirmation can bypass the "already confirmed" filter
        questions_needed: list[dict] = [
            {
                "field": f,
                "question": TRIAGE_FIELDS[f]["questions"][0],
                "blocking": TRIAGE_FIELDS[f]["blocking"],
                "priority": "high" if TRIAGE_FIELDS[f]["blocking"] else "medium",
                "is_confirmation": False,
            }
            for f in TRIAGE_FIELDS
            if f not in confirmed
        ]

        # For inferred product: add confirming question even though product IS in confirmed
        # (reuses cached _pm, no second identify() call)
        if _pm is not None and "product" in inferred_field_names and extracted.get("product"):
            clarification_q_dict = _product_identifier.build_clarification_question(_pm)
            questions_needed.insert(0, clarification_q_dict)

        # Filter: standard questions only for missing fields; confirmation questions always pass through
        questions_needed = [
            q for q in questions_needed
            if q["field"] not in confirmed
            or q.get("is_confirmation", False)
        ]

        extracted["_inferred_field_names"] = list(inferred_field_names)
        extracted["_rich_questions"] = questions_needed

        return ReasoningResult(
            questions_to_ask=[q["question"] for q in questions_needed],
            fields_confirmed_from_conversation=confirmed,
            discrepancies=[],
            field_values_extracted=extracted,
        )

    # ── Step 4: DECIDE ────────────────────────────────────────────────────────

    async def decide(self, reasoning: ReasoningResult) -> Decision:
        confirmed = set(reasoning.fields_confirmed_from_conversation)
        _inferred = set(reasoning.field_values_extracted.get("_inferred_field_names", []))
        confirmed_score = sum(0.5 if f in _inferred else 1.0 for f in confirmed)
        confidence = self.calculate_confidence(confirmed_score, 5)

        # Check if blocking fields are missing
        blocking_missing = [
            f for f in ("product", "error_message")
            if f not in confirmed
        ]

        # Use rich questions from reason() — includes confirmation questions for inferred fields.
        # Fallback to TRIAGE_FIELDS rebuild for legacy callers that bypass reason().
        _rich_questions = reasoning.field_values_extracted.get("_rich_questions", [])
        if _rich_questions:
            questions = [q["question"] for q in _rich_questions][:1]
        else:
            questions = [
                TRIAGE_FIELDS[f]["questions"][0]
                for f in ("product", "error_message", "environment", "started_at", "impact_scope")
                if f not in confirmed
            ][:1]

        thresh = self.dynamic_threshold if self.dynamic_threshold is not None else self.confidence_threshold
        ready = confidence >= thresh and not blocking_missing

        return Decision(
            should_ask_questions=bool(questions),
            questions=questions,
            ready_to_finalize=ready,
        )

    def apply_severity_rules(self, tier: str, environment: str | None) -> str:
        """Determine severity from client tier and environment.

        Called AFTER LLM refinement so environment reflects the final confirmed value.
        Platinum + production (or unknown) defaults to critical — safer to downgrade
        than to miss a critical incident.
        """
        env = environment or "production"  # Treat unknown as production (safe default)
        if tier == "platinum" and env in ("production", "unknown"):
            return "critical"
        if env == "production":
            return "high"  # gold and standard get high in production
        if tier == "platinum":
            return "high"  # platinum in staging still gets high
        if env == "staging":
            return "medium"
        return "low"

    # ── Step 5: GENERATE ──────────────────────────────────────────────────────

    async def generate(
        self,
        decision: Decision,
        loaded_data: LoadedData,
        reasoning: ReasoningResult,
    ) -> TriageReport:
        extracted = reasoning.field_values_extracted
        db = loaded_data.db_records
        clients_db = db.get("clients", {})
        products_db = db.get("products", {})

        # Get client info from db_records (orchestrator injects client_id)
        client_id = loaded_data.db_records.get("_client_id", "")
        client_data = clients_db.get(client_id, {})

        product = extracted.get("product")
        error_message = extracted.get("error_message", "")
        environment = extracted.get("environment", "production")  # Safe default
        started_at = extracted.get("started_at")
        impact_scope = extracted.get("impact_scope", "all_users")  # Safe default

        # Check for known incident
        known_incident = False
        if product:
            product_data = products_db.get(product, {})
            if product_data.get("active_incident"):
                known_incident = True

        recent_tickets = client_data.get("recent_tickets", [])
        tier = client_data.get("tier", "standard")

        # Calculate SLA deadline
        sla_hours = client_data.get("sla_hours", 24)
        sla_deadline = _calculate_sla_deadline(sla_hours)

        # Confidence: if known_incident, auto 1.0
        # Inferred fields count as 0.5 — they represent a guess, not a confirmation.
        _inferred = set(extracted.get("_inferred_field_names", []))
        confirmed_fields_set = set(reasoning.fields_confirmed_from_conversation)
        confirmed_count = len(confirmed_fields_set)
        if known_incident:
            confidence = 1.0
            confirmed_count = 5
        else:
            confirmed_score = sum(0.5 if f in _inferred else 1.0 for f in confirmed_fields_set)
            confidence = self.calculate_confidence(confirmed_score, 5)

        # Detect whether the conversation explicitly mentions this is a recurring issue
        full_text_for_recurring = " ".join(
            msg["content"] for msg in loaded_data.conversation_history if msg["role"] == "user"
        )
        recurring_mentioned = bool(RECURRING_KEYWORDS.search(full_text_for_recurring))

        # Use Claude to refine extraction if client is available
        if _claude and len(loaded_data.conversation_history) > 0:
            try:
                prompt = build_triage_prompt(
                    client_info=client_data,
                    product_status=products_db,
                    conversation_history=loaded_data.conversation_history,
                )
                messages = [{"role": "user", "content": "Based on the conversation, extract triage fields as JSON."}]
                raw = await _claude.complete(system=prompt, messages=messages, max_tokens=800)
                parsed = _claude.safe_parse_json(raw)
                if parsed:
                    # Only accept the LLM's product if it's a real NexaCloud product name.
                    # Clients often state non-existent products ("NexaAnalytics"); the LLM
                    # echoes those back. The ProductIdentifier's match is more reliable.
                    llm_product = parsed.get("product")
                    if llm_product and llm_product in VALID_PRODUCTS:
                        product = llm_product
                    # else: keep ProductIdentifier's match (already in `product`)
                    error_message = parsed.get("error_message") or error_message
                    # Only accept LLM environment if client explicitly stated it in conversation.
                    # Otherwise keep the safe "production" default — never let the LLM guess.
                    if reasoning.field_values_extracted.get("_env_from_conversation"):
                        environment = parsed.get("environment") or environment
                    started_at = parsed.get("started_at") or started_at
                    impact_scope = parsed.get("impact_scope") or impact_scope
                    if parsed.get("known_incident"):
                        known_incident = True
                        confidence = 1.0
                    # Re-check known_incident with potentially updated product
                    if product and not known_incident:
                        _pd = products_db.get(product, {})
                        if _pd.get("active_incident"):
                            known_incident = True
                            confidence = 1.0
                    # Ensure fields confirmed by ProductIdentifier/regex count toward confidence
                    # even if the LLM didn't re-extract them in its JSON response.
                    if extracted.get("product") and not parsed.get("product"):
                        parsed["product"] = extracted["product"]
                    if extracted.get("environment") and not parsed.get("environment"):
                        parsed["environment"] = extracted["environment"]
                    # Re-count confirmed fields from LLM output (inferred fields count as 0.5)
                    llm_confirmed_score = sum(
                        0.5 if f in _inferred else 1.0
                        for f in ("product", "error_message", "environment", "started_at", "impact_scope")
                        if parsed.get(f)
                    )
                    if not known_incident:
                        confidence = self.calculate_confidence(llm_confirmed_score, 5)
                    # Build questions from LLM output, but preserve any is_confirmation
                    # questions from decide() — Claude doesn't know about the inferred flag
                    # and would silently drop the confirmation question.
                    _rich_questions = extracted.get("_rich_questions", [])
                    _confirmation_questions = [
                        QuestionForClient(
                            field=q["field"],
                            question=q["question"],
                            blocking=q.get("blocking", False),
                            priority=q.get("priority", "medium"),
                        )
                        for q in _rich_questions
                        if q.get("is_confirmation")
                    ]
                    _llm_questions = [
                        QuestionForClient(**q) if isinstance(q, dict) else q
                        for q in parsed.get("questions_for_client", [])
                    ]
                    # Confirmation questions go first; LLM questions fill the rest
                    decision.questions = (_confirmation_questions + _llm_questions)[:2]
            except Exception as e:
                log.warning("triage_agent.claude_failed", error=str(e))

        # Check for recurring issue AFTER LLM refinement (error_message may have been updated)
        error_code_guess = _guess_error_code(error_message)
        error_prefix = error_code_guess.split("_")[0] if error_code_guess else ""
        recurring_count = sum(
            1 for t in recent_tickets
            if t.get("error") == error_code_guess
            or (error_prefix and (t.get("error") or "").startswith(error_prefix + "_"))
        )
        # Require 3+ history matches OR (2+ AND explicit "again/same/recurring" mention)
        recurring = recurring_count >= 3 or (recurring_count >= 2 and recurring_mentioned)

        # Auto-set severity AFTER LLM refinement (environment may have been updated)
        if _severity_scorer:
            sr = _severity_scorer.score({
                "tier": tier,
                "environment": environment,
                "error_message": error_message,
                "impact_scope": impact_scope,
                "recurring": recurring,
            })
            severity = sr.severity
        else:
            severity = self.apply_severity_rules(tier=tier, environment=environment)

        # Validate output
        agent_output = {"product": product, "confidence": confidence}
        validation = validate_agent_output("triage_agent", agent_output, loaded_data.db_records, confidence)

        # Build client info
        client_info = ClientInfo(
            company=client_data.get("company", "Unknown"),
            tier=client_data.get("tier", "standard"),
            sla_hours=sla_hours,
            sla_deadline=sla_deadline,
            csm=client_data.get("csm", ""),
            recent_tickets=len(recent_tickets),
            vip_flag=client_data.get("vip_flag", False),
        )

        # Build issue info
        _triage_inferred = {
            f: True for f in _inferred
            if f in ("product", "error_message", "environment", "started_at", "impact_scope")
        }
        issue_info = IssueInfo(
            product=product or "",
            error_message=error_message,
            environment=environment,  # type: ignore[arg-type]
            started_at=started_at or "unknown",
            impact_scope=impact_scope,
            known_incident=known_incident,
            recurring=recurring,
            mentioned_products=extracted.get("mentioned_products", []),
            inferred_fields=_triage_inferred,
        )

        # Build questions for client — standard questions for missing fields
        questions_for_client = [
            QuestionForClient(
                field=f,
                question=TRIAGE_FIELDS[f]["questions"][0],
                blocking=TRIAGE_FIELDS[f]["blocking"],
                priority="high" if TRIAGE_FIELDS[f]["blocking"] else "medium",
            )
            for f in TRIAGE_FIELDS
            if f not in reasoning.fields_confirmed_from_conversation and f not in (extracted or {})
        ]

        # Prepend confirmation questions for inferred fields into the report so the
        # orchestrator's response generation layer picks them up. Without this, the
        # orchestrator never sees the confirmation question and asks a generic one instead.
        _rich_qs = extracted.get("_rich_questions", [])
        _confirmation_qs_for_report = [
            QuestionForClient(
                field=q["field"],
                question=q["question"],
                blocking=q.get("blocking", False),
                priority=q.get("priority", "medium"),
            )
            for q in _rich_qs
            if q.get("is_confirmation")
        ]
        questions_for_client = _confirmation_qs_for_report + questions_for_client

        # If product is confirmed but error code is unrecognised (not a standard HTTP 4xx/5xx
        # or named code), replace all questions with a single targeted error-clarification question
        _error_val = (error_message or "").strip()
        if product and _error_val and not ERROR_CODE_PATTERN.search(_error_val):
            questions_for_client = [QuestionForClient(
                field="error_detail",
                question=(
                    f'Error "{_error_val}" isn\'t a code I recognise — could you share what '
                    f"the full API response looks like when this occurs? For example, any message "
                    f"field or description alongside the code."
                ),
                blocking=True,
                priority="high",
            )]

        report = TriageReport(
            session_id=loaded_data.session_id,
            client=client_info,
            issue=issue_info,
            severity=severity,  # type: ignore[arg-type]
            routing="escalation_agent" if known_incident else "diagnostic_agent",
            confidence=confidence,
            questions_for_client=questions_for_client[:1],
            hallucination_flags=validation.flags,
            completed=(
                confidence >= (self.dynamic_threshold if self.dynamic_threshold is not None else self.confidence_threshold)
                or (bool(product) and bool(error_message) and confidence >= 0.60)
            ),
        )

        self.log_reasoning(
            "GENERATE",
            f"product={product}, severity={severity}, known_incident={known_incident}, confidence={confidence:.2f}",
            f"recurring={recurring}, sla_deadline={sla_deadline}",
        )

        _threshold_used = self.dynamic_threshold if self.dynamic_threshold is not None else self.confidence_threshold
        _primary_cond = confidence >= _threshold_used
        _secondary_cond = bool(product) and bool(error_message) and confidence >= 0.60
        log.info(
            "triage_agent.generate_complete",
            completed=report.completed,
            confidence=round(confidence, 4),
            threshold_used=round(_threshold_used, 4),
            dynamic_threshold=self.dynamic_threshold,
            primary_condition_met=_primary_cond,
            secondary_condition_met=_secondary_cond,
            product=product or None,
            error_message=error_message or None,
            severity=severity,
            known_incident=known_incident,
            questions_remaining=len(questions_for_client),
        )

        return report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _guess_error_code(error_message: str) -> str:
    """Map an error message to its likely error_db key."""
    msg = (error_message or "").lower()
    if "401" in msg or "invalid_token" in msg or "unauthorized" in msg:
        return "401_invalid_token"
    if "429" in msg or "rate_limit" in msg or "too many" in msg:
        return "429_rate_limit_exceeded"
    if "403" in msg or "forbidden" in msg or "access_denied" in msg:
        return "403_access_denied"
    if "500" in msg or "internal error" in msg:
        return "500_internal_error"
    if "404" in msg or "not found" in msg:
        return "404_object_not_found"
    return ""


def _calculate_sla_deadline(sla_hours: int) -> str:
    now = datetime.now(tz=timezone.utc)
    from datetime import timedelta
    deadline = now + timedelta(hours=sla_hours)
    return deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
