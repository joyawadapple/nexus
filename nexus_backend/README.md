# Nexus — Intelligent Conversational Support Agent

Nexus is a multi-agent AI system that interviews enterprise clients about API issues, diagnoses root causes using a knowledge base, and produces structured support tickets. It was built as a full-stack assignment demonstrating LLM integration, multi-turn conversation management, structured data extraction, RAG, and sentiment-aware escalation.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Quick Start](#quick-start)
3. [System Architecture](#system-architecture)
4. [How the Orchestrator Decides](#how-the-orchestrator-decides)
5. [The Four Agents](#the-four-agents)
6. [Key Features](#key-features)
7. [Data Collection & Storage](#data-collection--storage)
8. [API Reference](#api-reference)
9. [Key Design Decisions](#key-design-decisions)
10. [Potential Improvements](#potential-improvements)
11. [Running Tests](#running-tests)

---

## What It Does

A user opens a support chat and describes a problem. Nexus takes over from there:

1. **Interviews** the client to collect the product, error, environment, and impact scope
2. **Diagnoses** the root cause using semantic search over a knowledge base (RAG)
3. **Generates** a step-by-step resolution plan with commands, risks, and verification steps
4. **Decides** whether to self-resolve or escalate to a human — factoring in severity, SLA, and client sentiment
5. **Produces** a structured JSON support ticket with the full reasoning trail

Everything happens in natural conversation. The client never fills out a form.

**Supported products:** NexaAuth · NexaStore · NexaMsg · NexaPay
**Test clients:** Acme Corp (Platinum) · GlobalRetail SA (Gold) · DevStartup Ltd (Standard)

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- An Anthropic API key

### Backend

```bash
cd nexus_backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create .env with your key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

uvicorn main:app --reload --port 8000
```

The server will:
1. Load all JSON databases (clients, products, errors, knowledge base)
2. Encode the knowledge base with sentence-transformers (~3s cold start)
3. Create `conversations/` and `tickets/` directories for session persistence
4. Be ready at `http://localhost:8000` — docs at `/docs`

### Frontend

```bash
cd nexus_frontend
npm install
npm run dev   # http://localhost:5173
```

Select a client from the dropdown → click **Start Session** → chat.

To return to the client selector at any time, click **← End Session** in the top-right corner of the header. This resets the UI locally — the backend session is preserved and visible in the Admin Dashboard.

### Optional: Voice (Deepgram STT + ElevenLabs TTS)

Add to `.env`:
```
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
```

Voice degrades gracefully — the app works fully without these keys.

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `MODEL` | No | `claude-sonnet-4-6` | Claude model to use |
| `DEEPGRAM_API_KEY` | No | — | Voice transcription (STT) |
| `ELEVENLABS_API_KEY` | No | — | Voice synthesis (TTS) |
| `LANGSMITH_TRACING` | No | `false` | Enable LangSmith tracing |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Client (React Frontend)                                            │
│  ┌─────────────┐  ┌───────────────────┐  ┌──────────────────────┐  │
│  │ Chat Window │  │ Agent Pipeline    │  │ Live Ticket Preview  │  │
│  │             │  │ [■■■□] Triage     │  │ (polls every 2s)     │  │
│  │ voice input │  │ [■■□□] Diagnostic │  │                      │  │
│  │ → Deepgram  │  │ [■□□□] Resolution │  │ ticket JSON appears  │  │
│  │ TTS output  │  │ [□□□□] Escalation │  │ as agents complete   │  │
│  └──────┬──────┘  └───────────────────┘  └──────────────────────┘  │
│  [← End Session] button in header — returns to client selector      │
└─────────┼───────────────────────────────────────────────────────────┘
          │ POST /conversation/message
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  NexusOrchestrator                                                  │
│                                                                     │
│  Phase 1 ── Sequential ─────────────────────────────────────────   │
│                                                                     │
│         ┌──────────────────────────┐                               │
│         │       Triage Agent       │  threshold: 0.90              │
│         │  collects: product★,     │                               │
│         │  error★, env, scope,     │                               │
│         │  started_at              │                               │
│         └────────────┬─────────────┘                               │
│                      │                                             │
│          ┌───────────┴────────────┐                                │
│          │                        │                                │
│   known_incident=True      confidence ≥ 0.90                       │
│          │                        │                                │
│          ▼                        ▼                                │
│   ┌─────────────┐    Phase 2 ── Parallel ──────────────────────    │
│   │  SHORTCUT   │                                                  │
│   │  skip diag  │   ┌───────────────────┐  ┌──────────────────┐   │
│   │  & resolve  │   │  Diagnostic Agent │  │ Resolution Agent │   │
│   └──────┬──────┘   │  threshold: 0.75  │  │ threshold: 0.80  │   │
│          │          │  RAG: root cause  │  │ RAG: fix steps   │   │
│          │          └─────────┬─────────┘  └────────┬─────────┘   │
│          │                   └──────────┬────────────┘             │
│          │                              │                          │
│          │          Phase 3 ── Sequential ─────────────────────    │
│          │                              │                          │
│          │                  ┌───────────▼──────────────┐           │
│          └─────────────────►│     Escalation Agent     │           │
│                             │     threshold: 0.95       │           │
│                             │  rules: SLA + tier +      │           │
│                             │  sentiment + confidence   │           │
│                             └───────────┬──────────────┘           │
│                                         │                          │
│                             ┌───────────▼──────────────┐           │
│                             │     assemble_ticket()    │           │
│                             │  → SupportTicket JSON    │           │
│                             │  → saved to disk         │           │
│                             └──────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
          │
          │ JSON file written to conversations/ and tickets/
          ▼
┌───────────────────────────┐   ┌───────────────────────────────────┐
│  RAG Engine               │   │  Knowledge Base (25 docs)         │
│  sentence-transformers    │   │  NexaAuth · NexaStore · NexaMsg   │
│  all-MiniLM-L6-v2         │   │  NexaPay · cross-product          │
│  in-memory cosine search  │   │  tagged: diagnostic / resolution  │
└───────────────────────────┘   └───────────────────────────────────┘
```

### Confidence thresholds

| Agent | Threshold | What counts toward confidence |
|---|---|---|
| Triage | 0.90 | `confirmed_fields / 5` — product★, error★, env, scope, started_at |
| Diagnostic | 0.75 | +0.40 error DB match, +0.25 RAG similarity > 0.80, +0.20 known version bug |
| Resolution | 0.80 | RAG similarity > 0.85 → 0.89, > 0.70 → 0.75, > 0.55 → 0.60 |
| Escalation | 0.95 | Deterministic rule score — never LLM-estimated |

★ = blocking fields. No routing happens until at least these two are confirmed.

---

## How the Orchestrator Decides

The orchestrator runs a priority-ordered decision tree on every turn. The first condition that matches determines what happens next.

```
Every turn, in order:

1. Did the client say "escalate" / "speak to a human" / "supervisor"?
   → Force escalation immediately, skip remaining checks.

2. Did triage detect a known active incident on the product?
   → Jump straight to escalation, skip diagnostic and resolution.

3. Did the client confirm the issue is resolved ("working now", "fixed")?
   → Mark session resolved, stop pipeline.

4. Is the session already complete or escalated?
   → Return cached result, do nothing.

5. Is triage incomplete (confidence < 0.90)?
   → Run triage agent. Ask for missing fields.
   → Stuck detection: if triage has run 3+ times with no confidence growth → escalate.

6. Is diagnostic incomplete (confidence < 0.75)?
   → Run diagnostic agent. Query RAG for root cause.
   → If diagnostic confidence is below minimum after running → escalate.

7. Is resolution incomplete (confidence < 0.80)?
   → Run resolution agent. Query RAG for fix steps.

8. Is escalation incomplete?
   → Run escalation agent. Apply all rules.

9. Everything done?
   → Assemble final ticket, write to disk, return complete.
```

### Signal detection

The orchestrator checks for intent signals in every message before running the decision tree:

| Signal | Trigger phrases | Action |
|---|---|---|
| Escalation | "escalate", "speak to someone", "supervisor", "this isn't working" | Force escalation |
| Resolved | "working now", "fixed", "resolved", "no more errors", "it's back" | Mark resolved |

### Stuck detection

If an agent runs 3 or more times without its confidence score growing, the `ProgressTracker` marks it as stuck and the orchestrator escalates automatically. This prevents infinite loops on ambiguous cases.

### Agent caching

Once an agent reaches its confidence threshold, its finding is cached in the session state. It never runs again for that session — even if the client sends more messages. This ensures Claude is only called when genuinely useful.

---

## The Four Agents

Every agent follows the same 6-step loop internally:

**LOAD → ANALYZE → REASON → DECIDE → GENERATE → RETURN**

| Step | What happens |
|---|---|
| LOAD | Fetch session state, conversation history, RAG context |
| ANALYZE | Count confirmed vs missing fields |
| REASON | Extract values from the conversation, detect discrepancies |
| DECIDE | Are we ready to finalize, or do we need more information? |
| GENERATE | Call Claude with a structured prompt, parse JSON response |
| RETURN | Cache result in session state, log reasoning |

---

### Agent 1 — Triage

**Role:** Identify the product, error, and situation before any routing decision.

**Collects:**
- `product` — which NexaCloud API (NexaAuth / NexaStore / NexaMsg / NexaPay)
- `error_message` — the specific error or code the client sees
- `environment` — production / staging / development (defaults to production)
- `impact_scope` — all users / subset / internal only
- `started_at` — when the issue began

**Also detects:**
- `severity` (critical / high / medium / low) via keyword + environment + scope analysis
- `known_incident` — checks product_db for active incidents; if found, triggers shortcut
- `recurring` — checks client's recent ticket history for the same error

**Transitions to:** Diagnostic + Resolution (parallel) once confidence ≥ 0.90, or directly to Escalation if `known_incident=True`.

---

### Agent 2 — Diagnostic

**Role:** Find the root cause using semantic search over the knowledge base.

**How it works:**
1. Queries RAG with `category=diagnostic` + `product={product from triage}`
2. Looks up the error code in the error database
3. Combines evidence into a confidence score:
   - RAG similarity > 0.80 → +0.25
   - Error found in error DB → +0.40
   - Known version bug detected → +0.20
4. Returns primary cause, alternative causes, and supporting evidence

**Transitions to:** Escalation if confidence remains below minimum, or waits for Resolution to complete.

---

### Agent 3 — Resolution

**Role:** Generate step-by-step fix instructions grounded in the knowledge base.

**How it works:**
1. Queries RAG with `category=resolution` + `product={product}`
2. Maps RAG similarity score to confidence:
   - Similarity > 0.85 → confidence 0.89
   - Similarity > 0.70 → confidence 0.75
   - Similarity > 0.55 → confidence 0.60
3. Returns a numbered resolution plan

Each step includes:
- **Action** — what to do
- **Command** — exact CLI or API call to run
- **Why** — the rationale
- **Verify** — how to confirm it worked
- **Risk** — none / low / medium / high

**Transitions to:** Escalation agent once complete (or if confidence too low).

---

### Agent 4 — Escalation

**Role:** Decide whether the issue can be self-resolved or needs a human.

**Decision inputs:**
- SLA deadline vs current time
- Client tier (platinum → faster escalation)
- Diagnostic and resolution confidence scores
- Detected sentiment (frustrated / urgent → lower escalation threshold)
- Whether the issue is a known incident
- Whether the issue is recurring (3+ times → always escalate)

**Sentiment adjustment:**

| Sentiment | Escalation bias | Effect |
|---|---|---|
| `calm` | +0.00 | Standard thresholds |
| `frustrated` | +0.10 | Escalates in more cases |
| `urgent` | +0.15 | Escalates in almost all critical cases |

**Outputs:** `self_resolve` · `escalated` · `pending`

---

## Key Features

### RAG — Retrieval-Augmented Generation

The knowledge base contains 25 documents across all four products, split into `diagnostic` and `resolution` categories. At startup, every document is encoded into a vector using `sentence-transformers/all-MiniLM-L6-v2`. Queries are encoded the same way and matched via cosine similarity (all in-memory, ~1ms per query).

Agents only receive KB documents relevant to the detected product and their category — no cross-product noise.

### Sentiment Analysis

Every client message is analyzed with VADER and a set of domain-specific keyword profiles. The result is stored in the session's `sentiment_history` and the current sentiment label (`calm`, `frustrated`, `urgent`, `positive`, `confused`, `anxious`) is updated each turn.

The escalation agent reads this profile and applies a bias to its decision thresholds. A technically-solvable issue will still be escalated if the client is clearly distressed or if their SLA is already at risk.

### Hallucination Guard

After every agent's `generate()` call, a validator cross-references the output against the databases:

- **Invented values** — if the agent names a product not in `product_db` or an error code not in `error_db`, the field is flagged as `INVENTED_VALUE` (severity: high). The ticket is demoted to `pending_review`.
- **Confidence inflation** — if the agent's claimed confidence exceeds the mathematically-justified value by more than 0.10, it is clamped and flagged as `CONFIDENCE_INFLATION` (severity: medium).

Flags are stored in `ticket.data_provenance.hallucination_flags`. The system never blocks ticket generation — it surfaces flags for human review instead.

### Voice Input / Output

- **Speech-to-text:** Deepgram (POST `/voice/transcribe`, audio/webm)
- **Text-to-speech:** ElevenLabs (POST `/voice/synthesize`)

Both are optional. If the API keys are absent, the voice button is disabled and the rest of the app works normally.

### Multi-Language Support

Nexus automatically detects the language of each client message using `langdetect` and stores it in the session as `detected_language`. All agent responses are then generated in that language — Claude is explicitly instructed to reply in the language of the client's most recent message. No configuration required; it works out of the box for any language Claude supports.

### Session Persistence

Completed and escalated sessions are written to disk automatically:

- `conversations/{session_id}.json` — full transcript, sentiment history, extracted fields, confidence scores, agent findings
- `tickets/{ticket_id}.json` — the final support ticket

On graceful shutdown, any still-active sessions are also flushed to disk.

### Admin Dashboard

Available at `/admin` in the frontend (link in the top-right header):

- Active sessions with confidence scores — sessions remain visible here even after clicking **← End Session** in the chat UI
- All generated tickets
- Escalated tickets requiring human review
- System metrics (turn count, escalation rate, hallucination rate)
- Per-session reasoning logs (expandable in the chat UI)

---

## Data Collection & Storage

### What Nexus collects per session

| Category | Fields |
|---|---|
| **Identity** | `session_id`, `client_id`, `created_at` |
| **Conversation** | Every message with role, content, and timestamp |
| **Triage fields** | product, error_message, environment, impact_scope, started_at |
| **Diagnostic** | root cause, confidence, supporting evidence, alternative causes |
| **Resolution** | steps with commands/risks/verification, estimated resolution time |
| **Escalation** | decision, reason, escalation path |
| **Sentiment** | label and VADER compound score per message |
| **Ticket** | Full `SupportTicket` JSON with data provenance and reasoning logs |

### Output format

Every completed session produces two JSON files:

**`conversations/{session_id}.json`**
```json
{
  "session_id": "...",
  "status": "complete",
  "conversation": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "extracted_data": {"product": "NexaAuth", "error_message": "401 invalid_token", ...},
  "confidence_breakdown": {"triage": 0.92, "diagnostic": 0.87, "resolution": 0.89, "escalation": 0.95},
  "agent_findings": { ... },
  "ticket": { ... }
}
```

**`tickets/{ticket_id}.json`**
```json
{
  "ticket_id": "TKT-...",
  "status": "self_resolve",
  "priority": "critical",
  "client": {"company": "Acme Corp", "tier": "platinum", "sla_hours": 4},
  "issue_summary": {"product": "NexaAuth", "error_message": "401 invalid_token", ...},
  "diagnosis": {"primary_cause": "...", "confidence": 0.87, ...},
  "resolution": {"steps": [...], "estimated_resolution_time": "10-15 minutes"},
  "escalation": {"decision": "self_resolve", "reason": "..."},
  "sentiment_analysis": {"detected": "calm", "tone_adjustment_applied": false},
  "nexus_summary": "Natural language summary of the incident and resolution."
}
```

---

## API Reference

### Conversation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/conversation/start` | Start a session (`client_id` or `api_key`) |
| `POST` | `/conversation/message` | Send a message, run the full agent pipeline |
| `GET` | `/conversation/status/{session_id}` | Poll agent progress and partial ticket |

### Tickets

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/tickets` | List all generated tickets |
| `GET` | `/tickets/{ticket_id}` | Retrieve a specific ticket |

### Admin

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/admin/sessions` | All active sessions with confidence scores |
| `GET` | `/admin/escalations` | Tickets flagged for human review |
| `GET` | `/admin/metrics` | System-wide metrics |
| `GET` | `/admin/reasoning/{session_id}` | Agent reasoning logs |

### Voice

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/voice/transcribe` | Audio → text (Deepgram) |
| `POST` | `/voice/synthesize` | Text → audio (ElevenLabs) |

---

## Key Design Decisions

### 1. Sequential-then-parallel agent execution

Triage must run before anything else — the diagnostic and resolution agents have no product or error to work with until triage completes. Once triage is done, diagnostic and resolution can run concurrently via `asyncio.gather()` since they query different RAG categories and don't depend on each other. Escalation always runs last because it needs all prior findings to make its decision.

This isn't an arbitrary pattern — it mirrors the actual dependency graph of a real support workflow.

### 2. Mathematical confidence, never LLM-estimated

Agent confidence scores are calculated from field completion counts, not generated by Claude. `confirmed_fields / total_fields` is deterministic and verifiable. Claude cannot inflate its own confidence. This is what makes the hallucination guard meaningful — it catches the gap between what the LLM claims and what the math supports.

### 3. Known incident shortcut

If triage detects an active incident in `product_db`, the orchestrator skips diagnostic and resolution entirely. Running those agents would waste Claude calls trying to "diagnose" a known platform-wide outage. The right response is to acknowledge the incident and escalate — and Nexus does exactly that in one step.

### 4. Sentiment-adjusted escalation thresholds

A technically-valid resolution plan is not always the right output. If a client is expressing urgency or frustration, delivering step-by-step instructions to a stressed enterprise contact is often wrong. Nexus detects this and lowers the escalation threshold — routing to a human even when the AI is confident it could solve the issue. This reflects how a skilled human support engineer would actually behave.

### 5. Non-blocking hallucination handling

Hallucination flags demote ticket status to `pending_review` but never block generation. Hard-blocking on hallucination would create failure modes for edge cases that the evaluator hasn't seen before. Flagging and demotion achieves the same safety outcome while keeping the system operational.

### 6. Stuck detection via confidence velocity

Rather than setting a fixed maximum turn count, the `ProgressTracker` watches whether confidence is actually growing between runs. An agent that has run 3 times with zero confidence change is genuinely stuck — not just slow. This produces fewer premature escalations than a hard turn limit.

---

## Potential Improvements

Listed in priority order based on product impact.

### 1. Product Knowledge Agent + Dynamic Knowledge Base

Add a dedicated **FAQ / product knowledge agent** that can answer general questions about NexaCloud products ("How does NexaAuth token rotation work?", "What are the NexaPay settlement windows?") using the knowledge base — not just diagnose issues. Pair this with a **hot-reload mechanism** so support engineers can add or update knowledge base articles without restarting the server, and a UI for uploading new documents that get immediately encoded into the RAG index.

### 2. Multi-Issue Sessions

Currently Nexus handles one issue per session. A common real-world pattern is a client reporting two separate bugs in the same message ("Our NexaAuth tokens are expiring AND our NexaStore uploads are failing"). Extend the triage agent to detect and split multi-issue queries, spawning parallel diagnostic/resolution sub-flows for each issue within the same session.

### 3. Escalation / De-escalation

Support bidirectional escalation state. If a case has been escalated to a CSM but the client messages again with new information that raises confidence above the threshold, Nexus should be able to pull the case back ("de-escalate") and attempt self-resolution again — updating the CSM in the process. This reflects how real support handoffs actually work.

### 4. UI Language Toggle

The backend already responds in the client's language (detected per-message via `langdetect`). The next step is making the **UI itself** available in multiple languages — labels, buttons, status messages, and the admin dashboard — rather than just the agent's conversational responses.

### 5. Session Resumption

Allow clients to reconnect to a previous session and continue where they left off, rather than starting fresh every time. This requires persisting session state in a way that survives server restarts (see DB improvement below) and a session lookup flow on the frontend. Design should be informed by actual usage patterns — understanding how often clients return mid-issue before building the UX.

### 6. Streaming Responses

Claude responses are currently returned in full after completion, adding noticeable latency on long resolution plans. Add **SSE (Server-Sent Events) streaming** from the backend and incremental rendering on the frontend so the client sees words appearing in real time rather than waiting for the full response.

### 7. Persistent Database + Vector Store

Replace the current file-based `StorageManager` and in-memory RAG index with production-grade storage:
- **PostgreSQL** — sessions, tickets, reasoning logs, audit trails, queryable history
- **Vector database** (Pinecone, Weaviate, or pgvector) — replace the in-memory sentence-transformers index with a persistent, scalable vector store that survives restarts and supports incremental updates

### 8. Authentication & Authorization

Add a proper **client login/logout flow** — currently clients authenticate via a static API key in a dropdown, which is suitable for demos but not production. This includes session tokens, refresh flows, and role-based access for the admin dashboard (support engineers vs. client-facing views).

### 9. LangSmith Evaluation Pipeline in CI

Nexus already has LangSmith tracing and a golden-case evaluator (`NexusEvaluator`). The next step is running these golden cases automatically on every pull request in CI, surfacing agent quality regressions (confidence scores, field extraction accuracy, turn efficiency) before they reach production.

### 10. Voice Sentiment Analysis

Sentiment is currently detected from **text** only (VADER + keywords). Extend this to analyze the **audio signal directly** — tone of voice, speaking pace, and acoustic stress markers — using a model like `speechbrain` or a Deepgram feature flag. A client who sounds distressed but uses calm words would still trigger the urgency escalation path.

### 11. Higher Quality TTS Voice

The current ElevenLabs integration uses a standard voice configuration. Upgrade to a more natural, expressive voice profile — one that adjusts pace and tone based on the content (calmer for reassurance, more precise for technical steps). This significantly improves the experience for clients using voice mode.

### 12. Upgraded Embedding Model

The current RAG engine uses `all-MiniLM-L6-v2` — a fast, lightweight model appropriate for local development. Once the infrastructure moves to cloud, upgrade to a higher-capacity embedding model (e.g. `text-embedding-3-large` from OpenAI, or a domain-fine-tuned model on NexaCloud support data) for meaningfully better RAG retrieval precision on ambiguous or technical queries.

---

## Running Tests

```bash
cd nexus_backend
pytest tests/ -v
# 35+ tests covering all four agents, the orchestrator, RAG engine, and evaluator
```

### Evaluation framework

`NexusEvaluator` measures 7 quality dimensions across golden test cases:

| Metric | Measures | Target |
|---|---|---|
| `field_extraction_accuracy` | All triage fields collected before routing | ≥ 0.90 |
| `question_relevance` | Questions only ask about unconfirmed fields | ≥ 0.85 |
| `redundancy_rate` | Same field asked more than once | ≤ 0.10 |
| `hallucination_rate` | Hallucination flags / total fields | ≤ 0.05 |
| `resolution_confidence` | Final resolution agent confidence score | ≥ 0.75 |
| `turn_efficiency` | Turns required to complete a ticket | ≤ 5 |
| `rag_faithfulness` | Resolution steps cite KB sources | ≥ 0.80 |

---

## Sample Conversations

See [`sample_conversations.md`](sample_conversations.md) for three annotated end-to-end dialogues:

1. **Self-resolved** — NexaAuth 401 JWT expiry, clean happy path, full resolution plan
2. **Escalated** — NexaPay financial discrepancy, frustrated client, sentiment bias triggered
3. **Known incident fast-track** — NexaStore CORS recurring issue, triage shortcuts directly to escalation
