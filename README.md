# Nexus — Intelligent Conversational Support Agent

Nexus is a multi-agent AI system that interviews enterprise clients about API issues, diagnoses root causes using a knowledge base, and produces structured support tickets — deciding in real time whether to self-resolve or escalate to a human.

Built as a full-stack assignment demonstrating LLM orchestration, multi-turn conversation management, RAG, sentiment-aware escalation, and persistent storage.

---

## Repository Structure

```
nexus/
├── nexus_backend/    # FastAPI backend — agents, RAG, storage, API
└── nexus_frontend/   # React 19 + TypeScript frontend — chat UI, admin dashboard
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (claude-sonnet-4-6) |
| Backend | FastAPI, Python 3.11+ |
| Agents | Custom multi-agent pipeline (Triage → Diagnostic → Resolution → Escalation) |
| RAG | sentence-transformers (all-MiniLM-L6-v2), in-memory cosine similarity |
| Sentiment | VADER + keyword heuristics |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Voice (optional) | Deepgram STT + ElevenLabs TTS |

---

## Quick Start

**Backend**
```bash
cd nexus_backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
uvicorn main:app --reload
```

**Frontend**
```bash
cd nexus_frontend
npm install
npm run dev
```

Open `http://localhost:5173` — select a client API key and start a session.

---

## Full Documentation

See [nexus_backend/README.md](nexus_backend/README.md) for the complete documentation:
architecture diagram, agent breakdown, orchestrator decision logic, API reference, design decisions, and potential improvements.

See [nexus_backend/sample_conversations.md](nexus_backend/sample_conversations.md) for three annotated end-to-end conversations showing self-resolve, escalation, and known-incident fast-track flows.
