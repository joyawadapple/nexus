"""
Nexus Backend — FastAPI application entry point.

Startup sequence:
1. Load all JSON databases
2. Initialize RAGEngine (encodes KB docs with sentence-transformers)
3. Instantiate ClaudeClient
4. Set up session infrastructure
5. Inject services into orchestrator and routers
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.nexus_orchestrator import NexusOrchestrator, set_services
from core.claude_client import ClaudeClient
from core.config import settings
from core.conversation_memory import ConversationMemory
from core.evaluator import NexusEvaluator
from core.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from core.monitor import AgentMonitor
from core.rag_engine import RAGEngine
from core.session_manager import SessionManager
from db import database
from routers.admin import router as admin_router
from routers.admin import set_services as admin_set
from routers.conversation import router as conversation_router
from routers.conversation import set_services as conv_set
from routers.tickets import router as tickets_router
from routers.tickets import set_services as tick_set
from routers.voice import router as voice_router
from routers.voice import set_services as voice_set

# Mirror LANGSMITH_* vars to the legacy LANGCHAIN_* names that some versions
# of the langsmith SDK still read internally for endpoint + tracing toggle.
_ls_endpoint = settings.LANGSMITH_ENDPOINT
if _ls_endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = _ls_endpoint
if settings.LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# langsmith.utils.get_env_var is lru_cache-decorated — clear it now so any
# call that happened before settings were loaded doesn't serve a stale value.
try:
    from langsmith.utils import get_env_var as _ls_get_env_var
    _ls_get_env_var.cache_clear()
except Exception:
    pass

# ── Structlog configuration ────────────────────────────────────────────────────
# Must be called before any logger is used. Enables DEBUG-level output with
# timestamped, coloured console rendering so log.debug() calls are visible.
_log_level = settings.LOG_LEVEL.upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.DEBUG),
    format="%(message)s",
)
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S.%f", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, _log_level, logging.DEBUG)
    ),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger("main")

# ── Third-party log suppression ───────────────────────────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("anthropic._base_client").setLevel(logging.WARNING)

# ── Uvicorn polling filter ─────────────────────────────────────────────────────
_POLLING_PATHS = (
    "/conversation/status/",
    "/admin/sessions",
    "/admin/tickets",
    "/admin/escalations",
    "/admin/metrics",
)


class _SuppressPollingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if '"GET ' in msg and '" 200 ' in msg:
            return not any(p in msg for p in _POLLING_PATHS)
        return True


# ── Readiness flag ─────────────────────────────────────────────────────────────
_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global _ready
    logging.getLogger("uvicorn.access").addFilter(_SuppressPollingFilter())
    log.info("nexus.startup")

    # 1. Load databases
    db_data = database.load_all()

    # 2. Initialize RAG engine
    rag_engine = RAGEngine(knowledge_base=db_data["knowledge_base"])

    # 3. Instantiate Claude client
    claude = ClaudeClient(
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.MODEL,
        timeout=settings.CLAUDE_TIMEOUT_SECONDS,
    )
    log.info("nexus.claude_ready", model=settings.MODEL)

    # 4. Session infrastructure
    from core.storage import StorageManager
    storage = StorageManager(base_dir=".")
    memory = ConversationMemory(storage=storage)
    session_manager = SessionManager(memory=memory)

    # 5. Monitor + evaluator
    monitor = AgentMonitor()
    evaluator = NexusEvaluator()

    # 6. Orchestrator
    set_services(claude=claude, db_data=db_data, rag_engine=rag_engine, monitor=monitor)
    orchestrator = NexusOrchestrator()

    # 6b. LangGraph + LangSmith (optional — enabled via LANGSMITH_TRACING=true)
    _graph_runner = None
    if settings.LANGSMITH_TRACING:
        from agents.nexus_graph import build_nexus_graph
        from agents.nexus_graph_runner import NexusGraphRunner
        compiled = build_nexus_graph(
            orchestrator=orchestrator,
            session_manager=session_manager,
            db_data=db_data,
        )
        _graph_runner = NexusGraphRunner(compiled)
        log.info("nexus.langgraph_enabled", project=settings.LANGSMITH_PROJECT)

    # 7. Inject into routers
    conv_set(session_manager=session_manager, memory=memory, orchestrator=orchestrator, graph_runner=_graph_runner, storage=storage)
    tick_set(session_manager=session_manager)
    admin_set(session_manager=session_manager, monitor=monitor, evaluator=evaluator)

    # 8. Voice service (Deepgram STT + ElevenLabs TTS)
    from core.voice_service import VoiceService
    voice_service = VoiceService(
        deepgram_key=settings.DEEPGRAM_API_KEY,
        elevenlabs_key=settings.ELEVENLABS_API_KEY,
        elevenlabs_voice_id=settings.ELEVENLABS_VOICE_ID,
    )
    voice_set(voice_service=voice_service)

    _ready = True
    log.info("nexus.ready")
    yield

    # Flush LangSmith traces before shutdown so no spans are lost
    if settings.LANGSMITH_TRACING:
        try:
            from langsmith import Client as LangSmithClient
            ls_client = LangSmithClient()
            await ls_client.flush()
            log.info("nexus.langsmith_flushed")
        except Exception as flush_err:
            log.warning("nexus.langsmith_flush_failed", error=str(flush_err))

    # Flush all in-progress sessions to disk on graceful shutdown
    active_sessions = memory.list_sessions()
    for s in active_sessions:
        if s.status not in ("complete", "escalated"):
            storage.save_session(s)
    if active_sessions:
        log.info("nexus.sessions_flushed", count=len(active_sessions))

    log.info("nexus.shutdown")


app = FastAPI(
    title="Nexus — NexaCloud AI Support System",
    description="Multi-agent technical support platform for NexaCloud enterprise clients.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    log.error(
        "nexus.unhandled_exception",
        error=str(exc),
        path=str(request.url.path),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "request_id": request_id},
    )


app.include_router(conversation_router)
app.include_router(tickets_router)
app.include_router(admin_router)
app.include_router(voice_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "nexus"}


@app.get("/ready")
async def ready():
    """Readiness probe — returns 503 until all services have initialised."""
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {"status": "ready", "model": settings.MODEL}


@app.get("/")
async def root():
    return {
        "service": "Nexus — NexaCloud AI Support System",
        "version": "1.0.0",
        "docs": "/docs",
    }
