"""
Nexus — Centralised application configuration.

All environment variables are declared here. Import `settings` from this module
instead of calling `os.getenv()` directly anywhere else in the codebase.

Usage:
    from core.config import settings
    print(settings.MODEL)
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    MODEL: str = "claude-sonnet-4-6"
    CLAUDE_TIMEOUT_SECONDS: float = 120.0

    # ── App ────────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "DEBUG"
    CORS_ORIGINS: list[str] = Field(default=["*"])
    ADMIN_API_KEY: str = ""

    # ── LangSmith (optional — enable via LANGSMITH_TRACING=true) ──────────────
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = ""
    LANGSMITH_PROJECT: str = "default"
    LANGSMITH_API_KEY: str = ""

    # ── Voice services (optional) ──────────────────────────────────────────────
    DEEPGRAM_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"


settings = Settings()
