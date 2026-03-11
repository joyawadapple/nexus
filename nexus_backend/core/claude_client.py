"""ClaudeClient — centralised wrapper for all Anthropic API calls in Nexus."""
from __future__ import annotations

import json
from typing import Any

import anyio
import anthropic
import structlog

from core.retry import with_retry

log = structlog.get_logger("claude_client")

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    TimeoutError,
)


class ClaudeClient:
    """
    Thin async wrapper around the Anthropic SDK.

    All Claude calls in Nexus go through this class.
    Default model: claude-sonnet-4-6

    Args:
        api_key:  Anthropic API key.
        model:    Model identifier (default: claude-sonnet-4-6).
        timeout:  Per-call timeout in seconds. Raises TimeoutError (retried via
                  with_retry) if Claude does not respond within this window.
                  Default: 120s.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        timeout: float = 120.0,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self._timeout = timeout
        self._last_input_tokens: int = 0
        self._last_output_tokens: int = 0

    @with_retry(max_attempts=3, base_delay=1.0, retryable_exceptions=_RETRYABLE)
    async def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> str:
        """
        Non-streaming completion — returns full text response.
        Primary method used by all Nexus agents for structured JSON output.

        Args:
            system:     System prompt string.
            messages:   Conversation history in Anthropic message format.
            max_tokens: Maximum tokens in the response (default: 2048).

        Returns:
            The text content of Claude's response, or an empty string on failure.

        Raises:
            TimeoutError: If Claude does not respond within ``self._timeout`` seconds.
            anthropic.APIError: On unrecoverable API errors after all retries.
        """
        with anyio.fail_after(self._timeout):
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        self._last_input_tokens = response.usage.input_tokens
        self._last_output_tokens = response.usage.output_tokens
        log.debug(
            "claude.complete",
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.content[0].text if response.content else ""

    @property
    def last_usage(self) -> dict[str, int]:
        """Token counts from the most recent complete() call.
        Best-effort for aggregate monitoring — not thread-safe across concurrent sessions."""
        return {"input_tokens": self._last_input_tokens, "output_tokens": self._last_output_tokens}

    @staticmethod
    def safe_parse_json(text: str) -> dict:
        """
        Safely parse a JSON string, stripping markdown code fences if present.
        Returns empty dict on failure rather than raising.
        """
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            log.warning("claude_client.json_parse_failed", raw=text[:200])
            return {}
