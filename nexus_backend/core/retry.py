import asyncio
from functools import wraps
from typing import Any, Callable, Tuple, Type

import structlog

log = structlog.get_logger("retry")


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Exponential backoff retry decorator for async functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    log.warning(
                        "retry.attempt",
                        func=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        delay_ms=int(delay * 1000),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
