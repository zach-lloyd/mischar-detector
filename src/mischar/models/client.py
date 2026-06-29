"""
ModelClient protocol and shared infrastructure for model backends.

Defines the ``ModelClient`` protocol that Ollama, MLX, and Gemini backends
implement. Also provides retry logic and JSON parsing helpers shared across
all backends.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

import structlog

from mischar.types import ModelResponse

log = structlog.get_logger("models")


# ---------------------------------------------------------------------------
# Protocol — the contract all backends must satisfy
# ---------------------------------------------------------------------------

# Use a Protocol for easier type checking.
class ModelClient(Protocol):
    """
    Common interface across Ollama, MLX, and Gemini backends.

    Any class with a ``name`` attribute and a ``generate`` method matching
    this signature satisfies the protocol — no inheritance required.
    
    Elipses at the end defines this as an interface without implementations,
    not a regular class.

    Use a protocol for easier type checking. If we just use an independent class 
    for each model, we'd have to use something like this:
    
    def __init__(self, attribution_client: OllamaClient | MLXClient | GeminiClient, ...)

    But with a protocol, we can use this format:

    def __init__(self, attribution_client: ModelClient, ...)
    """

    name: str


    def generate(
        self,
        prompt: str,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Generate a completion from the model.

        Args:
            prompt: The full prompt string to send to the model.
            json_schema: If provided, constrain the model's output to match
                this JSON schema (using the backend's native structured-output
                mode). The parsed result will be in ``ModelResponse.parsed_json``.
            temperature: Sampling temperature. 0.0 = deterministic.
            max_tokens: Maximum tokens to generate.

        Returns:
            A ``ModelResponse`` containing the raw text, optionally parsed
            JSON, and backend-specific metadata.

        Raises:
            ModelClientError: On infrastructure failure after retries are
                exhausted (timeouts, network errors, server errors).
        """
        ...


# ---------------------------------------------------------------------------
# Shared exception types
# ---------------------------------------------------------------------------


class ModelClientError(Exception):
    """
    Raised when a model backend fails after retries are exhausted.

    This is an infrastructure error, not a pipeline abstention. The pipeline
    surfaces it to the user rather than treating it as a classification result.

    Backend refers to the name of the model (e.g., "ollama", "gemini", etc.).
    """

    def __init__(self, backend: str, message: str, cause: Exception | None = None) -> None:
        self.backend = backend
        self.cause = cause
        super().__init__(f"[{backend}] {message}")


# ---------------------------------------------------------------------------
# Retry helper — used by all backends for transient failures
# ---------------------------------------------------------------------------

# Default retry parameters: 3 attempts, starting with a 2-second delay,
# doubling each time (2s, 4s, 8s).
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0


def retry_with_backoff(
    fn,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    context: str = "",
) -> Any:
    """
    Call ``fn()`` with exponential backoff on transient failures.

    Args:
        fn: Zero-argument callable to attempt.
        max_retries: Maximum number of retry attempts (not counting the
            initial attempt).
        base_delay: Initial delay in seconds; doubles after each failure.
        retryable_exceptions: Exception types that trigger a retry.
            Non-matching exceptions propagate immediately.
        context: Description for log messages (e.g. "ollama generate").

    Returns:
        Whatever ``fn()`` returns on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        # retryable_exceptions allows backend to specify which errors are worth 
        # retrying
        except retryable_exceptions as exc:
            last_exception = exc

            if attempt < max_retries:
                # Exponential backoff: 2s, 4s, 8s, ... to avoid hammering a 
                # struggling server
                delay = base_delay * (2 ** attempt)
                log.warning(
                    "retry_attempt",
                    context=context,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay_seconds=delay,
                    error=str(exc),
                )
                time.sleep(delay)
            else:
                # All retries exhausted — let the caller handle it.
                log.error(
                    "retries_exhausted",
                    context=context,
                    attempts=max_retries + 1,
                    error=str(exc),
                )

    raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# JSON parsing helper — shared across backends
# ---------------------------------------------------------------------------


def parse_json_response(text: str, backend: str) -> dict | None:
    """
    Attempt to parse a JSON object from model output.

    Models sometimes wrap JSON in markdown fences (```json ... ```) or
    include leading/trailing text. This function tries three different
    strategies for parsing JSON:

        1. Parse the text directly
        2. Strip markdown fences and try again
        3. Find the first { and last } and try paring that substring

    Args:
        text: Raw model output text.
        backend: Backend name for error logging.

    Returns:
        Parsed dict if JSON was found, None otherwise.
    """
    cleaned = text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    if cleaned.startswith("```"):
        # Remove opening fence. The opening fence is often accompanied by a language
        # tag like ```json, as well as a newline, so we can't just set cleaned = cleaned[3:].
        # In the event there is no newline (which shouldn't happen), cleaned will be
        # set to an empty string, the parse will fail, and None will be returned
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1 :]

        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object within the text by locating the first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    log.warning("json_parse_failed", backend=backend, text_preview=text[:200])

    return None
