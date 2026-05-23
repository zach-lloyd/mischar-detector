"""
Gemini model client for frontier API inference.

Wraps the ``google-genai`` SDK to call Gemini models (e.g. Gemini 3.1 Pro).
Used as the frontier baseline and as a fallback for attribution when the
local model struggles with complex passages.
"""

from __future__ import annotations

from mischar.logging import get_logger
from mischar.models.client import (
    ModelClientError,
    parse_json_response,
    retry_with_backoff,
)
from mischar.types import ModelResponse

log = get_logger("gemini")


class GeminiClient:
    """
    Client for Google's Gemini API via the google-genai SDK.

    Supports structured JSON output through Gemini's native
    ``response_mime_type`` parameter, which constrains the model to
    produce valid JSON.

    Args:
        api_key: Google AI Studio API key.
        model: Gemini model identifier (e.g. ``"gemini-3.1-pro"``).
        timeout_seconds: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-pro",
        timeout_seconds: int = 120,
    ) -> None:
        # Import at instantiation time so the rest of the codebase works
        # without google-genai installed (e.g. when only using Ollama/MLX).
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "GeminiClient requires the 'google-genai' package. "
                "Install with: pip install -e '.[local]'"
            ) from exc

        self.name = f"gemini:{model}"
        self._model = model
        self._timeout = timeout_seconds

        # The google-genai SDK uses a Client object that holds auth config.
        self._client = genai.Client(api_key=api_key)

        # Store the types module for building generation configs later.
        self._genai_types = genai.types

        log.info("gemini_client_initialized", model=model)

    def generate(
        self,
        prompt: str,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Send a prompt to Gemini and return the response.

        When ``json_schema`` is provided, Gemini is instructed to return
        JSON via its ``response_mime_type`` parameter, which provides
        server-side enforcement of the output format.

        Args:
            prompt: The full prompt text.
            json_schema: Optional JSON schema for structured output.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            ``ModelResponse`` with raw text and optionally parsed JSON.

        Raises:
            ``ModelClientError`` if the Gemini API is unreachable or
            returns errors after retries.
        """
        # Build the generation config. Gemini's SDK uses a typed config
        # object rather than loose kwargs.
        gen_config = self._genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # When structured JSON output is requested, tell Gemini to
        # constrain its response to valid JSON.
        if json_schema is not None:
            gen_config.response_mime_type = "application/json"

        def _call():
            """
            Inner function for retry_with_backoff.

            The reason for using this inner function is so that the necessary 
            arguments are baked in and retry_with_backoff can call it with no
            arguments.
            """
            return self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=gen_config,
            )

        # Retry on transient API errors: rate limits (429), server
        # errors (5xx), and network issues.
        try:
            response = retry_with_backoff(
                _call,
                retryable_exceptions=(Exception,),
                context=f"gemini generate ({self._model})",
            )
        except Exception as exc:
            raise ModelClientError(
                backend="gemini",
                message=f"Gemini API call failed: {exc}",
                cause=exc,
            ) from exc

        # Extract the text from Gemini's response object.
        try:
            text = response.text
        except (AttributeError, ValueError) as exc:
            raise ModelClientError(
                backend="gemini",
                message=f"Could not extract text from Gemini response: {exc}",
                cause=exc,
            ) from exc

        # Parse JSON if structured output was requested.
        parsed_json = None
        if json_schema is not None:
            parsed_json = parse_json_response(text, backend="gemini")

        return ModelResponse(
            text=text,
            parsed_json=parsed_json,
            raw_metadata={"model": self._model},
        )
