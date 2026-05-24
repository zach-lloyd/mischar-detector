"""
Ollama model client for locally-served Gemma 3 27B.

Connects to an Ollama instance running on localhost via its OpenAI-compatible
``/api/chat`` endpoint. Used for the prompted (non-fine-tuned) baseline.
"""

from __future__ import annotations

import httpx

from mischar.logging import get_logger
from mischar.models.client import (
    ModelClientError,
    parse_json_response,
    retry_with_backoff,
)
from mischar.types import ModelResponse

log = get_logger("ollama")


class OllamaClient:
    """
    Client for a locally-running Ollama model server.

    Ollama serves models via HTTP and supports structured JSON output through
    its ``format`` parameter. This client wraps that API with retries and
    timeout handling.

    Args:
        model_name: The Ollama model tag (e.g. ``"gemma3:27b"``).
        base_url: Ollama server URL. Defaults to localhost on the standard port.
        timeout_seconds: How long to wait for a response before giving up.
            Local 27B inference on an M3 Max can take a while on long prompts.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
    ) -> None:
        self.name = f"ollama:{model_name}"
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")

        # httpx client with a generous timeout — 27B models on local hardware
        # can be slow, especially on first load when weights are being read
        # from disk into unified memory.
        self._http = httpx.Client(timeout=httpx.Timeout(timeout_seconds))

        log.info(
            "ollama_client_initialized",
            model=model_name,
            base_url=self._base_url,
        )


    def generate(
        self,
        prompt: str,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Send a prompt to Ollama and return the response.

        Uses Ollama's ``/api/chat`` endpoint with a single user message.
        When ``json_schema`` is provided, Ollama constrains output to valid
        JSON matching the schema (native structured output support).

        Args:
            prompt: The full prompt text.
            json_schema: Optional JSON schema to constrain output format.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            ``ModelResponse`` with raw text and optionally parsed JSON.

        Raises:
            ``ModelClientError`` if Ollama is unreachable or returns errors
            after retries.
        """
        # Build the request payload for Ollama's /api/chat endpoint.
        payload: dict = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,  # We want the full response at once, not streamed tokens.
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # When a JSON schema is provided, Ollama uses it to constrain
        # generation so the output is guaranteed to be valid JSON matching
        # the schema structure.
        if json_schema is not None:
            payload["format"] = json_schema


        def _call() -> httpx.Response:
            """
            Inner function for retry_with_backoff to wrap.

            The reason for using this inner function is so that the necessary 
            arguments are baked in and retry_with_backoff can call it with no
            arguments.
            """
            return self._http.post(f"{self._base_url}/api/chat", json=payload)

        # Retry on transient HTTP errors (connection refused if Ollama is
        # still loading, timeouts on very long prompts, etc.)
        try:
            response = retry_with_backoff(
                _call,
                retryable_exceptions=(httpx.HTTPError,),
                context=f"ollama generate ({self._model_name})",
            )
        except httpx.HTTPError as exc:
            raise ModelClientError(
                backend="ollama",
                message=f"Failed to reach Ollama at {self._base_url}: {exc}",
                cause=exc,
            ) from exc

        # Check for non-200 status codes from Ollama.
        if response.status_code != 200:
            raise ModelClientError(
                backend="ollama",
                message=f"Ollama returned status {response.status_code}: {response.text[:500]}",
            )

        # Parse the response body. Ollama returns JSON with the model's
        # output in response["message"]["content"].
        try:
            data = response.json()
            text = data["message"]["content"]
        except (KeyError, ValueError) as exc:
            raise ModelClientError(
                backend="ollama",
                message=f"Unexpected response format from Ollama: {exc}",
                cause=exc,
            ) from exc

        # If we requested structured JSON output, try to parse it.
        parsed_json = None
        if json_schema is not None:
            parsed_json = parse_json_response(text, backend="ollama")

        return ModelResponse(
            text=text,
            parsed_json=parsed_json,
            raw_metadata={
                "model": data.get("model"),
                "total_duration_ns": data.get("total_duration"),
                "eval_count": data.get("eval_count"),
            },
        )


    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
