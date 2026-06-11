"""
Modal model client for fine-tuned adapter inference.

Connects to the deployed ``mischar-inference`` Modal app (see
``scripts/inference/serve_adapter.py``) and satisfies the ``ModelClient``
protocol, so the pipeline and eval harness can use a fine-tuned model on
a Modal GPU exactly like any local backend.

The deployment must exist before this client can connect:

    modal deploy src/mischar/scripts/inference/serve_adapter.py

Requires the ``modal`` package and an authenticated Modal CLI
(``modal token new``) on the local machine.
"""

from __future__ import annotations

from mischar.logging import get_logger
from mischar.models.client import (
    ModelClientError,
    parse_json_response,
    retry_with_backoff,
)
from mischar.types import ModelResponse

log = get_logger("modal_inference")

DEFAULT_APP_NAME = "mischar-inference"
DEFAULT_CLASS_NAME = "MischarClassifier"


class ModalInferenceClient:
    """
    Client for remote inference against the deployed Modal app.

    The first call after a period of inactivity is slow (cold start:
    container boot + model load, several minutes), but subsequent calls
    hit a warm container and return in seconds. The container scales to
    zero after the configured idle window, so no GPU cost accrues
    between eval runs.

    Args:
        base_model_id: HuggingFace ID of the base model
            (e.g. ``"google/gemma-3-27b-it"``).
        adapter_name: Name of the adapter directory in the
            ``mischar-adapters`` volume (e.g. ``"gemma3-27b-mischar-v2"``).
            Empty string or None serves the unadapted base model.
        app_name: Name of the deployed Modal app.
        class_name: Name of the serving class within the app.
        name_override: Optional display name for logs and cache keys.
    """

    def __init__(
        self,
        base_model_id: str,
        adapter_name: str | None = None,
        *,
        app_name: str = DEFAULT_APP_NAME,
        class_name: str = DEFAULT_CLASS_NAME,
        name_override: str | None = None,
    ) -> None:
        # Import here so the rest of the codebase doesn't require modal.
        try:
            import modal
        except ImportError as exc:
            raise ImportError(
                "ModalInferenceClient requires the 'modal' package. "
                "Install it with: pip install modal"
            ) from exc

        self._adapter_name = adapter_name or ""
        self._base_model_id = base_model_id

        if name_override:
            self.name = name_override
        elif self._adapter_name:
            self.name = f"modal:{self._adapter_name}"
        else:
            self.name = f"modal:{base_model_id.split('/')[-1]}"

        try:
            cls = modal.Cls.from_name(app_name, class_name)
            self._instance = cls(
                base_model_id=base_model_id,
                adapter_name=self._adapter_name,
            )
        except Exception as exc:
            raise ModelClientError(
                backend="modal",
                message=(
                    f"Could not connect to Modal app '{app_name}'. "
                    "Has it been deployed? Run: "
                    "modal deploy src/mischar/scripts/inference/serve_adapter.py "
                    f"({exc})"
                ),
                cause=exc,
            ) from exc

        log.info(
            "modal_client_initialized",
            app=app_name,
            base_model=base_model_id,
            adapter=self._adapter_name or "(none)",
        )

    def generate(
        self,
        prompt: str,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Generate a completion via the deployed Modal app.

        Like the MLX backend, Modal has no native JSON schema enforcement,
        so when ``json_schema`` is provided we append a JSON instruction
        to the prompt and parse the result. The fine-tuned models were
        trained to emit label-only JSON, so this is mostly a safety net
        for the prompted baseline.

        Args:
            prompt: The full prompt text.
            json_schema: Optional JSON schema (parsing hint only).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            ``ModelResponse`` with raw text and optionally parsed JSON.

        Raises:
            ModelClientError: If the remote call fails after retries.
        """
        effective_prompt = prompt
        if json_schema is not None:
            effective_prompt = (
                f"{prompt}\n\nRespond with a JSON object. "
                f"Do not include any text outside the JSON object."
            )

        def _call() -> str:
            return self._instance.generate.remote(
                prompt=effective_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        try:
            # Only one retry: a failed call may have already paid for a
            # cold start, and most genuine failures (bad adapter path,
            # OOM) won't be fixed by retrying.
            text = retry_with_backoff(
                _call,
                max_retries=1,
                context=f"modal generate ({self.name})",
            )
        except Exception as exc:
            raise ModelClientError(
                backend="modal",
                message=f"Modal generation failed: {exc}",
                cause=exc,
            ) from exc

        parsed_json = None
        if json_schema is not None:
            parsed_json = parse_json_response(text, backend="modal")

        return ModelResponse(
            text=text,
            parsed_json=parsed_json,
            raw_metadata={
                "base_model_id": self._base_model_id,
                "adapter_name": self._adapter_name,
            },
        )

    def close(self) -> None:
        """No persistent local resources to release."""
