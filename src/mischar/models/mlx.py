"""
MLX model client for fine-tuned adapter inference on Apple Silicon.

Loads a quantized base model (e.g. Gemma 3 27B 4-bit) plus a QLoRA adapter
via ``mlx-lm``, which runs natively on Apple Silicon unified memory. This is
the primary inference path for the fine-tuned classifier.

This module requires ``mlx`` and ``mlx-lm`` to be installed (part of the
``local`` optional dependency group). It will raise ``ImportError`` at
instantiation time if they're missing.
"""

from __future__ import annotations

from mischar.logging import get_logger
from mischar.models.client import (
    ModelClientError,
    parse_json_response,
    retry_with_backoff,
)
from mischar.types import ModelResponse

log = get_logger("mlx")


class MLXClient:
    """
    Client for local inference using MLX with optional LoRA adapters.

    Loads the model and adapter into unified memory on first instantiation.
    This can take 30-60 seconds for a 27B 4-bit model but only happens once.

    Args:
        model_path: Path or HuggingFace ID for the base model
            (e.g. ``"mlx-community/gemma-3-27b-it-4bit"``).
        adapter_path: Path to the QLoRA adapter directory. If None, the base
            model is used without adaptation (useful for prompted baseline
            comparison).
        name_override: Optional display name. If not provided, a name is
            generated from the model and adapter paths.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        *,
        name_override: str | None = None,
    ) -> None:
        # Import mlx-lm here rather than at module level so that the rest
        # of the codebase can be imported on non-Apple machines (e.g. for
        # testing or for running only the Gemini/Ollama backends).
        try:
            from mlx_lm import generate as mlx_generate
            from mlx_lm import load as mlx_load
        except ImportError as exc:
            raise ImportError(
                "MLXClient requires 'mlx' and 'mlx-lm' packages. "
                "Install them with: pip install -e '.[local]'"
            ) from exc

        self._mlx_generate = mlx_generate
        self._model_path = model_path
        self._adapter_path = adapter_path

        # Build a human-readable name for logging and cache keys.
        if name_override:
            self.name = name_override
        elif adapter_path:
            # Extract just the adapter directory name from the full path.
            adapter_name = adapter_path.rstrip("/").split("/")[-1]
            self.name = f"mlx:{adapter_name}"
        else:
            self.name = f"mlx:{model_path.split('/')[-1]}"

        log.info(
            "mlx_loading_model",
            model_path=model_path,
            adapter_path=adapter_path,
        )

        # Load the model and tokenizer into unified memory. This is the
        # slow step (~30-60s for 27B 4-bit). mlx_load handles downloading
        # from HuggingFace Hub if the model isn't cached locally.
        self._model, self._tokenizer = mlx_load(
            model_path,
            adapter_path=adapter_path,
        )

        log.info("mlx_model_loaded", name=self.name)


    def generate(
        self,
        prompt: str,
        json_schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Generate a completion using the loaded MLX model.

        Unlike Ollama and Gemini, MLX doesn't have native JSON schema
        enforcement. When ``json_schema`` is provided, we append an
        instruction to the prompt asking for JSON output, then parse
        the result. The fine-tuned model should reliably produce valid
        JSON since it was trained on structured output examples, but the 
        prompted baseline may be shakier, which is why fallback options are 
        included in the parse_json_response function.

        Args:
            prompt: The full prompt text.
            json_schema: Optional JSON schema (used as a parsing hint;
                MLX doesn't enforce it at the generation level).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            ``ModelResponse`` with raw text and optionally parsed JSON.

        Raises:
            ``ModelClientError`` if generation fails unexpectedly.
        """
        # If structured output is requested, append a JSON instruction
        # to the prompt. The fine-tuned model was trained to produce JSON
        # in this format, so this is mostly a safety net for the prompted
        # baseline.
        effective_prompt = prompt
        if json_schema is not None:
            effective_prompt = (
                f"{prompt}\n\nRespond with a JSON object. "
                f"Do not include any text outside the JSON object."
            )


        def _call() -> str:
            """
            Inner function for retry_with_backoff.

            The reason for using this inner function is so that the necessary 
            arguments are baked in and retry_with_backoff can call it with no
            arguments.
            """
            return self._mlx_generate(
                self._model,
                self._tokenizer,
                prompt=effective_prompt,
                temp=temperature,
                max_tokens=max_tokens,
            )

        try:
            text = retry_with_backoff(
                _call,
                # MLX generation is local and shouldn't fail transiently,
                # but we retry on any exception as a safety net.
                max_retries=1,
                context=f"mlx generate ({self.name})",
            )
        except Exception as exc:
            raise ModelClientError(
                backend="mlx",
                message=f"MLX generation failed: {exc}",
                cause=exc,
            ) from exc

        # Parse JSON if structured output was requested.
        parsed_json = None
        if json_schema is not None:
            parsed_json = parse_json_response(text, backend="mlx")

        return ModelResponse(
            text=text,
            parsed_json=parsed_json,
            raw_metadata={"model_path": self._model_path, "adapter_path": self._adapter_path},
        )
