"""
CLI entrypoint for the mischaracterization detector.

Provides a command-line interface for running the five-stage pipeline
on legal brief passages. Supports three input modes (inline passage,
file, or stdin) and two output formats (human-readable or JSON).

Usage examples::

    # Analyze a single passage
    mischar --passage "Under Smith v. Jones, 123 F.3d 456 (9th Cir. 2001), ..."

    # Analyze passages from a file (one passage per paragraph, separated by blank lines)
    mischar --file brief_excerpts.txt

    # Pipe from stdin
    echo "Under Smith v. Jones, 123 F.3d 456 ..." | mischar --stdin

    # Use a specific model and JSON output
    mischar --passage "..." --model gemma27b-prompted --output json

    # Run without caching
    mischar --passage "..." --no-cache
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mischar.cache import Cache
from mischar.config import Config, ModelConfig, Secrets, load_config, load_secrets
from mischar.constants import DISCLAIMER, PIPELINE_VERSION
from mischar.logging import configure_logging, get_logger
from mischar.models.client import ModelClient, ModelClientError
from mischar.models.embedding import EmbeddingClient
from mischar.pipeline import Pipeline
from mischar.stages.resolve import CourtListenerClient
from mischar.types import CitationResult

log = get_logger("cli")


def main() -> None:
    """
    CLI entrypoint. Parses arguments, initializes the pipeline, runs
    it on the provided input, and formats the output.
    """
    args = _parse_args()

    # Load configuration.
    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        _exit_with_error(f"Configuration error: {exc}", code=2)
    except Exception as exc:
        _exit_with_error(f"Invalid configuration: {exc}", code=2)

    # Load secrets from .env.
    try:
        from dotenv import load_dotenv

        load_dotenv()
        secrets = load_secrets()
    except OSError as exc:
        _exit_with_error(f"Missing secrets: {exc}", code=2)

    # Initialize logging.
    configure_logging(level=config.log_level)

    # Set seeds for reproducibility.
    _set_seeds()

    # Read input passage(s).
    passages = _read_input(args)

    # Override classifier model if --model was specified.
    classifier_model_name = args.model or config.classifier_model
    attribution_model_name = config.attribution_model

    # Construct clients and pipeline.
    try:
        pipeline = _build_pipeline(
            config=config,
            secrets=secrets,
            attribution_model_name=attribution_model_name,
            classifier_model_name=classifier_model_name,
            no_cache=args.no_cache,
        )
    except (ImportError, ModelClientError, OSError) as exc:
        _exit_with_error(f"Failed to initialize pipeline: {exc}", code=1)

    # Run the pipeline on each passage and collect results.
    all_results: list[CitationResult] = []
    exit_code = 0

    for passage in passages:
        try:
            results = pipeline.process_passage(passage)

            if not results:
                _print_no_citations(passage, args.output)
            else:
                all_results.extend(results)
        except ModelClientError as exc:
            # Infrastructure errors are reported inline and set a
            # non-zero exit code, but don't stop processing.
            log.error("pipeline_error", error=str(exc))
            _print_error(str(exc), args.output)
            exit_code = 1
        except Exception as exc:
            log.error("unexpected_error", exc_info=True)
            _print_error(f"Unexpected error: {exc}", args.output)
            exit_code = 1

    # Format and display results.
    if all_results:
        _print_results(all_results, args.output)

    # Clean up.
    pipeline._cache.close()

    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        prog="mischar",
        description=(
            "Legal citation mischaracterization detector. "
            "Analyzes whether legal briefs accurately characterize cited cases."
        ),
        epilog=DISCLAIMER,
    )

    # Input modes — mutually exclusive.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--passage",
        type=str,
        help="Analyze a single passage provided as a string.",
    )
    input_group.add_argument(
        "--file",
        type=Path,
        help=(
            "Analyze passages from a file. Passages are separated by "
            "blank lines (double newlines)."
        ),
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read passage(s) from stdin.",
    )

    # Options.
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Classifier model to use. Must be a key in the config's "
            "models section (e.g. gemma27b-tuned, gemma27b-prompted, "
            "gemini-3.1-pro). Defaults to config's classifier_model."
        ),
    )
    parser.add_argument(
        "--output",
        choices=["human", "json"],
        default="human",
        help="Output format. 'human' for readable text, 'json' for structured JSON. Default: human.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching. Forces fresh computation for every stage.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file. Default: config.yaml.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mischar {PIPELINE_VERSION}",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input reading
# ---------------------------------------------------------------------------


def _read_input(args: argparse.Namespace) -> list[str]:
    """
    Read passage(s) from the specified input source.

    For ``--file`` and ``--stdin``, passages are split on blank lines
    (two consecutive newlines). Each passage is stripped of leading/
    trailing whitespace.

    Args:
        args: The parsed command-line arguments.

    Returns:
        A list of non-empty passage strings.
    """
    if args.passage:
        raw_text = args.passage
    elif args.file:
        if not args.file.exists():
            _exit_with_error(f"File not found: {args.file}", code=2)
        raw_text = args.file.read_text(encoding="utf-8")
    elif args.stdin:
        raw_text = sys.stdin.read()
    else:
        _exit_with_error("No input provided. Use --passage, --file, or --stdin.", code=2)

    # Split on blank lines to get individual passages.
    passages = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

    if not passages:
        _exit_with_error("Input is empty or contains only whitespace.", code=2)

    return passages


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _build_pipeline(
    config: Config,
    secrets: Secrets,
    attribution_model_name: str,
    classifier_model_name: str,
    *,
    no_cache: bool = False,
) -> Pipeline:
    """
    Construct the full pipeline with all clients initialized.

    Looks up the model configurations by name, constructs the appropriate
    backend clients, and wires everything together.

    Args:
        config: Application configuration.
        secrets: API keys.
        attribution_model_name: Name of the attribution model (key in
            config.models).
        classifier_model_name: Name of the classifier model (key in
            config.models).
        no_cache: If True, construct a disabled cache.

    Returns:
        An initialized ``Pipeline`` ready to process passages.

    Raises:
        ValueError: If a model name isn't found in config.
        ImportError: If a backend's dependencies aren't installed.
        ModelClientError: If a backend fails to initialize.
    """
    # Validate model names.
    if attribution_model_name not in config.models:
        raise ValueError(
            f"Attribution model '{attribution_model_name}' not found in config. "
            f"Available: {list(config.models.keys())}"
        )
    if classifier_model_name not in config.models:
        raise ValueError(
            f"Classifier model '{classifier_model_name}' not found in config. "
            f"Available: {list(config.models.keys())}"
        )

    # Build model clients.
    attribution_client = _build_model_client(
        name=attribution_model_name,
        model_config=config.models[attribution_model_name],
        secrets=secrets,
        timeout=config.llm_timeout_seconds,
    )
    classifier_client = _build_model_client(
        name=classifier_model_name,
        model_config=config.models[classifier_model_name],
        secrets=secrets,
        timeout=config.llm_timeout_seconds,
    )

    # Build embedding client.
    embedding_client = EmbeddingClient(
        api_key=secrets.voyage_api_key,
        model=config.embedding_model,
    )

    # Build CourtListener client.
    courtlistener_client = CourtListenerClient(
        api_key=secrets.courtlistener_api_key,
        base_url=config.courtlistener_base_url,
        rate_limit_per_minute=config.courtlistener_rate_limit_per_minute,
        max_retries=config.courtlistener_max_retries,
        timeout_seconds=config.llm_timeout_seconds,
    )

    # Build cache.
    cache = Cache(
        path=config.cache_dir,
        enabled=not no_cache,
    )

    return Pipeline(
        config=config,
        attribution_client=attribution_client,
        classifier_client=classifier_client,
        embedding_client=embedding_client,
        courtlistener_client=courtlistener_client,
        cache=cache,
    )


def _build_model_client(
    name: str,
    model_config: ModelConfig,
    secrets: Secrets,
    timeout: int,
) -> ModelClient:
    """
    Construct a model client from its configuration.

    Routes to the appropriate backend (Ollama, MLX, Gemini, or Modal)
    based on the ``backend`` field in the model config.

    Args:
        name: The model's name (used as display name in logs/cache keys).
        model_config: The per-model configuration from config.yaml.
        secrets: API keys (needed for Gemini).
        timeout: Request timeout in seconds.

    Returns:
        A ``ModelClient`` instance for the specified backend.

    Raises:
        ValueError: If the backend is unrecognized.
        ImportError: If the backend's dependencies aren't installed.
    """
    if model_config.backend == "ollama":
        from mischar.models.ollama import OllamaClient

        return OllamaClient(
            model_name=model_config.ollama_model,
            timeout_seconds=timeout,
        )

    if model_config.backend == "mlx":
        from mischar.models.mlx import MLXClient

        return MLXClient(
            model_path=model_config.base_model_path,
            adapter_path=model_config.adapter_path,
            name_override=name,
        )

    if model_config.backend == "gemini":
        from mischar.models.gemini import GeminiClient

        return GeminiClient(
            api_key=secrets.gemini_api_key,
            model=model_config.api_model,
            timeout_seconds=timeout,
        )

    if model_config.backend == "modal":
        from mischar.models.modal_inference import ModalInferenceClient

        if not model_config.base_model_id:
            raise ValueError(
                f"Model '{name}' uses the modal backend but has no "
                "'base_model_id' configured."
            )

        return ModalInferenceClient(
            base_model_id=model_config.base_model_id,
            adapter_name=model_config.adapter_name,
            name_override=name,
        )

    raise ValueError(f"Unknown backend '{model_config.backend}' for model '{name}'")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_results(results: list[CitationResult], output_format: str) -> None:
    """
    Format and print pipeline results.

    Args:
        results: The pipeline's output CitationResults.
        output_format: Either "human" or "json".
    """
    if output_format == "json":
        _print_results_json(results)
    else:
        _print_results_human(results)


def _print_results_json(results: list[CitationResult]) -> None:
    """
    Print results as a JSON array to stdout.

    Converts CitationResult dataclasses to dicts, handling fields that
    aren't directly JSON-serializable (dates, tuples, etc.).

    Args:
        results: The pipeline's output CitationResults.
    """
    output = []
    for result in results:
        record = _citation_result_to_dict(result)
        output.append(record)

    print(json.dumps(output, indent=2, default=str))


def _print_results_human(results: list[CitationResult]) -> None:
    """
    Print results in a human-readable format to stdout.

    Each citation gets a block with the citation text, verdict (label or
    abstention reason), confidence, and supporting text.

    Args:
        results: The pipeline's output CitationResults.
    """
    print(f"\n{'='*60}")
    print(f"  Mischaracterization Detector v{PIPELINE_VERSION}")
    print(f"{'='*60}")

    for i, result in enumerate(results):
        print(f"\n--- Citation {i + 1} ---")
        print(f"  Citation: {result.citation.raw_text}")

        if result.resolved_case:
            print(f"  Resolved: {result.resolved_case.case_name}")

        if result.abstained:
            print(f"  Verdict:  ABSTAINED")
            print(f"  Reason:   {result.abstention.reason}")

            if result.abstention.details:
                print(f"  Details:  {result.abstention.details}")
        else:
            label = result.classification.label
            confidence = result.classification.confidence

            # Color-code the label for terminal readability.
            label_display = _format_label(label)
            print(f"  Verdict:  {label_display}")
            print(f"  Confidence: {confidence:.0%}")

            if result.classification.supporting_text:
                print(f"  Reasoning: {result.classification.supporting_text}")

        if result.claim:
            print(f"  Attributed claim: {result.claim.claim_text}")

        print(f"  Model: {result.model_used}")

    print(f"\n{'='*60}")
    print(f"  {DISCLAIMER}")
    print(f"{'='*60}\n")


def _print_no_citations(passage: str, output_format: str) -> None:
    """
    Report that no citations were found in a passage.

    Args:
        passage: The passage that had no citations.
        output_format: Either "human" or "json".
    """
    preview = passage[:80] + ("..." if len(passage) > 80 else "")

    if output_format == "json":
        print(json.dumps({
            "status": "no_citations_found",
            "passage_preview": preview,
        }, indent=2))
    else:
        print(f"\nNo citations found in passage: \"{preview}\"")


def _print_error(message: str, output_format: str) -> None:
    """
    Report an error during processing.

    Args:
        message: The error message.
        output_format: Either "human" or "json".
    """
    if output_format == "json":
        print(json.dumps({"error": message}, indent=2), file=sys.stderr)
    else:
        print(f"\nError: {message}", file=sys.stderr)


def _format_label(label: str) -> str:
    """
    Format a classification label for human-readable display.

    Uses descriptive language so non-technical readers understand
    the verdict at a glance.

    Args:
        label: One of the two classification labels.

    Returns:
        A formatted label string.
    """
    descriptions = {
        "accurate": "ACCURATE — Case supports the claim as stated",
        "mischaracterized": (
            "MISCHARACTERIZED — Claim misstates what the case held"
        ),
    }

    return descriptions.get(label, label.upper())


def _citation_result_to_dict(result: CitationResult) -> dict:
    """
    Convert a CitationResult to a JSON-serializable dict.

    Strips the full opinion text from the resolved case (too large for
    output) and converts non-serializable fields.

    Args:
        result: The CitationResult to convert.

    Returns:
        A JSON-serializable dict.
    """
    d = asdict(result)

    # Remove full_text from resolved_case — it's too large and not
    # useful in CLI output.
    if d.get("resolved_case") and "full_text" in d["resolved_case"]:
        d["resolved_case"]["full_text"] = (
            f"[{len(result.resolved_case.full_text)} chars — omitted from output]"
        )

    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: The seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Torch seed is set only if torch is available (it's not required
    # for inference — only for training on Modal).
    try:
        import torch
        
        torch.manual_seed(seed)
    except ImportError:
        pass


def _exit_with_error(message: str, code: int = 1) -> None:
    """
    Print an error message to stderr and exit.

    Args:
        message: The error message to display.
        code: The exit code (default 1, use 2 for input errors).
    """
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


if __name__ == "__main__":
    main()
