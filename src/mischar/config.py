"""Configuration loading and validation.

Loads ``config.yaml`` into a validated ``Config`` dataclass. Secrets are loaded
separately from ``.env`` via python-dotenv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

from mischar.constants import (
    DEFAULT_CHUNK_MAX_TOKENS,
    DEFAULT_CHUNK_OVERLAP_PARAGRAPHS,
    DEFAULT_COURTLISTENER_MAX_RETRIES,
    DEFAULT_COURTLISTENER_RATE_LIMIT,
    DEFAULT_GENERATION_MAX_TOKENS,
    DEFAULT_GENERATION_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_PINCITE_BOOST,
    DEFAULT_PINCITE_NEIGHBOR_WINDOW,
    DEFAULT_TOP_K,
)

# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a single model backend."""

    backend: str  # "ollama", "mlx", or "gemini"

    # Ollama fields
    ollama_model: str | None = None

    # MLX fields
    base_model_path: str | None = None
    adapter_path: str | None = None

    # Gemini fields
    api_model: str | None = None

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"ollama", "mlx", "gemini"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Top-level config (pydantic model for validation, then frozen)
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Top-level application configuration.

    Loaded from ``config.yaml``, validated by pydantic, and then treated
    as read-only for the lifetime of a run.
    """

    model_config = {"frozen": True}

    # Paths
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./cache")
    artifacts_dir: Path = Path("./artifacts")
    eval_runs_dir: Path = Path("./eval_runs")

    # Default model selection
    attribution_model: str = "gemma27b-prompted"
    classifier_model: str = "gemma27b-tuned"
    attribution_fallback: str | None = "gemini-3.1-pro"

    # Per-model configs
    models: dict[str, ModelConfig] = field(default_factory=dict)

    # Retrieval
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS
    chunk_overlap_paragraphs: int = DEFAULT_CHUNK_OVERLAP_PARAGRAPHS
    top_k: int = DEFAULT_TOP_K
    pincite_boost: float = DEFAULT_PINCITE_BOOST
    pincite_neighbor_window: int = DEFAULT_PINCITE_NEIGHBOR_WINDOW

    # Embedding
    embedding_model: str = "voyage-law-2"

    # CourtListener
    courtlistener_base_url: str = "https://www.courtlistener.com/api/rest/v3/"
    courtlistener_rate_limit_per_minute: int = DEFAULT_COURTLISTENER_RATE_LIMIT
    courtlistener_max_retries: int = DEFAULT_COURTLISTENER_MAX_RETRIES

    # LLM generation
    generation_temperature: float = DEFAULT_GENERATION_TEMPERATURE
    generation_max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS
    llm_timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS

    # Prompt versions
    attribution_prompt_version: str = "v1.0"
    classification_prompt_version: str = "v1.0"

    # Logging
    log_level: str = "INFO"
    log_rotation_days: int = 30


# ---------------------------------------------------------------------------
# Secrets (loaded from .env, not from config.yaml)
# ---------------------------------------------------------------------------


@dataclass
class Secrets:
    """API keys loaded from the ``.env`` file."""

    courtlistener_api_key: str
    voyage_api_key: str
    gemini_api_key: str


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_config(path: Path | str = "config.yaml") -> Config:
    """Load and validate configuration from a YAML file.

    Raises ``FileNotFoundError`` if the config file is missing and
    ``pydantic.ValidationError`` if the content is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return Config(**raw)


def load_secrets() -> Secrets:
    """Load API keys from environment variables (expected via ``.env``).

    Call ``dotenv.load_dotenv()`` before invoking this function.

    Raises ``EnvironmentError`` if any required key is missing.
    """
    import os

    keys = {
        "courtlistener_api_key": os.environ.get("COURTLISTENER_API_KEY"),
        "voyage_api_key": os.environ.get("VOYAGE_API_KEY"),
        "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
    }

    missing = [name for name, val in keys.items() if not val]
    if missing:
        raise OSError(
            f"Missing required environment variables: "
            f"{', '.join(k.upper() for k in missing)}. "
            f"Set them in your .env file."
        )

    return Secrets(**keys)  # type: ignore[arg-type]
