"""
Modal app definition for QLoRA fine-tuning.

Defines the shared infrastructure used by both the primary (Gemma 3 27B) and
secondary (Gemma 3 12B) training entrypoints:

- **Image**: Ubuntu 22.04 with PyTorch (CUDA 12.x), Transformers, TRL, PEFT,
  bitsandbytes, and Accelerate pre-installed.
- **Volumes**: One for training data (read), one for adapter outputs (read/write).
- **Secrets**: HuggingFace token for gated model access (Gemma requires
  acceptance of the Gemma Terms of Use on HuggingFace).

Training scripts import the ``app``, ``image``, ``training_data_volume``,
``adapter_volume``, and path constants from this module.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("mischar-training")

# ---------------------------------------------------------------------------
# Image — frozen Python environment for reproducible training
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.46",
        "peft>=0.13",
        "trl>=0.12",
        "bitsandbytes>=0.44",
        "accelerate>=1.0",
        "datasets>=3.0",
        "scikit-learn>=1.4",
        "numpy>=1.26",
        "pyyaml>=6.0",
        # sentencepiece / protobuf needed for Gemma tokenizer
        "sentencepiece",
        "protobuf",
    )
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

# Training data (JSONL files uploaded before training). Mount read-only in
# the training function; data construction scripts upload to it separately.
training_data_volume = modal.Volume.from_name(
    "mischar-training-data",
    create_if_missing=True,
)

# Adapter outputs. The training function writes checkpoints and final
# adapters here. User pulls them to local ``artifacts/adapters/`` afterward.
adapter_volume = modal.Volume.from_name(
    "mischar-adapters",
    create_if_missing=True,
)

# ---------------------------------------------------------------------------
# Volume mount paths (inside the container)
# ---------------------------------------------------------------------------

TRAINING_DATA_PATH = "/data"
ADAPTER_OUTPUT_PATH = "/adapters"

# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------

# H100 80GB is the target; A100 80GB is the fallback. The 27B model in
# 4-bit quantization peaks at ~20 GB model memory + ~15 GB for optimizer
# states and activations, fitting comfortably on either.
GPU_TYPE = "H100"
GPU_COUNT = 1

# Training timeout. 27B QLoRA for 3 epochs over ~10K examples should
# finish well within 4 hours; 6 hours gives comfortable headroom.
TRAINING_TIMEOUT_SECONDS = 6 * 60 * 60
