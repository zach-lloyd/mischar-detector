"""
Secondary fine-tuning entrypoint: Gemma 3 12B QLoRA.

Same training setup as the primary (27B) but targeting the smaller Gemma 3 12B
for the deployability story. The 12B model trains faster and requires less
memory, making it suitable for validating the training pipeline before
committing to the more expensive 27B run.

Ideally run the 12B first: if it converges poorly or shows degenerate output,
that's a signal to revisit hyperparameters before kicking off the 27B run.

Training data format is identical to the primary — same JSONL files with
``prompt`` and ``completion`` fields.

Usage:
    modal run src/mischar/scripts/training/train_secondary.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure (app, image, volumes, GPU config)
#
# These are defined inline rather than imported from a shared module because
# Modal copies each script into the container at /root/<script>.py, detached
# from the local directory structure. Cross-file imports break inside the
# container regardless of how they're written (absolute, relative, sys.path).
# Inlining ~50 lines is the simplest way to keep each script self-contained.
# ---------------------------------------------------------------------------

app = modal.App("mischar-training")

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

# Training data (JSONL files uploaded before training).
training_data_volume = modal.Volume.from_name(
    "mischar-training-data",
    create_if_missing=True,
)

# Adapter outputs. User pulls these to local artifacts/adapters/ afterward.
adapter_volume = modal.Volume.from_name(
    "mischar-adapters",
    create_if_missing=True,
)

# Volume mount paths (inside the container)
TRAINING_DATA_PATH = "/data"
ADAPTER_OUTPUT_PATH = "/adapters"

# H100 80GB is the target; A100 80GB is the fallback.
GPU_TYPE = "H100"
GPU_COUNT = 1

# 10 hours of headroom — long-context (8192) runs can be slow, and the
# timeout is only a ceiling (billing is per actual second used), so a
# generous limit costs nothing unless a run genuinely needs the time.
TRAINING_TIMEOUT_SECONDS = 10 * 60 * 60

# ---------------------------------------------------------------------------
# Constants — overrides from the primary entrypoint
# ---------------------------------------------------------------------------

# Gemma 3 12B IT is the smaller base model.
BASE_MODEL_ID = "google/gemma-3-12b-it"
# v2: binary accurate/mischaracterized labels with label-only completions.
ADAPTER_OUTPUT_NAME = "gemma3-12b-mischar-v2"

# Same QLoRA configuration as primary. No reason to change these for the
# 12B model — the adapter capacity should be proportionally similar.
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

# Same training hyperparameters. The 12B model is smaller so we can afford
# a larger per-device batch, reducing gradient accumulation steps while
# keeping the same effective batch size of 16.
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH = 2
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
MAX_LENGTH = 8192
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 50
EARLY_STOPPING_PATIENCE = 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(data_dir: str) -> tuple[list[dict], list[dict]]:
    """
    Load train and validation splits from JSONL files in the data volume.

    Expects ``train.jsonl`` and ``val.jsonl`` in ``data_dir``.

    Args:
        data_dir: Path to the directory containing the JSONL files.

    Returns:
        A tuple of (train_examples, val_examples).

    Raises:
        FileNotFoundError: If either file is missing.
    """
    train_path = Path(data_dir) / "train.jsonl"
    val_path = Path(data_dir) / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Upload train.jsonl to the mischar-training-data volume first."
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation data not found at {val_path}. "
            "Upload val.jsonl to the mischar-training-data volume first."
        )

    train_examples = _read_jsonl(train_path)
    val_examples = _read_jsonl(val_path)

    print(
        f"Loaded {len(train_examples)} training examples, "
        f"{len(val_examples)} validation examples"
    )

    return train_examples, val_examples


def _read_jsonl(path: Path) -> list[dict]:
    """
    Read a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of parsed dicts.
    """
    records = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at {path}:{line_num}: {exc}"
                ) from exc

    return records


def format_for_sft(example: dict) -> dict:
    """
    Format a training example into chat template messages for SFTTrainer.

    Args:
        example: A dict with ``prompt`` and ``completion`` fields.

    Returns:
        A dict with a ``messages`` field for the chat template.
    """
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


# ---------------------------------------------------------------------------
# Training function — runs on Modal
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        TRAINING_DATA_PATH: training_data_volume,
        ADAPTER_OUTPUT_PATH: adapter_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=TRAINING_TIMEOUT_SECONDS,
)
def train() -> str:
    """
    Fine-tune Gemma 3 12B with QLoRA on the mischaracterization dataset.

    Same procedure as the primary 27B training but with the smaller model
    and larger per-device batch size (4 instead of 2) to take advantage
    of the freed GPU memory.

    Returns:
        A summary string with training results.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
    )
    from trl import SFTConfig, SFTTrainer

    print("=" * 60)
    print(f"Starting QLoRA fine-tuning: {BASE_MODEL_ID}")
    print(f"GPU: {GPU_TYPE} x{GPU_COUNT}")
    print(f"LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}")
    eff_batch = PER_DEVICE_TRAIN_BATCH * GRADIENT_ACCUMULATION_STEPS
    print(f"LR={LEARNING_RATE}, epochs={NUM_EPOCHS}, effective_batch={eff_batch}")
    print("=" * 60)

    # ----- 1. Load tokenizer -----

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=hf_token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- 2. Load model in 4-bit -----

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False

    # ----- 3. Apply LoRA -----

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ----- 4. Load and format data -----

    train_examples, val_examples = load_training_data(TRAINING_DATA_PATH)

    train_formatted = [format_for_sft(ex) for ex in train_examples]
    val_formatted = [format_for_sft(ex) for ex in val_examples]

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Val dataset: {len(val_dataset)} examples")

    # ----- 5. Configure trainer -----

    output_dir = os.path.join(ADAPTER_OUTPUT_PATH, ADAPTER_OUTPUT_NAME)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        max_length=MAX_LENGTH,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        seed=42,
        data_seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    # ----- 6. Train -----

    print("Starting training...")
    train_result = trainer.train()

    # ----- 7. Save best adapter -----

    # The best checkpoint by val loss is already loaded (load_best_model_at_end=True).
    best_adapter_path = os.path.join(output_dir, "best")
    trainer.save_model(best_adapter_path)
    tokenizer.save_pretrained(best_adapter_path)
    print(f"Best adapter (by val loss) saved to {best_adapter_path}")

    # ----- 8. Log summary -----

    metrics = train_result.metrics
    summary = (
        f"Training complete.\n"
        f"  Total steps: {metrics.get('train_steps', 'N/A')}\n"
        f"  Final train loss: {metrics.get('train_loss', 'N/A'):.4f}\n"
        f"  Runtime: {metrics.get('train_runtime', 0):.0f}s\n"
        f"  Samples/sec: {metrics.get('train_samples_per_second', 0):.1f}\n"
        f"  Adapters saved to: {output_dir}\n"
    )
    print(summary)

    adapter_volume.commit()

    return summary


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    """
    Local entrypoint for ``modal run src/mischar/scripts/training/train_secondary.py``.

    Dispatches to the remote ``train`` function and prints the result.
    """
    result = train.remote()
    print(result)
