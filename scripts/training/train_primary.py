"""
Primary fine-tuning entrypoint: Gemma 3 27B QLoRA.

Runs on Modal with an H100 GPU. Fine-tunes a 4-bit quantized Gemma 3 27B
on the mischaracterization classification task using QLoRA (rank 32, alpha 64).

Training data format:
    JSONL files in the training data volume, each line a JSON object with:
    - ``prompt``: The classification prompt (claim + retrieved case text).
    - ``completion``: The target JSON output (label, confidence, supporting_text).

The script saves two adapter checkpoints to the adapter volume:
    1. The final adapter after all epochs.
    2. The best adapter by validation loss (from early stopping / checkpoint selection).

Usage:
    modal run scripts/training/train_primary.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from scripts.training.modal_app import (
    ADAPTER_OUTPUT_PATH,
    GPU_COUNT,
    GPU_TYPE,
    TRAINING_DATA_PATH,
    TRAINING_TIMEOUT_SECONDS,
    adapter_volume,
    app,
    image,
    training_data_volume,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model identifiers on HuggingFace. Gemma 3 27B IT (instruction-tuned) is
# the base model we fine-tune on top of.
BASE_MODEL_ID = "google/gemma-3-27b-it"
ADAPTER_OUTPUT_NAME = "gemma3-27b-mischar-v1"

# QLoRA configuration.
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
# Target all linear layers for maximum adapter expressiveness at this
# scale. PEFT's "all-linear" shorthand handles this automatically.
LORA_TARGET_MODULES = "all-linear"

# Training hyperparameters.
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
# Effective batch size = per_device_batch * gradient_accumulation.
# 2 * 8 = 16 effective batch size.
PER_DEVICE_TRAIN_BATCH = 2
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.03  # 3% of total steps
LR_SCHEDULER = "cosine"
MAX_SEQ_LENGTH = 2048
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 50

# Early stopping patience: stop if val loss doesn't improve for this many
# eval rounds. With eval every 500 steps over ~10K examples at effective
# batch 16, that's roughly 2-3 evaluations per epoch.
EARLY_STOPPING_PATIENCE = 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(data_dir: str) -> tuple[list[dict], list[dict]]:
    """
    Load train and validation splits from JSONL files in the data volume.

    Expects ``train.jsonl`` and ``val.jsonl`` in ``data_dir``. Each line
    is a JSON object with ``prompt`` and ``completion`` fields.

    Args:
        data_dir: Path to the directory containing the JSONL files.

    Returns:
        A tuple of (train_examples, val_examples), each a list of dicts.

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

    Skips blank lines. Raises on malformed JSON so training doesn't
    silently proceed with missing data.

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


# ---------------------------------------------------------------------------
# Formatting — converts raw examples into the chat format Gemma expects
# ---------------------------------------------------------------------------


def format_for_sft(example: dict) -> dict:
    """
    Format a training example into a single ``text`` field for SFTTrainer.

    Gemma's instruction-tuned models use a specific chat template. We
    format the classification prompt as a user turn and the target
    classification JSON as the model turn, then let the tokenizer's
    ``apply_chat_template`` handle the special tokens.

    Args:
        example: A dict with ``prompt`` and ``completion`` fields.

    Returns:
        A dict with a ``messages`` field containing the formatted
        conversation in the chat template format.
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
    Fine-tune Gemma 3 27B with QLoRA on the mischaracterization dataset.

    This function runs inside the Modal container with GPU access. It:
    1. Loads the base model in 4-bit quantization (bitsandbytes NF4).
    2. Applies LoRA adapters (rank 32, alpha 64, all linear layers).
    3. Loads training data from the data volume.
    4. Trains with TRL's SFTTrainer (cosine LR, warmup, early stopping).
    5. Saves final and best-by-val-loss adapters to the adapter volume.

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

    # Gemma tokenizer doesn't set a pad token by default. Use EOS so
    # padding tokens are ignored during loss computation.
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
        attn_implementation="flash_attention_2",
    )

    # Disable caching during training — incompatible with gradient
    # checkpointing and wastes memory.
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
        max_seq_length=MAX_SEQ_LENGTH,
        # Evaluation and checkpointing
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        # Reproducibility
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
    # Save it explicitly under a "best" subdirectory for clarity.
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

    # Commit volume changes so adapters persist after the container exits.
    adapter_volume.commit()

    return summary


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    """
    Local entrypoint for ``modal run scripts/training/train_primary.py``.

    Dispatches to the remote ``train`` function and prints the result.
    """
    result = train.remote()
    print(result)
