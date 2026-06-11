"""
Evaluation runner — ties dataset, pipeline, harness, and report together.

For each requested model, builds a pipeline with that model as the
classifier, runs the evaluation harness over the dataset, and writes a
combined report (``summary.md``, ``metrics.json``, ``run_manifest.json``)
plus per-example prediction files to a timestamped directory under
``eval_runs/``.

The per-example prediction files (``predictions-<model>.jsonl``) are what
enable cross-run analysis — e.g., "which examples did the fine-tuned 27B
get right that the baseline missed?" Predictions are keyed by example ID,
which is why test-set records must carry stable ``annotation_id`` values.

Usage:
    # Evaluate the default classifier on the real-brief test set:
    python -m mischar.scripts.eval.run_eval \\
        --dataset data/processed/annotated/real_briefs.jsonl

    # Compare several models in one run:
    python -m mischar.scripts.eval.run_eval \\
        --dataset data/processed/annotated/real_briefs.jsonl \\
        --models gemma27b-tuned,gemma12b-tuned,gemma27b-prompted-modal,gemini-3.1-pro

    # Sanity-check a model on the CaseHOLD val split:
    python -m mischar.scripts.eval.run_eval \\
        --dataset data/processed/casehold.jsonl \\
        --source casehold --split val --models gemma12b-tuned \\
        [--max-examples 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from mischar.cli import _build_pipeline
from mischar.config import load_config, load_secrets
from mischar.data.datasets import load_casehold, load_real_brief, validate_labels
from mischar.eval.harness import EvalRun, run_evaluation
from mischar.eval.report import build_report
from mischar.logging import configure_logging, get_logger
from mischar.types import EvalExample

log = get_logger("scripts.run_eval")

DATASET_LOADERS = {
    "casehold": load_casehold,
    "real_brief": load_real_brief,
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset(
    path: Path,
    source: str,
    split: str | None,
    max_examples: int | None,
) -> list[EvalExample]:
    """
    Load and validate the evaluation dataset.

    Args:
        path: Path to the dataset file.
        source: Which loader to use ("casehold" or "real_brief").
        split: Optional split filter ("train", "val", or "test").
        max_examples: Optional cap on the number of examples (smoke tests).

    Returns:
        A list of validated ``EvalExample`` objects.
    """
    loader = DATASET_LOADERS[source]
    examples = validate_labels(loader(path, split=split))

    if max_examples:
        examples = examples[:max_examples]

    return examples


# ---------------------------------------------------------------------------
# Prediction dumps
# ---------------------------------------------------------------------------


def _write_predictions(run: EvalRun, run_dir: Path) -> Path:
    """
    Write per-example predictions for one model to the run directory.

    One JSON object per line, keyed by ``example_id``, with the gold
    label alongside the prediction so the file is self-contained for
    cross-run comparisons.

    Args:
        run: The completed eval run for one model.
        run_dir: The report directory created by ``build_report``.

    Returns:
        The path of the written predictions file.
    """
    # Model names can contain characters unsuitable for filenames.
    safe_name = run.model_name.replace("/", "-").replace(":", "-")
    path = run_dir / f"predictions-{safe_name}.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for prediction, gold_label in zip(
            run.predictions, run.gold_labels, strict=True
        ):
            record = asdict(prediction)
            record["gold_label"] = gold_label
            record["correct"] = prediction.predicted_label == gold_label
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the evaluation harness over a dataset for one or more models.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the evaluation dataset file.",
    )
    parser.add_argument(
        "--source",
        choices=sorted(DATASET_LOADERS.keys()),
        default="real_brief",
        help="Dataset format/loader to use (default: real_brief).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Optional split filter. real_brief records default to 'test'; "
        "casehold records carry the split assigned during construction.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated classifier model names (keys in the config's "
        "models section). Defaults to the config's classifier_model.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Evaluate at most this many examples. Useful for smoke tests.",
    )
    parser.add_argument(
        "--no-dual-eval",
        action="store_true",
        help="Skip the gold-claim (dual evaluation) pass even when gold "
        "claims are present.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the pipeline cache. NOTE: classification results are "
        "cached by (claim, chunks, model, prompt version), so cached runs "
        "are cheap to repeat; disable only if you suspect stale results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file. Default: config.yaml.",
    )
    args = parser.parse_args()

    # ---- Config and secrets ----

    config = load_config(args.config)
    configure_logging(level=config.log_level)

    load_dotenv()
    secrets = load_secrets()

    # ---- Resolve model list ----

    model_names = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else [config.classifier_model]
    )

    unknown = [m for m in model_names if m not in config.models]
    if unknown:
        print(
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(config.models.keys()))}",
            file=sys.stderr,
        )
        sys.exit(2)

    # ---- Load dataset ----

    examples = _load_dataset(args.dataset, args.source, args.split, args.max_examples)
    if not examples:
        print(f"No examples loaded from {args.dataset}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loaded {len(examples)} examples from {args.dataset} "
        f"(source={args.source}, split={args.split or 'all'})"
    )

    # ---- Evaluate each model ----

    eval_runs = []

    for i, model_name in enumerate(model_names):
        print(f"\n[{i + 1}/{len(model_names)}] Evaluating {model_name}...")

        pipeline = _build_pipeline(
            config=config,
            secrets=secrets,
            attribution_model_name=config.attribution_model,
            classifier_model_name=model_name,
            no_cache=args.no_cache,
        )

        try:
            run = run_evaluation(
                pipeline,
                examples,
                run_dual_eval=not args.no_dual_eval,
            )
        finally:
            pipeline._cache.close()

        eval_runs.append(run)

        print(
            f"  macro F1: {run.metrics.macro_f1:.4f} | "
            f"abstention rate: {run.metrics.abstention_rate:.2%} | "
            f"n={run.metrics.n_examples}"
        )

    # ---- Write report and prediction dumps ----

    run_dir = build_report(
        eval_runs,
        config,
        config.eval_runs_dir,
        eval_set_paths={args.source: args.dataset},
    )

    for run in eval_runs:
        pred_path = _write_predictions(run, run_dir)
        print(f"Predictions written: {pred_path}")

    print(f"\nReport written to {run_dir}")
    print(f"  Summary: {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
