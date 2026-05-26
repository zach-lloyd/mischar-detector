"""
Evaluation report generation.

Writes three output files to a timestamped subdirectory of ``eval_runs/``:

- ``summary.md`` — human-readable narrative with headline numbers,
  per-label F1 tables, and abstention notes.
- ``metrics.json`` — machine-readable nested JSON with full metrics
  for every (source, model) combination.
- ``run_manifest.json`` — reproducibility record with config snapshot,
  model info, dependency versions, and eval set hashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from mischar.config import Config
from mischar.constants import Label
from mischar.eval.harness import EvalRun
from mischar.logging import get_logger
from mischar.types import MetricsBundle

log = get_logger("eval.report")


def build_report(
    eval_runs: list[EvalRun],
    config: Config,
    output_dir: Path,
    *,
    eval_set_paths: dict[str, Path] | None = None,
) -> Path:
    """
    Write evaluation report files to a timestamped output directory.

    Creates a new subdirectory under ``output_dir`` named with the current
    timestamp, then writes ``summary.md``, ``metrics.json``, and
    ``run_manifest.json`` into it.

    Args:
        eval_runs: Results from one or more ``run_evaluation`` calls,
            covering different (source, model) combinations.
        config: The application config used during the eval run.
        output_dir: Parent directory for eval runs (typically
            ``config.eval_runs_dir``).
        eval_set_paths: Optional mapping from source name to the path
            of the eval dataset file, used to compute content hashes
            for the run manifest.

    Returns:
        The path to the created report directory.
    """
    # Create timestamped output directory.
    run_id = str(uuid4())
    timestamp = datetime.now(UTC).isoformat()
    run_dir_name = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("report_writing", run_dir=str(run_dir), n_runs=len(eval_runs))

    # Organize runs by source and model for the nested report structure.
    organized = _organize_runs(eval_runs)

    # Write the three report files.
    _write_metrics_json(organized, run_id, timestamp, run_dir)
    _write_summary_md(organized, run_id, timestamp, run_dir)
    _write_run_manifest(
        config, run_id, timestamp, eval_runs, run_dir,
        eval_set_paths=eval_set_paths,
    )

    log.info("report_complete", run_dir=str(run_dir))

    return run_dir


# ---------------------------------------------------------------------------
# metrics.json
# ---------------------------------------------------------------------------


def _write_metrics_json(
    organized: dict[str, dict[str, EvalRun]],
    run_id: str,
    timestamp: str,
    run_dir: Path,
) -> None:
    """
    Write the structured metrics JSON file.

    Schema: top-level metadata plus a nested ``results[source][model]`` structure 
    containing the full ``MetricsBundle`` for each combination.

    Args:
        organized: Eval runs organized as ``{source: {model: EvalRun}}``.
        run_id: Unique identifier for this eval run.
        timestamp: ISO-format timestamp.
        run_dir: Directory to write into.
    """
    sources = sorted(organized.keys())
    models = sorted({
        model for source_runs in organized.values() for model in source_runs
    })

    results = {}
    for source in sources:
        results[source] = {}
        for model, run in organized.get(source, {}).items():
            entry: dict[str, Any] = {
                "metrics": _metrics_bundle_to_dict(run.metrics),
            }

            # Include dual eval results for real_brief source.
            if run.dual_eval is not None:
                entry["dual_eval"] = {
                    "gold_claim": _metrics_bundle_to_dict(
                        run.dual_eval.gold_claim_metrics
                    ),
                    "attributed_claim": _metrics_bundle_to_dict(
                        run.dual_eval.attributed_claim_metrics
                    ),
                }

            results[source][model] = entry

    output = {
        "run_id": run_id,
        "timestamp": timestamp,
        "models": models,
        "sources": sources,
        "results": results,
    }

    path = run_dir / "metrics.json"
    path.write_text(json.dumps(output, indent=2, default=str))

    log.debug("report_metrics_json_written", path=str(path))


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------


def _write_summary_md(
    organized: dict[str, dict[str, EvalRun]],
    run_id: str,
    timestamp: str,
    run_dir: Path,
) -> None:
    """
    Write the human-readable summary markdown file.

    Includes run metadata, headline macro F1 numbers, per-label F1
    tables, and abstention rate notes.

    Args:
        organized: Eval runs organized as ``{source: {model: EvalRun}}``.
        run_id: Unique identifier for this eval run.
        timestamp: ISO-format timestamp.
        run_dir: Directory to write into.
    """
    lines = []

    # Header.
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append(f"**Run ID:** {run_id}")
    lines.append(f"**Timestamp:** {timestamp}")

    git_sha = _get_git_sha()
    if git_sha:
        lines.append(f"**Git SHA:** {git_sha}")

    sources = sorted(organized.keys())
    models = sorted({
        model for source_runs in organized.values() for model in source_runs
    })
    lines.append(f"**Models evaluated:** {', '.join(models)}")
    lines.append(f"**Sources:** {', '.join(sources)}")
    lines.append("")

    # Headline numbers: macro F1 table.
    lines.append("## Macro F1 by Source and Model")
    lines.append("")
    lines.append(_build_headline_table(organized, sources, models))
    lines.append("")

    # Per-source detail sections.
    for source in sources:
        lines.append(f"## {source}")
        lines.append("")

        for model, run in sorted(organized.get(source, {}).items()):
            lines.append(f"### {model}")
            lines.append("")
            lines.append(f"- **Macro F1:** {run.metrics.macro_f1:.4f}")
            lines.append(f"- **Examples:** {run.metrics.n_examples}")
            lines.append(f"- **Abstention rate:** {run.metrics.abstention_rate:.2%}")

            if run.metrics.abstention_by_reason:
                reason_parts = [
                    f"{reason}: {count}"
                    for reason, count in sorted(run.metrics.abstention_by_reason.items())
                ]
                lines.append(f"- **Abstentions by reason:** {', '.join(reason_parts)}")

            lines.append("")

            # Per-label F1 table.
            lines.append(_build_per_label_table(run.metrics))
            lines.append("")

            # Dual eval results if present.
            if run.dual_eval is not None:
                lines.append("#### Dual Evaluation (Attribution Error Decomposition)")
                lines.append("")
                lines.append(
                    f"- **Gold claim macro F1:** "
                    f"{run.dual_eval.gold_claim_metrics.macro_f1:.4f}"
                )
                lines.append(
                    f"- **Attributed claim macro F1:** "
                    f"{run.dual_eval.attributed_claim_metrics.macro_f1:.4f}"
                )
                gap = (
                    run.dual_eval.gold_claim_metrics.macro_f1
                    - run.dual_eval.attributed_claim_metrics.macro_f1
                )
                lines.append(
                    f"- **Attribution gap:** {gap:+.4f} "
                    f"(positive = attribution errors are hurting performance)"
                )
                lines.append("")

    # Footer.
    lines.append("---")
    lines.append("")
    lines.append("Full metrics available in `metrics.json`. Reproducibility record in `run_manifest.json`.")

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines))

    log.debug("report_summary_md_written", path=str(path))


def _build_headline_table(
    organized: dict[str, dict[str, EvalRun]],
    sources: list[str],
    models: list[str],
) -> str:
    """
    Build a markdown table of macro F1 scores by source and model.

    Args:
        organized: Eval runs organized as ``{source: {model: EvalRun}}``.
        sources: Sorted list of source names.
        models: Sorted list of model names.

    Returns:
        A markdown table string.
    """
    # Header row.
    header = "| Source | " + " | ".join(models) + " |"
    separator = "|---" + "|---" * len(models) + "|"

    rows = [header, separator]
    for source in sources:
        cells = [source]
        for model in models:
            run = organized.get(source, {}).get(model)
            if run:
                cells.append(f"{run.metrics.macro_f1:.4f}")
            else:
                cells.append("—")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def _build_per_label_table(metrics: MetricsBundle) -> str:
    """
    Build a markdown table of per-label precision/recall/F1.

    Args:
        metrics: The metrics bundle to render.

    Returns:
        A markdown table string.
    """
    header = "| Label | Precision | Recall | F1 | Support |"
    separator = "|---|---|---|---|---|"
    rows = [header, separator]

    for label in Label.values():
        lm = metrics.per_label.get(label)
        if lm:
            rows.append(
                f"| {label} | {lm.precision:.4f} | {lm.recall:.4f} | "
                f"{lm.f1:.4f} | {lm.support} |"
            )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# run_manifest.json
# ---------------------------------------------------------------------------


def _write_run_manifest(
    config: Config,
    run_id: str,
    timestamp: str,
    eval_runs: list[EvalRun],
    run_dir: Path,
    *,
    eval_set_paths: dict[str, Path] | None = None,
) -> None:
    """
    Write the reproducibility manifest.

    Records everything needed to reproduce this eval run: config snapshot,
    model identifiers, prompt versions, Python dependency versions, and
    content hashes of the evaluation datasets.

    Args:
        config: The application config.
        run_id: Unique identifier for this eval run.
        timestamp: ISO-format timestamp.
        eval_runs: The eval runs (used to extract model names).
        run_dir: Directory to write into.
        eval_set_paths: Optional mapping from source name to eval dataset
            file path for content hashing.
    """
    # Config snapshot — serialize the pydantic model to a dict.
    config_snapshot = json.loads(config.model_dump_json())

    # Model info from the config's model definitions.
    models_info = {}
    for name, model_config in config.models.items():
        models_info[name] = json.loads(model_config.model_dump_json())

    # Prompt versions.
    prompt_versions = {
        "attribution": config.attribution_prompt_version,
        "classification": config.classification_prompt_version,
    }

    # Python dependencies.
    dependencies = _get_pip_freeze()

    # Eval set content hashes.
    eval_set_hashes = {}
    if eval_set_paths:
        for source, path in eval_set_paths.items():
            if path.exists():
                eval_set_hashes[source] = _file_sha256(path)

    # Git SHA.
    git_sha = _get_git_sha()

    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "config_snapshot": config_snapshot,
        "models": models_info,
        "embeddings": {
            "model": config.embedding_model,
        },
        "prompt_versions": prompt_versions,
        "dependencies": dependencies,
        "eval_set_hashes": eval_set_hashes,
    }

    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))

    log.debug("report_manifest_written", path=str(path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _organize_runs(
    eval_runs: list[EvalRun],
) -> dict[str, dict[str, EvalRun]]:
    """
    Organize eval runs into a nested dict by source and model.

    Args:
        eval_runs: Flat list of eval run results.

    Returns:
        A dict of ``{source: {model_name: EvalRun}}``.
    """
    organized: dict[str, dict[str, EvalRun]] = {}
    for run in eval_runs:
        if run.source not in organized:
            organized[run.source] = {}
        organized[run.source][run.model_name] = run

    return organized


def _metrics_bundle_to_dict(bundle: MetricsBundle) -> dict[str, Any]:
    """
    Convert a MetricsBundle to a JSON-serializable dict.

    The confusion matrix uses tuple keys ``(true, pred)`` which aren't
    valid JSON keys, so we convert them to ``"true->pred"`` strings.

    Args:
        bundle: The metrics bundle to serialize.

    Returns:
        A JSON-serializable dict.
    """
    d = asdict(bundle)

    # Convert tuple keys in confusion_matrix to strings.
    if "confusion_matrix" in d:
        cm = d["confusion_matrix"]
        d["confusion_matrix"] = {
            f"{true_label}->{pred_label}": count
            for (true_label, pred_label), count in cm.items()
        }

    return d


def _get_git_sha() -> str | None:
    """
    Get the current git commit SHA, or None if not in a git repo.

    Args:
        None.

    Returns:
        The short git SHA string, or None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            return result.stdout.strip()

        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _get_pip_freeze() -> dict[str, str]:
    """
    Get installed Python package versions via pip freeze.

    Args:
        None.

    Returns:
        A dict mapping package names to version strings.
    """
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {}

        deps = {}
        for line in result.stdout.strip().splitlines():
            if "==" in line:
                name, version = line.split("==", 1)
                deps[name.strip()] = version.strip()

        return deps
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def _file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file to hash.

    Returns:
        The 64-character hex digest string.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()
