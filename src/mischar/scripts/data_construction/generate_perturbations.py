"""
Generate perturbed (mischaracterized) examples from CaseHOLD source pairs.

Takes genuine "entails" examples from the CaseHOLD JSONL and applies
perturbation types from ``data.perturbation`` to produce synthetic
mischaracterizations across the three non-entails labels: partially_supports,
unrelated, and contradicts.

This prototype version:
- Uses only perturbation types that don't require case text (no
  CourtListener resolution needed): P3, U2, C1, C2, C3.
- Calls Gemma 27B via Ollama for all perturbations.
- Skips the spot-check gate (manual review after generation instead).

Usage:
    python -m mischar.scripts.data_construction.generate_perturbations \\
        --source data/processed/casehold-sample.jsonl \\
        --output data/processed/perturbations-sample.jsonl \\
        [--num-examples 10] \\
        [--ollama-model gemma3:27b]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from mischar.data.perturbation import (
    PerturbationType,
    SourceExample,
    generate_perturbation,
)
from mischar.models.ollama import OllamaClient


# ---------------------------------------------------------------------------
# Perturbation types that don't require case text, grouped by target label.
# ---------------------------------------------------------------------------

PROTOTYPE_TYPES_BY_LABEL: dict[str, list[PerturbationType]] = {
    "partially_supports": [
        PerturbationType.P3_INFLATE_STRENGTH,
    ],
    "unrelated": [
        PerturbationType.U2_OFF_TOPIC,
    ],
    "contradicts": [
        PerturbationType.C1_NEGATE,
        PerturbationType.C2_REVERSE_WINNER,
        PerturbationType.C3_OPPOSITE_FACTORS,
    ],
}


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _load_source_examples(path: Path) -> list[SourceExample]:
    """
    Load CaseHOLD JSONL and convert to SourceExample objects.

    The claim comes from ``metadata.claim`` and the case name is
    extracted from the citation context in the passage. Case text is
    left empty since we're using perturbation types that don't need it.

    Args:
        path: Path to the CaseHOLD JSONL file.

    Returns:
        A list of SourceExample objects.
    """
    examples = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            examples.append(
                SourceExample(
                    example_id=record["example_id"],
                    claim=record["metadata"]["claim"],
                    case_name="",  # Not available in CaseHOLD output.
                    case_text="",  # Deferred — not needed for prototype types.
                    citation_text=record["citation_text"],
                )
            )

    return examples


# ---------------------------------------------------------------------------
# Assignment — which source gets which perturbation type
# ---------------------------------------------------------------------------


def _assign_perturbations(
    sources: list[SourceExample],
    num_examples: int,
) -> list[tuple[SourceExample, PerturbationType]]:
    """
    Assign perturbation types to source examples.

    Distributes the requested number of examples as evenly as possible
    across partially_supports, unrelated, and contradicts. Within each
    label, cycles through available perturbation types.

    Args:
        sources: Available source examples.
        num_examples: Total number of perturbations to generate.

    Returns:
        A list of (source, perturbation_type) pairs.
    """
    labels = list(PROTOTYPE_TYPES_BY_LABEL.keys())
    per_label = num_examples // len(labels)
    remainder = num_examples % len(labels)

    assignments = []
    # Cycle through source examples so if more perturbations are requested than
    # sources, will repeat sources as needed.
    source_cycle = itertools.cycle(sources)

    for i, label in enumerate(labels):
        count = per_label + (1 if i < remainder else 0)
        types = PROTOTYPE_TYPES_BY_LABEL[label]
        # Cycle through type examples. Really only matters for "contradicts"
        # because "partially supports" and "unrelated" only have one type each.
        type_cycle = itertools.cycle(types)

        for _ in range(count):
            source = next(source_cycle)
            ptype = next(type_cycle)
            assignments.append((source, ptype))

    return assignments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate perturbed examples from CaseHOLD source pairs.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to CaseHOLD JSONL file (output of build_casehold_set.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/perturbations-sample.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of perturbations to generate (default: 10).",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="gemma3:27b",
        help="Ollama model tag to use (default: gemma3:27b).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    args = parser.parse_args()

    # ---- Load source examples ----

    sources = _load_source_examples(args.source)
    if not sources:
        print(f"No source examples found in {args.source}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(sources)} source examples from {args.source}")

    if len(sources) < args.num_examples:
        print(
            f"Warning: only {len(sources)} source examples available "
            f"for {args.num_examples} perturbations. Sources will be reused."
        )

    # ---- Set up assignments ----

    assignments = _assign_perturbations(sources, args.num_examples)

    print(f"Planned {len(assignments)} perturbations:")
    label_counts: dict[str, int] = {}
    for _, ptype in assignments:
        from mischar.data.perturbation import PERTURBATION_TARGET_LABELS

        label = PERTURBATION_TARGET_LABELS[ptype]
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # ---- Initialize Ollama client ----

    print(f"Connecting to Ollama ({args.ollama_model})...")
    client = OllamaClient(
        model_name=args.ollama_model,
        base_url=args.ollama_url,
        timeout_seconds=180,
    )

    # ---- Generate perturbations ----

    results = []
    failed = 0

    for i, (source, ptype) in enumerate(assignments):
        print(
            f"  [{i + 1}/{len(assignments)}] "
            f"{ptype.name} on example {source.example_id}...",
            end=" ",
            flush=True,
        )

        result = generate_perturbation(source, ptype, client)

        if result is None:
            print("FAILED")
            failed += 1
            continue

        print(f"OK ({result.target_label})")
        results.append(result)

    print(
        f"\nGeneration complete: {len(results)} succeeded, {failed} failed."
    )

    # ---- Write output ----

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for result in results:
            record = {
                "example_id": result.example_id,
                "source_example_id": result.source_example_id,
                "perturbation_type": result.perturbation_type,
                "target_label": result.target_label,
                "original_claim": result.original_claim,
                "perturbed_claim": result.perturbed_claim,
                "case_name": result.case_name,
                "citation_text": result.citation_text,
                "passage": result.passage,
                "metadata": result.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} perturbations to {args.output}")

    client.close()


if __name__ == "__main__":
    main()
