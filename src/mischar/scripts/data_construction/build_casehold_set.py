"""
Build the CaseHOLD evaluation set.

Downloads the CaseHOLD dataset from HuggingFace and converts it into
standardized JSONL that ``datasets.load_casehold()`` can read. Each
CaseHOLD example is a genuine "entails" pair — the holding accurately
describes the cited case.

The script:
1. Loads the dataset from HuggingFace (``casehold/casehold``).
2. For each example, extracts the correct holding and keeps it in its
   original parenthetical style ("holding that X") as the claim.
3. Reconstructs a natural passage by replacing the ``<HOLDING>`` placeholder
   with the holding in parenthetical form.
4. Runs eyecite on the reconstructed passage to extract the citation
   immediately preceding the holding — the same parser the pipeline uses,
   ensuring format consistency.
5. Filters out examples where eyecite can't parse the target citation.
6. Writes the output as JSONL.

Usage:
    python -m mischar.scripts.data_construction.build_casehold_set \\
        --output data/processed/casehold.jsonl \\
        [--split train] \\
        [--max-examples 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eyecite import get_citations
from eyecite.models import FullCaseCitation


# ---------------------------------------------------------------------------
# Passage reconstruction
# ---------------------------------------------------------------------------

# The <HOLDING> placeholder in citing_prompt, with its surrounding
# parentheses. CaseHOLD format is: "Case Name, Reporter (<HOLDING>)"
_HOLDING_PLACEHOLDER = "(<HOLDING>)"


def _reconstruct_passage(citing_prompt: str, holding: str) -> str:
    """
    Replace the ``<HOLDING>`` placeholder with the actual holding text.

    The holding is inserted in parenthetical form to match how it would
    appear in a real brief: ``(holding that X)``.

    Args:
        citing_prompt: The CaseHOLD citing prompt with ``<HOLDING>``
            placeholder.
        holding: The correct holding text.

    Returns:
        The passage with the holding inserted.
    """
    return citing_prompt.replace(_HOLDING_PLACEHOLDER, f"({holding})")


# ---------------------------------------------------------------------------
# Citation extraction via eyecite
# ---------------------------------------------------------------------------


def _find_target_citation(
    passage: str,
    holding_position: int,
) -> str | None:
    """
    Find the citation immediately preceding the holding in the passage.

    Runs eyecite on the full passage and returns the full case citation
    whose end position is closest to (and before) the holding insertion
    point. This is the citation that the holding characterizes.

    Args:
        passage: The reconstructed passage with the holding inserted.
        holding_position: The character offset where ``<HOLDING>`` was
            in the original citing_prompt.

    Returns:
        The citation's matched text, or None if eyecite can't find a
        full case citation before the holding position.
    """
    raw_citations = get_citations(passage)
    full_citations = [c for c in raw_citations if isinstance(c, FullCaseCitation)]

    if not full_citations:
        return None

    # Find the citation closest to (and before) the holding position.
    # The target citation appears immediately before the parenthetical.
    best = None
    best_distance = float("inf")

    for cite in full_citations:
        try:
            span = cite.span()
            end = span[1]
        except (AttributeError, TypeError):
            continue

        # Citation must end before or near the holding position.
        distance = holding_position - end

        if 0 <= distance < best_distance:
            best = cite
            best_distance = distance

    if best is None:
        return None

    return best.matched_text()


# ---------------------------------------------------------------------------
# Single example processing
# ---------------------------------------------------------------------------


def _process_example(row: dict) -> dict | None:
    """
    Process a single CaseHOLD example into the standardized format.

    Args:
        row: A dict from the HuggingFace dataset with fields:
            ``example_id``, ``citing_prompt``, ``holding_0`` through
            ``holding_4``, and ``label``.

    Returns:
        A dict ready for JSONL output with fields: ``example_id``,
        ``passage``, ``citation_text``, ``label``, ``metadata``.
        Returns None if the target citation can't be parsed.
    """
    example_id = row["example_id"]
    citing_prompt = row["citing_prompt"]
    label_idx = row["label"]
    correct_holding = row[f"holding_{label_idx}"]

    # Find where <HOLDING> is so we can identify the target citation.
    holding_offset = citing_prompt.find("<HOLDING>")
    if holding_offset == -1:
        return None

    # Reconstruct the passage with the holding inserted.
    passage = _reconstruct_passage(citing_prompt, correct_holding)

    # Extract the target citation using eyecite.
    citation_text = _find_target_citation(passage, holding_offset)
    if not citation_text:
        return None

    return {
        "example_id": str(example_id),
        "passage": passage,
        "citation_text": citation_text,
        "label": "entails",
        "metadata": {
            "source_dataset": "casehold",
            "claim": correct_holding,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CaseHOLD evaluation set from HuggingFace.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/casehold.jsonl"),
        help="Output JSONL file path (default: data/processed/casehold.jsonl).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="HuggingFace dataset split to use (e.g., 'train', 'test'). "
        "If not specified, uses all splits.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process. Useful for testing "
        "or when you don't need the full 53K dataset.",
    )
    args = parser.parse_args()

    # ---- Load from HuggingFace ----

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "The 'datasets' library is required. Install it with: "
            "pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading CaseHOLD from HuggingFace...")

    if args.split:
        # Include revision="refs/convert/parquet" to load from the auto-converted
        # parquet branch. This bypasses the loading script and avoids errors raised
        # by the datasets library.
        dataset = load_dataset(
            "casehold/casehold", split=args.split, revision="refs/convert/parquet"
        )
        rows = list(dataset)
    else:
        dataset = load_dataset("casehold/casehold", revision="refs/convert/parquet")
        rows = []
        for split_name in dataset:
            rows.extend(dataset[split_name])

    print(f"Loaded {len(rows)} examples from CaseHOLD.")

    if args.max_examples:
        rows = rows[: args.max_examples]
        print(f"Limiting to {len(rows)} examples.")

    # ---- Process examples ----

    output_records = []
    skipped = 0

    for i, row in enumerate(rows):
        result = _process_example(row)

        if result is None:
            skipped += 1
            continue

        output_records.append(result)

        if (i + 1) % 5000 == 0:
            print(
                f"  Processed {i + 1}/{len(rows)} "
                f"({len(output_records)} kept, {skipped} skipped)"
            )

    print(
        f"Processing complete: {len(output_records)} examples kept, "
        f"{skipped} skipped (eyecite couldn't parse target citation)."
    )

    # ---- Write output ----

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_records)} examples to {args.output}")


if __name__ == "__main__":
    main()
