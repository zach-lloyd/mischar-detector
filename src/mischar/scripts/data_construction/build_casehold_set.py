"""
Build the CaseHOLD accurate/mischaracterized example set.

Downloads the CaseHOLD dataset from HuggingFace and converts each entry
into a PAIR of standardized JSONL records:

1. An **accurate** example — the passage with the correct holding inserted.
2. A **mischaracterized** example — the same passage with one of the
   entry's incorrect holding choices inserted instead.

CaseHOLD is a multiple-choice dataset: each entry has five candidate
holdings (``holding_0`` ... ``holding_4``) and a label indicating which
one is correct. The four incorrect holdings are plausible-but-wrong
characterizations of the cited case, so they serve as natural
mischaracterization examples without any LLM generation. One incorrect
holding is chosen per entry with a seeded RNG for reproducibility.

The script:
1. Loads the dataset from HuggingFace (``casehold/casehold``).
2. For each entry, reconstructs two passages by replacing the
   ``<HOLDING>`` placeholder with the correct and an incorrect holding.
3. Runs eyecite on the reconstructed passage to extract the citation
   immediately preceding the holding — the same parser the pipeline uses,
   ensuring format consistency.
4. Filters out entries where eyecite can't parse the target citation
   (both pair members are dropped together).
5. Writes the output as JSONL, two records per kept entry.

Usage:
    python -m mischar.scripts.data_construction.build_casehold_set \\
        --output data/processed/casehold.jsonl \\
        [--split train] \\
        [--max-entries 5000] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
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

# Number of candidate holdings per CaseHOLD entry.
_NUM_HOLDINGS = 5


def _reconstruct_passage(citing_prompt: str, holding: str) -> str:
    """
    Replace the ``<HOLDING>`` placeholder with the actual holding text.

    The holding is inserted in parenthetical form to match how it would
    appear in a real brief: ``(holding that X)``.

    Args:
        citing_prompt: The CaseHOLD citing prompt with ``<HOLDING>``
            placeholder.
        holding: The holding text to insert (correct or incorrect).

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
# Single entry processing
# ---------------------------------------------------------------------------


def _process_entry(row: dict, rng: random.Random) -> list[dict] | None:
    """
    Process a single CaseHOLD entry into an accurate/mischaracterized pair.

    Both pair members share the same citation and source entry; they
    differ only in which holding is inserted into the passage and in
    their label. The incorrect holding is picked at random (seeded)
    from the entry's four wrong choices.

    Args:
        row: A dict from the HuggingFace dataset with fields:
            ``example_id``, ``citing_prompt``, ``holding_0`` through
            ``holding_4``, and ``label``.
        rng: Seeded RNG used to pick which incorrect holding to use.

    Returns:
        A list of two dicts ready for JSONL output (accurate first,
        mischaracterized second), each with fields: ``example_id``,
        ``passage``, ``citation_text``, ``label``, ``metadata``.
        Returns None if the target citation can't be parsed — in that
        case the whole entry is dropped so pairs stay balanced.
    """
    example_id = row["example_id"]
    citing_prompt = row["citing_prompt"]
    correct_idx = int(row["label"])
    correct_holding = row[f"holding_{correct_idx}"]

    # Pick one of the four incorrect holdings at random.
    wrong_indices = [i for i in range(_NUM_HOLDINGS) if i != correct_idx]
    wrong_idx = rng.choice(wrong_indices)
    wrong_holding = row[f"holding_{wrong_idx}"]

    if not correct_holding or not wrong_holding:
        return None

    # Find where <HOLDING> is so we can identify the target citation.
    holding_offset = citing_prompt.find("<HOLDING>")
    if holding_offset == -1:
        return None

    # Reconstruct both passages.
    accurate_passage = _reconstruct_passage(citing_prompt, correct_holding)
    mischar_passage = _reconstruct_passage(citing_prompt, wrong_holding)

    # Extract the target citation. The citation is identical in both
    # passages (only the parenthetical differs), so parsing the accurate
    # passage is sufficient.
    citation_text = _find_target_citation(accurate_passage, holding_offset)
    if not citation_text:
        return None

    accurate_record = {
        "example_id": f"{example_id}-acc",
        "passage": accurate_passage,
        "citation_text": citation_text,
        "label": "accurate",
        "metadata": {
            "source_dataset": "casehold",
            "source_entry_id": str(example_id),
            "claim": correct_holding,
        },
    }

    mischar_record = {
        "example_id": f"{example_id}-mis",
        "passage": mischar_passage,
        "citation_text": citation_text,
        "label": "mischaracterized",
        "metadata": {
            "source_dataset": "casehold",
            "source_entry_id": str(example_id),
            "claim": wrong_holding,
            "correct_claim": correct_holding,
            "wrong_holding_index": wrong_idx,
        },
    }

    return [accurate_record, mischar_record]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the CaseHOLD accurate/mischaracterized pair set "
            "from HuggingFace."
        ),
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
        "--max-entries",
        type=int,
        default=5000,
        help="Maximum number of CaseHOLD entries to process. Each kept "
        "entry produces TWO examples (one accurate, one mischaracterized). "
        "Default: 5000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for choosing which incorrect holding to use "
        "per entry (default: 42).",
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

    print(f"Loaded {len(rows)} entries from CaseHOLD.")

    if args.max_entries:
        rows = rows[: args.max_entries]
        print(f"Limiting to {len(rows)} entries.")

    # ---- Process entries ----

    rng = random.Random(args.seed)
    output_records = []
    skipped = 0

    for i, row in enumerate(rows):
        pair = _process_entry(row, rng)

        if pair is None:
            skipped += 1
            continue

        output_records.extend(pair)

        if (i + 1) % 1000 == 0:
            print(
                f"  Processed {i + 1}/{len(rows)} entries "
                f"({len(output_records)} examples kept, {skipped} entries skipped)"
            )

    print(
        f"Processing complete: {len(output_records)} examples "
        f"({len(output_records) // 2} pairs) kept, "
        f"{skipped} entries skipped (eyecite couldn't parse target citation)."
    )

    # ---- Write output ----

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_records)} examples to {args.output}")


if __name__ == "__main__":
    main()
