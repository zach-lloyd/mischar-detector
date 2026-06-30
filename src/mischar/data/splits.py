"""
Train/val/test split logic.

Splits evaluation examples into train, val, and test sets while
respecting the following constraint:

1. **No leakage**: No (claim, case) pair appears in more than one split.
   This prevents the model from memorizing specific examples during
   training and being tested on them.

Because grouping is by citation, the accurate/mischaracterized pair
generated from the same CaseHOLD entry always lands in the same split —
the pair members share a citation.

Note: under the current design, CaseHOLD-derived examples are split into
train/val only (the default test ratio is 0). The held-out test set is
the separately-constructed, hand-annotated real-brief dataset.
"""

from __future__ import annotations

import random
from collections import defaultdict

from mischar.logging import get_logger
from mischar.types import EvalExample

log = get_logger("data.splits")

# Default proportions for random splitting.
# Train/val only — the hand-annotated real-brief set serves as the test set.
DEFAULT_TRAIN_RATIO = 0.85
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.0


def assign_splits(
    examples: list[EvalExample],
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = 42,
) -> list[EvalExample]:
    """
    Assign train/val/test splits to a list of evaluation examples.

    Groups examples by their case identity (citation_text) so that
    all examples sharing the same case are assigned to the same split.

    This function mutates the ``split`` field on each example in place
    and also returns the list for convenience.

    Args:
        examples: The examples to split. Their ``split`` fields will
            be overwritten.
        train_ratio: Target proportion of case groups assigned to train.
        val_ratio: Target proportion of case groups assigned to val.
            The remainder (currently 0 since I'm using real briefs for testing) 
            goes to test.
        seed: Random seed for reproducible random assignment.

    Returns:
        The same list of examples with ``split`` fields assigned.
    """
    # Group examples by case identity. All examples citing the same case
    # must land in the same split to prevent leakage.
    case_groups = _group_by_case(examples)

    log.info(
        "splits_start",
        n_examples=len(examples),
        n_case_groups=len(case_groups),
    )

    # Compute target sizes across all case groups so that the final
    # proportions approximate the requested ratios.
    n_total_groups = len(case_groups)
    target_train = round(n_total_groups * train_ratio)
    target_val = round(n_total_groups * val_ratio)

    rng = random.Random(seed)

    case_group_keys = list(case_groups.keys())
    rng.shuffle(case_group_keys)
    train_filled = val_filled = test_filled = 0

    for case_key in case_group_keys:
        if train_filled < target_train:
            for example in case_groups[case_key]:
                example.split = "train"
            train_filled += 1
        elif val_filled < target_val:
            for example in case_groups[case_key]:
                example.split = "val"
            val_filled += 1
        else:
            for example in case_groups[case_key]:
                example.split = "test"
            test_filled += 1

    # Log split distribution.
    split_counts = {"train": 0, "val": 0, "test": 0}
    for example in examples:
        split_counts[example.split] = split_counts.get(example.split, 0) + 1

    log.info(
        "splits_complete",
        train=split_counts["train"],
        val=split_counts["val"],
        test=split_counts["test"]
    )

    return examples


def _group_by_case(examples: list[EvalExample]) -> dict[str, list[EvalExample]]:
    """
    Group examples by case identity.

    Uses ``citation_text`` as the grouping key since volume + reporter +
    page uniquely identifies a case. All examples referencing the same
    case are grouped together so they can be assigned to the same split, avoiding
    data leakage.

    Args:
        examples: The examples to group.

    Returns:
        A dict mapping case key (citation_text) to its examples.
    """
    groups: dict[str, list[EvalExample]] = defaultdict(list)

    for example in examples:
        # Normalize the citation text for grouping. Strip whitespace
        # and lowercase to catch trivial formatting differences.
        case_key = example.citation_text.strip().lower()
        groups[case_key].append(example)

    return dict(groups)
