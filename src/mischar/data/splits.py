"""
Train/val/test split logic with temporal awareness.

Splits evaluation examples into train, val, and test sets while
respecting two constraints:

1. **No leakage**: No (claim, case) pair appears in more than one split.
   This prevents the model from memorizing specific examples during
   training and being tested on them.

2. **Temporal split** (when possible): Pre-2020 cases go to train,
   post-2020 cases go to val/test. This mitigates the risk that the
   base model memorized older case text during pretraining, which
   would inflate eval scores on those cases.

When temporal information isn't available (``decided_year`` is None),
examples are assigned randomly with a fixed seed.

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

# Year threshold for temporal splitting. Cases decided before this
# year go to train; cases decided in this year or later go to val/test.
TEMPORAL_SPLIT_YEAR = 2020

# Default proportions for random splitting (when temporal info is absent).
# Train/val only — the hand-annotated real-brief set serves as the test set.
DEFAULT_TRAIN_RATIO = 0.85
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.0


def assign_splits(
    examples: list[EvalExample],
    *,
    temporal_split_year: int = TEMPORAL_SPLIT_YEAR,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = 42,
) -> list[EvalExample]:
    """
    Assign train/val/test splits to a list of evaluation examples.

    First groups examples by their case identity (citation_text) so that
    all examples sharing the same case are assigned to the same split.
    Then applies temporal splitting where possible and random splitting
    for the remainder.

    Proportions are enforced globally across all case groups. Post-2020
    cases always go to val/test (temporal purity for eval). Pre-2020
    cases go to train up to the target ratio; if there are more pre-2020
    cases than the train target allows, excess are spilled into val/test.
    Non-temporal cases fill whatever slots remain.

    This function mutates the ``split`` field on each example in place
    and also returns the list for convenience.

    Args:
        examples: The examples to split. Their ``split`` fields will
            be overwritten.
        temporal_split_year: Year threshold. Cases decided before this
            year go to train; cases decided in or after go to val/test.
        train_ratio: Target proportion of case groups assigned to train.
        val_ratio: Target proportion of case groups assigned to val.
            The remainder goes to test.
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

    # Classify each case group as temporal-train, temporal-eval, or
    # no-temporal-info based on the decided year.
    temporal_train_keys = []
    temporal_eval_keys = []
    no_temporal_keys = []

    for case_key, group_examples in case_groups.items():
        year = _get_decided_year(group_examples)

        if year is not None:
            if year < temporal_split_year:
                temporal_train_keys.append(case_key)
            else:
                temporal_eval_keys.append(case_key)
        else:
            no_temporal_keys.append(case_key)

    # Compute target sizes across ALL case groups so that the final
    # proportions approximate the requested ratios regardless of how
    # many cases fall on each side of the temporal threshold.
    n_total_groups = len(case_groups)
    target_train = round(n_total_groups * train_ratio)
    target_val = round(n_total_groups * val_ratio)
    target_test = n_total_groups - target_train - target_val

    rng = random.Random(seed)

    # Step 1: Temporal-eval keys (post-2020) always go to val/test.
    # This is the whole point of temporal splitting — we want post-2020
    # cases in eval to mitigate pretraining memorization.
    rng.shuffle(temporal_eval_keys)
    n_temporal_eval = len(temporal_eval_keys)
    n_temporal_val = max(1, round(n_temporal_eval * 0.5)) if n_temporal_eval > 1 else 0

    for case_key in temporal_eval_keys[:n_temporal_val]:
        for example in case_groups[case_key]:
            example.split = "val"

    for case_key in temporal_eval_keys[n_temporal_val:]:
        for example in case_groups[case_key]:
            example.split = "test"

    # Track how many val/test slots the temporal-eval keys consumed.
    val_filled = n_temporal_val
    test_filled = n_temporal_eval - n_temporal_val

    # Step 2: Temporal-train keys (pre-2020) go to train — but only
    # up to the target. If there are more pre-2020 cases than the train
    # target allows, spill the excess into val/test so proportions
    # stay approximately correct. The spilled cases lose temporal purity
    # but that's an acceptable tradeoff.
    rng.shuffle(temporal_train_keys)
    temporal_for_train = temporal_train_keys[:target_train]
    temporal_spill = temporal_train_keys[target_train:]

    for case_key in temporal_for_train:
        for example in case_groups[case_key]:
            example.split = "train"

    train_filled = len(temporal_for_train)

    # Distribute spilled temporal-train keys across val/test.
    for case_key in temporal_spill:
        if val_filled < target_val:
            for example in case_groups[case_key]:
                example.split = "val"
            val_filled += 1
        else:
            for example in case_groups[case_key]:
                example.split = "test"
            test_filled += 1

    if temporal_spill:
        log.info(
            "splits_temporal_spill",
            spilled=len(temporal_spill),
            reason="pre-2020 cases exceeded train target",
        )

    # Step 3: Non-temporal keys fill whatever slots remain in
    # train, then val, then test.
    rng.shuffle(no_temporal_keys)

    for case_key in no_temporal_keys:
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
        test=split_counts["test"],
        temporal_train_cases=len(temporal_train_keys),
        temporal_eval_cases=len(temporal_eval_keys),
        no_temporal_cases=len(no_temporal_keys),
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


def _get_decided_year(examples: list[EvalExample]) -> int | None:
    """
    Extract the decided year from a group of examples for the same case.

    Looks in each example's metadata for a ``decided_year`` field.
    Returns the first non-None value found, since all examples in the
    group reference the same case.

    Args:
        examples: Examples from the same case group.

    Returns:
        The decision year as an int, or None if not available.
    """
    for example in examples:
        year = example.metadata.get("decided_year")
        if year is not None:
            try:
                return int(year)
            except (ValueError, TypeError):
                continue

    return None
