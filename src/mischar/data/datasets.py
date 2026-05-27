"""
Dataset loaders for each evaluation source.

Each loader reads its source-specific format and converts records into
``EvalExample`` objects with a standardized shape that the evaluation
harness can process uniformly. The loaders handle format differences
across sources so the rest of the pipeline doesn't have to.

Supported sources:
- **CaseHOLD**: CaseHOLD-derived (claim, case) pairs with labels.
- **perturbation**: Synthetically perturbed examples from the generator.
- **hou**: Hou et al. published dataset.
- **charlotin**: Charlotin adversarial database.
- **real_brief**: Hand-annotated real brief citations (our JSONL format).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mischar.constants import Label
from mischar.logging import get_logger
from mischar.types import EvalExample

log = get_logger("data.datasets")


def load_casehold(path: Path, split: str | None = None) -> list[EvalExample]:
    """
    Load CaseHOLD-derived evaluation examples.

    Expects a JSONL file where each line is a JSON object with fields:
    ``example_id``, ``passage``, ``citation_text``, ``label``, ``split``,
    and optionally ``metadata``.

    CaseHOLD examples are genuine entails pairs — the citation accurately
    characterizes the case. They're used alongside perturbation examples
    to test the pipeline's ability to distinguish accurate from inaccurate
    characterizations.

    Args:
        path: Path to the CaseHOLD JSONL file.
        split: If provided, filter to only examples with this split
            value ("train", "val", or "test").

    Returns:
        A list of ``EvalExample`` objects.
    """
    examples = _load_jsonl(path, source="casehold")

    if split:
        examples = [ex for ex in examples if ex.split == split]

    log.info("dataset_loaded", source="casehold", count=len(examples), split=split)

    return examples


def load_perturbations(path: Path, split: str | None = None) -> list[EvalExample]:
    """
    Load synthetically perturbed evaluation examples.

    Expects a JSONL file where each line is a ``PerturbedExample``
    serialized as JSON with fields: ``example_id``, ``passage``,
    ``citation_text``, ``target_label`` (used as gold_label),
    ``perturbed_claim``, ``perturbation_type``, ``split``, and
    optionally other metadata.

    Args:
        path: Path to the perturbations JSONL file.
        split: If provided, filter to only examples with this split value.

    Returns:
        A list of ``EvalExample`` objects.
    """
    raw_records = _read_jsonl(path)
    examples = []

    for record in raw_records:
        example = EvalExample(
            example_id=record["example_id"],
            source="perturbation",
            passage=record["passage"],
            citation_text=record["citation_text"],
            gold_label=record["target_label"],
            gold_claim=None,
            split=record.get("split", "test"),
            metadata={
                "perturbation_type": record.get("perturbation_type"),
                "original_claim": record.get("original_claim"),
                "perturbed_claim": record.get("perturbed_claim"),
                "source_example_id": record.get("source_example_id"),
            },
        )
        examples.append(example)

    if split:
        examples = [ex for ex in examples if ex.split == split]

    log.info("dataset_loaded", source="perturbation", count=len(examples), split=split)

    return examples


def load_hou(path: Path, split: str | None = None) -> list[EvalExample]:
    """
    Load examples from the Hou et al. dataset.

    Expects a JSONL file with fields: ``example_id`` (or ``id``),
    ``passage`` (or ``text``), ``citation_text`` (or ``citation``),
    ``label``, and optionally ``split`` and ``metadata``.

    The Hou dataset's label scheme may differ from ours. This loader
    expects labels to already be mapped to our four-label scheme
    during the preprocessing step
    (``scripts/data_construction/build_hou_set.py``).

    Args:
        path: Path to the preprocessed Hou JSONL file.
        split: If provided, filter to only examples with this split value.

    Returns:
        A list of ``EvalExample`` objects.
    """
    raw_records = _read_jsonl(path)
    examples = []

    for record in raw_records:
        # Handle alternative field names from the raw Hou format.
        example = EvalExample(
            example_id=str(record.get("example_id") or record.get("id", "")),
            source="hou",
            passage=record.get("passage") or record.get("text", ""),
            citation_text=record.get("citation_text") or record.get("citation", ""),
            gold_label=record["label"],
            gold_claim=record.get("gold_claim"),
            split=record.get("split", "test"),
            metadata=record.get("metadata", {}),
        )
        examples.append(example)

    if split:
        examples = [ex for ex in examples if ex.split == split]

    log.info("dataset_loaded", source="hou", count=len(examples), split=split)

    return examples


def load_charlotin(path: Path, split: str | None = None) -> list[EvalExample]:
    """
    Load examples from the Charlotin adversarial database.

    Expects a CSV file with columns that include at minimum: a passage
    or text field, a citation field, and a label field. Column names
    are mapped during preprocessing
    (``scripts/data_construction/build_charlotin_set.py``), so this
    loader expects a preprocessed JSONL file with our standard field
    names.

    The Charlotin dataset is reported separately in evaluation results
    as a real-world adversarial stress test.

    Args:
        path: Path to the preprocessed Charlotin JSONL file.
        split: If provided, filter to only examples with this split value.

    Returns:
        A list of ``EvalExample`` objects.
    """
    examples = _load_jsonl(path, source="charlotin")

    if split:
        examples = [ex for ex in examples if ex.split == split]

    log.info("dataset_loaded", source="charlotin", count=len(examples), split=split)

    return examples


def load_real_brief(path: Path, split: str | None = None) -> list[EvalExample]:
    """
    Load hand-annotated real brief examples.

    Reads the annotation JSONL format (see blueprint section 4.3) and
    converts to ``EvalExample``. These are the only examples with
    ``gold_claim`` set, enabling the dual evaluation flow in the harness.

    Args:
        path: Path to the real_briefs.jsonl annotation file.
        split: If provided, filter to only examples with this split value.

    Returns:
        A list of ``EvalExample`` objects with ``gold_claim`` populated.
    """
    raw_records = _read_jsonl(path)
    examples = []

    for record in raw_records:
        example = EvalExample(
            example_id=record["annotation_id"],
            source="real_brief",
            passage=record["passage"],
            citation_text=record["citation_text_in_passage"],
            gold_label=record["label"],
            gold_claim=record.get("gold_claim"),
            split=record.get("split", "test"),
            metadata={
                "annotator": record.get("annotator"),
                "annotator_notes": record.get("annotator_notes"),
                "boundary_case": record.get("boundary_case", False),
                "cited_case": record.get("cited_case", {}),
                "source_info": record.get("source", {}),
            },
        )
        examples.append(example)

    if split:
        examples = [ex for ex in examples if ex.split == split]

    log.info("dataset_loaded", source="real_brief", count=len(examples), split=split)

    return examples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path, source: str) -> list[EvalExample]:
    """
    CaseHOLD and Charlotin use the same field names, so we define a helper function
    to create an EvalExample using those field names and then call it in load_casehold
    and load_charlotin to avoid duplicating the code.

    Args:
        path: Path to the JSONL file.
        source: The source name to set on each example.

    Returns:
        A list of ``EvalExample`` objects.
    """
    raw_records = _read_jsonl(path)
    examples = []

    for record in raw_records:
        example = EvalExample(
            example_id=str(record.get("example_id", "")),
            source=source,
            passage=record["passage"],
            citation_text=record["citation_text"],
            gold_label=record["label"],
            gold_claim=record.get("gold_claim"),
            split=record.get("split", "test"),
            metadata=record.get("metadata", {}),
        )
        examples.append(example)

    return examples


def _read_jsonl(path: Path) -> list[dict]:
    """
    Read a JSONL file into a list of dicts.

    Each line in the file should be a valid JSON object. Blank lines
    are skipped. Lines that fail to parse are logged and skipped.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of parsed dicts, one per valid line.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning(
                    "dataset_parse_error",
                    path=str(path),
                    line=line_num,
                    error=str(exc),
                )

    return records


def validate_labels(examples: list[EvalExample]) -> list[EvalExample]:
    """
    Filter examples to only those with valid gold labels.

    Logs a warning for any examples with unrecognized labels and
    excludes them from the returned list. This catches label-mapping
    errors in the preprocessing step.

    Args:
        examples: The examples to validate.

    Returns:
        The subset of examples with valid four-label gold labels.
    """
    valid_labels = set(Label.values())
    valid = []
    invalid_count = 0

    for example in examples:
        if example.gold_label in valid_labels:
            valid.append(example)
        else:
            invalid_count += 1
            log.warning(
                "dataset_invalid_label",
                example_id=example.example_id,
                label=example.gold_label,
            )

    if invalid_count > 0:
        log.warning("dataset_invalid_labels_total", count=invalid_count)

    return valid
