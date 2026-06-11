"""
Dataset loaders for each evaluation source.

Each loader reads its source-specific format and converts records into
``EvalExample`` objects with a standardized shape that the evaluation
harness can process uniformly. The loaders handle format differences
across sources so the rest of the pipeline doesn't have to.

Supported sources:
- **CaseHOLD**: Accurate/mischaracterized example pairs derived from
  CaseHOLD entries (used for train and val).
- **real_brief**: Hand-annotated real brief citations in our JSONL
  format (used as the held-out test set).
"""

from __future__ import annotations

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

    Each CaseHOLD entry contributes a pair of examples: an "accurate"
    example built from the entry's correct holding and a
    "mischaracterized" example built from one of the entry's incorrect
    holding choices. Both pair members share the same citation, so the
    split-assignment logic keeps them in the same split.

    Args:
        path: Path to the CaseHOLD JSONL file
            (output of ``build_casehold_set.py``).
        split: If provided, filter to only examples with this split
            value ("train", "val", or "test").

    Returns:
        A list of ``EvalExample`` objects.
    """
    raw_records = _read_jsonl(path)
    examples = []

    for record in raw_records:
        example = EvalExample(
            example_id=str(record.get("example_id", "")),
            source="casehold",
            passage=record["passage"],
            citation_text=record["citation_text"],
            gold_label=record["label"],
            gold_claim=record.get("gold_claim"),
            split=record.get("split", "train"),
            metadata=record.get("metadata", {}),
        )
        examples.append(example)

    if split:
        examples = [ex for ex in examples if ex.split == split]

    log.info("dataset_loaded", source="casehold", count=len(examples), split=split)

    return examples


def load_real_brief(path: Path, split: str | None = None) -> list[EvalExample]:
    """
    Load hand-annotated real brief examples.

    Reads the annotation format (see docs/annotation-guide.md) and
    converts to ``EvalExample``. The file may be strict JSONL (one
    object per line) or a sequence of pretty-printed JSON objects
    separated by commas — both are handled.

    Required fields per record: ``annotation_id``, ``passage``,
    ``citation_text_in_passage``, and ``label``. ``annotation_id`` must
    be a stable unique string — predictions are keyed by it, so
    cross-run per-example analysis breaks if IDs shift between runs.
    ``gold_claim`` is optional; when absent, the harness simply skips
    the gold-claim (dual evaluation) pass for that example.

    Args:
        path: Path to the annotation file.
        split: If provided, filter to only examples with this split
            value. Records without a ``split`` field default to "test".

    Returns:
        A list of ``EvalExample`` objects.

    Raises:
        ValueError: If any record is missing ``annotation_id``.
    """
    raw_records = _read_records(path)
    examples = []

    for i, record in enumerate(raw_records, start=1):
        annotation_id = record.get("annotation_id")
        if not annotation_id:
            raise ValueError(
                f"Record {i} in {path} is missing 'annotation_id'. "
                "Every annotation needs a stable unique ID so per-example "
                "results can be compared across eval runs. Add one to "
                "each record and reload."
            )

        example = EvalExample(
            example_id=str(annotation_id),
            source="real_brief",
            passage=record["passage"],
            citation_text=record["citation_text_in_passage"],
            gold_label=record["label"],
            gold_claim=record.get("gold_claim"),
            split=record.get("split", "test"),
            metadata={
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


def _read_records(path: Path) -> list[dict]:
    """
    Read annotation records from a file in either supported format.

    Tries strict JSONL first (one JSON object per line). If that yields
    nothing, falls back to parsing the whole file as a comma-separated
    sequence of pretty-printed JSON objects (the hand-annotation format),
    by wrapping the content in brackets and parsing it as a JSON array.
    ``strict=False`` permits literal newlines inside string values.

    Args:
        path: Path to the annotation file.

    Returns:
        A list of parsed dicts.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file can't be parsed in either format.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    # Attempt 1: strict JSONL. Done quietly (no per-line warnings) since
    # a wholesale failure just means the file is in the other format.
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records = []
            break

    if records:
        return records

    # Attempt 2: comma-separated pretty-printed JSON objects.
    if text.endswith(","):
        text = text[:-1]

    try:
        parsed = json.loads(f"[{text}]", strict=False)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse {path} as JSONL or as a comma-separated "
            f"sequence of JSON objects: {exc}"
        ) from exc

    if not isinstance(parsed, list):
        parsed = [parsed]

    log.info("dataset_parsed_lenient", path=str(path), count=len(parsed))

    return parsed


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
        The subset of examples with valid binary gold labels.
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
