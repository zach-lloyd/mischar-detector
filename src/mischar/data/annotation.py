"""
Annotation file IO and schema validation.

Handles reading and writing of hand-annotated real brief examples in
JSONL format (one JSON object per line). Every record is validated
against a schema at write time to catch malformed annotations early.

The annotation file is append-only — new annotations are added to the
end. This module never modifies or deletes existing records.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, field_validator

from mischar.constants import Label
from mischar.logging import get_logger

log = get_logger("data.annotation")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class AnnotationSourceInfo(BaseModel):
    """Source metadata for a real brief annotation."""

    recap_docket_id: str
    court: str
    filing_date: str
    document_url: str


class CitedCaseInfo(BaseModel):
    """Metadata about the cited case in an annotation."""

    courtlistener_id: str
    case_name: str
    citation: str


class AnnotationRecord(BaseModel):
    """
    Schema for a single annotation in the real_briefs.jsonl file.

    Validated at write time via pydantic. Required fields are enforced
    and the label must be one of the four valid values.
    """

    annotation_id: str
    annotated_at: str
    time_spent_minutes: int | None = None
    annotator: str

    source: AnnotationSourceInfo
    passage: str
    citation_text_in_passage: str
    citation_offset: list[int]

    cited_case: CitedCaseInfo

    gold_claim: str | None = None
    label: str
    annotator_notes: str | None = None
    boundary_case: bool = False

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """
        Ensure the label is one of the four valid classification labels.

        Args:
            v: The label value to validate.

        Returns:
            The validated label.

        Raises:
            ValueError: If the label is not one of the four valid values.
        """
        valid = set(Label.values())
        if v not in valid:
            raise ValueError(f"label must be one of {valid}, got '{v}'")

        return v

    @field_validator("citation_offset")
    @classmethod
    def validate_offset(cls, v: list[int]) -> list[int]:
        """
        Ensure the citation offset is a two-element list of non-negative ints.

        Args:
            v: The citation offset to validate.

        Returns:
            The validated offset.

        Raises:
            ValueError: If the offset is malformed.
        """
        if len(v) != 2:
            raise ValueError(f"citation_offset must have exactly 2 elements, got {len(v)}")
        if v[0] < 0 or v[1] < 0:
            raise ValueError(f"citation_offset values must be non-negative, got {v}")
        if v[0] >= v[1]:
            raise ValueError(f"citation_offset start must be less than end, got {v}")

        return v


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_annotations(path: Path) -> list[AnnotationRecord]:
    """
    Read and validate all annotation records from a JSONL file.

    Each line is parsed as JSON and validated against the
    ``AnnotationRecord`` schema. Invalid lines are logged and skipped.

    Args:
        path: Path to the annotations JSONL file.

    Returns:
        A list of validated ``AnnotationRecord`` objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
                record = AnnotationRecord(**raw)
                records.append(record)
            except (json.JSONDecodeError, Exception) as exc:
                log.warning(
                    "annotation_parse_error",
                    path=str(path),
                    line=line_num,
                    error=str(exc),
                )

    log.info("annotations_loaded", path=str(path), count=len(records))

    return records


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def append_annotation(path: Path, record: AnnotationRecord) -> None:
    """
    Append a single validated annotation record to the JSONL file.

    The record is validated by pydantic on construction, so this function
    just serializes and appends. Creates the file and parent directories
    if they don't exist.

    Args:
        path: Path to the annotations JSONL file.
        record: The validated annotation record to append.
    """
    # Ensure the directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize and append.
    line = record.model_dump_json()
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    log.info(
        "annotation_appended",
        path=str(path),
        annotation_id=record.annotation_id,
    )


def create_annotation(
    passage: str,
    citation_text_in_passage: str,
    citation_offset: tuple[int, int],
    label: str,
    source_info: dict[str, str],
    cited_case_info: dict[str, str],
    *,
    annotator: str = "user",
    gold_claim: str | None = None,
    annotator_notes: str | None = None,
    boundary_case: bool = False,
    time_spent_minutes: int | None = None,
) -> AnnotationRecord:
    """
    Create a new annotation record with auto-generated ID and timestamp.

    This is a convenience function for building valid ``AnnotationRecord``
    objects without having to manually construct the ID and timestamp.
    The record is validated on creation via pydantic.

    Args:
        passage: The full passage text from the brief.
        citation_text_in_passage: The specific citation string.
        citation_offset: Character offsets (start, end) of the citation
            within the passage.
        label: One of the four valid classification labels.
        source_info: Dict with keys ``recap_docket_id``, ``court``,
            ``filing_date``, ``document_url``.
        cited_case_info: Dict with keys ``courtlistener_id``,
            ``case_name``, ``citation``.
        annotator: Annotator identifier. Defaults to "user".
        gold_claim: The human-judged claim, if provided.
        annotator_notes: Free-text notes about the annotation.
        boundary_case: Whether this is a boundary/ambiguous case.
        time_spent_minutes: How long the annotation took.

    Returns:
        A validated ``AnnotationRecord``.

    Raises:
        pydantic.ValidationError: If any field is invalid.
    """
    return AnnotationRecord(
        annotation_id=str(uuid4()),
        annotated_at=datetime.now(UTC).isoformat(),
        time_spent_minutes=time_spent_minutes,
        annotator=annotator,
        source=AnnotationSourceInfo(**source_info),
        passage=passage,
        citation_text_in_passage=citation_text_in_passage,
        citation_offset=list(citation_offset),
        cited_case=CitedCaseInfo(**cited_case_info),
        gold_claim=gold_claim,
        label=label,
        annotator_notes=annotator_notes,
        boundary_case=boundary_case,
    )
