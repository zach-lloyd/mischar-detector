"""
Core domain types for the mischaracterization detector pipeline.

All cross-cutting data structures live here so that stages, models, eval, and
CLI modules can import from a single location without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Literal

from mischar.constants import PIPELINE_VERSION

# ---------------------------------------------------------------------------
# Citation parsing types
# ---------------------------------------------------------------------------


@dataclass
class ParsedCitation:
    """
    A citation extracted from a passage by eyecite.

    Carries the raw text, parsed components, and character-level position
    within the source passage.
    """

    raw_text: str
    case_name: str | None
    reporter: str
    volume: int
    page: int
    court: str | None
    year: int | None
    position_in_passage: tuple[int, int]  # (start_char, end_char)


# ---------------------------------------------------------------------------
# Resolution types
# ---------------------------------------------------------------------------


@dataclass
class ResolvedCase:
    """A case resolved via CourtListener with its full opinion text."""

    courtlistener_id: str
    case_name: str
    citation_string: str
    full_text: str
    decided_at: date | None
    court: str


# ---------------------------------------------------------------------------
# Attribution types
# ---------------------------------------------------------------------------


@dataclass
class AttributedClaim:
    """The proposition a passage attributes to a cited case."""

    claim_text: str
    confidence: float


# ---------------------------------------------------------------------------
# Retrieval types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A chunk of opinion text produced by paragraph-grouped chunking."""

    text: str
    token_count: int
    chunk_index: int  # 0-based position within opinion
    paragraph_range: tuple[int, int]  # paragraph indices [start, end] inclusive


@dataclass
class ChunkEmbedding:
    """A chunk paired with its embedding vector."""

    chunk: Chunk
    embedding: list[float]  # length 1024 for voyage-law-2


@dataclass
class RetrievalResult:
    """The top-K retrieved chunks for a claim, with similarity scores."""

    chunks: list[Chunk]
    scores: list[float]


# ---------------------------------------------------------------------------
# Classification types
# ---------------------------------------------------------------------------


@dataclass
class Classification:
    """A classification verdict for a (claim, cited-case) pair."""

    label: Literal["accurate", "mischaracterized"]
    confidence: float
    supporting_text: str


@dataclass
class Abstention:
    """
    Indicates the pipeline cannot produce a verdict for this citation.

    Abstentions are part of the pipeline's contract — they represent conditions
    where no meaningful classification is possible, distinct from infrastructure
    errors.
    """

    reason: Literal[
        "parsing-failed",
        "case-not-found",
        "text-not-retrieved",
        "attribution-failed",
        "case-too-long",
    ]
    details: str | None = None


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------


@dataclass
class CitationResult:
    """
    The pipeline's complete output for a single citation.

    Invariant: exactly one of ``classification`` or ``abstention`` is non-None.
    """

    citation: ParsedCitation
    resolved_case: ResolvedCase | None
    claim: AttributedClaim | None
    retrieval: RetrievalResult | None
    classification: Classification | None
    abstention: Abstention | None
    model_used: str
    pipeline_version: str = field(default=PIPELINE_VERSION)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


    @property
    def abstained(self) -> bool:
        """True if the pipeline abstained rather than classifying."""
        
        return self.abstention is not None


    def __post_init__(self) -> None:
        has_class = self.classification is not None
        has_abst = self.abstention is not None

        if has_class == has_abst:
            raise ValueError(
                "CitationResult must have exactly one of classification or abstention, "
                f"got classification={has_class}, abstention={has_abst}"
            )


# ---------------------------------------------------------------------------
# Evaluation types
# ---------------------------------------------------------------------------


@dataclass
class EvalExample:
    """A single example in an evaluation dataset."""

    example_id: str
    source: Literal["casehold", "real_brief"]
    passage: str
    citation_text: str
    gold_label: Literal["accurate", "mischaracterized"]
    gold_claim: str | None # Only used if real brief examples are lengthy passages that require LLM-automated claim extraction
    split: Literal["train", "val", "test"]
    metadata: dict = field(default_factory=dict)


@dataclass
class Prediction:
    """A pipeline prediction for a single eval example."""

    example_id: str
    predicted_label: str | None  # None if abstained
    abstention_reason: str | None
    confidence: float | None
    used_gold_claim: bool  # True if dual eval used gold claim instead of attributed claim


@dataclass
class LabelMetrics:
    """Precision/recall/F1 for a single label."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class MetricsBundle:
    """Aggregate evaluation metrics across all labels."""

    macro_f1: float
    per_label: dict[str, LabelMetrics]
    confusion_matrix: dict[tuple[str, str], int]  # (true, pred) -> count
    abstention_rate: float
    abstention_by_reason: dict[str, int]
    n_examples: int


# ---------------------------------------------------------------------------
# Model client types
# ---------------------------------------------------------------------------


@dataclass
class ModelResponse:
    """Response from any model backend."""

    text: str
    parsed_json: dict | None = None  # populated if json_schema was provided
    raw_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Embedding type alias
# ---------------------------------------------------------------------------

Embedding = list[float]
