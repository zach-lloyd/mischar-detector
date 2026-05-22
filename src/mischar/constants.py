"""
Constants for the mischaracterization detector pipeline.

Defines label enums, abstention reason codes, disclaimer text, and version strings
used throughout the library.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------


class Label(StrEnum):
    """Four-label classification scheme for citation characterization."""

    ENTAILS = "entails"
    PARTIALLY_SUPPORTS = "partially_supports"
    UNRELATED = "unrelated"
    CONTRADICTS = "contradicts"

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]


# ---------------------------------------------------------------------------
# Abstention reason codes
# ---------------------------------------------------------------------------


class AbstentionReason(StrEnum):
    """
    Reason codes for pipeline abstentions.

    These represent conditions where a meaningful verdict cannot be produced
    even with perfect infrastructure — they are part of the pipeline's contract,
    not error conditions.
    """

    PARSING_FAILED = "parsing-failed"
    CASE_NOT_FOUND = "case-not-found"
    TEXT_NOT_RETRIEVED = "text-not-retrieved"
    ATTRIBUTION_FAILED = "attribution-failed"
    CASE_TOO_LONG = "case-too-long"


# ---------------------------------------------------------------------------
# Evaluation sources
# ---------------------------------------------------------------------------


class EvalSource(StrEnum):
    """Dataset sources used in evaluation."""

    CASEHOLD = "casehold"
    PERTURBATION = "perturbation"
    HOU = "hou"
    REAL_BRIEF = "real_brief"
    CHARLOTIN = "charlotin"


# ---------------------------------------------------------------------------
# Prompt versions — bumped when prompt content changes
# ---------------------------------------------------------------------------

DEFAULT_ATTRIBUTION_PROMPT_VERSION = "v1.0"
DEFAULT_CLASSIFICATION_PROMPT_VERSION = "v1.0"

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "This tool is a research prototype for studying legal citation characterization. "
    "It is not a legal tool, does not provide legal advice, and should not be used as "
    "a substitute for professional legal analysis. Its outputs have not been validated "
    "for use in any legal proceeding. The author makes no warranty as to the accuracy "
    "or reliability of any classification produced."
)

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_MAX_TOKENS = 1200
DEFAULT_CHUNK_OVERLAP_PARAGRAPHS = 1
DEFAULT_TOP_K = 5
# When there is a cite to a specific page, boost the value of the chunk containing
# that page, as well as neighboring chunks. However, note that this will be 
# inherently somewhat fuzzy because we have to estimate where the cited page falls
# in the chunks
DEFAULT_PINCITE_BOOST = 0.15
DEFAULT_PINCITE_NEIGHBOR_WINDOW = 2
# Controls randomness of LLM output. Set to 0.0 so the model will be fully
# deterministic and always predict the highest probability token
DEFAULT_GENERATION_TEMPERATURE = 0.0
DEFAULT_GENERATION_MAX_TOKENS = 1024
# CourtListener's API allows 60 requests per minute on a free key
DEFAULT_COURTLISTENER_RATE_LIMIT = 60
DEFAULT_COURTLISTENER_MAX_RETRIES = 5
DEFAULT_LLM_TIMEOUT_SECONDS = 120
