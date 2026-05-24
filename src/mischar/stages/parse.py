"""
Stage 1: Citation parsing via eyecite.

Extracts legal citations from a passage of text, producing ``ParsedCitation``
objects with structured components (reporter, volume, page, pincite, etc.)
and character-level positions within the source passage.

This is a pure function with no external dependencies beyond eyecite — no
API calls, no caching needed.
"""

from __future__ import annotations

import re

from eyecite import get_citations
from eyecite.models import FullCaseCitation

from mischar.logging import get_logger
from mischar.types import ParsedCitation, Pincite

log = get_logger("parse")


def parse_citations(passage: str) -> list[ParsedCitation]:
    """
    Extract legal citations from a text passage.

    Uses eyecite to find and parse citations, then converts them into
    our domain's ``ParsedCitation`` type. Only full case citations are
    included — short-form citations (e.g. "Id. at 5") and supra
    references are skipped because they can't be independently resolved.

    Args:
        passage: The text to search for citations. Typically a paragraph
            or multi-paragraph excerpt from a legal brief.

    Returns:
        A list of ``ParsedCitation`` objects, one per citation found.
        Returns an empty list if no citations are parseable (the caller
        treats this as a ``parsing-failed`` abstention).
    """
    if not passage or not passage.strip():
        log.debug("parse_empty_passage")

        return []

    # eyecite returns a mix of citation types (FullCaseCitation,
    # ShortCaseCitation, SupraCitation, etc.). We only want full
    # citations because those are the ones we can resolve independently
    # via CourtListener.
    raw_citations = get_citations(passage)
    full_citations = [c for c in raw_citations if isinstance(c, FullCaseCitation)]

    if not full_citations:
        log.debug("parse_no_full_citations", total_found=len(raw_citations))

        return []

    results = []
    for cite in full_citations:
        parsed = _convert_eyecite(cite, passage)
        if parsed is not None:
            results.append(parsed)

    log.info(
        "parse_complete",
        total_eyecite=len(raw_citations),
        full_citations=len(full_citations),
        parsed=len(results),
    )

    return results


def _convert_eyecite(cite: FullCaseCitation, passage: str) -> ParsedCitation | None:
    """
    Convert an eyecite ``FullCaseCitation`` to our ``ParsedCitation`` type.

    Returns None if essential fields (reporter, volume, page) can't be
    extracted — this shouldn't happen for a FullCaseCitation but we
    handle it defensively.

    Args:
        cite: The eyecite citation object to convert.
        passage: The original passage text (used for position tracking).

    Returns:
        A ``ParsedCitation`` with structured components, or None if
        essential fields couldn't be extracted.
    """
    # Extract the citation's position in the passage. eyecite provides
    # a span() method that returns (start, end) character offsets.
    try:
        span = cite.span()
        position = (span[0], span[1])
    except (AttributeError, TypeError):
        # If span information isn't available, use a sentinel.
        position = (0, 0)

    # Volume and page are required for resolution. If eyecite couldn't
    # parse them (shouldn't happen for FullCaseCitation, but defensive),
    # skip this citation.
    try:
        volume = int(cite.groups["volume"])
        page = int(cite.groups["page"])
        reporter = cite.groups.get("reporter", "")
    except (KeyError, ValueError, TypeError) as exc:
        log.debug("parse_skip_citation", raw=str(cite), reason=str(exc))

        return None

    # Extract pincite if present (e.g. "at 462" or ", 462").
    pincite = _extract_pincite(cite)

    # Extract metadata fields. These may be None depending on how
    # much information eyecite could parse from the citation string.
    case_name = _extract_case_name(cite)
    court = _extract_court(cite)
    year = _extract_year(cite)

    return ParsedCitation(
        raw_text=cite.matched_text(),
        case_name=case_name,
        reporter=reporter,
        volume=volume,
        page=page,
        pincite=pincite,
        court=court,
        year=year,
        position_in_passage=position,
    )


def _extract_pincite(cite: FullCaseCitation) -> Pincite | None:
    """
    Extract a pinpoint citation (specific page reference) if present.

    eyecite parses pincites from the citation string and stores them
    in ``metadata.pin_cite``. Pincites look like "123 F.3d 456, 462"
    or "123 F.3d 456, at 462" in the raw citation text.

    We extract the numeric page number from eyecite's parsed string,
    and check for paragraph-level pincites (¶ notation).

    Args:
        cite: The eyecite citation object.

    Returns:
        A ``Pincite`` with page number (and optional paragraph number),
        or None if no pincite is present. None is normal — many
        citations reference a case generally rather than a specific page.
    """
    try:
        pin = cite.metadata.pin_cite
        if not pin:
            return None

        # Extract the page number from the pincite string.
        # Pincites can be "462", "at 462", "at 462-63", "at *5", etc.
        # We try to get the first numeric value.
        page_match = re.search(r"(\d+)", pin)
        if not page_match:
            return None

        page_number = int(page_match.group(1))

        # Check for paragraph-level pincites (¶ or "para.").
        para_match = re.search(r"[¶]\s*(\d+)", pin)
        paragraph_number = int(para_match.group(1)) if para_match else None

        return Pincite(
            raw=pin,
            page_number=page_number,
            paragraph_number=paragraph_number,
        )
    except (AttributeError, ValueError):
        return None


def _extract_case_name(cite: FullCaseCitation) -> str | None:
    """
    Extract the case name from eyecite's parsed citation metadata.

    eyecite (≥2.6) parses party names from the text preceding the
    reporter citation and stores them in ``metadata.plaintiff`` and
    ``metadata.defendant``. If both are present, we combine them
    into "Plaintiff v. Defendant" format. If only the defendant is
    present, we return that alone.

    Args:
        cite: The eyecite citation object.

    Returns:
        The case name as a string (e.g. "Smith v. Jones"), or None
        if eyecite couldn't parse it.
    """
    try:
        if hasattr(cite.metadata, "defendant") and cite.metadata.defendant:
            plaintiff = getattr(cite.metadata, "plaintiff", None) or ""
            defendant = cite.metadata.defendant

            if plaintiff:
                return f"{plaintiff} v. {defendant}"

            return defendant
    except AttributeError:
        pass

    return None


def _extract_court(cite: FullCaseCitation) -> str | None:
    """
    Read the court identifier from eyecite's parsed citation metadata.

    eyecite parses the parenthetical after the reporter citation
    (e.g. "(9th Cir. 2001)") and extracts the court portion into
    ``metadata.court``. We just read what eyecite already parsed.

    Args:
        cite: The eyecite citation object.

    Returns:
        The court string (e.g. "9th Cir."), or None if eyecite
        couldn't parse it.
    """
    try:
        court = cite.metadata.court

        return court if court else None
    except AttributeError:

        return None


def _extract_year(cite: FullCaseCitation) -> int | None:
    """
    Read the decision year from eyecite's parsed citation metadata.

    eyecite parses the parenthetical after the reporter citation
    (e.g. "(9th Cir. 2001)") and extracts the year into
    ``metadata.year``. We read it and convert to int.

    Args:
        cite: The eyecite citation object.

    Returns:
        The decision year as an integer (e.g. 2001), or None if
        eyecite couldn't parse it.
    """
    try:
        year = cite.metadata.year

        return int(year) if year else None
    except (AttributeError, ValueError, TypeError):
        return None
