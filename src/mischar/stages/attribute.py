"""
Stage 3: Claim attribution via LLM.

Given a passage and a citation within it, this stage uses an LLM to extract
the specific legal proposition (claim) the passage attributes to the cited
case. For example, if a brief says "Under Smith, the exclusionary rule does
not apply," the attributed claim is "The exclusionary rule does not apply."

Abstains when:
- The citation is a string cite (listed without discussion).
- The citation is a "see generally" reference with no specific proposition.
- The LLM cannot extract a clear claim from the context.
- The LLM returns malformed output after retry.
"""

from __future__ import annotations

from mischar.cache import Cache
from mischar.logging import get_logger
from mischar.models.client import ModelClient
from mischar.prompts.attribution import (
    ATTRIBUTION_JSON_SCHEMA,
    build_attribution_prompt,
)
from mischar.types import Abstention, AttributedClaim, ParsedCitation

log = get_logger("attribute")


def attribute_claim(
    passage: str,
    citation: ParsedCitation,
    client: ModelClient,
    cache: Cache,
    prompt_version: str,
) -> AttributedClaim | Abstention:
    """
    Extract the claim a passage attributes to a cited case.

    Checks the cache first (keyed by passage, citation position, model,
    and prompt version). On miss, calls the LLM with the attribution
    prompt and parses the structured JSON response.

    Args:
        passage: The full passage text from the legal brief.
        citation: The parsed citation to attribute.
        client: The LLM client to use (any backend satisfying ModelClient).
        cache: Pipeline cache instance.
        prompt_version: Attribution prompt version string (e.g. "v1.0").
            Included in the cache key so prompt changes invalidate
            cached results.

    Returns:
        An ``AttributedClaim`` with the extracted proposition and confidence,
        or an ``Abstention`` with reason ``attribution-failed``.
    """
    # Cache key includes the model name and prompt version so that
    # switching models or updating the prompt automatically invalidates
    # stale cached attributions.
    cache_key = Cache.make_key(
        passage,
        citation.position_in_passage,
        client.name,
        prompt_version,
    )

    cached = cache.get("attribute", cache_key)
    if cached is not None:
        log.debug("attribute_cache_hit", cite=citation.raw_text)

        return cached

    # Build the prompt and call the LLM.
    prompt = build_attribution_prompt(passage, citation.raw_text)
    log.info("attribute_call", cite=citation.raw_text, model=client.name)

    # First attempt.
    result = _call_and_parse(prompt, client)

    # If the first attempt returned malformed JSON, retry once with
    # a stricter instruction appended.
    if result is None:
        log.warning("attribute_retry", cite=citation.raw_text, reason="malformed JSON")
        stricter_prompt = (
            prompt + "\n\nIMPORTANT: You must respond with valid JSON only. No other text."
        )
        result = _call_and_parse(stricter_prompt, client)

    # If still no valid result, abstain.
    if result is None:
        abstention = Abstention(
            reason="attribution-failed",
            details=f"LLM returned malformed output for {citation.raw_text}",
        )
        cache.set("attribute", cache_key, abstention)

        return abstention

    # Check if the model itself decided to abstain (e.g., string cite,
    # see-generally reference).
    if result.get("abstain", False):
        abstention = Abstention(
            reason="attribution-failed",
            details=result.get("abstain_reason", "No specific proposition attributed"),
        )
        cache.set("attribute", cache_key, abstention)
        log.info(
            "attribute_abstained",
            cite=citation.raw_text,
            reason=abstention.details,
        )

        return abstention

    # Extract the claim and confidence from the parsed response.
    claim_text = result.get("claim", "").strip()
    confidence = float(result.get("confidence", 0.0))

    if not claim_text:
        abstention = Abstention(
            reason="attribution-failed",
            details=f"Empty claim for {citation.raw_text}",
        )
        cache.set("attribute", cache_key, abstention)

        return abstention

    # Low confidence is treated as an abstention — the model isn't sure
    # it found a real claim.
    if confidence < 0.5:
        abstention = Abstention(
            reason="attribution-failed",
            details=f"Low confidence ({confidence:.2f})",
        )
        cache.set("attribute", cache_key, abstention)

        return abstention

    claim = AttributedClaim(claim_text=claim_text, confidence=confidence)
    cache.set("attribute", cache_key, claim)

    log.info(
        "attribute_success",
        cite=citation.raw_text,
        confidence=confidence,
        claim_preview=claim_text[:80],
    )

    return claim


def _call_and_parse(prompt: str, client: ModelClient) -> dict | None:
    """
    Call the model and parse the JSON response.

    Args:
        prompt: The legal passage plus the citation.
        client: The LLM client to be called.

    Returns:
        The parsed dict, or None if the response isn't valid JSON.
        Catches model errors and returns None so the caller can retry or
        abstain.
    """
    try:
        response = client.generate(
            prompt=prompt,
            json_schema=ATTRIBUTION_JSON_SCHEMA,
            temperature=0.0,
        )

        return response.parsed_json
    except Exception:
        log.warning("attribute_model_error", exc_info=True)

        return None
