"""
Stage 5: Classification via LLM.

The final pipeline stage. Given a claim (from attribution) and relevant
case excerpts (from retrieval), the classifier determines whether the
case actually supports the claim.

This stage does not abstain — all abstention conditions (case not found,
text not retrieved, attribution failed, case too long) are caught by
earlier stages. If classification reaches this point, it must produce
a label.
"""

from __future__ import annotations

from mischar.cache import Cache
from mischar.constants import Label
from mischar.logging import get_logger
from mischar.models.client import ModelClient
from mischar.prompts.classification import (
    CLASSIFICATION_JSON_SCHEMA,
    build_classification_prompt,
)
from mischar.types import AttributedClaim, Classification, ResolvedCase, RetrievalResult

log = get_logger("classify")


def classify(
    claim: AttributedClaim,
    retrieval: RetrievalResult,
    case: ResolvedCase,
    client: ModelClient,
    cache: Cache,
    prompt_version: str,
) -> Classification:
    """
    Classify the relationship between a claim and cited case text.

    Builds a prompt from the claim and retrieved chunks, calls the LLM
    with structured JSON output, and parses the result into a
    ``Classification`` with label, confidence, and supporting text.

    Args:
        claim: The attributed claim (what the brief says the case held).
        retrieval: The top-K retrieved chunks from the case opinion.
        case: The resolved case (used for case name in the prompt).
        client: The LLM client (any backend satisfying ModelClient).
        cache: Pipeline cache instance.
        prompt_version: Classification prompt version string. Included
            in cache keys so prompt changes invalidate cached results.

    Returns:
        A ``Classification`` with one of four labels: entails,
        partially_supports, unrelated, or contradicts.
    """
    # Build cache key from all inputs that affect the classification.
    # Using chunk texts (not embeddings) because the classifier sees
    # text, not vectors.
    chunk_texts = [chunk.text for chunk in retrieval.chunks]
    cache_key = Cache.make_key(
        claim.claim_text,
        chunk_texts,
        client.name,
        prompt_version,
    )

    cached = cache.get("classify", cache_key)
    if cached is not None:
        log.debug("classify_cache_hit", case=case.case_name)

        return cached

    # Concatenate retrieved chunks into a single context block,
    # separated by dividers so the model can see chunk boundaries.
    retrieved_text = _format_retrieved_chunks(retrieval)

    # Build the classification prompt.
    prompt = build_classification_prompt(
        claim=claim.claim_text,
        retrieved_text=retrieved_text,
        case_name=case.case_name,
    )

    log.info(
        "classify_call",
        case=case.case_name,
        model=client.name,
        n_chunks=len(retrieval.chunks),
    )

    # Call the LLM with structured JSON output.
    result = _call_and_parse(prompt, client)

    # If the first attempt failed, retry with stricter instructions.
    if result is None:
        log.warning("classify_retry", case=case.case_name, reason="malformed JSON")
        stricter_prompt = (
            prompt + "\n\nIMPORTANT: You must respond with valid JSON only. No other text."
        )
        result = _call_and_parse(stricter_prompt, client)

    # If still no valid result, fall back to a low-confidence "unrelated"
    # classification with a logged warning. This is a last resort — structured 
    # output mode should prevent it.
    if result is None:
        log.error(
            "classify_fallback",
            case=case.case_name,
            reason="malformed JSON after retry",
        )
        classification = Classification(
            label="unrelated",
            confidence=0.1,
            supporting_text="Classification failed — model returned malformed output.",
        )
        cache.set("classify", cache_key, classification)

        return classification

    # Extract and validate the label.
    raw_label = result.get("label", "").lower().strip()
    valid_labels = Label.values()

    if raw_label not in valid_labels:
        log.warning(
            "classify_invalid_label",
            case=case.case_name,
            raw_label=raw_label,
        )
        # Default to "unrelated" with low confidence for invalid labels.
        raw_label = "unrelated"
        confidence = 0.1
    else:
        confidence = float(result.get("confidence", 0.5))
        # Clamp confidence to [0, 1].
        confidence = max(0.0, min(1.0, confidence))

    supporting_text = result.get("supporting_text", "")

    classification = Classification(
        label=raw_label,
        confidence=confidence,
        supporting_text=supporting_text,
    )

    cache.set("classify", cache_key, classification)

    log.info(
        "classify_success",
        case=case.case_name,
        label=raw_label,
        confidence=round(confidence, 3),
    )

    return classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_retrieved_chunks(retrieval: RetrievalResult) -> str:
    """
    Format retrieved chunks into a single text block for the prompt.

    Each chunk is labeled with its index and relevance score so the
    model has context about which parts are most relevant.

    Args:
        retrieval: The retrieved chunks.
    
    Returns:
        The retrieved chunks formatted as a string.
    """
    parts = []
    for i, (chunk, _score) in enumerate(
        zip(retrieval.chunks, retrieval.scores, strict=True)
    ):
        header = f"[Excerpt {i + 1}]"

        parts.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(parts)


def _call_and_parse(prompt: str, client: ModelClient) -> dict | None:
    """
    Call the model and parse the JSON response.

    Args:
        prompt: The claim and the cited chunks for classification.
        client: The LLM model to be called.

    Returns:
        The parsed dict, or None if the response isn't valid JSON.
    """
    try:
        response = client.generate(
            prompt=prompt,
            json_schema=CLASSIFICATION_JSON_SCHEMA,
            temperature=0.0,
        )

        return response.parsed_json
    except Exception:
        log.warning("classify_model_error", exc_info=True)

        return None
