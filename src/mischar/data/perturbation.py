"""
Perturbation generator for synthetic training data.

Transforms genuine "entails" (claim, case) pairs into perturbed versions
that fall under the other three labels: partially_supports, unrelated,
and contradicts. Each of the 11 perturbation types produces a specific
kind of mischaracterization that maps to a target label.

Perturbation types by target label:

**partially_supports** (claim overstates or drops nuance):
- P1: Drop a qualification from the holding.
- P2: Generalize the scope beyond what the case established.
- P3: Inflate the strength of the language ("may" → "must").
- P4: Drop the procedural posture or context.

**unrelated** (claim doesn't match the case's topic):
- U1: Wrong legal issue from the same case.
- U2: Completely off-topic claim.
- U3: Tangentially related but not what the case addressed.

**contradicts** (claim is opposite of what the case held):
- C1: Directly negate the holding.
- C2: Reverse which party prevailed.
- C3: Reverse the factors in a multi-factor test.
- C4: Fabricate an opposite holding.

The bulk generation orchestration (spot-checking, volume targets, model
routing) lives in ``scripts/data_construction/generate_perturbations.py``,
not here. This module provides the per-example generation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from mischar.logging import get_logger
from mischar.models.client import ModelClient, parse_json_response
from mischar.types import ModelResponse

log = get_logger("perturbation")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PerturbationType(StrEnum):
    """
    The 11 perturbation types, grouped by target label.

    The string value (e.g. "p1") is used as a compact identifier in
    dataset files and logs.
    """

    # partially_supports — claim overstates or drops nuance
    P1_DROP_QUALIFICATION = "p1"
    P2_GENERALIZE_SCOPE = "p2"
    P3_INFLATE_STRENGTH = "p3"
    P4_DROP_POSTURE = "p4"

    # unrelated — claim doesn't match the case's topic
    U1_WRONG_ISSUE_SAME_CASE = "u1"
    U2_OFF_TOPIC = "u2"
    U3_TANGENTIAL = "u3"

    # contradicts — claim is opposite of what the case held
    C1_NEGATE = "c1"
    C2_REVERSE_WINNER = "c2"
    C3_OPPOSITE_FACTORS = "c3"
    C4_HALLUCINATED_OPPOSITE = "c4"


# Mapping from perturbation type to target classification label.
PERTURBATION_TARGET_LABELS: dict[PerturbationType, str] = {
    PerturbationType.P1_DROP_QUALIFICATION: "partially_supports",
    PerturbationType.P2_GENERALIZE_SCOPE: "partially_supports",
    PerturbationType.P3_INFLATE_STRENGTH: "partially_supports",
    PerturbationType.P4_DROP_POSTURE: "partially_supports",
    PerturbationType.U1_WRONG_ISSUE_SAME_CASE: "unrelated",
    PerturbationType.U2_OFF_TOPIC: "unrelated",
    PerturbationType.U3_TANGENTIAL: "unrelated",
    PerturbationType.C1_NEGATE: "contradicts",
    PerturbationType.C2_REVERSE_WINNER: "contradicts",
    PerturbationType.C3_OPPOSITE_FACTORS: "contradicts",
    PerturbationType.C4_HALLUCINATED_OPPOSITE: "contradicts",
}


@dataclass
class SourceExample:
    """
    A genuine (claim, case) pair used as input for perturbation.

    These are real "entails" examples — the claim accurately describes
    the case's holding. The perturbation generator transforms the claim
    into a mischaracterization.
    """

    example_id: str
    claim: str
    case_name: str
    case_text: str
    citation_text: str
    decided_year: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PerturbedExample:
    """
    Output of the perturbation generator.

    Contains the perturbed claim (the mischaracterization), the target
    label it should receive, and provenance information linking back to
    the source example and perturbation type.
    """

    example_id: str
    source_example_id: str
    perturbation_type: str
    target_label: str
    original_claim: str
    perturbed_claim: str
    case_name: str
    citation_text: str
    passage: str  # Synthetic passage embedding the perturbed claim with the citation.
    decided_year: int | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


_PERTURBATION_PROMPTS: dict[PerturbationType, str] = {

    PerturbationType.P1_DROP_QUALIFICATION: """You are generating training data for 
    a legal citation mischaracterization detector.

Given a claim that accurately describes a case's holding, rewrite it to DROP an 
important qualification or limitation. The rewritten claim should be partially 
correct but misleadingly overbroad because it omits a key condition.

Example: "The exclusionary rule does not apply to evidence obtained in good-faith 
reliance on a warrant" → "The exclusionary rule does not apply to illegally obtained 
evidence" (drops the "good-faith reliance on a warrant" qualification).

## Original claim (accurate)
{claim}

## Case name
{case_name}

## Relevant case text
{case_text_excerpt}

Respond with a JSON object:
{{"perturbed_claim": "...", "dropped_qualification": "...", "passage": "..."}}

"perturbed_claim" is the rewritten claim with the qualification removed.
"dropped_qualification" is a brief description of what was dropped.
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using 
the perturbed claim. Use the citation: {citation_text}""",

    PerturbationType.P2_GENERALIZE_SCOPE: """You are generating training data for a 
    legal citation mischaracterization detector.

Given a claim that accurately describes a case's holding, rewrite it to GENERALIZE THE 
SCOPE beyond what the case actually established. The rewritten claim should extend the 
holding to a broader context than the case addressed.

Example: "Employment contracts in New York require a non-compete clause to be reasonable 
in scope" → "All contracts require non-compete clauses to be reasonable in scope" (generalizes 
from employment contracts in NY to all contracts everywhere).

## Original claim (accurate)
{claim}

## Case name
{case_name}

## Relevant case text
{case_text_excerpt}

Respond with a JSON object:
{{"perturbed_claim": "...", "generalization": "...", "passage": "..."}}

"perturbed_claim" is the rewritten claim with overgeneralized scope.
"generalization" is a brief description of how the scope was expanded.
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using the perturbed 
claim. Use the citation: {citation_text}""",

    PerturbationType.P3_INFLATE_STRENGTH: """You are generating training data for a legal citation 
    mischaracterization detector.

Given a claim that accurately describes a case's holding, rewrite it to INFLATE THE STRENGTH of 
the language. Change permissive language to mandatory, possibilities to certainties, 
considerations to requirements.

Example: "Courts may consider the totality of circumstances" → "Courts must consider the 
totality of circumstances" (inflates "may" to "must").

## Original claim (accurate)
{claim}

## Case name
{case_name}

Respond with a JSON object:
{{"perturbed_claim": "...", "inflation": "...", "passage": "..."}}

"perturbed_claim" is the rewritten claim with inflated language strength.
"inflation" is a brief description of the language change (e.g., "may → must").
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using 
the perturbed claim. Use the citation: {citation_text}""",

    PerturbationType.P4_DROP_POSTURE: """You are generating training data for a 
    legal citation mischaracterization detector.

Given a claim that accurately describes a case's holding, rewrite it to DROP 
THE PROCEDURAL POSTURE or context. State a holding that was specific to a 
procedural context (e.g., motion to dismiss, summary judgment, preliminary 
injunction) as if it were a general rule of law.

Example: "At the motion to dismiss stage, the court assumes all facts alleged 
in the complaint are true" → "Courts must assume all facts alleged in a complaint 
are true" (drops the procedural limitation).

## Original claim (accurate)
{claim}

## Case name
{case_name}

## Relevant case text
{case_text_excerpt}

Respond with a JSON object:
{{"perturbed_claim": "...", "dropped_posture": "...", "passage": "..."}}

"perturbed_claim" is the rewritten claim with procedural context removed.
"dropped_posture" is a brief description of the procedural context that was dropped.
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using the 
perturbed claim. Use the citation: {citation_text}""",

    PerturbationType.U1_WRONG_ISSUE_SAME_CASE: """You are generating training data 
    for a legal citation mischaracterization detector.

Given a claim that accurately describes a case's holding, create a NEW claim about a 
DIFFERENT LEGAL ISSUE that the same case did NOT address. The new claim should sound 
plausible for a case in this area of law but should be about an entirely different 
legal question.

Example: If the case held "warrantless searches of automobiles are permitted under 
the automobile exception," create a claim about a different Fourth Amendment issue the 
case did NOT address, like "thermal imaging of a home requires a warrant."

## Original claim (accurate — about the issue the case DID address)
{claim}

## Case name
{case_name}

## Relevant case text
{case_text_excerpt}

Respond with a JSON object:
{{"perturbed_claim": "...", "wrong_issue": "...", "passage": "..."}}

"perturbed_claim" is a claim about a legal issue the case did NOT address.
"wrong_issue" is a brief description of the unrelated legal issue.
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using the 
wrong-issue claim. Use the citation: {citation_text}""",

    PerturbationType.U2_OFF_TOPIC: """You are generating training data for a legal 
    citation mischaracterization detector.

Given a claim about a case, create a COMPLETELY OFF-TOPIC claim — one about an 
entirely different area of law that has nothing to do with the case. The claim 
should still sound like a legitimate legal proposition.

Example: If the case is about Fourth Amendment search and seizure, create a claim 
about contract formation, tax law, environmental regulation, or another unrelated 
area.

## Original claim (for context on the case's actual topic)
{claim}

## Case name
{case_name}

Respond with a JSON object:
{{"perturbed_claim": "...", "off_topic_area": "...", "passage": "..."}}

"perturbed_claim" is a plausible legal claim about an entirely unrelated area of law.
"off_topic_area" is the area of law the off-topic claim is about.
"passage" is a realistic 2-3 sentence brief excerpt that incorrectly cites the case for 
this off-topic claim. Use the citation: {citation_text}""",

    PerturbationType.U3_TANGENTIAL: """You are generating training data for a legal 
    citation mischaracterization detector.

Given a claim about a case, create a TANGENTIALLY RELATED claim — one that is in the 
same broad legal area but about a subtopic the case did not specifically address. 
The claim should be close enough that it seems plausible someone might cite this case, 
but far enough that the case doesn't actually support it.

Example: If the case held "the good-faith exception to the exclusionary rule applies 
when officers rely on a warrant," a tangential claim might be "the inevitable discovery 
doctrine permits admission of illegally obtained evidence" — related to the exclusionary 
rule but a different doctrine.

## Original claim (accurate)
{claim}

## Case name
{case_name}

## Relevant case text
{case_text_excerpt}

Respond with a JSON object:
{{"perturbed_claim": "...", "tangent_description": "...", "passage": "..."}}

"perturbed_claim" is a claim that is loosely related but not actually supported by the 
case.
"tangent_description" is a brief description of how the claim is tangentially related.
"passage" is a realistic 2-3 sentence brief excerpt that cites the case for this tangential 
claim. Use the citation: {citation_text}""",

    PerturbationType.C1_NEGATE: """You are generating training data for a legal citation 
    mischaracterization detector.

Given a claim that accurately describes a case's holding, DIRECTLY NEGATE it. The rewritten 
claim should state the exact opposite of what the case held.

Example: "The exclusionary rule applies to evidence obtained through illegal searches" → 
"The exclusionary rule does not apply to evidence obtained through illegal searches."

## Original claim (accurate)
{claim}

## Case name
{case_name}

Respond with a JSON object:
{{"perturbed_claim": "...", "negation_method": "...", "passage": "..."}}

"perturbed_claim" is the directly negated version of the original claim.
"negation_method" is a brief description of how the negation was applied.
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using 
the negated claim. 
Use the citation: {citation_text}""",

    PerturbationType.C2_REVERSE_WINNER: """You are generating training data for a 
    legal citation mischaracterization detector.

Given a claim that accurately describes a case's holding, REVERSE WHICH PARTY PREVAILED. 
If the case ruled in favor of the plaintiff, rewrite the claim to say the defendant 
prevailed, and vice versa. The legal reasoning should stay similar but the outcome flipped.

Example: "The court ruled that the employer violated Title VII by implementing a 
discriminatory hiring policy" → "The court ruled that the employer's hiring policy 
did not violate Title VII."

## Original claim (accurate)
{claim}

## Case name
{case_name}

Respond with a JSON object:
{{"perturbed_claim": "...", "reversal": "...", "passage": "..."}}

"perturbed_claim" is the claim with the outcome reversed.
"reversal" is a brief description of the reversal (e.g., "plaintiff win → defendant win").
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using the 
reversed-outcome claim. Use the citation: {citation_text}""",

    PerturbationType.C3_OPPOSITE_FACTORS: """You are generating training data for 
    a legal citation mischaracterization detector.

Given a claim that describes a case's holding in a multi-factor legal test, 
REVERSE WHICH FACTORS THE COURT FOUND SATISFIED OR UNSATISFIED. If the court found 
factor X weighed in favor, say it weighed against, and vice versa.

Example: "The court found that the first and third fair use factors weighed in the 
defendant's favor" → "The court found that the first and third fair use factors 
weighed against the defendant."

## Original claim (accurate)
{claim}

## Case name
{case_name}

Respond with a JSON object:
{{"perturbed_claim": "...", "reversed_factors": "...", "passage": "..."}}

"perturbed_claim" is the claim with factor outcomes reversed.
"reversed_factors" is a brief description of which factors were reversed.
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using the 
factor-reversed claim. Use the citation: {citation_text}""",

    PerturbationType.C4_HALLUCINATED_OPPOSITE: """You are generating training data 
    for a legal citation mischaracterization detector.

Given a claim that accurately describes a case's holding, FABRICATE AN OPPOSITE 
HOLDING that the case might plausibly have reached but did not. The fabricated 
holding should be the kind of thing a careless or dishonest brief writer might 
attribute to the case — it should sound authoritative and specific but be the 
opposite of what happened.

Example: If the case established that "reasonable suspicion is sufficient for a 
Terry stop," fabricate "the court held that probable cause is required for any 
investigatory detention, rejecting the reasonable suspicion standard."

## Original claim (accurate)
{claim}

## Case name
{case_name}

## Relevant case text
{case_text_excerpt}

Respond with a JSON object:
{{"perturbed_claim": "...", "fabrication_description": "...", "passage": "..."}}

"perturbed_claim" is the fabricated opposite holding.
"fabrication_description" is a brief description of what was fabricated.
"passage" is a realistic 2-3 sentence brief excerpt that cites the case using the 
fabricated claim. Use the citation: {citation_text}""",
}


# JSON schema for structured output from the LLM.
_PERTURBATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "perturbed_claim": {
            "type": "string",
            "description": "The perturbed (mischaracterized) version of the original "
            "claim.",
        },
        "passage": {
            "type": "string",
            "description": "A realistic brief excerpt using the perturbed claim with the "
            "citation.",
        },
    },
    "required": ["perturbed_claim", "passage"],
}


# Maximum characters of case text to include in the prompt.
# Keeps the prompt within reasonable context limits while giving
# the model enough context to produce realistic perturbations.
_MAX_CASE_TEXT_EXCERPT = 3000


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_perturbation(
    source: SourceExample,
    perturbation_type: PerturbationType,
    client: ModelClient,
) -> PerturbedExample | None:
    """
    Generate a single perturbed example from a source (claim, case) pair.

    Calls the LLM with the perturbation-type-specific prompt to transform
    the source's genuine claim into a mischaracterization of the specified
    type. The model routing (which client to use for which perturbation
    type) is handled upstream by the bulk generation script.

    Args:
        source: A genuine "entails" (claim, case) pair.
        perturbation_type: Which of the 11 perturbation types to apply.
        client: The LLM client to use for generation.

    Returns:
        A ``PerturbedExample`` with the transformed claim, target label,
        and synthetic passage, or None if generation failed (malformed
        output after retry).
    """
    target_label = PERTURBATION_TARGET_LABELS[perturbation_type]
    prompt = _build_perturbation_prompt(source, perturbation_type)

    log.info(
        "perturbation_generate",
        type=perturbation_type.value,
        source_id=source.example_id,
        model=client.name,
    )

    # First attempt.
    result = _call_and_parse(prompt, client)

    # Retry once on malformed output.
    if result is None:
        log.warning(
            "perturbation_retry",
            type=perturbation_type.value,
            source_id=source.example_id,
        )
        stricter_prompt = (
            prompt + "\n\nIMPORTANT: You must respond with valid JSON only. No other text."
        )
        result = _call_and_parse(stricter_prompt, client)

    if result is None:
        log.error(
            "perturbation_failed",
            type=perturbation_type.value,
            source_id=source.example_id,
        )

        return None

    perturbed_claim = result.get("perturbed_claim", "").strip()
    passage = result.get("passage", "").strip()

    if not perturbed_claim:
        log.warning(
            "perturbation_empty_claim",
            type=perturbation_type.value,
            source_id=source.example_id,
        )

        return None

    # If the model didn't generate a passage, create a minimal one.
    if not passage:
        passage = f"{perturbed_claim} {source.citation_text}."

    return PerturbedExample(
        example_id=str(uuid4()),
        source_example_id=source.example_id,
        perturbation_type=perturbation_type.value,
        target_label=target_label,
        original_claim=source.claim,
        perturbed_claim=perturbed_claim,
        case_name=source.case_name,
        citation_text=source.citation_text,
        passage=passage,
        decided_year=source.decided_year,
        metadata={
            k: v for k, v in result.items()
            if k not in ("perturbed_claim", "passage")
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_perturbation_prompt(
    source: SourceExample,
    perturbation_type: PerturbationType,
) -> str:
    """
    Build the prompt for a specific perturbation type and source example.

    Substitutes the source example's fields into the type-specific
    prompt template. Truncates case text to keep the prompt within
    reasonable context limits.

    Args:
        source: The source example providing claim and case text.
        perturbation_type: The perturbation type whose template to use.

    Returns:
        The complete prompt string.
    """
    template = _PERTURBATION_PROMPTS[perturbation_type]

    # Truncate case text excerpt to avoid blowing up the prompt.
    case_text_excerpt = source.case_text[:_MAX_CASE_TEXT_EXCERPT]
    if len(source.case_text) > _MAX_CASE_TEXT_EXCERPT:
        case_text_excerpt += "\n[... truncated ...]"

    return template.format(
        claim=source.claim,
        case_name=source.case_name,
        # Note that some perturbation prompts don't use case_text_excerpt in
        # their prompt templates. This still passes it in as an argument
        # regardless, but that's not an issue since unused keywords are just
        # ignored.
        case_text_excerpt=case_text_excerpt,
        citation_text=source.citation_text,
    )


def _call_and_parse(prompt: str, client: ModelClient) -> dict | None:
    """
    Call the model and parse the JSON response.

    Args:
        prompt: The perturbation prompt.
        client: The LLM client to call.

    Returns:
        The parsed dict, or None if the response isn't valid JSON.
    """
    try:
        response = client.generate(
            prompt=prompt,
            json_schema=_PERTURBATION_JSON_SCHEMA,
            temperature=0.7,  # Higher temperature for creative variation.
            max_tokens=1024,
        )

        return response.parsed_json
    except Exception:
        log.warning("perturbation_model_error", exc_info=True)

        return None
