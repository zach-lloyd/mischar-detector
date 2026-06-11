"""
Classification prompt template.

Generates prompts that instruct the LLM to classify the relationship between
a claimed legal proposition and the actual text of the cited case. This is
the core of Stage 5: does the case actually say what the brief claims it says?

Two labels are possible:
- **accurate**: The case text supports the claimed proposition as stated.
- **mischaracterized**: The claim misstates what the case held — it
  overstates the holding, drops a key qualification, addresses a topic
  the case didn't reach, or contradicts the case outright.
"""

from __future__ import annotations

# Included in cache keys so prompt changes invalidate cached classifications.
# v2.0: switched from the four-label scheme to binary accurate/mischaracterized.
CLASSIFICATION_PROMPT_VERSION = "v2.0"

# JSON schema for structured output.
# Only "label" is required. The fine-tuned models are trained on label-only
# completions, so confidence and supporting_text are optional extras that
# prompted (non-tuned) models may include. The classify stage applies
# defaults when they're absent.
CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["accurate", "mischaracterized"],
            "description": "Whether the claim accurately characterizes the cited case.",
        },
        "confidence": {
            "type": "number",
            "description": "Optional confidence from 0.0 to 1.0 in the assigned label.",
        },
        "supporting_text": {
            "type": "string",
            "description": "Optional brief explanation (1-3 sentences) citing specific "
                           "language from the case that supports the label assignment.",
        },
    },
    "required": ["label"],
}


def build_classification_prompt(
    claim: str,
    retrieved_text: str,
    case_name: str,
) -> str:
    """
    Build the classification prompt.

    The prompt presents the model with:
    1. The claim (what the brief says the case held).
    2. Relevant excerpts from the actual case text (retrieved chunks).
    3. Instructions for comparing them and assigning a binary label.

    Args:
        claim: The proposition attributed to the case (from Stage 3).
        retrieved_text: Concatenated relevant chunks from the case
            opinion (from Stage 4).
        case_name: Name of the cited case for context.

    Returns:
        The complete prompt string.
    """
    return f"""You are a legal research assistant evaluating whether a legal brief
accurately characterizes a cited case.

## Task

A legal brief claims that the case **{case_name}** stands for the following proposition:

**Claim:** "{claim}"

Below are relevant excerpts from the actual opinion in {case_name}. Based on these excerpts,
decide whether the claim accurately characterizes what the case actually says.

## Labels

- **accurate**: The case text supports the claim as stated. The proposition is an accurate
characterization of what the case held or established.
- **mischaracterized**: The claim misstates what the case held. This includes claims that
overstate the holding, drop an important qualification or limitation, generalize beyond the
case's actual scope, address a legal issue the case did not reach, or state the opposite of
what the case held.

## Important guidelines

1. Pay close attention to qualifications, limitations, and procedural posture. A claim that
drops a key qualification (e.g., "the court held X" when the court actually held "X only
when Y") is mischaracterized.
2. Consider the strength of the language. If the case says "may" but the claim says "must,"
that's a meaningful difference and the claim is mischaracterized.
3. If the excerpts do not address the topic of the claim at all, the claim is
mischaracterized — the case does not support it.
4. Base your classification only on the provided excerpts, not on your own knowledge of
the case.

## Case excerpts from {case_name}

{retrieved_text}

## Claim to evaluate

"{claim}"

Respond with a JSON object containing "label" — either "accurate" or "mischaracterized"."""
