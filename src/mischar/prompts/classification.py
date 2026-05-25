"""
Classification prompt template.

Generates prompts that instruct the LLM to classify the relationship between
a claimed legal proposition and the actual text of the cited case. This is
the core of Stage 5: does the case actually say what the brief claims it says?

Four labels are possible:
- **entails**: The case text fully supports the claimed proposition.
- **partially_supports**: The case addresses the topic but the claim is
  overstated, under-qualified, or missing important nuance.
- **unrelated**: The case doesn't address the topic of the claim at all.
- **contradicts**: The case says the opposite of what's claimed.
"""

from __future__ import annotations

# Included in cache keys so prompt changes invalidate cached classifications.
CLASSIFICATION_PROMPT_VERSION = "v1.0"

# JSON schema for structured output.
CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["entails", "partially_supports", "unrelated", "contradicts"],
            "description": "The relationship between the claim and the cited case text.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence from 0.0 to 1.0 in the assigned label.",
        },
        "supporting_text": {
            "type": "string",
            "description": "A brief explanation (1-3 sentences) citing specific language " +
                           "from the case that supports the label assignment.",
        },
    },
    "required": ["label", "confidence", "supporting_text"],
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
    3. Instructions for comparing them and assigning a label.

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
classify the relationship between the claim and what the case actually says.

## Labels

- **entails**: The case text fully supports the claim. The proposition is an accurate characterization 
of what the case held or established.
- **partially_supports**: The case addresses the same legal topic, but the claim is overstated, drops 
an important qualification, generalizes beyond the case's actual holding, or misses significant nuance. 
The claim is not wrong, but it's not fully accurate either.
- **unrelated**: The case text does not address the topic of the claim. The citation appears to be to 
the wrong case or the wrong part of the case.
- **contradicts**: The case text says the opposite of the claim, or the claim fundamentally misrepresents 
the case's holding.

## Important guidelines

1. Pay close attention to qualifications, limitations, and procedural posture. A claim that drops a key 
qualification (e.g., "the court held X" when the court actually held "X only when Y") should be classified 
as "partially_supports," not "entails."
2. Consider the strength of the language. If the case says "may" but the claim says "must," that's a 
meaningful difference.
3. If the excerpts don't contain enough information to evaluate the claim, classify as "unrelated" — the 
relevant portion of the case was not retrieved.
4. Base your classification only on the provided excerpts, not on your own knowledge of the case.

## Case excerpts from {case_name}

{retrieved_text}

## Claim to evaluate

"{claim}"

Respond with a JSON object containing "label", "confidence", and "supporting_text"."""
