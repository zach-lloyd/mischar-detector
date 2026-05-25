"""
Attribution prompt template.

Generates prompts that instruct the LLM to extract the specific legal
proposition (claim) that a passage attributes to a cited case. This is
the core of Stage 3: given "Smith v. Jones held that X" in a brief,
we need to isolate "X" as a standalone claim.

The prompt uses structured JSON output to ensure consistent parsing.
"""

from __future__ import annotations

# The version is included in cache keys so that changing the prompt
# automatically invalidates cached attributions.
ATTRIBUTION_PROMPT_VERSION = "v1.0"

# JSON schema for structured output. The model must return a JSON object
# matching this structure. Backends that support schema enforcement
# (Ollama, Gemini) will constrain output to match; MLX appends a text
# instruction instead.
ATTRIBUTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {
            "type": "string",
            "description": "The specific legal proposition attributed to the cited case.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence from 0.0 to 1.0 that the passage makes a clear, " +
                           "specific claim about what the cited case held.",
        },
        "abstain": {
            "type": "boolean",
            "description": "True if the citation is a string cite, see-generally, or " +
                           "otherwise does not attribute a specific proposition.",
        },
        "abstain_reason": {
            "type": "string",
            "description": "If abstain is true, a brief explanation of why no claim could be extracted.",
        },
    },
    "required": ["claim", "confidence", "abstain"],
}


def build_attribution_prompt(passage: str, citation_text: str) -> str:
    """
    Build the attribution prompt for a (passage, citation) pair.

    The prompt asks the model to:
    1. Read the passage and identify the cited case.
    2. Extract the specific legal proposition the passage attributes to
       that case (the "claim").
    3. Express confidence in whether a clear claim exists.
    4. Abstain if the citation is a string cite, see-generally reference,
       or other context where no specific proposition is being asserted.

    Args:
        passage: The surrounding text from the legal brief.
        citation_text: The specific citation string within the passage.

    Returns:
        The complete prompt string.
    """
    return f"""You are a legal research assistant analyzing how a legal brief characterizes a cited case.

## Task

Read the passage below from a legal brief. It contains a citation to a case: {citation_text}

Your job is to extract the specific legal proposition (the "claim") that this passage 
attributes to the cited case. The claim should be a single, self-contained sentence 
describing what the passage says the case held, established, or stands for.

## Rules

1. The claim must be stated as a proposition, not a quotation from the passage. Rephrase it in neutral, 
declarative language.
2. If the passage says "Smith held that X," the claim is X (rephrased as a standalone proposition).
3. If the citation is a string cite (listed without discussion), a "see generally" reference, or otherwise 
does not attribute a specific holding or proposition to the case, set "abstain" to true.
4. Do not infer or add information not present in the passage. The claim must be grounded in what the 
passage actually says about the case.
5. Set confidence between 0.0 and 1.0 based on how clearly the passage attributes a specific proposition 
   to the cited case:
   - 0.9-1.0: Passage explicitly states what the case held (e.g., "Smith held that...")
   - 0.7-0.9: Passage clearly implies the case's holding (e.g., "Under Smith, ...")
   - 0.5-0.7: Passage references the case in support of a point but the specific attribution is ambiguous
   - Below 0.5: Abstain — the attribution is too unclear to extract a meaningful claim

## Passage

{passage}

## Citation to analyze

{citation_text}

Respond with a JSON object containing "claim", "confidence", "abstain", and optionally "abstain_reason"."""
