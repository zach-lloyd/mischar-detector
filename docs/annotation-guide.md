# Annotation Guide

## Labels

**entails** — The brief's characterization of the cited case is accurate. The case does hold what the brief says it holds. Minor paraphrasing is fine; the substance must be correct.

**partially_supports** — The brief's characterization is *directionally* correct but overstates, omits a key qualification, or drops important context. The case is relevant to the proposition, but someone reading just the brief would come away with a misleading impression of what the case actually held.

**unrelated** — The case does not address the legal issue the brief cites it for. The brief's characterization isn't wrong in the sense of being *opposite* — the case simply has nothing to do with the claimed proposition.

**contradicts** — The case holds the opposite of what the brief claims. The brief says the case supports proposition X, but the case actually supports not-X (or explicitly rejects X).

## Decision tree

1. Does the case actually address the legal issue the brief cites it for?
   - **No** → **unrelated**
   - **Yes** → continue
2. Is the brief's characterization of the holding substantively accurate?
   - **Yes** → **entails**
   - **No** → continue
3. Is the brief's characterization directionally correct but misleading (overstated, missing qualifications, wrong scope)?
   - **Yes** → **partially_supports**
   - **No** → **contradicts**

## Boundary cases

**entails vs. partially_supports**: If the brief uses stronger language than the case ("must" vs. "may", "all contracts" vs. "employment contracts in New York"), that's partially_supports. If the brief paraphrases the holding in different words but preserves the substance, that's entails.

**partially_supports vs. contradicts**: Ask whether the case *leans toward* the brief's proposition. If the case supports a weaker version of the claim, that's partially_supports. If the case affirmatively rejects or holds the opposite, that's contradicts.

**unrelated vs. contradicts**: If the case addresses a completely different legal topic, that's unrelated — even if the case's actual holding happens to be inconsistent with the brief's claim. Contradicts requires that the case *addresses the same issue* but reaches the opposite conclusion.

## Fields to record

For each annotation, record a JSON object with these fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `passage` | string | yes | The text from the brief containing the mischaracterization. Include enough surrounding context to understand the claim being made (typically 2-4 sentences). |
| `citation_text_in_passage` | string | yes | The specific citation string within the passage (e.g., "Smith v. Jones, 123 F.3d 456 (9th Cir. 2001)"). |
| `citation_offset` | [int, int] | yes | Character offsets [start, end] of the citation within the passage. |
| `label` | string | yes | One of: `partially_supports`, `unrelated`, `contradicts`. |
| `gold_claim` | string | no | What the brief claims the case held, stated as a standalone proposition. Valuable but optional — skip if the brief doesn't make a clear, extractable claim. |
| `annotator_notes` | string | no | Your reasoning, especially for boundary cases. |
| `boundary_case` | bool | no | Set to `true` if you were genuinely uncertain between two labels. |
| `source` | object | yes | `{"recap_docket_id": "", "court": "...", "filing_date": "...", "document_url": "..."}`. For Charlotin entries, use the court from the CSV and the PDF link as document_url. `recap_docket_id` can be empty. |
| `cited_case` | object | yes | `{"courtlistener_id": "", "case_name": "...", "citation": "..."}`. `courtlistener_id` can be empty if you haven't resolved via CourtListener. |

## Charlotin-specific workflow

1. Open the Charlotin CSV. Filter to rows where "Hallucination Items" contains "Misrepresented: Case Law".
2. For each candidate, open the source document (PDF link in the Pointer or Source column).
3. In the court's order, find where it describes the misrepresentation. The court usually quotes or paraphrases the brief's claim and explains why it's wrong.
4. Extract the passage, citation, and label. Use the court's explanation to inform your label choice.
5. Append the annotation to `data/processed/annotated/charlotin.jsonl` — one JSON object per line, matching the fields above.

Note: The court documents describe the *error* but don't always quote the brief verbatim. When the court paraphrases rather than quotes, reconstruct the passage as faithfully as you can from the court's description and note this in `annotator_notes`.
