# Annotation Guide

## Labels

**accurate** — The brief's characterization of the cited case is accurate. The case does hold what the brief says it holds. Minor paraphrasing is fine; the substance must be correct.

**mischaracterized** — The brief's characterization misstates what the case held. This covers the full range of misstatement: overstating the holding or dropping a key qualification, citing the case for an issue it doesn't address at all, or claiming the opposite of what the case held.

## Decision tree

1. Does the case actually address the legal issue the brief cites it for?
   - **No** → **mischaracterized**
   - **Yes** → continue
2. Is the brief's characterization of the holding substantively accurate, including its qualifications, scope, and strength of language?
   - **Yes** → **accurate**
   - **No** → **mischaracterized**

## Boundary cases

**Paraphrase vs. overstatement**: If the brief uses stronger language than the case ("must" vs. "may", "all contracts" vs. "employment contracts in New York"), that's mischaracterized. If the brief paraphrases the holding in different words but preserves the substance, that's accurate.

**Directionally correct claims**: A claim that is *directionally* right but overstated, missing a qualification, or stated at the wrong scope is still mischaracterized — someone reading just the brief would come away with a misleading impression of what the case held. When you label one of these, note the nature of the misstatement (e.g., "dropped qualification", "overstated scope", "wrong issue", "opposite holding") in `annotator_notes` — this preserves the finer-grained information for error analysis even though the label is binary.

## Fields to record

For each annotation, record a JSON object with these fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `annotation_id` | string | yes | A stable unique ID for this annotation (e.g., `charlotin-0001`). Predictions are keyed by this ID, so it must never change once eval runs have used it — don't renumber when inserting or deleting records. |
| `passage` | string | yes | The text from the brief containing the mischaracterization. Include enough surrounding context to understand the claim being made (typically 2-4 sentences). |
| `citation_text_in_passage` | string | yes | The specific citation string within the passage (e.g., "Smith v. Jones, 123 F.3d 456 (9th Cir. 2001)"). |
| `label` | string | yes | One of: `accurate`, `mischaracterized`. |
| `gold_claim` | string | no | What the brief claims the case held, stated as a standalone proposition. Not currently tracked — passages are short enough that attribution error is unlikely to be a major factor. If present, it enables the harness's dual-evaluation pass. |
| `annotator_notes` | string | no | Your reasoning, especially for boundary cases. |
| `boundary_case` | bool | no | Set to `true` if you were genuinely uncertain between two labels. |
| `source` | object | yes | `{"recap_docket_id": "", "court": "...", "filing_date": "...", "document_url": "..."}`. For Charlotin entries, use the court from the CSV and the PDF link as document_url. `recap_docket_id` can be empty. |
| `cited_case` | object | yes | `{"courtlistener_id": "", "case_name": "...", "citation": "..."}`. `courtlistener_id` can be empty if you haven't resolved via CourtListener. |

### Example record

```json
{
  "annotation_id": "charlotin-0001",
  "passage": "Counsel cited Cooter & Gell to support a heightened Rule 11 standard for civil rights cases, arguing that courts must apply greater scrutiny before imposing sanctions on civil rights plaintiffs. See Cooter & Gell v. Hartmarx Corp., 496 U.S. 384 (1990).",
  "citation_text_in_passage": "Cooter & Gell v. Hartmarx Corp., 496 U.S. 384 (1990)",
  "label": "mischaracterized",
  "gold_claim": "Courts must apply a heightened Rule 11 standard before imposing sanctions on civil rights plaintiffs.",
  "annotator_notes": "Wrong issue. Cooter & Gell is an antitrust case about Rule 11 sanctions procedure; it does not address civil rights or a heightened standard. Passage reconstructed from court's description — brief was not quoted verbatim.",
  "boundary_case": false,
  "source": {
    "recap_docket_id": "",
    "court": "W.D. Oklahoma",
    "filing_date": "2026-05-21",
    "document_url": "https://www.damiencharlotin.com/documents/2175/Dalton_Gage_Hill_v._Oklahoma_County_Criminal_Justice_Authority_et_al._USA_21_May_2025.pdf"
  },
  "cited_case": {
    "courtlistener_id": "",
    "case_name": "Cooter & Gell v. Hartmarx Corp.",
    "citation": "496 U.S. 384 (1990)"
  }
}
```

## Charlotin-specific workflow

1. Open the Charlotin CSV. Filter to rows where "Hallucination Items" contains "Misrepresented: Case Law".
2. For each candidate, open the source document (PDF link in the Pointer or Source column).
3. In the court's order, find where it describes the misrepresentation. The court usually quotes or paraphrases the brief's claim and explains why it's wrong.
4. Extract the passage, citation, and label. Use the court's explanation to inform your label choice.
5. Append the annotation to `data/processed/annotated/charlotin.jsonl` — one JSON object per line, matching the fields above.

Note: The court documents describe the *error* but don't always quote the brief verbatim. When the court paraphrases rather than quotes, reconstruct the passage as faithfully as you can from the court's description and note this in `annotator_notes`.
