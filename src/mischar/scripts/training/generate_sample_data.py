"""
Generate a small sample training dataset for Modal dry-run testing.

Produces 20 train examples and 5 val examples covering all four labels,
using ``build_classification_prompt()`` to generate realistic prompts.
The case text and claims are synthetic but structurally realistic.

Usage:
    python -m mischar.scripts.training.generate_sample_data

Outputs:
    src/mischar/data/sample_data/train.jsonl
    src/mischar/data/sample_data/val.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from mischar.prompts.classification import build_classification_prompt

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("src/mischar/data/sample_data")

# ---------------------------------------------------------------------------
# Synthetic examples — (case_name, claim, case_excerpt, label, confidence,
# supporting_text) tuples grouped by label
# ---------------------------------------------------------------------------

EXAMPLES = [
    # ---- entails ----
    {
        "case_name": "Smith v. Johnson",
        "claim": (
            "A warrantless search of an automobile is permitted under "
            "the automobile exception when there is probable cause to "
            "believe the vehicle contains contraband."
        ),
        "case_excerpt": (
            "We hold that when law enforcement officers have probable "
            "cause to believe that a vehicle contains contraband or "
            "evidence of a crime, a warrantless search of the vehicle "
            "is permitted under the automobile exception to the Fourth "
            "Amendment warrant requirement. The mobility of vehicles "
            "and the reduced expectation of privacy therein justify "
            "this exception."
        ),
        "label": "entails",
        "confidence": 0.95,
        "supporting_text": (
            "The case directly holds that warrantless vehicle searches "
            "are permitted with probable cause under the automobile "
            "exception."
        ),
    },
    {
        "case_name": "Garcia v. State Board of Education",
        "claim": (
            "Public school students retain their First Amendment rights "
            "to free speech while on school grounds."
        ),
        "case_excerpt": (
            "It can hardly be argued that either students or teachers "
            "shed their constitutional rights to freedom of speech or "
            "expression at the schoolhouse gate. Students in public "
            "schools retain their First Amendment protections, subject "
            "to reasonable limitations necessary to maintain order and "
            "discipline."
        ),
        "label": "entails",
        "confidence": 0.92,
        "supporting_text": (
            "The court explicitly states that students retain First "
            "Amendment rights at school."
        ),
    },
    {
        "case_name": "Thornton v. United States",
        "claim": (
            "The exclusionary rule requires suppression of evidence "
            "obtained through an unconstitutional search."
        ),
        "case_excerpt": (
            "Evidence obtained in violation of the Fourth Amendment "
            "must be excluded from the prosecution's case-in-chief. "
            "The exclusionary rule serves to deter unlawful police "
            "conduct by removing the incentive to disregard "
            "constitutional protections."
        ),
        "label": "entails",
        "confidence": 0.93,
        "supporting_text": (
            "The case states that evidence from unconstitutional "
            "searches must be excluded."
        ),
    },
    {
        "case_name": "Patel v. Commissioner of Internal Revenue",
        "claim": (
            "Business expenses are deductible under Section 162 only "
            "if they are ordinary and necessary."
        ),
        "case_excerpt": (
            "Under Section 162 of the Internal Revenue Code, a "
            "taxpayer may deduct all the ordinary and necessary "
            "expenses paid or incurred during the taxable year in "
            "carrying on any trade or business. Both conditions — "
            "ordinary and necessary — must be satisfied."
        ),
        "label": "entails",
        "confidence": 0.97,
        "supporting_text": (
            "The case confirms both the ordinary and necessary "
            "requirements for Section 162 deductions."
        ),
    },
    {
        "case_name": "Rivera v. Metro Transit Authority",
        "claim": (
            "A common carrier owes the highest duty of care to its "
            "passengers."
        ),
        "case_excerpt": (
            "Common carriers are held to the highest degree of care "
            "toward their passengers. This duty requires the carrier "
            "to exercise extraordinary vigilance to protect passengers "
            "from harm during transit."
        ),
        "label": "entails",
        "confidence": 0.94,
        "supporting_text": (
            "The case explicitly establishes the highest duty of care "
            "standard for common carriers."
        ),
    },
    {
        "case_name": "Williams v. Consolidated Industries",
        "claim": (
            "Employers are strictly liable for workplace injuries "
            "under the workers' compensation system."
        ),
        "case_excerpt": (
            "The workers' compensation framework imposes strict "
            "liability on employers for injuries arising out of and "
            "in the course of employment, regardless of fault. This "
            "no-fault system provides the exclusive remedy for covered "
            "workplace injuries."
        ),
        "label": "entails",
        "confidence": 0.96,
        "supporting_text": (
            "The case confirms strict liability under workers' "
            "compensation for workplace injuries."
        ),
    },

    # ---- partially_supports ----
    {
        "case_name": "Anderson v. Liberty Lobby",
        "claim": (
            "Summary judgment must be granted whenever the moving "
            "party presents evidence in its favor."
        ),
        "case_excerpt": (
            "Summary judgment is appropriate only when the moving "
            "party demonstrates that there is no genuine dispute as "
            "to any material fact and the movant is entitled to "
            "judgment as a matter of law. The court must view the "
            "evidence in the light most favorable to the nonmoving "
            "party and draw all reasonable inferences in that party's "
            "favor."
        ),
        "label": "partially_supports",
        "confidence": 0.85,
        "supporting_text": (
            "The case addresses summary judgment but requires absence "
            "of genuine factual disputes, not merely evidence in the "
            "movant's favor. The claim drops the 'no genuine dispute' "
            "qualification."
        ),
    },
    {
        "case_name": "Chen v. Pacific Insurance Co.",
        "claim": (
            "Insurance contracts must always be construed against "
            "the insurer."
        ),
        "case_excerpt": (
            "When the language of an insurance policy is ambiguous, "
            "it is to be construed against the insurer who drafted "
            "the policy. However, when the terms are clear and "
            "unambiguous, the court must enforce the policy as "
            "written without resort to rules of construction."
        ),
        "label": "partially_supports",
        "confidence": 0.88,
        "supporting_text": (
            "The case only applies contra proferentem when policy "
            "language is ambiguous. The claim drops the ambiguity "
            "prerequisite, making it overbroad."
        ),
    },
    {
        "case_name": "Douglas v. State Personnel Board",
        "claim": (
            "Government employees cannot be terminated without a "
            "full evidentiary hearing."
        ),
        "case_excerpt": (
            "A public employee with a property interest in continued "
            "employment is entitled to due process before termination. "
            "At minimum, due process requires notice of the charges, "
            "an explanation of the evidence, and an opportunity to "
            "respond. A full evidentiary hearing is required only in "
            "cases where the initial procedures are inadequate to "
            "resolve the factual disputes."
        ),
        "label": "partially_supports",
        "confidence": 0.82,
        "supporting_text": (
            "The case requires due process but only mandates a full "
            "hearing when initial procedures are inadequate. The "
            "claim inflates this to all terminations."
        ),
    },
    {
        "case_name": "Baxter v. Regional Medical Center",
        "claim": (
            "Hospitals are liable for the negligence of all "
            "physicians who practice within their facilities."
        ),
        "case_excerpt": (
            "A hospital may be held vicariously liable for the "
            "negligence of physicians who are its employees or agents. "
            "However, liability does not extend to independent "
            "contractor physicians who merely hold staff privileges "
            "at the hospital, unless the patient reasonably believed "
            "the physician was an employee of the hospital."
        ),
        "label": "partially_supports",
        "confidence": 0.80,
        "supporting_text": (
            "The case limits hospital liability to employed or "
            "apparent-agent physicians. The claim overgeneralizes "
            "to all physicians."
        ),
    },
    {
        "case_name": "Morrison v. National Bank",
        "claim": (
            "Banks must verify the identity of every person who "
            "enters the premises."
        ),
        "case_excerpt": (
            "Financial institutions must implement reasonable "
            "customer identification procedures for account opening "
            "and significant transactions, as required by the Bank "
            "Secrecy Act. These procedures include verifying the "
            "identity of any person seeking to open an account."
        ),
        "label": "partially_supports",
        "confidence": 0.78,
        "supporting_text": (
            "The case requires identity verification for account "
            "opening, not for every person entering the premises. "
            "The claim vastly overstates the scope."
        ),
    },

    # ---- unrelated ----
    {
        "case_name": "Katz v. United States",
        "claim": (
            "Landlords must provide tenants with 60 days notice "
            "before increasing rent."
        ),
        "case_excerpt": (
            "The Fourth Amendment protects people, not places. What "
            "a person knowingly exposes to the public, even in his "
            "own home or office, is not a subject of Fourth Amendment "
            "protection. But what he seeks to preserve as private, "
            "even in an area accessible to the public, may be "
            "constitutionally protected."
        ),
        "label": "unrelated",
        "confidence": 0.97,
        "supporting_text": (
            "The case addresses Fourth Amendment privacy protections. "
            "It has nothing to do with landlord-tenant law or rent "
            "increases."
        ),
    },
    {
        "case_name": "Fletcher v. Peck",
        "claim": (
            "Environmental impact assessments are required before "
            "any federal construction project."
        ),
        "case_excerpt": (
            "The question presented is whether a state legislature "
            "can repeal a grant of land made by a previous "
            "legislature. We hold that the Contract Clause of the "
            "Constitution prohibits states from passing laws that "
            "impair the obligation of contracts, including grants "
            "of land."
        ),
        "label": "unrelated",
        "confidence": 0.98,
        "supporting_text": (
            "The case is about the Contract Clause and state land "
            "grants. Environmental assessments are not discussed."
        ),
    },
    {
        "case_name": "Harper v. Virginia Board of Elections",
        "claim": (
            "Corporations are required to hold annual shareholder "
            "meetings."
        ),
        "case_excerpt": (
            "We conclude that a state violates the Equal Protection "
            "Clause of the Fourteenth Amendment whenever it makes "
            "the affluence of the voter or payment of any fee an "
            "electoral standard. Voter qualifications have no "
            "relation to wealth."
        ),
        "label": "unrelated",
        "confidence": 0.99,
        "supporting_text": (
            "The case addresses voting rights and poll taxes. "
            "Corporate governance is entirely unrelated."
        ),
    },
    {
        "case_name": "Wickard v. Filburn",
        "claim": (
            "Physicians must obtain board certification before "
            "performing surgical procedures."
        ),
        "case_excerpt": (
            "Even if an individual's activity is local and may not "
            "be regarded as commerce, it may still be reached by "
            "Congress if it exerts a substantial economic effect on "
            "interstate commerce. The appellee's own consumption of "
            "wheat, though trivial by itself, contributes to the "
            "overall supply and demand of the commodity."
        ),
        "label": "unrelated",
        "confidence": 0.98,
        "supporting_text": (
            "The case concerns the Commerce Clause and agricultural "
            "regulation. Medical licensing is not addressed."
        ),
    },
    {
        "case_name": "Marbury v. Madison",
        "claim": (
            "Tenants have an implied warranty of habitability in "
            "residential leases."
        ),
        "case_excerpt": (
            "It is emphatically the province and duty of the judicial "
            "department to say what the law is. Those who apply the "
            "rule to particular cases must of necessity expound and "
            "interpret that rule. If two laws conflict with each "
            "other, the courts must decide on the operation of each."
        ),
        "label": "unrelated",
        "confidence": 0.99,
        "supporting_text": (
            "The case establishes judicial review. It does not "
            "address landlord-tenant law or habitability."
        ),
    },

    # ---- contradicts ----
    {
        "case_name": "Miranda v. Arizona",
        "claim": (
            "Police are not required to inform suspects of their "
            "rights before custodial interrogation."
        ),
        "case_excerpt": (
            "The prosecution may not use statements stemming from "
            "custodial interrogation of the defendant unless it "
            "demonstrates the use of procedural safeguards effective "
            "to secure the privilege against self-incrimination. "
            "Prior to any questioning, the person must be warned that "
            "he has a right to remain silent, that any statement he "
            "does make may be used as evidence against him, and that "
            "he has a right to the presence of an attorney."
        ),
        "label": "contradicts",
        "confidence": 0.98,
        "supporting_text": (
            "The case holds the exact opposite — police must inform "
            "suspects of their rights before custodial interrogation."
        ),
    },
    {
        "case_name": "Brown v. Board of Education",
        "claim": (
            "Racially segregated public schools satisfy the Equal "
            "Protection Clause."
        ),
        "case_excerpt": (
            "We conclude that in the field of public education the "
            "doctrine of separate but equal has no place. Separate "
            "educational facilities are inherently unequal. The "
            "plaintiffs are deprived of the equal protection of the "
            "laws guaranteed by the Fourteenth Amendment."
        ),
        "label": "contradicts",
        "confidence": 0.99,
        "supporting_text": (
            "The case explicitly holds that segregated schools are "
            "inherently unequal and violate equal protection."
        ),
    },
    {
        "case_name": "Gideon v. Wainwright",
        "claim": (
            "Indigent defendants in felony cases do not have a "
            "constitutional right to appointed counsel."
        ),
        "case_excerpt": (
            "Any person hauled into court who is too poor to hire a "
            "lawyer cannot be assured a fair trial unless counsel is "
            "provided for him. The Sixth Amendment's guarantee of "
            "counsel is fundamental and essential to a fair trial, "
            "and is made obligatory upon the states by the Fourteenth "
            "Amendment."
        ),
        "label": "contradicts",
        "confidence": 0.97,
        "supporting_text": (
            "The case establishes that indigent defendants have a "
            "constitutional right to counsel — the opposite of the "
            "claim."
        ),
    },
    {
        "case_name": "New York Times Co. v. Sullivan",
        "claim": (
            "Public officials can recover damages for defamation "
            "without proving actual malice."
        ),
        "case_excerpt": (
            "The constitutional guarantees of free speech require a "
            "federal rule that prohibits a public official from "
            "recovering damages for a defamatory falsehood relating "
            "to his official conduct unless he proves that the "
            "statement was made with actual malice — that is, with "
            "knowledge that it was false or with reckless disregard "
            "of whether it was false or not."
        ),
        "label": "contradicts",
        "confidence": 0.96,
        "supporting_text": (
            "The case requires proof of actual malice for public "
            "official defamation claims — directly contradicting "
            "the claim."
        ),
    },
    {
        "case_name": "Mapp v. Ohio",
        "claim": (
            "States are free to admit evidence obtained through "
            "illegal searches in their own courts."
        ),
        "case_excerpt": (
            "We hold that all evidence obtained by searches and "
            "seizures in violation of the Constitution is, by that "
            "same authority, inadmissible in a state court. The "
            "exclusionary rule, which bars the use of illegally "
            "obtained evidence, is an essential part of the Fourth "
            "and Fourteenth Amendments."
        ),
        "label": "contradicts",
        "confidence": 0.97,
        "supporting_text": (
            "The case holds that illegally obtained evidence is "
            "inadmissible in state courts, contradicting the claim."
        ),
    },
    {
        "case_name": "Lopez v. Federal Housing Authority",
        "claim": (
            "The Fair Housing Act prohibits discrimination in the "
            "sale or rental of housing based on race."
        ),
        "case_excerpt": (
            "Congress enacted the Fair Housing Act to provide, within "
            "constitutional limitations, for fair housing throughout "
            "the United States. The Act prohibits discrimination in "
            "the sale, rental, and financing of dwellings on the "
            "basis of race, color, religion, sex, or national origin."
        ),
        "label": "entails",
        "confidence": 0.95,
        "supporting_text": (
            "The case directly confirms the Fair Housing Act's "
            "prohibition on race-based housing discrimination."
        ),
    },
    {
        "case_name": "Taylor v. City of Springfield",
        "claim": (
            "Municipal governments are always liable for injuries "
            "caused by negligent road maintenance."
        ),
        "case_excerpt": (
            "A municipality may be held liable for injuries caused "
            "by its failure to maintain roadways in a reasonably "
            "safe condition, provided the plaintiff demonstrates that "
            "the municipality had actual or constructive notice of "
            "the dangerous condition and a reasonable opportunity to "
            "remedy it."
        ),
        "label": "partially_supports",
        "confidence": 0.83,
        "supporting_text": (
            "The case allows municipal liability but requires notice "
            "and opportunity to remedy. The claim drops these "
            "conditions by saying 'always liable'."
        ),
    },
    {
        "case_name": "United States v. Curtiss-Wright Export Corp.",
        "claim": (
            "Zoning regulations must comply with the Americans with "
            "Disabilities Act."
        ),
        "case_excerpt": (
            "The President possesses plenary and exclusive power in "
            "the field of international relations, a power which does "
            "not require as a basis for its exercise an act of "
            "Congress. The broad statement that the federal government "
            "can exercise no powers except those specifically "
            "enumerated in the Constitution has only a limited truth."
        ),
        "label": "unrelated",
        "confidence": 0.98,
        "supporting_text": (
            "The case addresses presidential foreign affairs power. "
            "Zoning and disability law are not discussed."
        ),
    },
    {
        "case_name": "Roe v. Wade",
        "claim": (
            "States have unrestricted authority to prohibit abortion "
            "at any stage of pregnancy."
        ),
        "case_excerpt": (
            "The right of personal privacy includes the abortion "
            "decision, but this right is not unqualified and must be "
            "considered against important state interests in "
            "regulation. During the first trimester, the abortion "
            "decision must be left to the medical judgment of the "
            "pregnant woman's attending physician."
        ),
        "label": "contradicts",
        "confidence": 0.95,
        "supporting_text": (
            "The case holds that states cannot prohibit abortion "
            "without restriction, especially in the first trimester "
            "— contradicting the claim of unrestricted authority."
        ),
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Generate train.jsonl and val.jsonl from the synthetic examples.

    Splits examples into 20 train + 5 val, ensuring at least one of
    each label appears in val. Builds prompts using the actual
    ``build_classification_prompt()`` function.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Split: take the last example of each label group for val (5 total),
    # the rest go to train (20 total). This guarantees all four labels
    # appear in both splits.
    by_label: dict[str, list[dict]] = {}
    for ex in EXAMPLES:
        by_label.setdefault(ex["label"], []).append(ex)

    train_examples = []
    val_examples = []

    for label, group in by_label.items():
        # Last example of each label goes to val.
        val_examples.append(group[-1])

        # The rest go to train.
        train_examples.extend(group[:-1])

    # Val has 4 examples (one per label). Add one more to reach 5
    val_examples.append(train_examples.pop(0))

    print(f"Train: {len(train_examples)} examples")
    print(f"Val: {len(val_examples)} examples")

    # Build JSONL records using the actual prompt template.
    _write_jsonl(DATA_DIR / "train.jsonl", train_examples)
    _write_jsonl(DATA_DIR / "val.jsonl", val_examples)

    print(f"Written to {DATA_DIR / 'train.jsonl'} and {DATA_DIR / 'val.jsonl'}")


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    """
    Write examples to a JSONL file with prompt and completion fields.

    Each example is converted to a training record by:
    1. Building the classification prompt via ``build_classification_prompt()``.
    2. Formatting the target output as a JSON string.

    Args:
        path: Output file path.
        examples: List of example dicts with case_name, claim,
            case_excerpt, label, confidence, supporting_text.
    """
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            prompt = build_classification_prompt(
                claim=ex["claim"],
                retrieved_text=ex["case_excerpt"],
                case_name=ex["case_name"],
            )

            completion = json.dumps({
                "label": ex["label"],
                "confidence": ex["confidence"],
                "supporting_text": ex["supporting_text"],
            })

            record = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
