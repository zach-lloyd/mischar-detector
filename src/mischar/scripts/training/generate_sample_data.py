"""
Generate a small sample training dataset for Modal dry-run testing.

Produces 21 train examples and 5 val examples covering both labels,
using ``build_classification_prompt()`` to generate realistic prompts.
The case text and claims are synthetic but structurally realistic.

Completions are label-only JSON (``{"label": "accurate"}`` or
``{"label": "mischaracterized"}``), matching the format produced by
``build_training_data.py``.

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
# Synthetic examples — (case_name, claim, case_excerpt, label) dicts.
# "accurate" examples state the holding correctly; "mischaracterized"
# examples overstate, drop qualifications, miss the topic, or contradict.
# ---------------------------------------------------------------------------

EXAMPLES = [
    # ---- accurate ----
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
        "label": "accurate",
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
        "label": "accurate",
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
        "label": "accurate",
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
        "label": "accurate",
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
        "label": "accurate",
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
        "label": "accurate",
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
        "label": "accurate",
    },
    {
        "case_name": "Hernandez v. Coastal Shipping Co.",
        "claim": (
            "A seaman injured in the course of employment may recover "
            "maintenance and cure regardless of fault."
        ),
        "case_excerpt": (
            "The doctrine of maintenance and cure entitles a seaman "
            "injured while in the service of his ship to food, lodging, "
            "and medical care, without regard to the negligence of the "
            "employer or the seaman's own contributory fault, short of "
            "willful misconduct."
        ),
        "label": "accurate",
    },
    {
        "case_name": "In re Estate of Donovan",
        "claim": (
            "A holographic will is valid only if the material "
            "provisions are in the testator's own handwriting."
        ),
        "case_excerpt": (
            "Under the governing statute, a holographic will may be "
            "admitted to probate if the signature and the material "
            "provisions are in the handwriting of the testator. Typed "
            "or preprinted text cannot supply the material terms."
        ),
        "label": "accurate",
    },
    {
        "case_name": "Nakamura v. Pacific Airlines",
        "claim": (
            "Under the Montreal Convention, a carrier is liable for "
            "passenger injuries caused by an accident on board an "
            "international flight."
        ),
        "case_excerpt": (
            "The Montreal Convention provides that the carrier is "
            "liable for damage sustained in case of death or bodily "
            "injury of a passenger upon condition only that the "
            "accident which caused the death or injury took place on "
            "board the aircraft or in the course of embarking or "
            "disembarking."
        ),
        "label": "accurate",
    },
    {
        "case_name": "Okafor v. City of Riverton",
        "claim": (
            "A municipality cannot be held liable under Section 1983 "
            "on a theory of respondeat superior."
        ),
        "case_excerpt": (
            "We reaffirm that a municipality may not be held liable "
            "under Section 1983 solely because it employs a tortfeasor. "
            "Liability attaches only when execution of the government's "
            "policy or custom inflicts the injury."
        ),
        "label": "accurate",
    },
    {
        "case_name": "Vance v. Greenfield Hospital",
        "claim": (
            "Expert testimony is required to establish the standard of "
            "care in a medical malpractice action except where the "
            "negligence is within the common knowledge of laypersons."
        ),
        "case_excerpt": (
            "In medical malpractice actions, the plaintiff must "
            "ordinarily present expert testimony to establish the "
            "applicable standard of care. An exception exists where "
            "the alleged negligence is so apparent that a layperson "
            "could recognize it without specialized knowledge, such as "
            "a surgical instrument left in the patient's body."
        ),
        "label": "accurate",
    },

    # ---- mischaracterized: drops qualifications / overstates ----
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
    },

    # ---- mischaracterized: case doesn't address the topic ----
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
    },

    # ---- mischaracterized: contradicts the holding ----
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
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
        "label": "mischaracterized",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Generate train.jsonl and val.jsonl from the synthetic examples.

    Splits the 26 examples into 21 train + 5 val, ensuring both labels
    appear in both splits. Builds prompts using the actual
    ``build_classification_prompt()`` function.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    by_label: dict[str, list[dict]] = {}
    for ex in EXAMPLES:
        by_label.setdefault(ex["label"], []).append(ex)

    train_examples = []
    val_examples = []

    # Take the last two accurate and last three mischaracterized examples
    # for val (5 total); the rest go to train (20 total). This guarantees
    # both labels appear in both splits.
    val_examples.extend(by_label["accurate"][-2:])
    train_examples.extend(by_label["accurate"][:-2])

    val_examples.extend(by_label["mischaracterized"][-3:])
    train_examples.extend(by_label["mischaracterized"][:-3])

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
    2. Formatting the target output as a label-only JSON string.

    Args:
        path: Output file path.
        examples: List of example dicts with case_name, claim,
            case_excerpt, and label.
    """
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            prompt = build_classification_prompt(
                claim=ex["claim"],
                retrieved_text=ex["case_excerpt"],
                case_name=ex["case_name"],
            )

            completion = json.dumps({"label": ex["label"]})

            record = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
