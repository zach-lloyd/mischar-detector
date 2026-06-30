# Fine-Tuned Legal Mischaracterization Detector

This repo implements a fine-tuned version of Gemma 3 12B that exhibits improved performance at classifying accurate vs mischaracterized case law citations in real-world legal briefs.

## Headline Result

Fine-tuning Gemma 3 12B on accurate and mischaracterized legal holdings generated from the CaseHOLD dataset resulted in a ~20% improvement in macro F1 score, from 0.69 to 0.83.

<p align="center">
  <img src="assets/results-real-brief.svg" alt="Fine-tuned vs. prompted baseline on real briefs" width="720">
</p>

## Problem

LLM hallucinations in legal filings are a growing problem that even the most prestigious law firms have fallen victim to (see, e.g., [Sullivan & Cromwell law firm apologizes for AI 'hallucinations' in court filing](https://www.reuters.com/legal/litigation/sullivan-cromwell-law-firm-apologizes-ai-hallucinations-court-filing-2026-04-21/)). These hallucinations manifest themselves in a few different ways. Sometimes, the LLM will cite to a case or statute that simply does not exist at all. Other times, the LLM might provide a fabricated quote from a real case. However, arguably the most pernicious category of legal hallucinations is mischaracterizations: when the LLM cites to a real legal authority, does not fabricate a quote, but cites the authority in support of a proposition or claim that the authority actually does not support. Consider the following two citations to Roe v. Wade:

1. In Roe v. Wade, 410 U.S. 113, the Supreme Court established the trimester framework for determining the extent to which states could regulate abortion.

2. The Supreme Court has held that states may not prohibit a woman's access to an abortion at any point during her pregnancy, but that they may regulate access to abortion in ways reasonably related to maternal health. Roe v. Wade, 410 U.S. 113.

The first is accurate, but the second is not fully supported by the holding in Roe. 

These types of hallucinations are especially pernicious because they can be so hard to detect. For those not intimately familiar with a cited case, determining whether the citation mischaracterizes the case's holding requires locating the court's opinion and carefully reviewing it against the citation, a process that can quickly become cumbersome and inefficient for judges charged with reviewing numerous briefs, each containing numerous citations.

## Potential Solution

Train an LLM on accurate and mischaracterized descriptions of legal holdings and use it to classify citations as either "Accurate" or "Mischaracterized". Through fine-tuning, an open-source model can be trained on domain-specific data, with the goal of improving its performance on a specific task. Here, I train Gemma 3 12B on a dataset of accurate and mischaracterized legal holdings sourced from the CaseHOLD dataset and then test whether the fine-tuned model outperforms the base model at classifying citations as "Accurate" or "Mischaracterized".

I chose Gemma 3 12B for this project because it is a high-quality open-weight model that, because it is developed by an American company (Alphabet/Google), does not present as many significant security issues as Chinese open-weight models like Qwen and DeepSeek. It is also built on Gemini technology and has both larger and smaller variants, thus presenting future opportunities to compare performance of small vs medium vs large vs frontier-sized variants within the same family of models.

## Pipeline Architecture

1. As input, take a citation to a legal case and a claim purportedly supported by the cited case.

2. Parse the citation using Eyecite.

3. Resolve the cited case using CourtListener's API.
   a. If the case cannot be resolved, abstain from classifying the claim.

4. Chunk and embed the text of the case.

5. Retrieve the top-K chunks most relevant to the claim.

6. Based on the retrieved chunks, classify the claim as "Accurate" or "Mischaracterized".

```mermaid
flowchart TB
  subgraph infer["Inference pipeline"]
    direction TB
    A["Brief claim + cited case"] --> B["Parse citation<br/>eyecite"]
    B --> C["Resolve case<br/>CourtListener API"]
    C --> D["Retrieve evidence<br/>chunk · embed · top-k"]
    D --> E["Classify<br/>baseline vs fine-tuned Gemma 3 12B"]
    E --> F["Verdict<br/>accurate / mischaracterized + excerpt"]
    B -. no parseable citation .-> X(["Abstain"])
    C -. case not found .-> X
  end
  subgraph train["Offline fine-tuning · Modal"]
    direction TB
    T1["CaseHOLD pairs<br/>accurate / mischaracterized"] --> T2["Build training data<br/>resolve · retrieve · label"]
    T2 --> T3["QLoRA fine-tune<br/>Gemma 3 12B"]
    T3 --> T4["LoRA adapter"]
  end
  T4 -. loaded by .-> E
```

## Data

**Training/Evaluation Data:** 6,894 examples sourced from 3,447 entries from the CaseHOLD dataset. For each entry, one "Accurate" example is created using the correct holding, and one "Mischaracterized" example is created by randomly choosing one of the incorrect holdings (each CaseHOLD entry is in multiple choice format). The examples are split 85/15 between the training and evaluation sets. Each pair derived from a CaseHOLD entry is assigned to one set; pairs are never split between the training and evaluation sets.

**Test Data:** 189 examples sourced from real-world briefs. For this, I used Damien Charlotin's database of AI hallucination cases. I compiled 104 examples of mischaracterizations, manually verifying and annotating each one. I then reviewed the filings available in Charlotin's database and pulled 85 accurate citations from those filings (manually verifying their accuracy by reviewing each claim against the text of the cited case).

**Temporal Split:** One concern is that the model may have memorized older cases during pretraining. To try to mitigate this, when building my test set I primarily focused on the most recent examples in Charlotin's database, resulting in a test set with examples that are largely from the second half of 2025 and first half of 2026 and thus less likely to be contained in model pretraining data.

## Future Work/Improvements

- Compare fine-tuned Gemma 3 12B to Gemma 3 4B, Gemma 3 27B, and Gemini 3.1 Pro.
- Replace Gemma 3 models with the newly-released Gemma 4 models to see whether the result still holds.
- Analyze and revise real brief test set to reduce abstention rate.
- Expand size fo real brief test set for more robust performance analysis.
  

