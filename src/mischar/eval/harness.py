"""
Evaluation harness.

Wraps the ``Pipeline`` to run it over evaluation datasets and collect
structured predictions. Handles the dual-evaluation flow for real_brief
examples. real_brief examples are hand annotated with a "gold claim", a 
human judgment of what the brief is asserting with respect to the cited case. 
In these cases, the evaluation harness runs once with the pipeline's attributed 
claim and once with the gold claim substituted in. The purpose of this is to determine, 
if the pipeline gets a classification wrong, whether that is because the attribution 
stage extracted a bad claim or because the classifier mislabeled a correctly extracted 
claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mischar.eval.metrics import compute_metrics
from mischar.logging import get_logger
from mischar.pipeline import Pipeline
from mischar.types import (
    Abstention,
    AttributedClaim,
    CitationResult,
    EvalExample,
    MetricsBundle,
    Prediction,
)

log = get_logger("eval.harness")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class EvalRun:
    """
    Results from evaluating a single (source, model) combination.

    Contains the per-example predictions and computed metrics. For
    real_brief sources, ``dual_eval`` holds the gold-claim evaluation
    alongside the normal attributed-claim evaluation.
    """

    source: str
    model_name: str
    predictions: list[Prediction]
    gold_labels: list[str]
    metrics: MetricsBundle
    dual_eval: DualEvalResult | None = None


@dataclass
class DualEvalResult:
    """
    Side-by-side results from attributed-claim vs gold-claim evaluation.

    Only produced for real_brief examples that have a ``gold_claim``
    field. Comparing the two MetricsBundles decomposes error into
    attribution error (claim extraction mistakes) vs classifier error
    (classification mistakes given a correct claim).
    """

    gold_claim_predictions: list[Prediction]
    gold_claim_metrics: MetricsBundle
    attributed_claim_predictions: list[Prediction]
    attributed_claim_metrics: MetricsBundle


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_evaluation(
    pipeline: Pipeline,
    dataset: list[EvalExample],
    *,
    run_dual_eval: bool = True,
) -> EvalRun:
    """
    Run the pipeline over all examples in an evaluation dataset.

    For each example, runs the full pipeline on the example's passage
    and matches the output to the example's citation. Collects
    predictions and computes aggregate metrics.

    For real_brief examples with gold claims, optionally runs a second
    pass that substitutes the gold claim (bypassing attribution) to
    isolate classifier error from attribution error.

    Args:
        pipeline: The initialized pipeline to evaluate.
        dataset: List of evaluation examples to process.
        run_dual_eval: If True and examples have gold claims, also run
            the gold-claim evaluation pass. Defaults to True.

    Returns:
        An ``EvalRun`` with predictions, metrics, and optional dual eval
        results.
    """
    if not dataset:
        log.warning("eval_empty_dataset")

        return EvalRun(
            source="unknown",
            model_name="unknown",
            predictions=[],
            gold_labels=[],
            metrics=compute_metrics([], []),
        )

    source = dataset[0].source
    model_name = pipeline._classifier_client.name

    log.info(
        "eval_run_start",
        source=source,
        model=model_name,
        n_examples=len(dataset),
    )

    # Normal evaluation: run full pipeline on each example.
    predictions = []
    gold_labels = []

    for example in dataset:
        prediction = _evaluate_example(pipeline, example, use_gold_claim=False)
        predictions.append(prediction)
        gold_labels.append(example.gold_label)

    metrics = compute_metrics(predictions, gold_labels)

    log.info(
        "eval_run_complete",
        source=source,
        model=model_name,
        macro_f1=round(metrics.macro_f1, 4),
        abstention_rate=round(metrics.abstention_rate, 4),
    )

    # Dual evaluation for real_brief examples with gold claims.
    dual_eval = None
    has_gold_claims = any(ex.gold_claim is not None for ex in dataset)

    if run_dual_eval and has_gold_claims:
        dual_eval = _run_dual_eval(pipeline, dataset, predictions, gold_labels)

    return EvalRun(
        source=source,
        model_name=model_name,
        predictions=predictions,
        gold_labels=gold_labels,
        metrics=metrics,
        dual_eval=dual_eval,
    )


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------


def _evaluate_example(
    pipeline: Pipeline,
    example: EvalExample,
    *,
    use_gold_claim: bool,
) -> Prediction:
    """
    Run the pipeline on a single eval example and produce a prediction.

    In normal mode (``use_gold_claim=False``), runs the full pipeline
    via ``process_passage`` and matches the result to the example's
    citation. In gold-claim mode, bypasses attribution and injects
    the gold claim directly.

    Args:
        pipeline: The pipeline to evaluate.
        example: The evaluation example to process.
        use_gold_claim: If True, skip attribution and use the example's
            gold claim. Requires ``example.gold_claim`` to be non-None.

    Returns:
        A ``Prediction`` with the pipeline's label (or abstention reason).
    """
    try:
        if use_gold_claim and example.gold_claim:
            return _evaluate_with_gold_claim(pipeline, example)

        return _evaluate_normal(pipeline, example)
    except Exception:
        # Infrastructure errors (network, model failures) are logged
        # and treated as evaluation-time failures, distinct from
        # pipeline abstentions. The eval continues; this example is
        # marked with a special abstention reason.
        log.error(
            "eval_example_error",
            example_id=example.example_id,
            exc_info=True,
        )

        return Prediction(
            example_id=example.example_id,
            predicted_label=None,
            abstention_reason="evaluation-error",
            confidence=None,
            used_gold_claim=use_gold_claim,
        )


def _evaluate_normal(pipeline: Pipeline, example: EvalExample) -> Prediction:
    """
    Run the full pipeline on an example and match the output citation.

    Calls ``process_passage`` and then finds the ``CitationResult``
    whose citation text best matches the example's ``citation_text``.

    Args:
        pipeline: The pipeline to evaluate.
        example: The evaluation example.

    Returns:
        A ``Prediction`` derived from the matched ``CitationResult``.
    """
    results = pipeline.process_passage(example.passage)

    if not results:
        # No citations parsed from the passage.
        return Prediction(
            example_id=example.example_id,
            predicted_label=None,
            abstention_reason="parsing-failed",
            confidence=None,
            used_gold_claim=False,
        )

    # Match the eval example's citation to one of the pipeline results.
    matched = _match_citation_result(example.citation_text, results)

    if matched is None:
        # The specific citation from the eval example wasn't found
        # in the pipeline output. This can happen if eyecite doesn't
        # parse it as a FullCaseCitation.
        log.warning(
            "eval_citation_not_matched",
            example_id=example.example_id,
            citation_text=example.citation_text,
        )

        return Prediction(
            example_id=example.example_id,
            predicted_label=None,
            abstention_reason="parsing-failed",
            confidence=None,
            used_gold_claim=False,
        )

    return _citation_result_to_prediction(matched, example.example_id, used_gold_claim=False)


def _evaluate_with_gold_claim(
    pipeline: Pipeline,
    example: EvalExample,
) -> Prediction:
    """
    Evaluate an example using the gold claim, bypassing attribution.

    Runs parse → resolve → (skip attribute, inject gold claim) →
    retrieve → classify. This isolates classifier error from
    attribution error for the dual-evaluation analysis.

    Args:
        pipeline: The pipeline to evaluate.
        example: The evaluation example (must have ``gold_claim`` set).

    Returns:
        A ``Prediction`` with ``used_gold_claim=True``.
    """
    # Stage 1: Parse.
    citations = pipeline.parse(example.passage)
    if not citations:
        return Prediction(
            example_id=example.example_id,
            predicted_label=None,
            abstention_reason="parsing-failed",
            confidence=None,
            used_gold_claim=True,
        )

    # Match to the eval example's citation.
    matched_citation = _match_parsed_citation(example.citation_text, citations)
    if matched_citation is None:
        return Prediction(
            example_id=example.example_id,
            predicted_label=None,
            abstention_reason="parsing-failed",
            confidence=None,
            used_gold_claim=True,
        )

    # Stage 2: Resolve.
    resolved = pipeline.resolve(matched_citation)
    if isinstance(resolved, Abstention):
        return Prediction(
            example_id=example.example_id,
            predicted_label=None,
            abstention_reason=resolved.reason,
            confidence=None,
            used_gold_claim=True,
        )

    # Stage 3: Skip attribution — inject gold claim directly.
    gold_claim = AttributedClaim(
        claim_text=example.gold_claim,
        confidence=1.0,
    )

    # Stage 4: Retrieve.
    retrieval = pipeline.retrieve(resolved, gold_claim)
    if isinstance(retrieval, Abstention):
        return Prediction(
            example_id=example.example_id,
            predicted_label=None,
            abstention_reason=retrieval.reason,
            confidence=None,
            used_gold_claim=True,
        )

    # Stage 5: Classify.
    classification = pipeline.classify(gold_claim, retrieval, resolved)

    return Prediction(
        example_id=example.example_id,
        predicted_label=classification.label,
        abstention_reason=None,
        confidence=classification.confidence,
        used_gold_claim=True,
    )


# ---------------------------------------------------------------------------
# Dual eval orchestration
# ---------------------------------------------------------------------------


def _run_dual_eval(
    pipeline: Pipeline,
    dataset: list[EvalExample],
    attributed_predictions: list[Prediction],
    gold_labels: list[str],
) -> DualEvalResult:
    """
    Run the gold-claim evaluation pass and combine with normal results.

    Only processes examples that have a ``gold_claim``. Examples without
    gold claims are included in both sets using their normal-pass
    prediction (so metrics are computed over the same example set).

    Args:
        pipeline: The pipeline to evaluate.
        dataset: The full evaluation dataset.
        attributed_predictions: Predictions from the normal (attributed
            claim) pass.
        gold_labels: Ground-truth labels matching the dataset.

    Returns:
        A ``DualEvalResult`` with metrics for both evaluation modes.
    """
    log.info("eval_dual_eval_start", n_examples=len(dataset))

    gold_claim_predictions = []
    for example, normal_pred in zip(dataset, attributed_predictions, strict=True):
        if example.gold_claim is not None:
            # Run with gold claim substituted.
            prediction = _evaluate_example(pipeline, example, use_gold_claim=True)
            gold_claim_predictions.append(prediction)
        else:
            # No gold claim available — reuse the normal prediction so
            # both sets have the same number of examples.
            gold_claim_predictions.append(normal_pred)

    gold_claim_metrics = compute_metrics(gold_claim_predictions, gold_labels)
    attributed_claim_metrics = compute_metrics(attributed_predictions, gold_labels)

    log.info(
        "eval_dual_eval_complete",
        gold_claim_f1=round(gold_claim_metrics.macro_f1, 4),
        attributed_claim_f1=round(attributed_claim_metrics.macro_f1, 4),
    )

    return DualEvalResult(
        gold_claim_predictions=gold_claim_predictions,
        gold_claim_metrics=gold_claim_metrics,
        attributed_claim_predictions=attributed_predictions,
        attributed_claim_metrics=attributed_claim_metrics,
    )


# ---------------------------------------------------------------------------
# Citation matching helpers
# ---------------------------------------------------------------------------


def _match_citation_result(
    citation_text: str,
    results: list[CitationResult],
) -> CitationResult | None:
    """
    Find the CitationResult whose citation best matches the given text.

    Tries exact match first, then falls back to substring containment.
    If multiple results match, returns the first (order matches passage
    position).

    Args:
        citation_text: The citation string from the eval example.
        results: The pipeline's output CitationResults.

    Returns:
        The matching ``CitationResult``, or None if no match found.
    """
    # Exact match on raw_text.
    for result in results:
        if result.citation.raw_text == citation_text:
            return result

    # Substring fallback — the eval example's citation_text might be
    # slightly different from what eyecite extracts (e.g., surrounding
    # punctuation differences).
    normalized_target = citation_text.lower().strip()
    for result in results:
        normalized_raw = result.citation.raw_text.lower().strip()

        if normalized_target in normalized_raw or normalized_raw in normalized_target:
            return result

    return None


def _match_parsed_citation(
    citation_text: str,
    citations: list,
) -> object | None:
    """
    Find the ParsedCitation that best matches the given citation text.

    Same matching logic as ``_match_citation_result`` but operates on
    ``ParsedCitation`` objects instead of ``CitationResult``.

    Args:
        citation_text: The citation string from the eval example.
        citations: The parsed citations from the passage.

    Returns:
        The matching ``ParsedCitation``, or None if no match found.
    """
    # Exact match.
    for cite in citations:
        if cite.raw_text == citation_text:
            return cite

    # Substring fallback.
    normalized_target = citation_text.lower().strip()
    for cite in citations:
        normalized_raw = cite.raw_text.lower().strip()
        
        if normalized_target in normalized_raw or normalized_raw in normalized_target:
            return cite

    return None


def _citation_result_to_prediction(
    result: CitationResult,
    example_id: str,
    *,
    used_gold_claim: bool,
) -> Prediction:
    """
    Convert a ``CitationResult`` into a ``Prediction`` for metrics.

    Args:
        result: The pipeline's output for a single citation.
        example_id: The eval example ID to attach to the prediction.
        used_gold_claim: Whether the gold claim was used (for dual eval
            tracking).

    Returns:
        A ``Prediction`` with label, confidence, and abstention info.
    """
    if result.abstained:
        return Prediction(
            example_id=example_id,
            predicted_label=None,
            abstention_reason=result.abstention.reason,
            confidence=None,
            used_gold_claim=used_gold_claim,
        )

    return Prediction(
        example_id=example_id,
        predicted_label=result.classification.label,
        abstention_reason=None,
        confidence=result.classification.confidence,
        used_gold_claim=used_gold_claim,
    )
