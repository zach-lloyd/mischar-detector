"""
Evaluation metrics computation.

Computes macro F1, per-label precision/recall/F1, confusion matrix,
and abstention rate from a list of predictions and their corresponding
ground-truth labels. Abstentions are excluded from per-label metrics
but tracked separately in the abstention rate and reason breakdown.
"""

from __future__ import annotations

from collections import Counter

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

from mischar.constants import Label
from mischar.types import LabelMetrics, MetricsBundle, Prediction


def compute_metrics(
    predictions: list[Prediction],
    gold_labels: list[str],
) -> MetricsBundle:
    """
    Compute aggregate evaluation metrics from predictions and ground truth.

    Abstentions (predictions with ``predicted_label=None``) are excluded
    from precision/recall/F1 computation but counted in the abstention
    rate. This means metrics reflect only the pipeline's quality on
    examples it actually classified, while the abstention rate tells you
    how often it declined to classify.

    Args:
        predictions: Pipeline predictions, one per eval example. Order
            must match ``gold_labels``.
        gold_labels: Ground-truth labels, one per eval example. Must be
            one of the four valid ``Label`` values.

    Returns:
        A ``MetricsBundle`` with macro F1, per-label metrics, confusion
        matrix, and abstention statistics.
    """
    n_examples = len(predictions)

    if n_examples == 0:
        return _empty_metrics_bundle()

    # Separate abstentions from actual predictions.
    abstentions = [p for p in predictions if p.predicted_label is None]
    abstention_rate = len(abstentions) / n_examples

    # Count abstention reasons.
    abstention_by_reason: dict[str, int] = Counter(
        p.abstention_reason for p in abstentions if p.abstention_reason
    )

    # Filter to only non-abstained predictions for metric computation.
    classified_preds = []
    classified_golds = []
    for pred, gold in zip(predictions, gold_labels, strict=True):
        if pred.predicted_label is not None:
            classified_preds.append(pred.predicted_label)
            classified_golds.append(gold)

    # If everything abstained, return metrics with zero F1.
    if not classified_preds:
        return MetricsBundle(
            macro_f1=0.0,
            per_label={
                label: LabelMetrics(precision=0.0, recall=0.0, f1=0.0, support=0)
                       for label in Label.values()
            },
            confusion_matrix={},
            abstention_rate=abstention_rate,
            abstention_by_reason=dict(abstention_by_reason),
            n_examples=n_examples,
        )

    # Compute per-label and macro metrics using scikit-learn.
    label_names = Label.values()
    per_label = _compute_per_label_metrics(classified_golds, classified_preds, label_names)
    macro_f1 = _compute_macro_f1(classified_golds, classified_preds, label_names)
    cm = _build_confusion_matrix(classified_golds, classified_preds, label_names)

    return MetricsBundle(
        macro_f1=macro_f1,
        per_label=per_label,
        confusion_matrix=cm,
        abstention_rate=abstention_rate,
        abstention_by_reason=dict(abstention_by_reason),
        n_examples=n_examples,
    )


def _compute_per_label_metrics(
    y_true: list[str],
    y_pred: list[str],
    label_names: list[str],
) -> dict[str, LabelMetrics]:
    """
    Compute precision, recall, and F1 for each label.

    Uses scikit-learn with ``zero_division=0`` so labels absent from
    both predictions and ground truth get 0.0 rather than warnings.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: The ordered list of valid label values.

    Returns:
        A dict mapping each label name to its ``LabelMetrics``.
    """
    # Compute per-label scores. zero_division=0 avoids warnings when
    # a label has no predictions or no ground-truth examples.
    precisions = precision_score(
        y_true, y_pred, labels=label_names, average=None, zero_division=0
    )
    recalls = recall_score(
        y_true, y_pred, labels=label_names, average=None, zero_division=0
    )
    f1s = f1_score(
        y_true, y_pred, labels=label_names, average=None, zero_division=0
    )

    # Count support (number of ground-truth examples) per label.
    support_counts = Counter(y_true)

    per_label = {}
    for i, label in enumerate(label_names):
        per_label[label] = LabelMetrics(
            precision=float(precisions[i]),
            recall=float(recalls[i]),
            f1=float(f1s[i]),
            support=support_counts.get(label, 0),
        )

    return per_label


def _compute_macro_f1(
    y_true: list[str],
    y_pred: list[str],
    label_names: list[str],
) -> float:
    """
    Compute macro-averaged F1 across all labels.

    Macro averaging gives equal weight to each label regardless of
    support, which matters when label distributions are imbalanced
    (as they are here — entails examples outnumber contradicts).

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: The ordered list of valid label values.

    Returns:
        Macro F1 as a float between 0.0 and 1.0.
    """
    return float(f1_score(
        y_true, y_pred, labels=label_names, average="macro", zero_division=0
    ))


def _build_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    label_names: list[str],
) -> dict[tuple[str, str], int]:
    """
    Build a confusion matrix as a dict keyed by (true_label, predicted_label).

    This representation is more convenient for serialization and lookup
    than a 2D numpy array, especially since the labels are strings.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: The ordered list of valid label values.

    Returns:
        A dict mapping ``(true_label, predicted_label)`` to count.
    """
    cm_array = sk_confusion_matrix(y_true, y_pred, labels=label_names)
    cm_dict = {}

    for i, true_label in enumerate(label_names):
        for j, pred_label in enumerate(label_names):
            count = int(cm_array[i][j])
            if count > 0:
                cm_dict[(true_label, pred_label)] = count

    return cm_dict


def _empty_metrics_bundle() -> MetricsBundle:
    """
    Return a MetricsBundle for an empty evaluation run (zero examples).

    Args:
        None.

    Returns:
        A ``MetricsBundle`` with all metrics zeroed out.
    """
    return MetricsBundle(
        macro_f1=0.0,
        per_label={
            label: LabelMetrics(precision=0.0, recall=0.0, f1=0.0, support=0)
            for label in Label.values()
        },
        confusion_matrix={},
        abstention_rate=0.0,
        abstention_by_reason={},
        n_examples=0,
    )
