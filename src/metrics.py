"""Metric definitions, threshold selection and model ranking utilities."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer

from config import (
    DEFAULT_THRESHOLD,
    MIN_PRECISION,
    MIN_SPECIFICITY,
    RELAXED_MIN_PRECISION,
    RELAXED_MIN_SPECIFICITY,
)


def specificity_score(y_true, y_pred) -> float:
    """TN / (TN + FP); returns 0.0 when there are no observed negatives."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


specificity_scorer = make_scorer(specificity_score, greater_is_better=True)


def _positive_probabilities(y_prob) -> np.ndarray:
    probabilities = np.asarray(y_prob, dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] != 2:
            raise ValueError("Expected binary predict_proba output with two columns.")
        return probabilities[:, 1]
    return probabilities


def metrics_at_threshold(
    y_true, y_prob, threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, float]:
    """Calculate all requested classification metrics at a fixed threshold."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = _positive_probabilities(y_prob)
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = specificity_score(y_true, y_pred)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def select_threshold(
    y_true,
    y_prob,
    min_specificity: float = MIN_SPECIFICITY,
    min_precision: float = MIN_PRECISION,
) -> Dict:
    """Choose a recall-focused threshold from validation/OOF probabilities only.

    A candidate must satisfy pre-declared specificity and precision floors. The
    floors are relaxed in documented stages if no candidate qualifies. Among
    eligible thresholds we maximise recall, then F1, then specificity.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = _positive_probabilities(y_prob)
    candidates = np.unique(
        np.round(
            np.concatenate(
                [np.linspace(0.01, 0.99, 199), probabilities, [DEFAULT_THRESHOLD]]
            ),
            4,
        )
    )

    evaluated = [metrics_at_threshold(y_true, probabilities, t) for t in candidates]
    constraint_stages = [
        (min_specificity, min_precision),
        (RELAXED_MIN_SPECIFICITY, RELAXED_MIN_PRECISION),
        (0.30, 0.10),
        (0.0, 0.0),
    ]

    selected = None
    used_floor = None
    for spec_floor, prec_floor in constraint_stages:
        eligible = [
            item
            for item in evaluated
            if item["specificity"] >= spec_floor and item["precision"] >= prec_floor
        ]
        if eligible:
            selected = sorted(
                eligible,
                key=lambda item: (
                    -item["recall"],
                    -item["f1_score"],
                    -item["specificity"],
                    abs(item["threshold"] - DEFAULT_THRESHOLD),
                ),
            )[0]
            used_floor = {"specificity": spec_floor, "precision": prec_floor}
            break

    if selected is None:
        raise RuntimeError("No decision threshold could be selected.")

    default_metrics = metrics_at_threshold(y_true, probabilities, DEFAULT_THRESHOLD)
    selected["reason"] = (
        "Maximum validation recall among thresholds meeting the pre-declared "
        f"specificity >= {used_floor['specificity']:.2f} and precision >= "
        f"{used_floor['precision']:.2f} floors; ties broken by F1 and specificity."
    )
    selected["default_threshold_metrics"] = default_metrics
    selected["applied_constraint_floor"] = used_floor
    selected["recall_change_vs_default"] = round(
        selected["recall"] - default_metrics["recall"], 6
    )
    return selected


def choose_best_model(comparison: pd.DataFrame) -> Tuple[str, str]:
    """Rank models using recall first, then PR-AUC, F1 and negative-case quality."""
    data = comparison.copy()
    qualified = data[
        (data["specificity"] >= MIN_SPECIFICITY)
        & (data["precision"] >= MIN_PRECISION)
    ]
    reason = (
        "Models meeting specificity >= 0.50 and precision >= 0.20 were ranked by "
        "recall, then PR-AUC, F1-score, specificity and precision."
    )
    if qualified.empty:
        qualified = data[
            (data["specificity"] >= RELAXED_MIN_SPECIFICITY)
            & (data["precision"] >= RELAXED_MIN_PRECISION)
        ]
        reason = (
            "No model met the strict floors; models meeting specificity >= 0.40 "
            "and precision >= 0.15 were ranked by recall, PR-AUC, F1, specificity "
            "and precision."
        )
    if qualified.empty:
        qualified = data
        reason = "No model met any floor; ranked all models by recall and PR-AUC."

    ranked = qualified.sort_values(
        by=["recall", "pr_auc", "f1_score", "specificity", "precision"],
        ascending=False,
    )
    return str(ranked.iloc[0]["model"]), reason


def curve_coordinates(y_true, y_prob) -> Dict[str, List[float]]:
    """Compute compact ROC/PR coordinates for the saved dashboard artefacts."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    probabilities = _positive_probabilities(y_prob)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    return {
        "fpr": [round(float(value), 6) for value in fpr],
        "tpr": [round(float(value), 6) for value in tpr],
        "precision": [round(float(value), 6) for value in precision],
        "recall": [round(float(value), 6) for value in recall],
    }
