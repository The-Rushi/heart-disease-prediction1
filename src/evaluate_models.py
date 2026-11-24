"""Evaluation tables, comparison charts and serialised dashboard artefacts."""

import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config import COMPARISON_PATH, EVALUATION_PATH, FIGURES_DIR, MODEL_RESULTS_DIR


def build_comparison_table(validation_results: Dict) -> pd.DataFrame:
    """Build the five-model comparison table from validation/OOF predictions."""
    rows = []
    for model_name, result in validation_results.items():
        metrics = result["validation_metrics"]
        rows.append(
            {
                "model": model_name,
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1_score": metrics["f1_score"],
                "false_negative_rate": metrics["false_negative_rate"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "cv_best_mean_recall": result["best_cv_mean_recall"],
            }
        )
    return pd.DataFrame(rows)


def _bar_chart(
    comparison: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    values = comparison[column].astype(float)
    ax.bar(comparison["model"], values, color="#2E86AB")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, float(values.max()) * 1.15))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    for index, value in enumerate(values):
        ax.text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def plot_metric_comparisons(comparison: pd.DataFrame) -> None:
    """Generate the required metric comparison charts."""
    charts = [
        ("accuracy", "Accuracy by Model", "accuracy_comparison.png", "Accuracy"),
        ("precision", "Precision by Model", "precision_comparison.png", "Precision"),
        ("recall", "Recall / Sensitivity by Model", "recall_comparison.png", "Recall"),
        ("f1_score", "F1-Score by Model", "f1_comparison.png", "F1-score"),
        (
            "false_negative_rate",
            "False Negative Rate by Model",
            "false_negative_rate_comparison.png",
            "False Negative Rate",
        ),
    ]
    for column, title, filename, ylabel in charts:
        _bar_chart(comparison, column, title, filename, ylabel)


def plot_roc_curves(curves: Dict, comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for _, row in comparison.iterrows():
        name = row["model"]
        ax.plot(curves[name]["fpr"], curves[name]["tpr"], lw=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Out-of-Fold Validation", weight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=160)
    plt.close(fig)


def plot_precision_recall_curves(curves: Dict, comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for _, row in comparison.iterrows():
        name = row["model"]
        ax.plot(curves[name]["recall"], curves[name]["precision"], lw=2, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - Out-of-Fold Validation", weight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=160)
    plt.close(fig)


def plot_confusion_matrices(validation_results: Dict, comparison: pd.DataFrame) -> None:
    names = comparison["model"].tolist()
    fig, axes = plt.subplots(1, len(names), figsize=(17, 3.8))
    for ax, name in zip(axes, names):
        metrics = validation_results[name]["validation_metrics"]
        matrix = np.array(
            [
                [metrics["true_negative"], metrics["false_positive"]],
                [metrics["false_negative"], metrics["true_positive"]],
            ]
        )
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["0", "1"]
        )
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=10, weight="bold")
    fig.suptitle("Confusion Matrices at Selected Thresholds", weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def save_evaluation_artifacts(
    validation_results: Dict,
    curves: Dict,
    comparison: pd.DataFrame,
    metadata: Dict,
    final_test_metrics: Dict | None = None,
) -> None:
    """Persist every metric and curve used by the Streamlit dashboard."""
    for directory in (FIGURES_DIR, MODEL_RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(COMPARISON_PATH, index=False)
    plot_metric_comparisons(comparison)
    plot_roc_curves(curves, comparison)
    plot_precision_recall_curves(curves, comparison)
    plot_confusion_matrices(validation_results, comparison)

    serialisable = {
        "selected_model": metadata["selected_model"],
        "decision_threshold": metadata["decision_threshold"],
        "model_selection_reason": metadata["model_selection_method"],
        "validation_results": validation_results,
        "curves": curves,
        "final_test_metrics": final_test_metrics,
    }
    EVALUATION_PATH.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    lines = [
        "HEART DISEASE PREDICTION - VALIDATION RESULTS",
        "=" * 55,
        "",
        f"Selected model: {metadata['selected_model']}",
        f"Selected threshold: {metadata['decision_threshold']:.4f}",
        "",
        comparison.to_string(index=False),
        "",
        "Metrics are computed from out-of-fold predictions on the training set.",
        "The final test set is evaluated once, after model and threshold selection.",
    ]
    if final_test_metrics:
        lines.extend(
            [
                "",
                "FINAL TEST METRICS",
                "-" * 20,
                *[f"{key}: {value}" for key, value in final_test_metrics.items()],
            ]
        )
    (MODEL_RESULTS_DIR / "validation_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
