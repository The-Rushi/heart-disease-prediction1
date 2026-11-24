"""Train, tune, compare and persist the five CHD prediction models.

Run from the project root:
    python src/train_models.py

The test set is only used once, after the model and its threshold have been
selected from training data.
"""

import argparse
import json
import platform
from copy import deepcopy
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    DEFAULT_THRESHOLD,
    METADATA_PATH,
    MIN_PRECISION,
    MODELS_DIR,
    MIN_SPECIFICITY,
    OBSERVED_INPUT_RANGES,
    RANDOM_STATE,
)
from evaluate_models import build_comparison_table, save_evaluation_artifacts
from metrics import (
    choose_best_model,
    curve_coordinates,
    metrics_at_threshold,
    select_threshold,
    specificity_scorer,
)
from preprocessing import (
    dataset_summary,
    load_dataset,
    separate_features_target,
    stratified_train_test_split,
)

SCORING = {
    "recall": "recall",
    "precision": "precision",
    "specificity": specificity_scorer,
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def _trim_grid(grid: Dict, smoke_test: bool) -> Dict:
    """Keep one value per parameter for a fast end-to-end smoke test."""
    if not smoke_test:
        return grid
    return {
        key: [value[0]] if isinstance(value, (list, tuple)) else [value]
        for key, value in grid.items()
    }


def build_model_specs(y_train: pd.Series, smoke_test: bool = False) -> List[Dict]:
    """Define the five required algorithms and modest, reproducible grids."""
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    specs = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                solver="liblinear", max_iter=2000, random_state=RANDOM_STATE
            ),
            "grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
            "scale_numeric": True,
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "grid": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [3, 5, 8, None],
                "classifier__min_samples_split": [2, 10, 20],
                "classifier__min_samples_leaf": [1, 5, 10],
                "classifier__class_weight": [None, "balanced"],
            },
            "scale_numeric": False,
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 8, 12],
                "classifier__min_samples_split": [2, 10],
                "classifier__min_samples_leaf": [1, 4],
                "classifier__max_features": ["sqrt", 0.5],
                "classifier__class_weight": [None, "balanced"],
            },
            "scale_numeric": False,
        },
        {
            "name": "Support Vector Machine",
            "estimator": SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
            ),
            "grid": [
                {
                    "classifier__kernel": ["rbf"],
                    "classifier__C": [0.1, 1.0, 10.0],
                    "classifier__gamma": ["scale", 0.01, 0.1],
                    "classifier__class_weight": [None, "balanced"],
                },
                {
                    "classifier__kernel": ["linear"],
                    "classifier__C": [0.1, 1.0, 10.0],
                    "classifier__class_weight": [None, "balanced"],
                },
            ],
            "scale_numeric": True,
        },
        {
            "name": "XGBoost",
            "estimator": xgboost.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            "grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
                "classifier__scale_pos_weight": [1.0, round(positive_weight, 3)],
            },
            "scale_numeric": False,
        },
    ]

    for spec in specs:
        if isinstance(spec["grid"], list):
            spec["grid"] = [_trim_grid(grid, smoke_test) for grid in spec["grid"]]
        else:
            spec["grid"] = _trim_grid(spec["grid"], smoke_test)
    return specs


def _to_builtin(value):
    """Convert NumPy scalars and nested containers for JSON serialisation."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_training_pipeline(
    smoke_test: bool = False,
    evaluate_final_test: bool = True,
    save_artifacts: bool = True,
) -> Dict:
    """Execute the complete training and evaluation workflow."""
    frame = load_dataset()
    dataset_info = dataset_summary(frame)
    X, y = separate_features_target(frame)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)

    print(f"Loaded {len(frame)} records; positives: {int(y.sum())}")
    print(f"Train: {len(X_train)} | Test (untouched until final evaluation): {len(X_test)}")
    print(f"Train positive rate: {y_train.mean():.4f} | Test positive rate: {y_test.mean():.4f}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: Dict = {}
    curves: Dict = {}
    fitted_estimators: Dict = {}

    for spec in build_model_specs(y_train, smoke_test):
        name = spec["name"]
        print(f"\n[{name}] tuning with {CV_FOLDS}-fold stratified CV ...")
        pipeline = _make_pipeline(spec)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["grid"],
            scoring=SCORING,
            refit="recall",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        best_index = int(search.best_index_)
        best_cv_recall = float(search.cv_results_["mean_test_recall"][best_index])

        # Out-of-fold probabilities: every fold refits preprocessing and model.
        oof_probability = cross_val_predict(
            clone(search.best_estimator_),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        threshold_info = select_threshold(y_train, oof_probability)
        threshold = threshold_info["threshold"]
        validation_metrics = metrics_at_threshold(y_train, oof_probability, threshold)

        results[name] = {
            "validation_metrics": validation_metrics,
            "default_threshold_metrics": threshold_info["default_threshold_metrics"],
            "threshold_reason": threshold_info["reason"],
            "applied_constraint_floor": threshold_info["applied_constraint_floor"],
            "recall_change_vs_default": threshold_info["recall_change_vs_default"],
            "best_cv_mean_recall": best_cv_recall,
            "best_cv_mean_precision": float(
                search.cv_results_["mean_test_precision"][best_index]
            ),
            "best_cv_mean_specificity": float(
                search.cv_results_["mean_test_specificity"][best_index]
            ),
            "best_cv_mean_roc_auc": float(
                search.cv_results_["mean_test_roc_auc"][best_index]
            ),
            "best_cv_mean_pr_auc": float(
                search.cv_results_["mean_test_pr_auc"][best_index]
            ),
            "best_hyperparameters": _to_builtin(search.best_params_),
        }
        curves[name] = curve_coordinates(y_train, oof_probability)
        fitted_estimators[name] = search.best_estimator_

        print(
            f"  best CV recall={best_cv_recall:.3f} | selected threshold={threshold:.3f} "
            f"| OOF recall={validation_metrics['recall']:.3f} "
            f"| specificity={validation_metrics['specificity']:.3f}"
        )

    comparison = build_comparison_table(results)
    best_name, selection_reason = choose_best_model(comparison)
    best_result = results[best_name]
    threshold = best_result["validation_metrics"]["threshold"]
    best_pipeline = fitted_estimators[best_name]

    print(f"\nSelected model: {best_name}")
    print(f"Selection reason: {selection_reason}")

    final_test_metrics = None
    if evaluate_final_test:
        final_probability = best_pipeline.predict_proba(X_test)[:, 1]
        final_test_metrics = metrics_at_threshold(y_test, final_probability, threshold)
        print("\nFINAL TEST METRICS (evaluated once)")
        for key, value in final_test_metrics.items():
            print(f"  {key}: {value}")

    transformed_features = list(best_pipeline.named_steps["preprocessor"].get_feature_names_out())

    metadata = {
        "project_title": "Heart Disease Prediction",
        "subtitle": (
            "A Machine Learning Approach for 10-Year Coronary Heart Disease "
            "Prediction Using the Framingham Heart Study Dataset"
        ),
        "selected_model": best_name,
        "default_threshold": DEFAULT_THRESHOLD,
        "decision_threshold": float(threshold),
        "threshold_selection": {
            "data": "Out-of-fold predictions on the training set only",
            "strict_min_specificity": MIN_SPECIFICITY,
            "strict_min_precision": MIN_PRECISION,
            "selected_threshold": float(threshold),
            "default_threshold_metrics": best_result["default_threshold_metrics"],
            "reason": best_result["threshold_reason"],
            "applied_constraint_floor": best_result["applied_constraint_floor"],
            "recall_change_vs_default": best_result["recall_change_vs_default"],
            "recall_at_selected_threshold": best_result["validation_metrics"]["recall"],
            "specificity_at_selected_threshold": best_result["validation_metrics"][
                "specificity"
            ],
            "note": (
                "Increasing recall can increase false positives; false negatives "
                "are reduced, not eliminated."
            ),
        },
        "model_selection_method": selection_reason,
        "feature_names": list(X.columns),
        "feature_dtypes": {
            name: str(dtype) for name, dtype in X_train.dtypes.items()
        },
        "transformed_feature_names": transformed_features,
        "training_configuration": {
            "random_state": RANDOM_STATE,
            "test_size": 0.20,
            "stratified_split": True,
            "cross_validator": "StratifiedKFold",
            "cv_folds": CV_FOLDS,
            "search": "GridSearchCV",
            "primary_cv_metric": "recall",
            "monitored_cv_metrics": [
                "recall",
                "precision",
                "specificity",
                "roc_auc",
                "pr_auc",
            ],
            "preprocessing_fit_scope": "Inside each training/CV fold",
            "target_column": "TenYearCHD",
        },
        "selected_hyperparameters": best_result["best_hyperparameters"],
        "validation_metrics": best_result["validation_metrics"],
        "final_test_metrics": final_test_metrics,
        "dataset_information": dataset_info,
        "input_ranges": {
            name: [float(lower), float(upper)]
            for name, (lower, upper) in OBSERVED_INPUT_RANGES.items()
        },
        "age_validation": {
            "observed_min": 32,
            "observed_max": 70,
            "rule": "Reject any age outside the observed 32-70 range.",
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, BEST_MODEL_PATH)
        METADATA_PATH.write_text(
            json.dumps(_to_builtin(metadata), indent=2), encoding="utf-8"
        )
        save_evaluation_artifacts(
            results,
            curves,
            comparison,
            metadata,
            final_test_metrics=final_test_metrics,
        )
        print(f"\nSaved pipeline: {BEST_MODEL_PATH}")
        print(f"Saved metadata:  {METADATA_PATH}")
        print(f"Saved comparison and dashboard artefacts.")

    return {
        "comparison": comparison,
        "metadata": metadata,
        "results": results,
        "curves": curves,
        "final_test_metrics": final_test_metrics,
    }


def _make_pipeline(spec: Dict):
    """Combine fold-fitted preprocessing with the model under tuning."""
    from sklearn.pipeline import Pipeline

    from preprocessing import build_preprocessor

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(spec["scale_numeric"])),
            ("classifier", deepcopy(spec["estimator"])),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one hyperparameter combination per model without saving artefacts.",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Stop before evaluating the untouched test set.",
    )
    args = parser.parse_args()
    run_training_pipeline(
        smoke_test=args.smoke_test,
        evaluate_final_test=not args.skip_final_test,
        save_artifacts=not args.smoke_test,
    )
