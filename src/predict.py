"""Load the saved pipeline and make validated single-record predictions.

This module never trains a model. The Streamlit application imports these
functions so the website and training pipeline share identical preprocessing.
"""

import argparse
import json
from typing import Dict

import joblib
import pandas as pd

from config import (
    BEST_MODEL_PATH,
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    METADATA_PATH,
    NUMERIC_FEATURES,
)


def load_bundle(model_path=None, metadata_path=None):
    """Load the fitted preprocessing+model pipeline and its metadata."""
    model_path = model_path or BEST_MODEL_PATH
    metadata_path = metadata_path or METADATA_PATH
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Trained artefacts not found. Run `python src/train_models.py` first."
        )
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return pipeline, metadata


def validate_input(values: Dict, metadata: Dict) -> pd.DataFrame:
    """Validate all 15 inputs and return a one-row DataFrame in training order."""
    missing = [name for name in FEATURE_COLUMNS if name not in values]
    if missing:
        raise ValueError(f"Missing input features: {missing}")

    row = {}
    for name in FEATURE_COLUMNS:
        value = values[name]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            if name == "education":
                value = 0
            else:
                raise ValueError(f"Missing value for required feature: {name}")
        row[name] = value

    for name in BINARY_FEATURES:
        if row[name] not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1.")

    education = row["education"]
    if education not in (0, 1, 2, 3, 4):
        raise ValueError("education must be 0 (unknown) or one of 1, 2, 3, 4.")

    age = row["age"]
    age_rule = metadata.get("age_validation", {})
    age_min = int(age_rule.get("observed_min", 32))
    age_max = int(age_rule.get("observed_max", 70))
    if not isinstance(age, (int, float)) or isinstance(age, bool):
        raise ValueError("age must be numeric.")
    if not age_min <= age <= age_max:
        raise ValueError(
            f"age must be between {age_min} and {age_max}; the model has not been "
            "validated outside the observed dataset age range."
        )

    input_ranges = metadata.get("input_ranges", {})
    for name in NUMERIC_FEATURES:
        value = row[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be numeric.")
        if name in input_ranges:
            lower, upper = input_ranges[name]
            if not float(lower) <= float(value) <= float(upper):
                raise ValueError(
                    f"{name} must be between {lower} and {upper} "
                    "(observed training-data range)."
                )

    features = pd.DataFrame(
        [[row[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS
    )
    # Match the exact dtypes seen during training so fitted imputers accept the row.
    for name, dtype in metadata.get("feature_dtypes", {}).items():
        features[name] = features[name].astype(dtype)
    return features


def predict_one(values: Dict, pipeline=None, metadata=None) -> Dict:
    """Return a neutral, threshold-based prediction for one validated record."""
    if pipeline is None or metadata is None:
        pipeline, metadata = load_bundle()

    features = validate_input(values, metadata)
    probability = float(pipeline.predict_proba(features)[0, 1])
    threshold = float(metadata["decision_threshold"])
    predicted_class = int(probability >= threshold)

    if predicted_class == 1:
        outcome = "Model predicts a 10-year CHD outcome."
    else:
        outcome = "Model does not predict a 10-year CHD outcome at this threshold."

    return {
        "predicted_class": predicted_class,
        "probability": probability,
        "threshold": threshold,
        "outcome": outcome,
        "selected_model": metadata["selected_model"],
        "interpretation": (
            f"Model-estimated probability: {probability:.1%}. The decision "
            f"threshold is {threshold:.2f}. This is an academic machine-learning "
            "estimate, not a clinically validated risk score or diagnosis."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict from a JSON object of 15 features.")
    parser.add_argument("--json", required=True, help="JSON object containing all 15 features.")
    args = parser.parse_args()
    result = predict_one(json.loads(args.json))
    print(json.dumps(result, indent=2))
