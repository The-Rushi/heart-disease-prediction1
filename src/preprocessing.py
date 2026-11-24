"""Dataset loading, validation and leakage-safe preprocessing pipelines."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    EDUCATION_CATEGORIES,
    EDUCATION_MISSING_CODE,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

NA_VALUES = ["NA", "N/A", "na", "NA ", "", "?", "null", "NULL"]


def load_dataset(path=None) -> pd.DataFrame:
    """Load the Framingham CSV and validate its schema."""
    if path is None:
        from config import DATA_PATH

        path = DATA_PATH

    frame = pd.read_csv(path, na_values=NA_VALUES)
    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = frame[expected_columns]
    invalid_target = set(frame[TARGET_COLUMN].dropna().unique()) - {0, 1}
    if invalid_target:
        raise ValueError(f"Target column contains unexpected values: {invalid_target}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")
    return frame


def separate_features_target(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the 15 predictor columns and the binary target."""
    X = frame[FEATURE_COLUMNS].copy()
    y = frame[TARGET_COLUMN].astype(int).copy()
    return X, y


def stratified_train_test_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create an untouched 20% test set while preserving class proportions."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Build a ColumnTransformer fitted only inside training/CV folds.

    Numeric: median imputation, optional standardisation.
    Binary: most-frequent imputation; values are already 0/1.
    Education: explicit unknown category plus one-hot encoding, avoiding the
    incorrect assumption that the four education codes are a continuous scale.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    binary_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    education_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=EDUCATION_MISSING_CODE,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    categories=EDUCATION_CATEGORIES,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("education", education_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def dataset_summary(frame: pd.DataFrame) -> dict:
    """Compute objective dataset facts used in metadata and the README."""
    counts = frame[TARGET_COLUMN].value_counts().to_dict()
    return {
        "n_records": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "n_features": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "negative_cases": int(counts.get(0, 0)),
        "positive_cases": int(counts.get(1, 0)),
        "positive_class_proportion": round(float(frame[TARGET_COLUMN].mean()), 6),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": {
            column: int(value)
            for column, value in frame.isna().sum().items()
            if value > 0
        },
        "age_range": [int(frame["age"].min()), int(frame["age"].max())],
    }
