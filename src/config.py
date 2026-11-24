"""Shared configuration for the Heart Disease Prediction project.

All paths, column definitions, random seeds and validation rules live here so the
training code, evaluation code and Streamlit application use exactly the same
contract.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "framingham.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_RESULTS_DIR = REPORTS_DIR / "model_results"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
EVALUATION_PATH = MODELS_DIR / "evaluation_results.json"

TARGET_COLUMN = "TenYearCHD"

FEATURE_COLUMNS = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

BINARY_FEATURES = [
    "male",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
]

NUMERIC_FEATURES = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

CATEGORICAL_FEATURES = ["education"]

EDUCATION_LEVELS = [1, 2, 3, 4]
EDUCATION_MISSING_CODE = 0
EDUCATION_CATEGORIES = [[EDUCATION_MISSING_CODE, *EDUCATION_LEVELS]]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.50

MIN_SPECIFICITY = 0.50
MIN_PRECISION = 0.20
RELAXED_MIN_SPECIFICITY = 0.40
RELAXED_MIN_PRECISION = 0.15

OBSERVED_INPUT_RANGES = {
    "age": (32, 70),
    "cigsPerDay": (0, 70),
    "totChol": (107, 696),
    "sysBP": (83.5, 295),
    "diaBP": (48, 142.5),
    "BMI": (15.54, 56.8),
    "heartRate": (44, 143),
    "glucose": (40, 394),
}

FEATURE_LABELS = {
    "male": "Sex (1 = male, 0 = female)",
    "age": "Age (years)",
    "education": "Education level (1-4; 0 = unknown/missing)",
    "currentSmoker": "Current smoker (0 = no, 1 = yes)",
    "cigsPerDay": "Cigarettes per day",
    "BPMeds": "Blood pressure medication (0 = no, 1 = yes)",
    "prevalentStroke": "Previous stroke (0 = no, 1 = yes)",
    "prevalentHyp": "Hypertension (0 = no, 1 = yes)",
    "diabetes": "Diabetes (0 = no, 1 = yes)",
    "totChol": "Total cholesterol (mg/dL)",
    "sysBP": "Systolic blood pressure (mm Hg)",
    "diaBP": "Diastolic blood pressure (mm Hg)",
    "BMI": "Body Mass Index (kg/m^2)",
    "heartRate": "Heart rate (beats/min)",
    "glucose": "Blood glucose (mg/dL)",
}
