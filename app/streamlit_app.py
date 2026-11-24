"""Professional Streamlit interface for the saved CHD prediction pipeline.

The app only loads trained artefacts. It never trains or re-evaluates models.
"""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    COMPARISON_PATH,
    EVALUATION_PATH,
    FEATURE_LABELS,
    METADATA_PATH,
)
from predict import load_bundle, predict_one  # noqa: E402

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: #f6f9fc; }
        .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
        .hero {
            background: linear-gradient(135deg, #123c69 0%, #2e86ab 100%);
            color: white; padding: 1.6rem 2rem; border-radius: 14px;
            margin-bottom: 1.2rem;
        }
        .hero h1 { margin-bottom: 0.2rem; font-size: 2.1rem; }
        .hero p { margin-bottom: 0; opacity: 0.92; }
        .section-title { color: #123c69; margin-top: 0.4rem; }
        .card {
            background: white; border: 1px solid #dce6ef; border-radius: 12px;
            padding: 1rem; margin: 0.35rem 0; box-shadow: 0 2px 8px rgba(18,60,105,0.05);
        }
        .card .label { color: #5b7288; font-size: 0.82rem; }
        .card .value { color: #123c69; font-size: 1.4rem; font-weight: 700; }
        .footer-note { color: #5b7288; font-size: 0.82rem; margin-top: 1.5rem; }
        div[data-testid="stSidebarNav"] { background: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def get_evaluation_data():
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    comparison = __import__("pandas").read_csv(COMPARISON_PATH)
    return evaluation, metadata, comparison


def metric_card(label, value):
    st.markdown(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def require_artifacts():
    if not METADATA_PATH.exists() or not EVALUATION_PATH.exists():
        st.error(
            "Trained artefacts are missing. Run `python src/train_models.py` "
            "from the project root first."
        )
        st.stop()


def yes_no(label, help_text=""):
    return st.selectbox(label, ["No", "Yes"], help=help_text, index=0)


def render_home(metadata):
    st.markdown(
        """
        <div class="hero">
            <h1>Heart Disease Prediction</h1>
            <p>Machine Learning-Based 10-Year CHD Prediction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project overview")
    st.write(
        "This academic project compares five machine-learning algorithms that "
        "estimate whether a participant developed coronary heart disease (CHD) "
        "within ten years, using baseline health and demographic information "
        "from the Framingham Heart Study dataset."
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### What the system predicts")
        st.write(
            "- Binary outcome: `TenYearCHD = 1` means a recorded CHD event during "
            "the ten-year follow-up period.\n"
            "- `TenYearCHD = 0` means no recorded event during that period.\n"
            "- The output is a model-estimated probability and a threshold-based "
            "decision, not a clinical diagnosis."
        )
        st.markdown("#### Algorithms compared")
        st.write(
            "1. Logistic Regression\n"
            "2. Decision Tree Classifier\n"
            "3. Random Forest Classifier\n"
            "4. Support Vector Machine\n"
            "5. XGBoost Classifier"
        )
    with right:
        st.markdown("#### Dataset at a glance")
        info = metadata["dataset_information"]
        metric_card("Records", f"{info['n_records']:,}")
        metric_card("Input features", info["n_features"])
        metric_card("Positive cases", f"{info['positive_cases']} ({info['positive_class_proportion']:.1%})")
        metric_card("Selected model", metadata["selected_model"])
        metric_card("Decision threshold", f"{metadata['decision_threshold']:.3f}")

    st.markdown("### Priority: fewer false negatives")
    st.write(
        "The workflow selects a decision threshold from out-of-fold training "
        "predictions to improve recall while maintaining pre-declared specificity "
        "and precision floors. Higher recall can increase false positives; both "
        "effects are reported in the dashboard."
    )

    st.warning(
        "**Medical disclaimer:** This is an academic machine-learning "
        "demonstration based on a historical dataset. It is not a clinically "
        "validated risk score, does not diagnose heart disease, and is not a "
        "substitute for professional medical advice."
    )


def render_prediction(pipeline, metadata):
    st.markdown("## Prediction")
    st.write(
        "Enter baseline measurements for one participant. All 15 inputs match "
        "the training feature order and preprocessing pipeline."
    )

    with st.form("prediction_form"):
        st.markdown("#### Demographics and lifestyle")
        c1, c2, c3 = st.columns(3)
        with c1:
            male = yes_no("Sex", "Choose Male for code 1 or Female for code 0.")
            male_value = 1 if male == "Yes" else 0
        with c2:
            age = st.number_input(
                "Age (years)", min_value=32, max_value=70, value=49, step=1,
                help="The training dataset contains ages 32-70. Values outside this range are rejected.",
            )
        with c3:
            education_label = st.selectbox(
                "Education", ["Unknown / missing", "1", "2", "3", "4"]
            )
            education_value = 0 if education_label.startswith("Unknown") else int(education_label)

        c1, c2, c3 = st.columns(3)
        with c1:
            smoker = yes_no("Current smoker")
            current_smoker = 1 if smoker == "Yes" else 0
        with c2:
            cigs = st.number_input(
                "Cigarettes per day", min_value=0, max_value=70, value=0, step=1
            )
        with c3:
            bp_meds = yes_no("Blood pressure medication")
            bp_meds_value = 1 if bp_meds == "Yes" else 0

        st.markdown("#### Medical history")
        c1, c2, c3 = st.columns(3)
        with c1:
            stroke = yes_no("Previous stroke")
            stroke_value = 1 if stroke == "Yes" else 0
        with c2:
            hyp = yes_no("Hypertension")
            hyp_value = 1 if hyp == "Yes" else 0
        with c3:
            diabetes = yes_no("Diabetes")
            diabetes_value = 1 if diabetes == "Yes" else 0

        st.markdown("#### Clinical measurements")
        c1, c2, c3 = st.columns(3)
        with c1:
            tot_chol = st.number_input("Total cholesterol (mg/dL)", 107.0, 696.0, 235.0, 1.0)
        with c2:
            sys_bp = st.number_input("Systolic BP (mm Hg)", 83.5, 295.0, 132.0, 0.5)
        with c3:
            dia_bp = st.number_input("Diastolic BP (mm Hg)", 48.0, 142.5, 83.0, 0.5)

        c1, c2, c3 = st.columns(3)
        with c1:
            bmi = st.number_input("BMI (kg/m^2)", 15.54, 56.8, 25.8, 0.01)
        with c2:
            heart_rate = st.number_input("Heart rate (beats/min)", 44.0, 143.0, 76.0, 1.0)
        with c3:
            glucose = st.number_input("Blood glucose (mg/dL)", 40.0, 394.0, 82.0, 1.0)

        submitted = st.form_submit_button("Predict", width="stretch")

    if submitted:
        values = {
            "male": male_value,
            "age": age,
            "education": education_value,
            "currentSmoker": current_smoker,
            "cigsPerDay": cigs,
            "BPMeds": bp_meds_value,
            "prevalentStroke": stroke_value,
            "prevalentHyp": hyp_value,
            "diabetes": diabetes_value,
            "totChol": tot_chol,
            "sysBP": sys_bp,
            "diaBP": dia_bp,
            "BMI": bmi,
            "heartRate": heart_rate,
            "glucose": glucose,
        }
        try:
            result = predict_one(values, pipeline=pipeline, metadata=metadata)
        except ValueError as error:
            st.error(f"Invalid input: {error}")
            st.stop()

        st.markdown("### Prediction result")
        positive = result["predicted_class"] == 1
        if positive:
            st.error(f"**{result['outcome']}**")
        else:
            st.success(f"**{result['outcome']}**")

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Model-estimated probability", f"{result['probability']:.1%}")
        with c2:
            metric_card("Decision threshold", f"{result['threshold']:.3f}")
        with c3:
            metric_card("Model", result["selected_model"])

        st.info(result["interpretation"])
        st.write(
            "A positive decision means the model estimated a probability at or "
            "above the selected threshold. It does not mean the person definitely "
            "has or will develop heart disease."
        )
        st.warning(
            "**Disclaimer:** Academic ML demonstration only. Not a clinically "
            "validated risk score, diagnosis, or medical advice."
        )


def _bar_figure(comparison, column, title, yaxis):
    figure = go.Figure(
        go.Bar(
            x=comparison["model"],
            y=comparison[column],
            marker_color="#2E86AB",
            text=comparison[column].round(3),
            textposition="outside",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=yaxis,
        yaxis_range=[0, 1],
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=20, t=50, b=80),
    )
    return figure


def render_dashboard(evaluation, metadata, comparison):
    st.markdown("## Model Performance Dashboard")
    st.write(
        "All values are loaded from saved evaluation artefacts. Validation "
        "metrics come from out-of-fold training predictions at each model's "
        "selected threshold."
    )

    st.markdown("### Selected model")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Model", metadata["selected_model"])
    with c2:
        metric_card("Threshold", f"{metadata['decision_threshold']:.3f}")
    with c3:
        metric_card("CV folds", metadata["training_configuration"]["cv_folds"])

    st.markdown("### Validation metrics at selected threshold")
    validation = metadata["validation_metrics"]
    cols = st.columns(4)
    labels = [
        ("Recall", "recall"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
        ("F1-score", "f1_score"),
        ("Accuracy", "accuracy"),
        ("False Negative Rate", "false_negative_rate"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
    ]
    for index, (label, key) in enumerate(labels):
        with cols[index % 4]:
            metric_card(label, f"{validation[key]:.3f}")

    if metadata.get("final_test_metrics"):
        with st.expander("Final test-set metrics (evaluated once after selection)"):
            test = metadata["final_test_metrics"]
            t_cols = st.columns(4)
            for index, (label, key) in enumerate(labels):
                with t_cols[index % 4]:
                    metric_card(label, f"{test[key]:.3f}")
            st.caption(
                f"Confusion matrix: TN={test['true_negative']}, "
                f"FP={test['false_positive']}, FN={test['false_negative']}, "
                f"TP={test['true_positive']}"
            )

    st.markdown("### Five-model comparison")
    st.dataframe(comparison.round(4), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _bar_figure(comparison, "recall", "Recall by Model", "Recall"),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            _bar_figure(comparison, "false_negative_rate", "False Negative Rate", "FNR"),
            width="stretch",
        )
    with c1:
        st.plotly_chart(
            _bar_figure(comparison, "precision", "Precision by Model", "Precision"),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            _bar_figure(comparison, "f1_score", "F1-Score by Model", "F1-score"),
            width="stretch",
        )

    st.markdown("### ROC and Precision-Recall curves")
    selected = metadata["selected_model"]
    curves = evaluation["curves"]
    roc = go.Figure()
    for name in comparison["model"]:
        roc.add_trace(
            go.Scatter(
                x=curves[name]["fpr"],
                y=curves[name]["tpr"],
                mode="lines",
                name=name,
                line=dict(width=2),
            )
        )
    roc.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(dash="dash", color="grey", width=1),
        )
    )
    roc.update_layout(
        title="ROC Curves - Out-of-Fold Validation",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=460,
    )

    pr = go.Figure()
    for name in comparison["model"]:
        pr.add_trace(
            go.Scatter(
                x=curves[name]["recall"],
                y=curves[name]["precision"],
                mode="lines",
                name=name,
                line=dict(width=2),
            )
        )
    pr.update_layout(
        title="Precision-Recall Curves - Out-of-Fold Validation",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=460,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(roc, width="stretch")
    with c2:
        st.plotly_chart(pr, width="stretch")

    st.markdown("### Confusion matrix for the selected model")
    metrics = evaluation["validation_results"][selected]["validation_metrics"]
    matrix = [
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ]
    heatmap = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    heatmap.update_layout(
        title=f"{selected} - Validation Confusion Matrix",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(heatmap, width="stretch")

    st.info(metadata["model_selection_method"])
    st.caption(
        "Increasing recall can increase false positives. False negatives are "
        "reduced, not eliminated."
    )


def render_dataset(metadata):
    st.markdown("## About the Dataset")
    info = metadata["dataset_information"]
    st.write(
        "This project uses the provided Framingham Heart Study CSV. It contains "
        "baseline health and demographic measurements plus a binary ten-year "
        "coronary heart disease outcome."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Records", f"{info['n_records']:,}")
    with c2:
        metric_card("Predictor variables", info["n_features"])
    with c3:
        metric_card("Target", info["target_column"])

    st.markdown("### Class balance")
    st.write(
        f"Negative cases: {info['negative_cases']:,} | Positive cases: "
        f"{info['positive_cases']:,} ({info['positive_class_proportion']:.1%}). "
        "This imbalance is why recall, specificity, precision, ROC-AUC and "
        "PR-AUC are reported instead of accuracy alone."
    )

    st.markdown("### Missing-value handling")
    st.write(
        "No records are deleted. Numeric and binary fields use median/mode "
        "imputation fitted inside each training or cross-validation fold. "
        "Education is encoded as an ordered category with an explicit unknown "
        "level rather than treated as a continuous number."
    )
    missing_rows = [
        {"column": key, "missing_values": value}
        for key, value in info["missing_values"].items()
    ]
    st.dataframe(missing_rows, hide_index=True, width="stretch")

    st.markdown("### Limitations")
    st.write(
        "- The historical `TenYearCHD` outcome is not identical to the MDCalc "
        "Framingham Hard CHD Risk Score.\n"
        "- The model is validated only for participants aged 32-70 resembling "
        "this cohort.\n"
        "- Glucose and other missing values were imputed, which adds uncertainty.\n"
        "- The dataset represents a specific historical population and should "
        "not be treated as a general diagnostic tool for all heart disease.\n"
        "- The application is an academic demonstration, not a clinical product."
    )

    st.warning(
        "**Disclaimer:** Not a substitute for medical diagnosis or professional "
        "advice. Do not use this output to make healthcare decisions."
    )


def main():
    require_artifacts()
    pipeline, metadata = get_model_bundle()
    evaluation, loaded_metadata, comparison = get_evaluation_data()

    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Choose a section",
        ["Home / Overview", "Prediction", "Model Performance", "About Dataset"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Selected model: {metadata['selected_model']}")
    st.sidebar.caption(f"Threshold: {metadata['decision_threshold']:.3f}")

    if page == "Home / Overview":
        render_home(metadata)
    elif page == "Prediction":
        render_prediction(pipeline, metadata)
    elif page == "Model Performance":
        render_dashboard(evaluation, loaded_metadata, comparison)
    else:
        render_dataset(metadata)

    st.markdown(
        '<div class="footer-note">Heart Disease Prediction | Academic Computational '
        'Intelligence Project | Not for clinical use</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
