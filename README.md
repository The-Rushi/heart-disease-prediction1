# Heart Disease Prediction

## Machine Learning-Based 10-Year CHD Prediction

**Subtitle:** A Machine Learning Approach for 10-Year Coronary Heart Disease Prediction Using the Framingham Heart Study Dataset

This is a complete academic project that explores the Framingham dataset, preprocesses it without data leakage, tunes and compares five classifiers, reduces false negatives with a validation-selected decision threshold, and serves the saved model through a professional Streamlit website.

### Important scope note

The target `TenYearCHD` is a historical ten-year coronary heart disease outcome from this dataset. It is **not** a general diagnostic system for heart attacks, heart failure or all cardiovascular diseases, and it is **not identical** to the MDCalc Framingham Hard CHD Risk Score.

## 1. Problem statement

Given 15 baseline demographic and health variables, predict whether a participant recorded a CHD event during a ten-year follow-up period. Because missed positive cases are costly in screening scenarios, the primary modelling objective is to improve **recall / sensitivity** while maintaining reasonable **specificity** and **precision**. Accuracy alone is not used for model selection.

## 2. Dataset description

The provided `framingham.csv` is stored at `data/framingham.csv` and is never replaced.

| Property | Value |
|---|---|
| Records | 4,240 |
| Columns | 16 |
| Predictors | 15 |
| Target | `TenYearCHD` |
| Negative cases | 3,596 |
| Positive cases | 644 |
| Positive proportion | ~15.2% |
| Age range | 32-70 years |
| Exact duplicate rows | 0 |

Missing values found in the file: `education` 105, `cigsPerDay` 29, `BPMeds` 53, `totChol` 50, `BMI` 19, `heartRate` 1, `glucose` 388.

## 3. Project structure

```text
heart-disease-prediction/
├── data/
│   └── framingham.csv
├── notebooks/
│   └── heart_disease_analysis.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── metrics.py
│   └── predict.py
├── models/
│   ├── best_model.joblib
│   ├── model_metadata.json
│   ├── model_comparison.csv
│   └── evaluation_results.json
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── figures/
│   └── model_results/
├── requirements.txt
├── README.md
└── .gitignore
```

## 4. Technologies used

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly, Streamlit, Joblib, Jupyter and ipykernel.

## 5. Installation

Run from a terminal in the project folder:

```bash
cd ~/heart-disease-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd $HOME\heart-disease-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The XGBoost wheel is large; installation may take several minutes on a slow connection. If activation is not desired, `python3 -m pip install --user -r requirements.txt` also works.

## 6. How to run the notebook

```bash
cd ~/heart-disease-prediction
source .venv/bin/activate
jupyter notebook
```

Open `notebooks/heart_disease_analysis.ipynb`. It contains introduction, dataset inspection, 15 EDA visualisations with interpretations, data-quality analysis, leakage-safe preprocessing, and the modelling workflow. `RUN_TRAINING` is `False` by default so the notebook loads saved artefacts instead of retraining; set it to `True` to reproduce the full five-model training from the notebook.

### Google Colab

1. Zip the `heart-disease-prediction` folder locally and unzip it on your computer.
2. In Colab, use **File → Upload notebook** and select `notebooks/heart_disease_analysis.ipynb`. Do not double-click the file in Colab's Files panel; that downloads it.
3. Run the first cell. It asks you to upload `heart-disease-prediction.zip` if the full project is not already under `/content`, extracts it, sets `PROJECT_ROOT`, and installs `requirements.txt`.

If you prefer Google Drive, place the folder at `MyDrive/heart-disease-prediction`. The setup cell mounts Drive automatically when the project is not found under `/content`. The saved models, comparison CSV, dashboard JSON and figures are read from that same folder, so keep the full extracted project together.

## 7. How to train the models

```bash
cd ~/heart-disease-prediction
source .venv/bin/activate
python src/train_models.py
```

This writes:

- `models/best_model.joblib` - complete fitted preprocessing + prediction pipeline.
- `models/model_metadata.json` - model, threshold, features, configuration, metrics and dataset facts.
- `models/model_comparison.csv` - five-model comparison table.
- `models/evaluation_results.json` - validation metrics, curves and confusion matrices used by the dashboard.
- `reports/figures/` - comparison charts and curves.
- `reports/model_results/validation_report.txt` - text report.

A fast syntax/workflow check without saving artefacts:

```bash
python src/train_models.py --smoke-test --skip-final-test
```

## 8. How to launch the website

```bash
cd ~/heart-disease-prediction
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`). The website has four sections: Home / Overview, Prediction, Model Performance and About Dataset. It loads the saved model and results; it does not retrain or re-evaluate models at startup.

### Deploy live with Streamlit Cloud

1. Initialise and push the full project folder to a GitHub repository:

```bash
cd /home/rushikesh/heart-disease-prediction
git init
git add .
git commit -m "Heart disease prediction app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

2. Sign in at `share.streamlit.io` with GitHub.
3. Select **New app**, choose the repository, branch `main`, and main file `app/streamlit_app.py`.
4. Click **Deploy**. The included `requirements.txt`, `runtime.txt` and `.streamlit/config.toml` are picked up automatically.
5. Share the public URL shown in the Streamlit Cloud dashboard.
6. If the app is already deployed, push the updated `requirements.txt` and use **Reboot app** in Streamlit Cloud so the pinned dependencies are reinstalled.

`scikit-learn` is pinned to `1.7.2` and `joblib` to `1.6.0` because `models/best_model.joblib` was serialised with those versions. A newer scikit-learn can change imputer internals and break prediction. If your Cloud app uses Python 3.14, select Python 3.11, 3.12 or 3.13 in **Advanced settings**; Streamlit Cloud may ignore `runtime.txt` for Python selection.

Keep `models/`, `reports/` and `data/framingham.csv` in the repository: the live app reads those saved artefacts at startup. After retraining locally, commit and push the regenerated artefacts to update the live site.

Optional CLI prediction from a JSON object:

```bash
python src/predict.py --json '{"male":1,"age":49,"education":2,"currentSmoker":0,"cigsPerDay":0,"BPMeds":0,"prevalentStroke":0,"prevalentHyp":0,"diabetes":0,"totChol":235,"sysBP":132,"diaBP":83,"BMI":25.8,"heartRate":76,"glucose":82}'
```

## 9. Model evaluation methodology

1. **Split:** 80% training / 20% test, stratified by `TenYearCHD`, `random_state=42`.
2. **Leakage prevention:** all imputation, encoding and scaling live inside scikit-learn `Pipeline` / `ColumnTransformer` objects and are fitted only on training data or inside each CV fold.
3. **Preprocessing:** numeric features use median imputation; binary features use mode imputation; education uses an explicit unknown category and one-hot encoding. Logistic Regression and SVM additionally use `StandardScaler`; tree models do not.
4. **Tuning:** `GridSearchCV` with five-fold `StratifiedKFold`. Recall is the refit metric; precision, specificity, ROC-AUC and PR-AUC are monitored.
5. **Validation predictions:** out-of-fold probabilities from the training set are used for threshold selection and all model comparison metrics. The test set is not used for tuning, model choice or threshold choice.
6. **Metrics:** accuracy, precision, recall, specificity, F1-score, false-negative rate, ROC-AUC, PR-AUC and confusion matrix.
7. **Threshold:** candidates are evaluated on out-of-fold probabilities. The selected threshold maximises recall among thresholds meeting specificity >= 0.50 and precision >= 0.20, with documented relaxation stages if needed. Ties are broken by F1 and specificity. The default 0.50 threshold is reported for comparison.
8. **Final evaluation:** after the model and threshold are fixed, the saved pipeline is evaluated once on the untouched test set.

Formulae: `Recall = TP/(TP+FN)`, `Specificity = TN/(TN+FP)`, `FNR = FN/(TP+FN)`, `Precision = TP/(TP+FP)`, `F1 = 2*Precision*Recall/(Precision+Recall)`.

## 10. Best model selection methodology

Models are ranked on out-of-fold validation results using a lexicographic rule: eligible models must meet specificity >= 0.50 and precision >= 0.20; among them, higher recall wins, followed by PR-AUC, F1-score, specificity and precision. If no model meets the strict floors, the documented relaxed floors (0.40 / 0.15) are used. The algorithm is not chosen by popularity or accuracy.

## 11. Results

Validation metrics below are computed from out-of-fold predictions on the training set, each at its own selected threshold.

| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1-score | FNR | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4288 | 0.5625 | 0.2316 | **0.8117** | 0.5179 | 0.3603 | 0.1884 | 0.7301 | **0.3491** |
| Decision Tree | 0.4806 | 0.5545 | 0.2186 | 0.7515 | 0.5193 | 0.3387 | 0.2485 | 0.6777 | 0.2502 |
| Random Forest | 0.3292 | 0.5486 | 0.2233 | 0.7961 | 0.5043 | 0.3488 | 0.2039 | 0.7150 | 0.3257 |
| Support Vector Machine | 0.1062 | 0.5486 | 0.2257 | 0.8117 | 0.5016 | 0.3532 | 0.1884 | 0.7228 | 0.2990 |
| XGBoost | 0.4061 | 0.5457 | 0.2206 | 0.7864 | 0.5026 | 0.3445 | 0.2136 | 0.7140 | 0.3245 |

**Selected model:** Logistic Regression (`C=0.01`, `class_weight='balanced'`). It tied with SVM on recall (0.8117) but had higher PR-AUC, F1-score and specificity, and it satisfied the pre-declared specificity/precision floors. Accuracy was not used for selection.

**Threshold trade-off:** at the default 0.50 threshold, validation recall was 0.7029, specificity 0.6524 and precision 0.2658. At the selected 0.4288 threshold, recall rose to 0.8117 while specificity fell to 0.5179 and precision to 0.2316. False negatives were reduced, not eliminated, and false positives increased.

**Final test metrics** (single evaluation after selection): accuracy 0.5672, precision 0.2233, recall 0.7442, specificity 0.5355, F1-score 0.3435, false-negative rate 0.2558, ROC-AUC 0.6956, PR-AUC 0.2911. Confusion matrix: TN=385, FP=334, FN=33, TP=96.

The authoritative machine-readable artefacts are `models/model_comparison.csv`, `models/model_metadata.json`, `models/evaluation_results.json` and `reports/model_results/validation_report.txt`. The dashboard reads these same files.

## 12. Limitations

- Historical single-cohort dataset; performance may not transfer to other populations or time periods.
- Ages outside 32-70 are rejected because the model has not been validated for them.
- 388 glucose values are missing; imputation adds uncertainty.
- The selected threshold intentionally trades extra false positives for fewer false negatives.
- Correlations and EDA comparisons are descriptive, not medical or causal conclusions.
- This is not a clinically validated tool.

## 13. Disclaimer

**This project is an academic machine-learning demonstration. It is not a medical device, not a clinically validated risk score, and not a substitute for diagnosis or professional medical advice. Predictions must not be used for healthcare decisions.**
