# End-to-End Credit Card Fraud Detection on Imbalanced Data

A production-style, resume-ready Data Science project that solves **credit card fraud detection** as a highly imbalanced binary classification problem using robust statistics, multiple ML strategies, explainability, API serving, and an interactive dashboard.

## 1) Project Overview

Fraud detection is a rare-event classification challenge where only a tiny fraction of transactions are fraudulent. This project demonstrates an end-to-end workflow that prioritizes:

- practical model performance on imbalanced data
- statistically grounded analysis
- explainability with SHAP
- deployment readiness through FastAPI + Streamlit

## 2) Problem Statement

Given transaction-level features (`Time`, `Amount`, and anonymized PCA-like features `V1` to `V28`), predict whether a transaction is fraudulent (`Class = 1`) or not (`Class = 0`).

## 3) Why Fraud Detection is Challenging

- Extreme class imbalance (fraud << non-fraud)
- False negatives are costly (missed fraud)
- False positives create review overhead and customer friction
- Accuracy alone can be misleading

## 4) Why Imbalance Matters

If a model predicts every transaction as non-fraud, it can still show very high accuracy but detect no fraud. For this reason, this project emphasizes:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (Average Precision)
- Confusion Matrix
- threshold tuning based on precision-recall tradeoff

## 5) Tech Stack

- Python
- Pandas, NumPy, SciPy
- Scikit-learn
- imbalanced-learn (`SMOTE`, `SMOTEENN`)
- XGBoost (with fallback model if unavailable)
- Matplotlib, Seaborn
- SHAP
- FastAPI + Uvicorn
- Streamlit
- Joblib, Pydantic
- Pytest
- Docker

## 6) Folder Structure

```text
credit-card-fraud-detection/
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- Dockerfile
+-- docker-compose.yml
+-- app.py
+-- data/
¦   +-- raw/
¦   +-- processed/
¦   +-- sample/
+-- notebooks/
¦   +-- 01_data_understanding.ipynb
¦   +-- 02_eda_statistics.ipynb
¦   +-- 03_modeling_imbalanced_data.ipynb
¦   +-- 04_explainability_shap.ipynb
+-- src/
¦   +-- __init__.py
¦   +-- config.py
¦   +-- logger.py
¦   +-- exception.py
¦   +-- utils.py
¦   +-- data_ingestion.py
¦   +-- data_validation.py
¦   +-- preprocessing.py
¦   +-- feature_engineering.py
¦   +-- statistical_analysis.py
¦   +-- model_training.py
¦   +-- model_evaluation.py
¦   +-- explainability.py
¦   +-- prediction_pipeline.py
+-- backend/
¦   +-- fastapi_app.py
+-- dashboard/
¦   +-- streamlit_app.py
+-- artifacts/
¦   +-- models/
¦   +-- metrics/
¦   +-- plots/
¦   +-- shap/
+-- tests/
    +-- test_data.py
    +-- test_model.py
    +-- test_api.py
```

## 7) Dataset Setup

Use the Kaggle dataset: **Credit Card Fraud Detection**.

Place the CSV at:

```text
data/raw/creditcard.csv
```

Expected schema:
- features: `Time`, `V1`...`V28`, `Amount`
- target: `Class`

## 8) Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## 9) Run Training Pipeline

This single command runs ingestion, validation, statistics, EDA plots, model comparison, threshold tuning, and SHAP artifact generation:

```bash
python app.py train --data-path data/raw/creditcard.csv
```

Optional quick-run mode:

```bash
python app.py train --data-path data/raw/creditcard.csv --sample-size 120000 --disable-smoteenn
```

Generated outputs include:
- `artifacts/models/best_model.joblib`
- `artifacts/models/model_metadata.json`
- `artifacts/metrics/model_comparison.csv`
- `artifacts/metrics/best_model_metrics.json`
- `artifacts/metrics/statistics_summary.json`
- plots and SHAP images under `artifacts/plots/` and `artifacts/shap/`

## 10) Run API (FastAPI)

```bash
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /`
- `GET /health`
- `POST /predict`
- `POST /predict_batch`

## 11) Run Dashboard (Streamlit)

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard sections:
- project overview
- class imbalance summary
- EDA plots
- model performance table and key metrics
- confusion matrix / ROC / PR visuals
- sample transaction prediction form
- SHAP explainability visuals

## 12) Statistical Analysis Included

Implemented in notebooks and `src/statistical_analysis.py`:

- mean, median, std, variance
- quartiles and IQR
- skewness and kurtosis
- class ratio and imbalance ratio
- IQR-based outlier detection
- correlation with target
- fraud vs non-fraud comparison
- hypothesis-style tests (`t-test`, `Mann-Whitney U`)
- interpretation of false positives/false negatives business impact

## 13) Modeling Included

- Dummy baseline
- Logistic Regression
- Logistic Regression (`class_weight='balanced'`)
- Random Forest (`class_weight='balanced_subsample'`)
- XGBoost (if installed) or HistGradientBoosting fallback
- SMOTE + Logistic Regression
- SMOTEENN + Random Forest (conditionally enabled for manageable dataset sizes)

## 14) Evaluation Strategy

Main focus metrics:
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

Threshold tuning is done via precision-recall curve (F1 optimization with recall floor).

## 15) Explainability

SHAP outputs include:
- global importance (bar)
- global summary (beeswarm)
- local waterfall explanations for high-risk transactions

Artifacts are saved to `artifacts/shap/`.

## 16) Testing

Run:

```bash
pytest -q
```

Tests cover:
- ingestion/validation sanity
- prediction output schema sanity
- API endpoint response sanity

## 17) Docker

Build and run API + dashboard:

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

## 18) Resume-Ready Summary

This project demonstrates:

- practical statistics for fraud analytics
- imbalanced classification strategy design
- model benchmarking and threshold tuning
- explainable AI with SHAP
- production-style modular code in Python
- deployment skills via FastAPI, Streamlit, and Docker

## 19) Future Improvements

- cost-sensitive learning with explicit fraud-loss matrix
- drift monitoring and periodic model retraining
- feature store and orchestration (Airflow/Prefect)
- model registry and CI/CD deployment pipeline
- real-time inference with message queue integration
