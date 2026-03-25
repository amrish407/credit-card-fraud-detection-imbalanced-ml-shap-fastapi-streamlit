from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is importable when Streamlit runs from dashboard/ context.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.prediction_pipeline import FraudPredictor
from src.utils import load_json

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_data
def load_metrics_table() -> pd.DataFrame:
    if config.MODEL_COMPARISON_FILE.exists():
        return pd.read_csv(config.MODEL_COMPARISON_FILE)
    return pd.DataFrame()


@st.cache_data
def load_statistics() -> dict:
    if config.STATISTICS_FILE.exists():
        return load_json(config.STATISTICS_FILE)
    return {}


def show_image_if_exists(image_path: Path, caption: str) -> None:
    if image_path.exists():
        try:
            st.image(str(image_path), caption=caption, use_container_width=True)
        except TypeError:
            st.image(str(image_path), caption=caption, use_column_width=True)
    else:
        st.info(f"Image not found: {image_path}")


def render_overview() -> None:
    st.title("End-to-End Credit Card Fraud Detection")
    st.markdown(
        """
        This dashboard demonstrates a full imbalanced-classification project with:
        - descriptive statistics and hypothesis-style comparisons
        - model benchmarking (baseline + classical ML + imbalance methods)
        - threshold tuning with precision-recall tradeoff
        - explainability artifacts (SHAP)
        - deployment-ready API integration
        """
    )


def render_class_imbalance() -> None:
    st.subheader("Class Imbalance Summary")
    stats_summary = load_statistics()

    imbalance = stats_summary.get("imbalance_summary", {})
    if imbalance:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{imbalance.get('total_transactions', 0):,}")
        col2.metric("Fraud Count", f"{imbalance.get('fraud_count', 0):,}")
        col3.metric(
            "Imbalance Ratio (Non-Fraud:Fraud)",
            f"{imbalance.get('imbalance_ratio_non_fraud_to_fraud', 0):.2f}:1",
        )
    else:
        st.info("Run training first to generate statistics artifacts.")

    show_image_if_exists(config.PLOTS_DIR / "class_distribution.png", "Class Distribution")


def render_eda_plots() -> None:
    st.subheader("EDA Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        show_image_if_exists(config.PLOTS_DIR / "amount_distribution.png", "Amount Distribution")
    with col2:
        show_image_if_exists(config.PLOTS_DIR / "amount_by_class_boxplot.png", "Amount by Class")


def render_model_performance() -> None:
    st.subheader("Model Performance")
    comparison_df = load_metrics_table()

    if comparison_df.empty:
        st.warning("Model comparison file not available. Run `python app.py train` first.")
        return

    st.dataframe(comparison_df, use_container_width=True)

    if config.BEST_METRICS_FILE.exists():
        best_metrics = load_json(config.BEST_METRICS_FILE)
        st.markdown("### Best Model Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", best_metrics.get("model_name", "N/A"))
        col2.metric("PR-AUC (Tuned)", f"{best_metrics.get('pr_auc_tuned', 0):.4f}")
        col3.metric("Recall (Tuned)", f"{best_metrics.get('recall_tuned', 0):.4f}")
        col4.metric("Precision (Tuned)", f"{best_metrics.get('precision_tuned', 0):.4f}")

    col_left, col_right = st.columns(2)
    with col_left:
        show_image_if_exists(config.PLOTS_DIR / "best_model_confusion_matrix.png", "Confusion Matrix")
        show_image_if_exists(config.PLOTS_DIR / "best_model_roc_curve.png", "ROC Curve")
    with col_right:
        show_image_if_exists(config.PLOTS_DIR / "best_model_pr_curve.png", "Precision-Recall Curve")
        show_image_if_exists(config.PLOTS_DIR / "model_comparison_pr_auc.png", "Model Comparison (PR-AUC)")


def render_prediction_tool() -> None:
    st.subheader("Sample Transaction Prediction")

    try:
        predictor = FraudPredictor()
    except Exception as error:
        st.error(f"Model is not available yet: {error}")
        return

    with st.form("prediction_form"):
        transaction = {}
        transaction["Time"] = st.number_input("Time", value=0.0, step=1.0)

        col1, col2 = st.columns(2)
        with col1:
            for index in range(1, 15):
                transaction[f"V{index}"] = st.number_input(f"V{index}", value=0.0, format="%.6f")
        with col2:
            for index in range(15, 29):
                transaction[f"V{index}"] = st.number_input(f"V{index}", value=0.0, format="%.6f")

        transaction["Amount"] = st.number_input("Amount", value=100.0, min_value=0.0, step=1.0)

        submitted = st.form_submit_button("Predict Fraud Risk")

    if submitted:
        result = predictor.predict_single(transaction)
        st.success("Prediction complete")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Class", str(result.prediction))
        col2.metric("Fraud Probability", f"{result.fraud_probability:.6f}")
        col3.metric("Threshold", f"{result.decision_threshold:.4f}")

        if result.prediction == 1:
            st.warning("This transaction is flagged as potential fraud. Recommend manual review.")
        else:
            st.info("This transaction is predicted as non-fraud under the current threshold.")


def render_explainability() -> None:
    st.subheader("Explainability (SHAP)")
    shap_summary_path = config.SHAP_DIR / "shap_summary.json"

    if shap_summary_path.exists():
        shap_summary = load_json(shap_summary_path)
        st.write("SHAP status:", shap_summary.get("status", "unknown"))
    else:
        st.info("SHAP summary not found. Run training first.")

    col1, col2 = st.columns(2)
    with col1:
        show_image_if_exists(config.SHAP_DIR / "shap_summary_bar.png", "Global Feature Importance (SHAP Bar)")
        show_image_if_exists(config.SHAP_DIR / "shap_local_waterfall_1.png", "Local Explanation - Example 1")
    with col2:
        show_image_if_exists(config.SHAP_DIR / "shap_summary_beeswarm.png", "Global SHAP Beeswarm")
        show_image_if_exists(config.SHAP_DIR / "shap_local_waterfall_2.png", "Local Explanation - Example 2")


render_overview()

st.markdown("---")
render_class_imbalance()

st.markdown("---")
render_eda_plots()

st.markdown("---")
render_model_performance()

st.markdown("---")
render_prediction_tool()

st.markdown("---")
render_explainability()
