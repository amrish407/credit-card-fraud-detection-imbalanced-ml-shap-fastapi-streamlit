from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.logger import get_logger
from src.utils import save_plot

logger = get_logger(__name__)


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def tune_threshold_by_f1(
    y_true: pd.Series,
    y_prob: np.ndarray,
    min_recall: float = 0.0,
) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return {"best_threshold": 0.5, "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0}

    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1] + 1e-12
    f1_scores = numerator / denominator

    valid_indices = np.where(recall[:-1] >= min_recall)[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(len(thresholds))

    candidate_f1 = f1_scores[valid_indices]
    best_local_idx = int(np.argmax(candidate_f1))
    best_idx = int(valid_indices[best_local_idx])

    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
    }


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    filepath: Path,
    title: str = "Confusion Matrix",
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0.5, 1.5], ["Non-Fraud (0)", "Fraud (1)"])
    plt.yticks([0.5, 1.5], ["Non-Fraud (0)", "Fraud (1)"], rotation=0)
    save_plot(filepath)


def save_roc_pr_plots(y_true: pd.Series, y_prob: np.ndarray, roc_path: Path, pr_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    save_plot(roc_path)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}", linewidth=2)
    baseline = np.mean(y_true)
    plt.hlines(baseline, xmin=0, xmax=1, colors="gray", linestyles="--", label=f"Baseline = {baseline:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    save_plot(pr_path)


def save_model_comparison_plot(comparison_df: pd.DataFrame, filepath: Path) -> None:
    if comparison_df.empty:
        logger.warning("Comparison dataframe is empty. Skipping model comparison plot.")
        return

    plot_df = comparison_df.sort_values("pr_auc_tuned", ascending=False)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=plot_df,
        x="model_name",
        y="pr_auc_tuned",
        hue="model_name",
        palette="viridis",
        dodge=False,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.title("Model Comparison by PR-AUC (Tuned Threshold)")
    plt.xlabel("Model")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=35, ha="right")
    save_plot(filepath)
