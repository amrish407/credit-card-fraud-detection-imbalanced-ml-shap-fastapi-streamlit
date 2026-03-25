import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime

# Avoids loky physical-core probing (wmic) noise on newer Windows setups.
# Keeping it slightly below os.cpu_count() bypasses the physical-core branch.
_cpu_count = os.cpu_count() or 1
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(_cpu_count - 1, 1)))

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.feature_engineering import FeatureEngineeringTransformer
from src.logger import get_logger
from src.model_evaluation import (
    evaluate_predictions,
    save_confusion_matrix_plot,
    save_model_comparison_plot,
    save_roc_pr_plots,
    tune_threshold_by_f1,
)
from src.utils import save_dataframe, save_json, save_model

logger = get_logger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r".*Could not find the number of physical cores.*",
    category=UserWarning,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency runtime guard
    XGBOOST_AVAILABLE = False


@dataclass
class TrainingConfig:
    random_state: int = config.RANDOM_STATE
    min_recall_for_threshold: float = 0.60
    enable_smoteenn: bool = True
    smoteenn_max_rows: int = 120_000
    random_forest_n_jobs: int = -1
    xgboost_n_jobs: int = -1
    retry_single_thread_on_permission_error: bool = True


class ModelTrainer:
    def __init__(self, training_config: TrainingConfig | None = None):
        self.config = training_config or TrainingConfig()

    @staticmethod
    def _predict_probabilities(model, features: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[:, 1]

        if hasattr(model, "decision_function"):
            scores = model.decision_function(features)
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score == min_score:
                return np.zeros_like(scores, dtype=float)
            return (scores - min_score) / (max_score - min_score)

        predictions = model.predict(features)
        return predictions.astype(float)

    def _build_models(self, class_ratio: float, train_rows: int) -> dict[str, ImbPipeline]:
        models: dict[str, ImbPipeline] = {
            "dummy_baseline": ImbPipeline(
                steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
            "logistic_regression": ImbPipeline(
                steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1500,
                            random_state=self.config.random_state,
                        ),
                    ),
                ]
            ),
            "logistic_balanced": ImbPipeline(
                steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1500,
                            class_weight="balanced",
                            random_state=self.config.random_state,
                        ),
                    ),
                ]
            ),
            "random_forest_balanced": ImbPipeline(
                steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=None,
                            class_weight="balanced_subsample",
                            random_state=self.config.random_state,
                            n_jobs=self.config.random_forest_n_jobs,
                        ),
                    ),
                ]
            ),
            "smote_logistic": ImbPipeline(
                steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    ("scaler", StandardScaler()),
                    ("smote", SMOTE(random_state=self.config.random_state)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1500,
                            random_state=self.config.random_state,
                        ),
                    ),
                ]
            ),
        }

        if self.config.enable_smoteenn and train_rows <= self.config.smoteenn_max_rows:
            models["smoteenn_random_forest"] = ImbPipeline(
                steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    ("smoteenn", SMOTEENN(random_state=self.config.random_state)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=180,
                            random_state=self.config.random_state,
                            n_jobs=self.config.random_forest_n_jobs,
                        ),
                    ),
                ]
            )
        else:
            logger.info(
                "Skipping SMOTEENN model due to dataset size (%s rows). Limit=%s.",
                train_rows,
                self.config.smoteenn_max_rows,
            )

        if XGBOOST_AVAILABLE:
            models["xgboost"] = ImbPipeline(
                steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=320,
                            max_depth=5,
                            learning_rate=0.06,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="binary:logistic",
                            eval_metric="aucpr",
                            random_state=self.config.random_state,
                            n_jobs=self.config.xgboost_n_jobs,
                            scale_pos_weight=max(class_ratio, 1.0),
                            reg_lambda=1.0,
                        ),
                    ),
                ]
            )
        else:
            logger.info("xgboost package not available. Using HistGradientBoosting as fallback.")
            models["hist_gradient_boosting"] = ImbPipeline(
                steps=[
                    ("feature_engineering", FeatureEngineeringTransformer()),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_depth=6,
                            learning_rate=0.05,
                            max_iter=250,
                            random_state=self.config.random_state,
                        ),
                    ),
                ]
            )

        return models

    @staticmethod
    def _fit_with_warning_suppression(model_pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Could not find the number of physical cores.*",
                category=UserWarning,
            )
            model_pipeline.fit(X_train, y_train)

    def _fit_with_fallback(
        self,
        model_name: str,
        model_pipeline: ImbPipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> str:
        try:
            self._fit_with_warning_suppression(model_pipeline, X_train, y_train)
            return "primary"
        except PermissionError as error:
            if not self.config.retry_single_thread_on_permission_error:
                raise

            error_text = str(error).lower()
            if "access is denied" not in error_text:
                raise

            try:
                model_pipeline.set_params(model__n_jobs=1)
            except Exception:
                raise

            logger.warning(
                "Model %s hit a multiprocessing permission error. Retrying with model__n_jobs=1.",
                model_name,
            )
            self._fit_with_warning_suppression(model_pipeline, X_train, y_train)
            return "fallback_single_thread"

    def train_and_compare(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, object]:
        class_counts = y_train.value_counts().to_dict()
        fraud_count = max(int(class_counts.get(1, 1)), 1)
        non_fraud_count = int(class_counts.get(0, 1))
        class_ratio = non_fraud_count / fraud_count

        model_candidates = self._build_models(class_ratio=class_ratio, train_rows=len(X_train))

        comparison_rows: list[dict[str, object]] = []
        fitted_models: dict[str, object] = {}
        probabilities_by_model: dict[str, np.ndarray] = {}

        for model_name, model_pipeline in model_candidates.items():
            logger.info("Training model: %s", model_name)
            start_time = time.time()
            try:
                fit_mode = self._fit_with_fallback(model_name, model_pipeline, X_train, y_train)
                y_prob = self._predict_probabilities(model_pipeline, X_test)

                default_metrics = evaluate_predictions(y_test, y_prob, threshold=0.5)
                threshold_info = tune_threshold_by_f1(
                    y_test,
                    y_prob,
                    min_recall=self.config.min_recall_for_threshold,
                )
                tuned_metrics = evaluate_predictions(
                    y_test,
                    y_prob,
                    threshold=threshold_info["best_threshold"],
                )

                fit_seconds = time.time() - start_time

                row = {
                    "model_name": model_name,
                    "fit_mode": fit_mode,
                    "fit_seconds": round(fit_seconds, 3),
                    "threshold_tuned": float(threshold_info["best_threshold"]),
                    "precision_default": default_metrics["precision"],
                    "recall_default": default_metrics["recall"],
                    "f1_default": default_metrics["f1_score"],
                    "roc_auc_default": default_metrics["roc_auc"],
                    "pr_auc_default": default_metrics["pr_auc"],
                    "precision_tuned": tuned_metrics["precision"],
                    "recall_tuned": tuned_metrics["recall"],
                    "f1_tuned": tuned_metrics["f1_score"],
                    "roc_auc_tuned": tuned_metrics["roc_auc"],
                    "pr_auc_tuned": tuned_metrics["pr_auc"],
                    "tp_tuned": tuned_metrics["true_positives"],
                    "fp_tuned": tuned_metrics["false_positives"],
                    "fn_tuned": tuned_metrics["false_negatives"],
                    "tn_tuned": tuned_metrics["true_negatives"],
                }
                comparison_rows.append(row)
                fitted_models[model_name] = model_pipeline
                probabilities_by_model[model_name] = y_prob

                logger.info(
                    "Finished %s | PR-AUC(tuned)=%.4f | Recall(tuned)=%.4f | Precision(tuned)=%.4f",
                    model_name,
                    row["pr_auc_tuned"],
                    row["recall_tuned"],
                    row["precision_tuned"],
                )
            except Exception as error:
                logger.exception("Model %s failed. Error: %s", model_name, error)

        comparison_df = pd.DataFrame(comparison_rows)
        if comparison_df.empty:
            raise RuntimeError("No model finished successfully. Please inspect logs.")

        comparison_df = comparison_df.sort_values(
            by=["pr_auc_tuned", "recall_tuned", "precision_tuned"],
            ascending=False,
        ).reset_index(drop=True)

        best_row = comparison_df.iloc[0]
        best_model_name = str(best_row["model_name"])
        best_threshold = float(best_row["threshold_tuned"])
        best_model = fitted_models[best_model_name]
        best_probabilities = probabilities_by_model[best_model_name]

        save_dataframe(comparison_df, config.MODEL_COMPARISON_FILE)
        save_model_comparison_plot(comparison_df, config.PLOTS_DIR / "model_comparison_pr_auc.png")

        save_confusion_matrix_plot(
            y_true=y_test,
            y_prob=best_probabilities,
            threshold=best_threshold,
            filepath=config.PLOTS_DIR / "best_model_confusion_matrix.png",
            title=f"Confusion Matrix ({best_model_name}, threshold={best_threshold:.3f})",
        )
        save_roc_pr_plots(
            y_true=y_test,
            y_prob=best_probabilities,
            roc_path=config.PLOTS_DIR / "best_model_roc_curve.png",
            pr_path=config.PLOTS_DIR / "best_model_pr_curve.png",
        )

        save_model(best_model, config.BEST_MODEL_FILE)

        best_metrics = comparison_df.iloc[0].to_dict()
        metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "best_model_name": best_model_name,
            "decision_threshold": best_threshold,
            "target_column": config.TARGET_COLUMN,
            "input_features": X_train.columns.tolist(),
            "model_artifact_path": str(config.BEST_MODEL_FILE),
            "comparison_file": str(config.MODEL_COMPARISON_FILE),
        }

        save_json(best_metrics, config.BEST_METRICS_FILE)
        save_json(metadata, config.MODEL_METADATA_FILE)

        logger.info("Best model: %s | threshold=%.4f", best_model_name, best_threshold)

        return {
            "comparison_df": comparison_df,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_threshold": best_threshold,
            "best_probabilities": best_probabilities,
            "y_test": y_test,
            "metadata": metadata,
        }
