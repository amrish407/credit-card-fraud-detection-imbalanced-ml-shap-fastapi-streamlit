import argparse
import os
from contextlib import nullcontext
from numbers import Number
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.data_ingestion import DataIngestion, DataIngestionConfig
from src.data_validation import DataValidator
from src.explainability import ModelExplainer
from src.logger import get_logger
from src.model_training import ModelTrainer, TrainingConfig
from src.prediction_pipeline import FraudPredictor
from src.preprocessing import SplitConfig, split_train_test
from src.statistical_analysis import StatisticalAnalyzer
from src.utils import ensure_directories, save_json

logger = get_logger(__name__)


def _prepare_directories() -> None:
    ensure_directories(
        [
            config.PROCESSED_DIR,
            config.SAMPLE_DIR,
            config.MODELS_DIR,
            config.METRICS_DIR,
            config.PLOTS_DIR,
            config.SHAP_DIR,
            config.LOG_DIR,
        ]
    )


def _initialize_mlflow(enable_mlflow: bool):
    if not enable_mlflow:
        logger.info("MLflow tracking disabled by --disable-mlflow flag.")
        return None

    env_toggle = os.getenv("ENABLE_MLFLOW", "true").strip().lower()
    if env_toggle in {"0", "false", "no", "off"}:
        logger.info("MLflow tracking disabled via ENABLE_MLFLOW environment variable.")
        return None

    try:
        import dagshub
        import mlflow
    except ImportError:
        logger.warning("dagshub/mlflow packages are unavailable. Skipping experiment tracking.")
        return None

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "amrish407")
    repo_name = os.getenv(
        "DAGSHUB_REPO_NAME",
        "credit-card-fraud-detection-imbalanced-ml-shap-fastapi-streamlit",
    )
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-card-fraud-detection")

    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow tracking enabled for DagsHub repo %s/%s", repo_owner, repo_name)
        return mlflow
    except Exception as error:
        logger.warning(
            "Could not initialize DagsHub MLflow tracking. Continuing without MLflow. Error: %s",
            error,
        )
        return None


def _log_metrics_if_numeric(mlflow_module, metrics: dict, prefix: str = "") -> None:
    if not mlflow_module:
        return

    for key, value in metrics.items():
        if isinstance(value, Number):
            metric_name = f"{prefix}{key}" if prefix else key
            try:
                mlflow_module.log_metric(metric_name, float(value))
            except Exception:
                continue


def _log_artifact_if_exists(mlflow_module, artifact_path: Path, artifact_group: str | None = None) -> None:
    if not mlflow_module:
        return

    if not artifact_path.exists():
        return

    try:
        if artifact_path.is_dir():
            mlflow_module.log_artifacts(str(artifact_path), artifact_path=artifact_group)
        else:
            mlflow_module.log_artifact(str(artifact_path), artifact_path=artifact_group)
    except Exception as error:
        logger.warning("Failed to log artifact %s to MLflow: %s", artifact_path, error)


def _save_eda_plots(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        data=dataframe,
        x=config.TARGET_COLUMN,
        hue=config.TARGET_COLUMN,
        palette="Set2",
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.title("Class Distribution (Non-Fraud vs Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "class_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    ax = sns.boxplot(
        data=dataframe,
        x=config.TARGET_COLUMN,
        y="Amount",
        hue=config.TARGET_COLUMN,
        palette="Set2",
        dodge=False,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.title("Transaction Amount by Class")
    plt.xlabel("Class")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "amount_by_class_boxplot.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=dataframe, x="Amount", hue=config.TARGET_COLUMN, bins=60, kde=False, element="step")
    plt.title("Amount Distribution by Class")
    plt.xlabel("Amount")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "amount_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()


def _stratified_sample(dataframe: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size >= len(dataframe):
        return dataframe

    sample_fraction = sample_size / len(dataframe)
    sampled_df = (
        dataframe.groupby(config.TARGET_COLUMN, group_keys=False)
        .sample(frac=sample_fraction, random_state=config.RANDOM_STATE)
        .reset_index(drop=True)
    )
    return sampled_df


def run_training(
    data_path: Path,
    sample_size: int | None,
    test_size: float,
    enable_smoteenn: bool,
    enable_mlflow: bool,
) -> None:
    _prepare_directories()

    mlflow_module = _initialize_mlflow(enable_mlflow)
    run_context = (
        mlflow_module.start_run(run_name=f"train_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}")
        if mlflow_module
        else nullcontext()
    )

    with run_context:
        if mlflow_module:
            mlflow_module.log_params(
                {
                    "data_path": str(data_path),
                    "sample_size": sample_size if sample_size is not None else -1,
                    "test_size": test_size,
                    "enable_smoteenn": int(enable_smoteenn),
                    "target_column": config.TARGET_COLUMN,
                }
            )

        ingestion = DataIngestion(DataIngestionConfig(data_path=data_path))
        dataframe = ingestion.load_data()

        if sample_size:
            dataframe = _stratified_sample(dataframe, sample_size)
            logger.info("Using stratified sample for training: %s rows", len(dataframe))

        validator = DataValidator()
        validation_summary = validator.validate(dataframe)

        if validation_summary.duplicate_rows > 0:
            dataframe = dataframe.drop_duplicates().reset_index(drop=True)
            logger.info("Dropped duplicate rows. New shape: %s", dataframe.shape)

        save_json(validation_summary.__dict__, config.METRICS_DIR / "validation_summary.json")
        ingestion.save_sample(dataframe)

        if mlflow_module:
            mlflow_module.log_metric("row_count", float(len(dataframe)))
            mlflow_module.log_metric("duplicate_rows", float(validation_summary.duplicate_rows))

        _save_eda_plots(dataframe)

        analyzer = StatisticalAnalyzer()
        statistics_summary = analyzer.generate_report(dataframe)
        imbalance_ratio = statistics_summary["imbalance_summary"]["imbalance_ratio_non_fraud_to_fraud"]
        logger.info("Imbalance ratio (non-fraud:fraud) = %.2f", imbalance_ratio)

        if mlflow_module:
            _log_metrics_if_numeric(mlflow_module, statistics_summary["imbalance_summary"], prefix="imbalance_")

        split_config = SplitConfig(test_size=test_size, random_state=config.RANDOM_STATE, stratify=True)
        X_train, X_test, y_train, y_test = split_train_test(dataframe, split_config=split_config)

        trainer = ModelTrainer(
            TrainingConfig(
                random_state=config.RANDOM_STATE,
                min_recall_for_threshold=0.60,
                enable_smoteenn=enable_smoteenn,
            )
        )

        training_outputs = trainer.train_and_compare(X_train, y_train, X_test, y_test)

        explainer = ModelExplainer()
        shap_summary = explainer.generate_shap_artifacts(
            model_pipeline=training_outputs["best_model"],
            X_train=X_train,
            X_test=X_test,
        )

        final_run_summary = {
            "data_path": str(data_path),
            "rows_used": int(len(dataframe)),
            "best_model": training_outputs["best_model_name"],
            "decision_threshold": training_outputs["best_threshold"],
            "shap_status": shap_summary.get("status", "unknown"),
            "metrics_file": str(config.MODEL_COMPARISON_FILE),
            "best_model_file": str(config.BEST_MODEL_FILE),
        }
        save_json(final_run_summary, config.METRICS_DIR / "run_summary.json")

        if mlflow_module:
            best_metrics = training_outputs["comparison_df"].iloc[0].to_dict()
            _log_metrics_if_numeric(mlflow_module, best_metrics, prefix="best_")

            _log_artifact_if_exists(mlflow_module, config.MODEL_COMPARISON_FILE, "metrics")
            _log_artifact_if_exists(mlflow_module, config.BEST_METRICS_FILE, "metrics")
            _log_artifact_if_exists(mlflow_module, config.STATISTICS_FILE, "metrics")
            _log_artifact_if_exists(mlflow_module, config.BEST_MODEL_FILE, "model")
            _log_artifact_if_exists(mlflow_module, config.PLOTS_DIR, "plots")
            _log_artifact_if_exists(mlflow_module, config.SHAP_DIR, "shap")

        logger.info("Training pipeline completed successfully.")
        logger.info("Best model: %s", training_outputs["best_model_name"])
        logger.info("Decision threshold: %.4f", training_outputs["best_threshold"])


def run_single_prediction(data_path: Path, sample_index: int) -> None:
    ingestion = DataIngestion(DataIngestionConfig(data_path=data_path))
    dataframe = ingestion.load_data()

    if sample_index < 0 or sample_index >= len(dataframe):
        raise IndexError(f"sample-index must be between 0 and {len(dataframe) - 1}")

    features = dataframe.drop(columns=[config.TARGET_COLUMN]).iloc[sample_index].to_dict()
    actual_label = int(dataframe.iloc[sample_index][config.TARGET_COLUMN])

    predictor = FraudPredictor()
    result = predictor.predict_single(features)

    logger.info("Prediction complete for sample index %s", sample_index)
    logger.info("Actual label: %s", actual_label)
    logger.info("Predicted label: %s", result.prediction)
    logger.info("Fraud probability: %.6f", result.fraud_probability)
    logger.info("Decision threshold: %.6f", result.decision_threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-End Credit Card Fraud Detection Project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run complete training and artifact pipeline")
    train_parser.add_argument(
        "--data-path",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Path to input CSV dataset",
    )
    train_parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for quick experimentation",
    )
    train_parser.add_argument("--test-size", type=float, default=config.TEST_SIZE, help="Test split ratio")
    train_parser.add_argument(
        "--disable-smoteenn",
        action="store_true",
        help="Disable SMOTEENN model in comparison",
    )
    train_parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable DagsHub MLflow tracking for this run",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict one row from the dataset")
    predict_parser.add_argument(
        "--data-path",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Path to input CSV dataset",
    )
    predict_parser.add_argument("--sample-index", type=int, default=0, help="Row index in dataset")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        run_training(
            data_path=args.data_path,
            sample_size=args.sample_size,
            test_size=args.test_size,
            enable_smoteenn=not args.disable_smoteenn,
            enable_mlflow=not args.disable_mlflow,
        )
    elif args.command == "predict":
        run_single_prediction(data_path=args.data_path, sample_index=args.sample_index)


if __name__ == "__main__":
    main()
