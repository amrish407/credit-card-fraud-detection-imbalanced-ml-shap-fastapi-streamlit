from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SHAP_DIR = ARTIFACTS_DIR / "shap"
LOG_DIR = ROOT_DIR / "logs"

TARGET_COLUMN = "Class"
BASE_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
ENGINEERED_FEATURES = BASE_FEATURES + ["Hour", "LogAmount", "AmountToMeanRatio"]

BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_FILE = METRICS_DIR / "model_comparison.csv"
BEST_METRICS_FILE = METRICS_DIR / "best_model_metrics.json"
STATISTICS_FILE = METRICS_DIR / "statistics_summary.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
