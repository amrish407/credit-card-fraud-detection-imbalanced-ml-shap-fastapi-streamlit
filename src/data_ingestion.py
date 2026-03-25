from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config
from src.exception import FraudDetectionException
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    data_path: Path = config.RAW_DATA_PATH
    target_column: str = config.TARGET_COLUMN


class DataIngestion:
    def __init__(self, ingestion_config: DataIngestionConfig | None = None):
        self.config = ingestion_config or DataIngestionConfig()

    def load_data(self) -> pd.DataFrame:
        if not self.config.data_path.exists():
            raise FraudDetectionException(
                "Dataset not found.",
                context=f"Expected file at: {self.config.data_path}",
            )

        dataframe = pd.read_csv(self.config.data_path)
        if self.config.target_column not in dataframe.columns:
            raise FraudDetectionException(
                "Target column missing in dataset.",
                context=f"Expected column: {self.config.target_column}",
            )

        dataframe[self.config.target_column] = dataframe[self.config.target_column].astype(int)
        logger.info("Loaded dataset with shape %s", dataframe.shape)
        return dataframe

    def save_sample(self, dataframe: pd.DataFrame, sample_size: int = 5000) -> Path:
        sample_size = min(sample_size, len(dataframe))
        sample_df = dataframe.sample(n=sample_size, random_state=config.RANDOM_STATE)

        sample_path = config.SAMPLE_DIR / "creditcard_sample.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(sample_path, index=False)

        logger.info("Saved sample dataset with %s rows to %s", sample_size, sample_path)
        return sample_path
