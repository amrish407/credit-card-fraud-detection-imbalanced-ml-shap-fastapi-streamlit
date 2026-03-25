from dataclasses import dataclass

import pandas as pd

from src import config
from src.exception import FraudDetectionException
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationSummary:
    is_valid: bool
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_values: dict[str, int]
    missing_columns: list[str]
    invalid_target_values: list[int]
    class_distribution: dict[str, int]


class DataValidator:
    def __init__(self, target_column: str = config.TARGET_COLUMN):
        self.target_column = target_column
        self.expected_columns = config.BASE_FEATURES + [self.target_column]

    def validate(self, dataframe: pd.DataFrame) -> ValidationSummary:
        missing_columns = [column for column in self.expected_columns if column not in dataframe.columns]
        if missing_columns:
            raise FraudDetectionException(
                "Dataset schema validation failed.",
                context=f"Missing columns: {missing_columns}",
            )

        duplicate_rows = int(dataframe.duplicated().sum())
        missing_values = dataframe.isna().sum().to_dict()

        unique_target_values = sorted(dataframe[self.target_column].dropna().unique().tolist())
        invalid_target_values = [int(value) for value in unique_target_values if value not in [0, 1]]

        class_distribution_series = dataframe[self.target_column].value_counts().sort_index()
        class_distribution = {str(int(index)): int(value) for index, value in class_distribution_series.items()}

        summary = ValidationSummary(
            is_valid=(len(missing_columns) == 0 and len(invalid_target_values) == 0),
            row_count=int(dataframe.shape[0]),
            column_count=int(dataframe.shape[1]),
            duplicate_rows=duplicate_rows,
            missing_values={key: int(value) for key, value in missing_values.items()},
            missing_columns=missing_columns,
            invalid_target_values=invalid_target_values,
            class_distribution=class_distribution,
        )

        logger.info(
            "Validation complete | rows=%s columns=%s duplicates=%s",
            summary.row_count,
            summary.column_count,
            summary.duplicate_rows,
        )
        return summary
