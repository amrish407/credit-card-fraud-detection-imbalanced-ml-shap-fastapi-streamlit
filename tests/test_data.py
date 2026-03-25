from pathlib import Path

import pytest

from src import config
from src.data_ingestion import DataIngestion, DataIngestionConfig
from src.data_validation import DataValidator


@pytest.mark.skipif(not config.RAW_DATA_PATH.exists(), reason="Dataset file is not present")
def test_data_ingestion_and_validation():
    ingestion = DataIngestion(DataIngestionConfig(data_path=Path(config.RAW_DATA_PATH)))
    dataframe = ingestion.load_data()

    assert not dataframe.empty
    assert config.TARGET_COLUMN in dataframe.columns
    assert set(config.BASE_FEATURES).issubset(set(dataframe.columns))

    validator = DataValidator()
    summary = validator.validate(dataframe)

    assert summary.is_valid
    assert summary.row_count > 0
    assert summary.class_distribution.get("0", 0) > 0
    assert summary.class_distribution.get("1", 0) > 0
