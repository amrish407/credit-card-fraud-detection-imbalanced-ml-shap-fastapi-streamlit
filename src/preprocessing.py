from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


@dataclass
class SplitConfig:
    test_size: float = config.TEST_SIZE
    random_state: int = config.RANDOM_STATE
    stratify: bool = True


def split_features_target(dataframe: pd.DataFrame, target_column: str = config.TARGET_COLUMN):
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    return X, y


def split_train_test(
    dataframe: pd.DataFrame,
    split_config: SplitConfig | None = None,
    target_column: str = config.TARGET_COLUMN,
):
    split_config = split_config or SplitConfig()
    X, y = split_features_target(dataframe, target_column=target_column)

    return train_test_split(
        X,
        y,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        stratify=y if split_config.stratify else None,
    )
