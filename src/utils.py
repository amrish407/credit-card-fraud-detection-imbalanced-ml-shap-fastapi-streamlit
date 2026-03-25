import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(filepath: Path) -> dict[str, Any]:
    with filepath.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_dataframe(dataframe: pd.DataFrame, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(filepath, index=False)


def save_model(model: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path) -> Any:
    return joblib.load(filepath)


def save_plot(filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160, bbox_inches="tight")
    plt.close()


def to_serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value
