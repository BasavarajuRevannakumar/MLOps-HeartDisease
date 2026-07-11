"""
Module to load the UCI Heart Disease dataset.
"""

from typing import List

import pandas as pd

from src.utils.config import RAW_DATASET
from src.utils.logger import get_logger

logger = get_logger(__name__)

COLUMN_NAMES: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_data() -> pd.DataFrame:
    """
    Load the official UCI Cleveland Heart Disease dataset.

    Returns
    -------
    pd.DataFrame
        Clean dataframe.
    """

    logger.info("Loading dataset...")

    df = pd.read_csv(
        RAW_DATASET,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
    )

    logger.info("Dataset loaded successfully.")

    logger.info(f"Dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    dataframe = load_data()
    print(dataframe.head())
