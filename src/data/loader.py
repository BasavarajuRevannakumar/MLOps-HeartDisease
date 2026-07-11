"""
Module to load the UCI Heart Disease dataset.
"""

from typing import List

import pandas as pd

from src.utils.config import RAW_DATASET
from src.utils.logger import get_logger
from src.data.validator import validate_dataframe

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

    validate_dataframe(df)
    df = df.apply(pd.to_numeric)
    df["target"] = (df["target"] > 0).astype(int)

    logger.info("Missing values per column:")
    logger.info(df.isnull().sum())

    logger.info("Target distribution:")
    logger.info(df["target"].value_counts())

    return df


if __name__ == "__main__":
    dataframe = load_data()
    print(dataframe.head())
    print(dataframe.info())
    print(dataframe["target"].value_counts())

