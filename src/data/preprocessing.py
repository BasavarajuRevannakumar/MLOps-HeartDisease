"""
Data preprocessing utilities for the Heart Disease MLOps project.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature groups
NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

TARGET = "target"


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataframe into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split dataset into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def create_numerical_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline for numerical features.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

def create_categorical_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline for categorical features.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

def create_preprocessor() -> ColumnTransformer:
    """
    Create the complete preprocessing pipeline.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                create_numerical_pipeline(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                create_categorical_pipeline(),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor
