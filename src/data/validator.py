"""
 Utilities for dataset validation.
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the loaded dataframe.

    Raises
    ------
    ValueError
        If dataset does not match expected format.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.shape[1] != 14:
        raise ValueError(
            f"Expected 14 columns but found {df.shape[1]}."
        )
