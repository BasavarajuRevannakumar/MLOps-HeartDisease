"""
Prediction utilities for the Heart Disease API.
"""

from pathlib import Path

import joblib
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"

# Load once during application startup
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict(features: dict):
    """
    Predict heart disease from input features.
    """

    df = pd.DataFrame([features])

    X = preprocessor.transform(df)

    prediction = int(model.predict(X)[0])

    probability = float(model.predict_proba(X)[0][1])

    label = ( "Heart Disease"
        if prediction == 1
        else "No Heart Disease"
    )


    return {
        "prediction": prediction,
        "label": label,
        "probability": round(probability, 4),
    }