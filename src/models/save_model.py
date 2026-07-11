"""
Module to save the model
"""

import joblib


def save_model(model, filepath):
    """
    Save a trained model to disk.
    """
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load a trained model from disk.
    """
    return joblib.load(filepath)