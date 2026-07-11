"""
Model evaluation utilities.
"""

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model : sklearn estimator
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    return metrics

def plot_confusion_matrix(model, X_test, y_test, save_path):
    """
    Plot and save the confusion matrix.
    """

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.show()


def cross_validate_model(model, X, y):
    """
    Perform 5-fold cross-validation.
    """

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
    )

    return {
        "Mean Accuracy": scores.mean(),
        "Std Accuracy": scores.std(),
    }    

    