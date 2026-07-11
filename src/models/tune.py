"""
Hyperparameter tuning utilities.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_logistic_regression(X_train, y_train):
    """
    Tune Logistic Regression using GridSearchCV.
    """

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    return grid

def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest using RandomizedSearchCV.
    """

    model = RandomForestClassifier(
        random_state=42,
    )

    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=15,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search

