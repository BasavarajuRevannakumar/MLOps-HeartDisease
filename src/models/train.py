from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(
    X_train,
    y_train,
):
    """
    Train a Logistic Regression classifier.
    """

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    model.fit(X_train, y_train)

    return model

def train_random_forest(
    X_train,
    y_train,
):
    """
    Train a Random Forest classifier.
    """

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
    )

    model.fit(X_train, y_train)

    return model    