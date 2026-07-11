import mlflow
import mlflow.sklearn


def log_experiment(
    model_name,
    model,
    metrics,
    params,
):
    """
    Log model, parameters and metrics to MLflow.
    """

    mlflow.set_experiment("Heart Disease Prediction")

    with mlflow.start_run(run_name=model_name):

        mlflow.log_params(params)

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )