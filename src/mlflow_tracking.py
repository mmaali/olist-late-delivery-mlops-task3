# MLflow tracking and model registration


import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# Project root directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Experiment 2 model and artifact paths

MODEL_PATH = PROJECT_ROOT / "artifacts" / "final_logistic_model_experiment2.joblib"

PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessor_experiment2.joblib"

THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "final_threshold_experiment2.json"

FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "feature_list_experiment2.txt"


# MLflow server address

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# MLflow experiment name

EXPERIMENT_NAME = "Olist Late Delivery - Experiment 2"


# Registered model name

REGISTERED_MODEL_NAME = "olist_late_delivery_model"


def register_experiment_2_model():
    # Connect to the MLflow tracking server

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create or get the MLflow experiment

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load the trained Experiment 2 model

    model = joblib.load(MODEL_PATH)

    # Start a new MLflow run

    with mlflow.start_run(run_name="Experiment_2_Final_Logistic_Regression"):
        # Log model parameters

        mlflow.log_param("model_type", "Logistic Regression")

        mlflow.log_param("experiment", "Experiment 2")

        mlflow.log_param("original_features", 20)

        mlflow.log_param("processed_features", 84)

        mlflow.log_param("classification_threshold", 0.35)

        # Log final test metrics

        mlflow.log_metric("accuracy", 0.687)

        mlflow.log_metric("balanced_accuracy", 0.612)

        mlflow.log_metric("late_precision", 0.110)

        mlflow.log_metric("late_recall", 0.527)

        mlflow.log_metric("late_f1", 0.183)

        # Log preprocessing artifact

        mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="preprocessing")

        # Log threshold configuration

        mlflow.log_artifact(str(THRESHOLD_PATH), artifact_path="configuration")

        # Log feature list

        mlflow.log_artifact(str(FEATURE_LIST_PATH), artifact_path="configuration")

        # Log and register the trained model

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="experiment_2_model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Get the run ID

        run_id = mlflow.active_run().info.run_id

        print("MLflow run completed successfully.")

        print(f"Run ID: {run_id}")

        print(f"Model URI: {model_info.model_uri}")

    # Connect to the MLflow Model Registry

    client = MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")

    if versions:
        latest_version = max(versions, key=lambda version: int(version.version))

        # Assign the Production alias

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "Production", latest_version.version
        )

        print("Model registered successfully.")

        print(f"Model version: {latest_version.version}")

        print("Model alias: Production")


if __name__ == "__main__":
    register_experiment_2_model()
