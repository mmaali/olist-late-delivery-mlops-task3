from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app import main as api

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "final_logistic_model_experiment2.joblib"
)


@pytest.fixture(scope="module")
def client():
    """
    Use the local model during API tests and initialize
    the application only once for this test module.
    """

    patcher = pytest.MonkeyPatch()

    def load_local_model(_model_uri):
        if not LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {LOCAL_MODEL_PATH}")

        return joblib.load(LOCAL_MODEL_PATH)

    patcher.setattr(
        api.mlflow.sklearn,
        "load_model",
        load_local_model,
    )

    try:
        with TestClient(api.app) as test_client:
            yield test_client
    finally:
        patcher.undo()


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == api.MODEL_NAME
    assert data["model_version"] == api.MODEL_VERSION
    assert data["threshold"] == api.MODEL_THRESHOLD
    assert data["mlflow_model"] == api.MLFLOW_MODEL_URI


def test_predict_invalid_input(client):
    invalid_order = {
        "item_count": -1,
        "total_price": 100.0,
        "total_freight": 20.0,
        "payment_count": 1,
        "total_payment": 120.0,
        "payment_installments": 1,
        "unique_products": 1,
        "average_product_weight": 500.0,
        "average_product_photos": 3.0,
        "customer_state": "SP",
        "seller_state": "SP",
        "customer_zip_code_prefix": 1000,
        "seller_zip_code_prefix": 1100,
        "order_purchase_timestamp": "2018-01-01 10:00:00",
        "order_approved_at": "2018-01-01 10:30:00",
        "order_estimated_delivery_date": "2018-01-10",
    }

    response = client.post("/predict", json=invalid_order)

    assert response.status_code == 422


def test_predict_valid_input(client):
    valid_order = {
        "item_count": 1,
        "total_price": 100.0,
        "total_freight": 20.0,
        "payment_count": 1,
        "total_payment": 120.0,
        "payment_installments": 1,
        "unique_products": 1,
        "average_product_weight": 500.0,
        "average_product_photos": 3.0,
        "customer_state": "SP",
        "seller_state": "SP",
        "customer_zip_code_prefix": 1000,
        "seller_zip_code_prefix": 1100,
        "order_purchase_timestamp": "2018-01-01 10:00:00",
        "order_approved_at": "2018-01-01 10:30:00",
        "order_estimated_delivery_date": "2018-01-10",
    }

    response = client.post("/predict", json=valid_order)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in {"Late", "On Time"}
    assert 0 <= result["late_probability"] <= 1
    assert result["model_version"] == api.MODEL_VERSION


def test_predict_batch_valid_input(client):
    order = {
        "item_count": 1,
        "total_price": 100.0,
        "total_freight": 20.0,
        "payment_count": 1,
        "total_payment": 120.0,
        "payment_installments": 1,
        "unique_products": 1,
        "average_product_weight": 500.0,
        "average_product_photos": 3.0,
        "customer_state": "SP",
        "seller_state": "SP",
        "customer_zip_code_prefix": 1000,
        "seller_zip_code_prefix": 1100,
        "order_purchase_timestamp": "2018-01-01 10:00:00",
        "order_approved_at": "2018-01-01 10:30:00",
        "order_estimated_delivery_date": "2018-01-10",
    }

    response = client.post("/predict-batch", json=[order, order])

    assert response.status_code == 200
    result = response.json()
    assert len(result["predictions"]) == 2
    assert result["model_version"] == api.MODEL_VERSION
    assert all(0 <= item["late_probability"] <= 1 for item in result["predictions"])


def test_predict_missing_model(client, monkeypatch):
    original_model = api.model

    try:
        monkeypatch.setattr(api, "model", None)

        valid_order = {
            "item_count": 1,
            "total_price": 100.0,
            "total_freight": 20.0,
            "payment_count": 1,
            "total_payment": 120.0,
            "payment_installments": 1,
            "unique_products": 1,
            "average_product_weight": 500.0,
            "average_product_photos": 3.0,
            "customer_state": "SP",
            "seller_state": "SP",
            "customer_zip_code_prefix": 1000,
            "seller_zip_code_prefix": 1100,
            "order_purchase_timestamp": "2018-01-01 10:00:00",
            "order_approved_at": "2018-01-01 10:30:00",
            "order_estimated_delivery_date": "2018-01-10",
        }

        response = client.post("/predict", json=valid_order)

        assert response.status_code == 503
        assert response.json()["detail"] == "MLflow model is not loaded."

    finally:
        api.model = original_model


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["request_count"] > 0
    assert data["error_count"] > 0
    assert data["error_rate"] > 0
    assert data["average_latency_seconds"] >= 0
    assert data["prediction_distribution"].get("Late", 0) >= 1
