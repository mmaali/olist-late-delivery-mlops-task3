# FastAPI service for the Olist late delivery prediction model


import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import load_geolocation
from src.monitoring import metrics, prediction_drift, timer
from src.pipeline import run_inference

# Project root and configuration file

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# Load the project configuration

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)


# MLflow tracking server

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI", "models:/olist_late_delivery_model@Production"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Build the paths used by the inference pipeline

PREPROCESSOR_PATH = PROJECT_ROOT / CONFIG["paths"]["preprocessor"]

THRESHOLD_PATH = PROJECT_ROOT / CONFIG["paths"]["threshold"]

GEOLOCATION_PATH = PROJECT_ROOT / CONFIG["paths"]["geolocation"]


# Basic model information

MODEL_NAME = CONFIG["model"]["name"]

MODEL_VERSION = CONFIG["model"]["version"]

MODEL_THRESHOLD = float(CONFIG["model"]["threshold"])


# Configure application logging

LOG_LEVEL = getattr(logging, CONFIG["logging"]["level"].upper(), logging.INFO)

LOG_FILE = PROJECT_ROOT / CONFIG["logging"]["file"]

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger("olist_api")

logger.setLevel(LOG_LEVEL)

logger.propagate = False


# Avoid adding duplicate logging handlers

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)


# Create the FastAPI application

app = FastAPI(
    title="Olist Late Delivery Prediction API",
    version=MODEL_VERSION,
    description=(
        "API for predicting whether an Olist order will be delivered late or on time."
    ),
)


@app.middleware("http")
async def record_request_metrics(request, call_next):
    started_at = timer()

    try:
        response = await call_next(request)
    except Exception:
        metrics.record_request(timer() - started_at)
        metrics.record_error()
        raise

    metrics.record_request(timer() - started_at)
    if response.status_code >= 400:
        metrics.record_error()

    return response


# Objects loaded when the application starts

geolocation_data = None

model = None


# Load required data and the registered MLflow model


@app.on_event("startup")
def startup_event():
    global geolocation_data
    global model

    # Load geolocation data

    geolocation_data = load_geolocation(GEOLOCATION_PATH)

    logger.info("Geolocation data loaded successfully.")

    # Load the registered model from MLflow

    model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)

    logger.info("Registered model loaded from MLflow: %s", MLFLOW_MODEL_URI)


# Request schema for one new order


class OrderRequest(BaseModel):
    item_count: int = Field(..., ge=0)

    total_price: float = Field(..., ge=0)

    total_freight: float = Field(..., ge=0)

    payment_count: int = Field(..., ge=0)

    total_payment: float = Field(..., ge=0)

    payment_installments: int = Field(..., ge=0)

    unique_products: int = Field(..., ge=0)

    average_product_weight: float = Field(..., ge=0)

    average_product_photos: float = Field(..., ge=0)

    customer_state: str = Field(..., min_length=2, max_length=2)

    seller_state: str = Field(..., min_length=2, max_length=2)

    customer_zip_code_prefix: int = Field(..., ge=0)

    seller_zip_code_prefix: int = Field(..., ge=0)

    order_purchase_timestamp: str

    order_approved_at: str

    order_estimated_delivery_date: str

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


# Response schema for one prediction


class PredictionResponse(BaseModel):
    prediction: str

    late_probability: float

    model_version: str


# Response schema for multiple predictions


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

    model_version: str


# Health check endpoint


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
def service_metrics():
    snapshot = metrics.snapshot()
    snapshot["prediction_drift"] = prediction_drift(
        snapshot,
        baseline_late_rate=CONFIG["monitoring"]["baseline_late_prediction_rate"],
        minimum_predictions=CONFIG["monitoring"]["minimum_predictions_for_drift_check"],
        threshold=CONFIG["monitoring"]["late_prediction_rate_deviation_threshold"],
    )
    return snapshot


# Model information endpoint


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": MODEL_THRESHOLD,
        "mlflow_model": MLFLOW_MODEL_URI,
    }


# Run the inference pipeline for incoming orders


def predict_orders(order_records):
    if geolocation_data is None:
        raise HTTPException(status_code=503, detail="Geolocation data is not loaded.")

    if model is None:
        raise HTTPException(status_code=503, detail="MLflow model is not loaded.")

    order_data = pd.DataFrame(order_records)

    start_time = time.perf_counter()

    try:
        results = run_inference(
            order_data=order_data,
            geolocation=geolocation_data,
            model=model,
            preprocessor_path=PREPROCESSOR_PATH,
            threshold_path=THRESHOLD_PATH,
        )

    except ValueError as error:
        logger.warning("Invalid prediction input: %s", error)

        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        logger.error("Required project file was not found: %s", error)

        raise HTTPException(status_code=500, detail=str(error))

    except Exception:
        logger.exception("Unexpected prediction error.")

        raise HTTPException(
            status_code=500, detail="An unexpected prediction error occurred."
        )

    latency_ms = (time.perf_counter() - start_time) * 1000
    metrics.record_predictions(result["prediction"] for result in results)

    logger.info(
        "Prediction request completed | "
        "input=%s | output=%s | "
        "latency_ms=%.2f | model_version=%s",
        order_records,
        results,
        latency_ms,
        MODEL_VERSION,
    )

    return results


# Predict delivery status for one order


@app.post("/predict", response_model=PredictionResponse)
def predict_order(order: OrderRequest):
    records = [order.model_dump()]

    results = predict_orders(records)

    result = results[0]

    return {
        "prediction": result["prediction"],
        "late_probability": result["late_probability"],
        "model_version": MODEL_VERSION,
    }


# Predict delivery status for multiple orders


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(orders: list[OrderRequest]):
    if not orders:
        raise HTTPException(status_code=400, detail="The order list cannot be empty.")

    records = [order.model_dump() for order in orders]

    results = predict_orders(records)

    predictions = []

    for result in results:
        predictions.append(
            {
                "prediction": result["prediction"],
                "late_probability": (result["late_probability"]),
                "model_version": MODEL_VERSION,
            }
        )

    return {"predictions": predictions, "model_version": MODEL_VERSION}
