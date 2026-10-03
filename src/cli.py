"""Run registered-model inference from a JSON file or standard input."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from src.data import load_geolocation
from src.pipeline import run_inference

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_order_records(input_path):
    if input_path == "-":
        payload = json.load(sys.stdin)
    else:
        with open(input_path, encoding="utf-8") as input_file:
            payload = json.load(input_file)

    if isinstance(payload, dict):
        return [payload]

    if (
        isinstance(payload, list)
        and payload
        and all(isinstance(record, dict) for record in payload)
    ):
        return payload

    raise ValueError("Input must be one order object or a non-empty list of orders.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict late delivery for one or more Olist orders."
    )
    parser.add_argument(
        "input",
        help="JSON file containing an order or order list; use '-' for stdin.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("olist_cli")

    try:
        with open(CONFIG_PATH, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        records = load_order_records(args.input)
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        model_uri = os.getenv(
            "MLFLOW_MODEL_URI",
            "models:/olist_late_delivery_model@Production",
        )
        model = mlflow.sklearn.load_model(model_uri)
        geolocation = load_geolocation(PROJECT_ROOT / config["paths"]["geolocation"])

        results = run_inference(
            order_data=pd.DataFrame(records),
            geolocation=geolocation,
            model=model,
            preprocessor_path=PROJECT_ROOT / config["paths"]["preprocessor"],
            threshold_path=PROJECT_ROOT / config["paths"]["threshold"],
        )
        response = [
            {**result, "model_version": config["model"]["version"]}
            for result in results
        ]
        sys.stdout.write(json.dumps(response, indent=2) + "\n")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        logger.error("Inference failed: %s", error)
        return 1
    except Exception:
        logger.exception("Inference failed unexpectedly.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
