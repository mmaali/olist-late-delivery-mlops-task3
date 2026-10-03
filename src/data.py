# Functions for loading project data

from pathlib import Path

import pandas as pd

# Load a CSV file and return it as a DataFrame


def load_csv(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file was not found: {path}")

    return pd.read_csv(path)


# Load the Olist geolocation dataset


def load_geolocation(file_path):
    geolocation = load_csv(file_path)

    required_columns = [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    ]

    missing_columns = [
        column for column in required_columns if column not in geolocation.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing geolocation columns: {missing_columns}")

    return geolocation


# Load a new order from a CSV file


def load_order(file_path):
    order_data = load_csv(file_path)

    if order_data.empty:
        raise ValueError("The input order file is empty.")

    return order_data
