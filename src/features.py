# Libraries used for data manipulation and mathematical calculations

import numpy as np
import pandas as pd

# These are the 20 features used by the final Experiment 2 model

FEATURE_COLUMNS = [
    # Order and payment information
    "item_count",
    "total_price",
    "total_freight",
    "payment_count",
    "total_payment",
    "payment_installments",
    # Product information
    "unique_products",
    "average_product_weight",
    "average_product_photos",
    # Customer and seller location
    "customer_state",
    "seller_state",
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    # Geographical features
    "same_state",
    "distance_km",
    # Time-related features
    "estimated_delivery_days",
    "approval_delay_hours",
    "order_day_of_week",
    "order_month",
    "is_weekend",
]


# Convert ZIP-code prefixes to a consistent five-digit format


def standardize_zip(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", np.nan)
        .str.zfill(5)
    )


# Create one average latitude and longitude for each ZIP-code prefix


def create_geo_lookup(geolocation):
    geolocation = geolocation.copy()

    # Standardize the ZIP-code prefix in the geolocation data
    geolocation["geolocation_zip_code_prefix"] = standardize_zip(
        geolocation["geolocation_zip_code_prefix"]
    )

    # Calculate the average coordinates for each ZIP-code prefix
    geo_by_zip = geolocation.groupby("geolocation_zip_code_prefix", as_index=False)[
        ["geolocation_lat", "geolocation_lng"]
    ].mean()

    return geo_by_zip


# Add the geographical distance between the customer and seller


def add_distance_feature(df, geo_by_zip):
    df = df.copy()

    # Standardize customer and seller ZIP-code prefixes
    df["customer_zip_temp"] = standardize_zip(df["customer_zip_code_prefix"])

    df["seller_zip_temp"] = standardize_zip(df["seller_zip_code_prefix"])

    # Prepare customer coordinates
    customer_geo = geo_by_zip.rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_temp",
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
        }
    )

    # Add customer coordinates to the order data
    df = df.merge(
        customer_geo,
        on="customer_zip_temp",
        how="left",
    )

    # Prepare seller coordinates
    seller_geo = geo_by_zip.rename(
        columns={
            "geolocation_zip_code_prefix": "seller_zip_temp",
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
        }
    )

    # Add seller coordinates to the order data
    df = df.merge(
        seller_geo,
        on="seller_zip_temp",
        how="left",
    )

    # Convert latitude and longitude from degrees to radians
    customer_lat = np.radians(df["customer_lat"])
    customer_lng = np.radians(df["customer_lng"])

    seller_lat = np.radians(df["seller_lat"])
    seller_lng = np.radians(df["seller_lng"])

    # Calculate the differences between customer and seller coordinates
    latitude_difference = seller_lat - customer_lat
    longitude_difference = seller_lng - customer_lng

    # Apply the Haversine formula
    haversine_value = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(customer_lat)
        * np.cos(seller_lat)
        * np.sin(longitude_difference / 2) ** 2
    )

    # Calculate the distance in kilometers
    df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(haversine_value))

    # Remove temporary columns that are no longer needed
    df.drop(
        columns=[
            "customer_zip_temp",
            "seller_zip_temp",
            "customer_lat",
            "customer_lng",
            "seller_lat",
            "seller_lng",
        ],
        inplace=True,
    )

    return df


# Check that all required input columns are available before feature engineering


def check_required_columns(order_data):
    required_columns = [
        "item_count",
        "total_price",
        "total_freight",
        "payment_count",
        "total_payment",
        "payment_installments",
        "unique_products",
        "average_product_weight",
        "average_product_photos",
        "customer_state",
        "seller_state",
        "customer_zip_code_prefix",
        "seller_zip_code_prefix",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    ]

    missing_columns = [
        column for column in required_columns if column not in order_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


# Build all prediction-time features required by the Experiment 2 model


def build_features(order_data, geolocation):
    order_data = order_data.copy()

    # Check that the input contains all columns needed for feature engineering
    check_required_columns(order_data)

    # Convert the date columns to datetime values
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        order_data[column] = pd.to_datetime(
            order_data[column],
            errors="coerce",
        )

    # Calculate the estimated delivery duration in days
    order_data["estimated_delivery_days"] = (
        order_data["order_estimated_delivery_date"]
        - order_data["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Calculate how many hours passed before the order was approved
    order_data["approval_delay_hours"] = (
        order_data["order_approved_at"] - order_data["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    # Create calendar-based features from the purchase date
    order_data["order_day_of_week"] = order_data[
        "order_purchase_timestamp"
    ].dt.dayofweek

    order_data["order_month"] = order_data["order_purchase_timestamp"].dt.month

    order_data["is_weekend"] = (
        order_data["order_purchase_timestamp"].dt.dayofweek >= 5
    ).astype(int)

    # Create the geographical distance feature
    geo_by_zip = create_geo_lookup(geolocation)

    order_data = add_distance_feature(
        order_data,
        geo_by_zip,
    )

    # Create a feature that indicates whether customer and seller
    # are located in the same state
    order_data["same_state"] = (
        order_data["customer_state"] == order_data["seller_state"]
    ).astype(int)

    # Keep only the features expected by the final model
    order_features = order_data[FEATURE_COLUMNS].copy()

    return order_features
