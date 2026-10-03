# Functions for validating prediction input data

import pandas as pd

# Columns that must be available before feature engineering

REQUIRED_ORDER_COLUMNS = [
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


# Brazilian state codes used in the Olist dataset

ALLOWED_STATES = {
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
}


# Check that all required columns are available


def validate_required_columns(order_data):
    missing_columns = [
        column for column in REQUIRED_ORDER_COLUMNS if column not in order_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True


# Check that the input DataFrame contains data


def validate_not_empty(order_data):
    if order_data.empty:
        raise ValueError("The input order data is empty.")

    return True


# Check numerical columns for invalid negative values


def validate_numeric_values(order_data):
    numeric_columns = [
        "item_count",
        "total_price",
        "total_freight",
        "payment_count",
        "total_payment",
        "payment_installments",
        "unique_products",
        "average_product_weight",
        "average_product_photos",
        "customer_zip_code_prefix",
        "seller_zip_code_prefix",
    ]

    for column in numeric_columns:
        if column in order_data.columns:
            values = pd.to_numeric(
                order_data[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(f"Column '{column}' contains invalid numeric values.")

            if (values < 0).any():
                raise ValueError(f"Column '{column}' contains negative values.")

    return True


# Check that customer and seller state values are valid


def validate_states(order_data):
    for column in ["customer_state", "seller_state"]:
        if column not in order_data.columns:
            continue

        states = order_data[column].astype(str).str.upper().str.strip()

        invalid_states = sorted(set(states) - ALLOWED_STATES)

        if invalid_states:
            raise ValueError(f"Invalid state values in '{column}': {invalid_states}")

    return True


# Check that the date columns can be converted to datetime


def validate_dates(order_data):
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    ]

    for column in date_columns:
        dates = pd.to_datetime(
            order_data[column],
            errors="coerce",
        )

        if dates.isna().any():
            raise ValueError(f"Column '{column}' contains invalid date values.")

    return True


# Run all basic validation checks for a new order


def validate_order(order_data):
    validate_not_empty(order_data)
    validate_required_columns(order_data)
    validate_numeric_values(order_data)
    validate_states(order_data)
    validate_dates(order_data)

    return True
