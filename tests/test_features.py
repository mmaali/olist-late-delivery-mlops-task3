# Unit tests for feature engineering


import pandas as pd

from src.features import FEATURE_COLUMNS, build_features

# Create a small example geolocation dataset


def create_geolocation():
    return pd.DataFrame(
        [
            {
                "geolocation_zip_code_prefix": 1000,
                "geolocation_lat": -23.55,
                "geolocation_lng": -46.63,
            },
            {
                "geolocation_zip_code_prefix": 1100,
                "geolocation_lat": -23.56,
                "geolocation_lng": -46.64,
            },
        ]
    )


# Create a valid order for feature engineering


def create_order():
    return pd.DataFrame(
        [
            {
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
        ]
    )


# Test that feature engineering returns the expected 20 features


def test_build_features():
    order = create_order()

    geolocation = create_geolocation()

    features = build_features(order, geolocation)

    assert features.shape[1] == 20


# Test that the generated features contain the expected columns


def test_feature_columns():
    order = create_order()

    geolocation = create_geolocation()

    features = build_features(order, geolocation)

    expected_columns = [
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
        "same_state",
        "distance_km",
        "estimated_delivery_days",
        "approval_delay_hours",
        "order_day_of_week",
        "order_month",
        "is_weekend",
    ]

    assert list(features.columns) == expected_columns


def test_feature_list_excludes_leakage_columns():
    leakage_columns = {
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "delivery_days",
        "review_score",
        "review_count",
    }

    assert leakage_columns.isdisjoint(FEATURE_COLUMNS)
