# Unit tests for the input validation functions


import pandas as pd
import pytest

from src.data_validation import validate_with_great_expectations
from src.validation import (
    validate_dates,
    validate_not_empty,
    validate_numeric_values,
    validate_required_columns,
    validate_states,
)

# Create a valid example order for testing


def create_valid_order():
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


# Test that a valid DataFrame is not considered empty


def test_validate_not_empty():
    order = create_valid_order()

    assert validate_not_empty(order) is True


# Test that all required columns are available


def test_validate_required_columns():
    order = create_valid_order()

    assert validate_required_columns(order) is True


# Test that invalid state values are rejected


def test_invalid_state():
    order = create_valid_order()

    order.loc[0, "customer_state"] = "XX"

    with pytest.raises(ValueError):
        validate_states(order)


# Test that invalid dates are rejected


def test_invalid_date():
    order = create_valid_order()

    order.loc[0, "order_approved_at"] = "invalid-date"

    with pytest.raises(ValueError):
        validate_dates(order)


def test_invalid_numeric_range():
    order = create_valid_order()
    order.loc[0, "total_price"] = -1

    with pytest.raises(ValueError, match="negative values"):
        validate_numeric_values(order)


def test_great_expectations_accept_valid_order():
    assert validate_with_great_expectations(create_valid_order()) is True


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [("customer_state", "XX"), ("total_price", -1), ("item_count", None)],
)
def test_great_expectations_reject_invalid_order(column, invalid_value):
    order = create_valid_order()
    order.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match="Great Expectations validation failed"):
        validate_with_great_expectations(order)
