# Data validation using Great Expectations


import great_expectations as gx
from great_expectations import expectations as gxe

# Validate incoming order data before feature engineering


def validate_with_great_expectations(order_data):
    # Create a Great Expectations context
    context = gx.get_context()

    # Create a temporary pandas data source
    data_source = context.data_sources.add_pandas(name="order_data_source")

    # Create a data asset for the incoming DataFrame
    data_asset = data_source.add_dataframe_asset(name="order_data_asset")

    # Create a batch definition for the whole DataFrame
    batch_definition = data_asset.add_batch_definition_whole_dataframe("order_batch")

    # Create the batch from the incoming data
    batch = batch_definition.get_batch(batch_parameters={"dataframe": order_data})

    # Define the expected input columns
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
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    ]

    # Create the expectation suite
    suite = gx.ExpectationSuite(name="order_input_expectations")

    # Check that the required columns exist
    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(column_set=expected_columns, exact_match=True)
    )

    # Check that important numerical values are not negative
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
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=column, min_value=0)
        )

    # Check that customer states are valid
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="customer_state",
            value_set=[
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
            ],
        )
    )

    # Check that seller states are valid
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="seller_state",
            value_set=[
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
            ],
        )
    )

    # Check that important input columns do not contain missing values
    required_non_null_columns = [
        "item_count",
        "total_price",
        "total_freight",
        "payment_count",
        "total_payment",
        "customer_state",
        "seller_state",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    ]

    for column in required_non_null_columns:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column))

    # Run all expectations against the incoming data
    validation_result = batch.validate(suite)

    # Reject the data if any expectation fails
    if not validation_result["success"]:
        raise ValueError("Great Expectations validation failed.")

    return True
