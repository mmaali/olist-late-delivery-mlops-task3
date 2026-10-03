# Main inference pipeline
# This module connects validation, data validation,
# feature engineering, preprocessing, and model prediction.


from src.data_validation import validate_with_great_expectations
from src.features import build_features
from src.prediction import load_threshold, predict
from src.preprocessing import prepare_features
from src.validation import validate_order

# Run the complete prediction pipeline for new orders


def run_inference(order_data, geolocation, model, preprocessor_path, threshold_path):
    # Step 1: Run the basic input validation

    validate_order(order_data)

    # Step 2: Run Great Expectations validation

    validate_with_great_expectations(order_data)

    # Step 3: Build the Experiment 2 prediction features

    features = build_features(order_data, geolocation)

    # Step 4: Apply the fitted Experiment 2 preprocessor

    processed_features = prepare_features(features, preprocessor_path)

    # Step 5: Load the selected classification threshold

    threshold = load_threshold(threshold_path)

    # Step 6: Generate the final prediction

    results = predict(processed_features, model, threshold)

    return results
