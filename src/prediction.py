# Functions for loading the trained model
# and making prediction results


import json
from pathlib import Path

import joblib
import numpy as np

# Load the final Experiment 2 model


def load_model(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file was not found: {path}")

    return joblib.load(path)


# Load the selected classification threshold


def load_threshold(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Threshold file was not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    if "threshold" not in threshold_data:
        raise ValueError("Threshold value was not found in the JSON file.")

    threshold = float(threshold_data["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")

    return threshold


# Get the probability of the Late class


def get_late_probability(processed_features, model):
    probabilities = model.predict_proba(processed_features)

    classes = list(model.classes_)

    if "Late" not in classes:
        raise ValueError("The model does not contain a 'Late' class.")

    late_class_index = classes.index("Late")

    late_probability = probabilities[:, late_class_index]

    return late_probability


# Convert the Late probability into a final prediction


def apply_threshold(late_probabilities, threshold):
    predictions = np.where(late_probabilities >= threshold, "Late", "On Time")

    return predictions


# Make predictions for processed features


def predict(processed_features, model, threshold):
    late_probabilities = get_late_probability(processed_features, model)

    predictions = apply_threshold(late_probabilities, threshold)

    results = []

    for prediction, probability in zip(predictions, late_probabilities):
        results.append(
            {
                "prediction": prediction,
                "late_probability": float(probability),
            }
        )

    return results


# Load the model and threshold, then make predictions


def make_prediction(processed_features, model_path, threshold_path):
    model = load_model(model_path)

    threshold = load_threshold(threshold_path)

    results = predict(processed_features, model, threshold)

    return results
