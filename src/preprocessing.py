# Functions for loading and applying the fitted preprocessing pipeline

from pathlib import Path

import joblib

# Load the fitted preprocessor saved during Task 2


def load_preprocessor(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file was not found: {path}")

    return joblib.load(path)


# Apply the already fitted preprocessor to new features


def preprocess_features(features, preprocessor):
    if features.empty:
        raise ValueError("The feature data is empty.")

    processed_features = preprocessor.transform(features)

    return processed_features


# Load the preprocessor and transform the features


def prepare_features(features, file_path):
    preprocessor = load_preprocessor(file_path)

    processed_features = preprocess_features(
        features,
        preprocessor,
    )

    return processed_features
