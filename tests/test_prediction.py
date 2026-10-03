# Unit tests for the prediction functions


import numpy as np

from src.prediction import (
    apply_threshold,
)

# Test the classification threshold


def test_apply_threshold():
    probabilities = np.array(
        [
            0.20,
            0.35,
            0.80,
        ]
    )

    predictions = apply_threshold(probabilities, 0.35)

    assert list(predictions) == [
        "On Time",
        "Late",
        "Late",
    ]
