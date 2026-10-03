import time
from collections import Counter
from threading import Lock


class Metrics:
    def __init__(self):
        self.lock = Lock()
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.predictions = Counter()

    def record_request(self, latency: float, prediction: str | None = None):
        with self.lock:
            self.request_count += 1
            self.total_latency += latency

            if prediction:
                self.predictions[prediction] += 1

    def record_error(self):
        with self.lock:
            self.error_count += 1

    def record_predictions(self, predictions):
        with self.lock:
            self.predictions.update(predictions)

    def snapshot(self):
        with self.lock:
            avg_latency = (
                self.total_latency / self.request_count if self.request_count else 0.0
            )

            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": (
                    self.error_count / self.request_count if self.request_count else 0.0
                ),
                "average_latency_seconds": avg_latency,
                "prediction_distribution": dict(self.predictions),
            }


metrics = Metrics()


def prediction_drift(snapshot, baseline_late_rate, minimum_predictions, threshold):
    distribution = snapshot["prediction_distribution"]
    prediction_count = sum(distribution.values())
    late_rate = (
        distribution.get("Late", 0) / prediction_count if prediction_count else None
    )
    deviation = abs(late_rate - baseline_late_rate) if late_rate is not None else None

    return {
        "predictions_seen": prediction_count,
        "baseline_late_prediction_rate": baseline_late_rate,
        "observed_late_prediction_rate": late_rate,
        "absolute_rate_deviation": deviation,
        "drift_alert": (
            prediction_count >= minimum_predictions
            and deviation is not None
            and deviation > threshold
        ),
    }


def timer():
    return time.perf_counter()
