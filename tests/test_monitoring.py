import pytest

from src.monitoring import Metrics, prediction_drift


def test_metrics_snapshot_tracks_requests_errors_latency_and_predictions():
    metrics = Metrics()
    metrics.record_request(0.2)
    metrics.record_request(0.4)
    metrics.record_error()
    metrics.record_predictions(["Late", "On Time", "Late"])

    snapshot = metrics.snapshot()

    assert snapshot["request_count"] == 2
    assert snapshot["error_count"] == 1
    assert snapshot["error_rate"] == 0.5
    assert snapshot["average_latency_seconds"] == pytest.approx(0.3)
    assert snapshot["prediction_distribution"] == {"Late": 2, "On Time": 1}


def test_prediction_drift_alert_waits_for_minimum_sample():
    snapshot = {"prediction_distribution": {"Late": 39, "On Time": 60}}

    result = prediction_drift(snapshot, 0.27448, 100, 0.10)

    assert result["predictions_seen"] == 99
    assert result["drift_alert"] is False


def test_prediction_drift_alerts_on_material_rate_change():
    snapshot = {"prediction_distribution": {"Late": 40, "On Time": 60}}

    result = prediction_drift(snapshot, 0.27448, 100, 0.10)

    assert result["predictions_seen"] == 100
    assert result["observed_late_prediction_rate"] == 0.4
    assert result["absolute_rate_deviation"] == pytest.approx(0.12552)
    assert result["drift_alert"] is True
