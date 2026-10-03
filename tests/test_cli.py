import json

import pytest

from src.cli import load_order_records


def test_load_single_order_json(tmp_path):
    input_path = tmp_path / "order.json"
    order = {"item_count": 1}
    input_path.write_text(json.dumps(order), encoding="utf-8")

    assert load_order_records(input_path) == [order]


def test_load_order_list_json(tmp_path):
    input_path = tmp_path / "orders.json"
    orders = [{"item_count": 1}, {"item_count": 2}]
    input_path.write_text(json.dumps(orders), encoding="utf-8")

    assert load_order_records(input_path) == orders


@pytest.mark.parametrize("payload", [[], [1], "invalid"])
def test_reject_invalid_json_payload(tmp_path, payload):
    input_path = tmp_path / "invalid.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Input must be"):
        load_order_records(input_path)
