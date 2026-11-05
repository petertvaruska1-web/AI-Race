import json
import math

import pytest
import torch

from airace_ml.worker.speed import gpu_name, measure_speed, parse_connect_link


def test_the_speed_test_measures_tokens_a_second_and_remembers_it(tmp_path):
    speed = measure_speed(torch.device("cpu"), seconds=0.3, cache_dir=tmp_path)
    assert speed > 0 and math.isfinite(speed)
    cached = json.loads((tmp_path / "speed.json").read_text())
    assert cached["CPU"] == speed
    # A second call reuses the measurement.
    assert measure_speed(torch.device("cpu"), seconds=0.3, cache_dir=tmp_path) == speed


def test_a_computer_without_a_graphics_card_is_named_cpu():
    assert gpu_name(torch.device("cpu")) == "CPU"


def test_a_game_link_gives_the_worker_address_and_the_key():
    link = "airace-helper://connect?url=https%3A%2F%2Fabc.trycloudflare.com&key=u1.mac"
    assert parse_connect_link(link) == ("wss://abc.trycloudflare.com/worker", "u1.mac")
    local = "airace-helper://connect?url=http%3A%2F%2Flocalhost%3A8787&key=k"
    assert parse_connect_link(local) == ("ws://localhost:8787/worker", "k")
    assert parse_connect_link("ws://127.0.0.1:8787/worker") == ("ws://127.0.0.1:8787/worker", None)
    with pytest.raises(ValueError, match="game link"):
        parse_connect_link("airace-helper://connect?url=ftp%3A%2F%2Fx&key=k")
    with pytest.raises(ValueError, match="game link"):
        parse_connect_link("https://example.com")
