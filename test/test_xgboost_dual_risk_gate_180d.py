from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest_xgboost_dual_risk_gate_180d import (  # noqa: E402
    FIVE_MINUTES,
    PAIRS,
    combine_channel_gates,
    validate_target_rates,
)


def test_dual_gate_pauses_only_the_pair_triggered_by_either_channel():
    start, end = 1_000_000, 1_000_000 + 3 * FIVE_MINUTES
    timestamps = list(range(start, end, FIVE_MINUTES))
    long_gate = {pair: {ts: True for ts in timestamps} for pair in PAIRS}
    short_gate = {pair: {ts: True for ts in timestamps} for pair in PAIRS}
    long_gate["BTC-FDUSD"][timestamps[0]] = False
    short_gate["ETH-FDUSD"][timestamps[1]] = False
    combined = combine_channel_gates([long_gate, short_gate], start, end)
    assert combined["BTC-FDUSD"] == {
        timestamps[0]: False, timestamps[1]: True, timestamps[2]: True,
    }
    assert combined["ETH-FDUSD"] == {
        timestamps[0]: True, timestamps[1]: False, timestamps[2]: True,
    }


def test_dual_gate_requires_at_least_one_channel():
    with pytest.raises(ValueError, match="At least one"):
        combine_channel_gates([], 0, FIVE_MINUTES)


def test_pair_specific_channels_ignore_the_unrelated_pair():
    start, end = 0, 3 * FIVE_MINUTES
    btc = {
        "BTC-FDUSD": {0: True, FIVE_MINUTES: False, 2 * FIVE_MINUTES: True},
        "ETH-FDUSD": {},
    }
    eth = {
        "BTC-FDUSD": {},
        "ETH-FDUSD": {0: True, FIVE_MINUTES: True, 2 * FIVE_MINUTES: False},
    }
    combined = combine_channel_gates([btc, eth], start, end)
    assert combined["BTC-FDUSD"] == btc["BTC-FDUSD"]
    assert combined["ETH-FDUSD"] == eth["ETH-FDUSD"]


def test_applicable_channel_missing_timestamp_is_fail_closed():
    btc = {"BTC-FDUSD": {0: True, 2 * FIVE_MINUTES: True}, "ETH-FDUSD": {}}
    eth = {
        "BTC-FDUSD": {},
        "ETH-FDUSD": {0: True, FIVE_MINUTES: True, 2 * FIVE_MINUTES: True},
    }
    combined = combine_channel_gates([btc, eth], 0, 3 * FIVE_MINUTES)
    assert combined["BTC-FDUSD"][FIVE_MINUTES] is False


def test_pair_without_an_applicable_channel_is_rejected():
    channel = {"BTC-FDUSD": {0: True}, "ETH-FDUSD": {}}
    with pytest.raises(ValueError, match="No applicable risk channel for ETH-FDUSD"):
        combine_channel_gates([channel], 0, FIVE_MINUTES)


def test_target_quality_rejects_overbroad_labels():
    frame = pd.DataFrame({
        "pair": ["BTC-FDUSD"] * 10 + ["ETH-FDUSD"] * 10,
        "target_long": [1.0] * 20,
        "target_short": [0.0, 1.0] * 10,
    })
    with pytest.raises(RuntimeError, match="outside 5%-40%"):
        validate_target_rates(frame)
