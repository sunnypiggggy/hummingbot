from __future__ import annotations

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from grid_xgboost_risk_gate import (
    MIN_RISK_OFF_SECONDS,
    MODEL_VERSION,
    PairGateState,
    advance_pair_gate,
    build_contract,
    combine_pair_channels,
    fail_closed_pair,
    state_from_dict,
    state_to_dict,
    load_runtime_xgboost_gate,
)
from plot_xgboost_risk_gate_timing import build_timing_events
from build_xgboost_risk_gate_signal import ensure_live_cache
from tune_xgboost_momentum_stop_v2 import xgb_configurations
from validate_grid_live import Candidate, InventoryExitPolicy, simulate


def step(state, probability, ts, pair="BTC-FDUSD"):
    return advance_pair_gate(
        pair=pair, probability=probability, entry_threshold=0.8,
        recovery_threshold=0.6, signal_ts=ts, previous=state,
        model_version="test-model",
    )


def test_hysteresis_entry_boundary_and_four_hour_two_bar_recovery():
    state, signal = step(PairGateState(), 0.8, 0)
    assert state.risk_off_active and signal["transition"] == "enter"

    state, signal = step(state, 0.59, 3 * 3600)
    assert state.risk_off_active and state.consecutive_recovery_bars == 1
    state, signal = step(state, 0.59, MIN_RISK_OFF_SECONDS)
    assert not state.risk_off_active and signal["transition"] == "recover"


def test_recovery_boundary_and_invalid_high_reset_counter():
    state, _ = step(PairGateState(), 0.9, 0)
    state, _ = step(state, 0.5, 3600)
    assert state.consecutive_recovery_bars == 1
    state, _ = step(state, 0.6, 2 * 3600)
    assert state.consecutive_recovery_bars == 0
    state, _ = step(state, 0.5, 4 * 3600)
    state, signal = step(state, 0.5, 5 * 3600)
    assert signal["transition"] == "recover"


def test_pair_state_is_independent_and_restart_round_trip_is_exact():
    btc, _ = step(PairGateState(), 0.9, 100, "BTC-FDUSD")
    eth, eth_signal = step(PairGateState(), 0.1, 100, "ETH-FDUSD")
    assert btc.risk_off_active
    assert not eth.risk_off_active and eth_signal["buy_enabled"]
    assert state_from_dict(state_to_dict(btc)) == btc


def test_fail_closed_and_contract_have_no_sell_action():
    failed = fail_closed_pair(
        pair="BTC-FDUSD", signal_ts=100,
        model_version="test-model", reason="source_stale",
    )
    assert failed["risk_off_active"] and not failed["buy_enabled"]
    contract = build_contract(
        generated_at=100, valid_until=250, model_version="test-model",
        model_sha256="a" * 64, feature_sha256="b" * 64,
        data_sha256="c" * 64, source_healthy=False, deployment_allowed=False,
        pair_signals={"BTC-FDUSD": failed, "ETH-FDUSD": {**failed, "pair": "ETH-FDUSD"}},
        last_complete_1h={"BTC-FDUSD": 0, "ETH-FDUSD": 0},
        last_complete_4h={"BTC-FDUSD": 0, "ETH-FDUSD": 0},
    )
    encoded = str(contract).lower()
    assert contract["schema"] == "grid-xgboost-long-risk-gate-v1"
    assert contract["short_spike_enabled"] is False
    assert contract["market_sell_action"] is False
    assert contract["deployment_allowed"] is False
    assert all(not item["buy_enabled"] for item in contract["pairs"].values())
    assert all(item["reason"] == "source_unhealthy" for item in contract["pairs"].values())
    assert "stop_excess_inventory" not in encoded
    assert "taker" not in encoded
    assert contract["market_sell_action"] is False


def test_long_only_channel_remains_pair_local():
    clear = {
        "probability": 0.2, "entry_threshold": 0.8, "recovery_threshold": 0.6,
        "risk_off_active": False, "transition": "clear",
    }
    active = {**clear, "probability": 0.9, "risk_off_active": True, "transition": "enter"}
    btc = combine_pair_channels(
        pair="BTC-FDUSD", channels={"long": active},
        signal_ts=100, model_version="test-model",
    )
    eth = combine_pair_channels(
        pair="ETH-FDUSD", channels={"long": clear},
        signal_ts=100, model_version="test-model",
    )
    assert btc["risk_off_active"] and not btc["buy_enabled"]
    assert btc["active_channels"] == ["long"]
    assert not eth["risk_off_active"] and eth["buy_enabled"]


def test_long_runtime_loader_accepts_authorized_contract_and_fails_closed_when_stale(tmp_path):
    clear = {
        "probability": 0.2, "entry_threshold": 0.8, "recovery_threshold": 0.6,
        "risk_off_active": False, "transition": "clear",
    }
    pair_signals = {
        pair: combine_pair_channels(
            pair=pair, channels={"long": clear},
            signal_ts=100, model_version=MODEL_VERSION,
        ) for pair in ("BTC-FDUSD", "ETH-FDUSD")
    }
    contract = build_contract(
        generated_at=100, valid_until=250, model_version=MODEL_VERSION,
        model_sha256="a" * 64, feature_sha256="b" * 64, data_sha256="c" * 64,
        source_healthy=True, deployment_allowed=True, pair_signals=pair_signals,
        last_complete_1h={"BTC-FDUSD": 99, "ETH-FDUSD": 99},
        last_complete_4h={"BTC-FDUSD": 96, "ETH-FDUSD": 96},
    )
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    healthy = load_runtime_xgboost_gate(
        path, now=datetime.fromtimestamp(120, timezone.utc),
        expected_model_sha256="a" * 64, expected_feature_sha256="b" * 64,
    )
    assert healthy["runtime_gate_healthy"]
    assert all(item["buy_enabled"] for item in healthy["pairs"].values())
    stale = load_runtime_xgboost_gate(path, now=datetime.fromtimestamp(400, timezone.utc))
    assert not stale["runtime_gate_healthy"]
    assert all(not item["buy_enabled"] for item in stale["pairs"].values())


def test_runtime_loader_rejects_any_short_spike_contract(tmp_path):
    clear = {
        "probability": 0.2, "entry_threshold": 0.8, "recovery_threshold": 0.6,
        "risk_off_active": False, "transition": "clear",
    }
    pair_signals = {
        pair: combine_pair_channels(
            pair=pair, channels={"long": clear}, signal_ts=100, model_version=MODEL_VERSION,
        ) for pair in ("BTC-FDUSD", "ETH-FDUSD")
    }
    contract = build_contract(
        generated_at=100, valid_until=250, model_version=MODEL_VERSION,
        model_sha256="a" * 64, feature_sha256="b" * 64, data_sha256="c" * 64,
        source_healthy=True, deployment_allowed=True, pair_signals=pair_signals,
        last_complete_1h={"BTC-FDUSD": 99, "ETH-FDUSD": 99},
        last_complete_4h={"BTC-FDUSD": 96, "ETH-FDUSD": 96},
    )
    contract["short_spike_enabled"] = True
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    failed = load_runtime_xgboost_gate(path, now=datetime.fromtimestamp(120, timezone.utc))
    assert not failed["runtime_gate_healthy"]
    assert "short-spike channel is forbidden" in failed["reason"]


def test_live_candle_cache_is_seeded_without_mutating_research_files(tmp_path):
    seed = tmp_path / "research"
    live = tmp_path / "live"
    seed.mkdir()
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        (seed / f"binance_{pair}_5m.csv").write_text("timestamp,close\n1,2\n", encoding="utf-8")
    ensure_live_cache(live, seed)
    (live / "binance_BTC-FDUSD_5m.csv").write_text("changed", encoding="utf-8")
    assert (seed / "binance_BTC-FDUSD_5m.csv").read_text(encoding="utf-8") == "timestamp,close\n1,2\n"


def _candles(prices):
    times = [index * 300 for index in range(len(prices))]
    return pd.DataFrame({
        "timestamp": times, "open": prices, "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices], "close": prices,
        "volume": [100.0] * len(prices),
    })


def test_buy_gate_does_not_enable_momentum_stop_exit():
    candles = {
        "BTC-FDUSD": _candles([100.0] * 80),
        "ETH-FDUSD": _candles([50.0] * 80),
    }
    gate = {
        "BTC-FDUSD": {ts: False for ts in candles["BTC-FDUSD"].timestamp},
        "ETH-FDUSD": {ts: True for ts in candles["ETH-FDUSD"].timestamp},
    }
    trades = []
    result, _, pairs = simulate(
        candles, Candidate(0.03, 0.006, 0.006, 0.015, 1800),
        maker_fee=0.0, taker_fee=0.001, order_refresh_seconds=7200,
        technical_buy_gate=gate, momentum_stop_timeline=None, trade_log=trades,
        risk_breakers_enabled=False, cost_floor_enabled=True,
        inventory_exit_policy=InventoryExitPolicy(10, 172800, 0.0, 0.0, 0.5, 0.75),
    )
    assert result["momentum_stop_enabled"] is False
    assert result["momentum_stop_exits"] == 0
    assert not any(t["reason"] == "momentum_stop_exit" for t in trades)
    assert pairs["BTC-FDUSD"]["technical_risk_off_seconds"] > 0
    assert pairs["ETH-FDUSD"]["technical_risk_off_seconds"] == 0


def test_xgboost_parameter_space_stays_at_forty_unique_with_anchors():
    configs = xgb_configurations()
    assert len(configs) == 40
    assert len({tuple(sorted(item.items())) for item in configs}) == 40
    assert configs[0]["kind"] == "legacy"
    assert configs[1]["kind"] == "regularized_anchor"


def test_plotly_timing_events_include_model_recovery_and_weekly_reset():
    states = pd.DataFrame([
        {"fold": 1, "pair": "BTC-FDUSD", "signal_ts": 100, "probability": .9, "entry_threshold": .8, "recovery_threshold": .6},
        {"fold": 1, "pair": "BTC-FDUSD", "signal_ts": 200, "probability": .5, "entry_threshold": .8, "recovery_threshold": .6},
        {"fold": 2, "pair": "ETH-FDUSD", "signal_ts": 300, "probability": .9, "entry_threshold": .8, "recovery_threshold": .6},
        {"fold": 2, "pair": "ETH-FDUSD", "signal_ts": 350, "probability": .7, "entry_threshold": .8, "recovery_threshold": .6},
    ])
    intervals = pd.DataFrame([
        {"fold": 1, "pair": "BTC-FDUSD", "start_ts": 100, "end_ts": 200, "duration_hours": 4, "end_reason": "recover"},
        {"fold": 2, "pair": "ETH-FDUSD", "start_ts": 300, "end_ts": 400, "duration_hours": 3, "end_reason": "weekly_reinitialization"},
    ])
    events = build_timing_events(states, intervals)
    assert set(events.event) == {"ENTER", "EXIT_RECOVER", "EXIT_WEEK_RESET"}
    assert len(events) == 4
