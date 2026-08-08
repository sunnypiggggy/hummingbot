import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.build_v22_grid_dca_forced_exit_v2 import (
    PACKAGE_ID, POLICY, canonical_hash, executable_qty, floor_qty, inventory_overlay,
)


OUTPUT = Path("results/backtests/v22_grid_dca_forced_exit_v2")
FROZEN = Path("results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/application_bundle")


def epoch(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def test_execution_filter_rounds_down_and_preserves_dust() -> None:
    assert floor_qty("BTC-FDUSD", .001239) == .00123
    quantity, dust = executable_qty("ETH-FDUSD", .01009, 2000)
    assert quantity == .01
    assert 0 < dust < .0001
    quantity, dust = executable_qty("BTC-USDT", .00001, 40_000)
    assert quantity == 0 and dust == .00001


def test_policy_is_versioned_offline_execution_overlay() -> None:
    assert PACKAGE_ID == "ethbtc-forced-exit"
    assert POLICY["package_id"] == PACKAGE_ID
    assert POLICY["execution_policy_version"] == "v22-risk-off-forced-exit-v2"
    assert POLICY["technical_cooldown_seconds"] == 0
    assert POLICY["required_healthy_guard_cycles"] == 3
    assert len(canonical_hash(POLICY)) == 64


def test_dca_inventory_exits_when_first_bar_is_already_risk_off() -> None:
    frame = pd.DataFrame([
        {"timestamp": 1000, "open": 10_000.0, "close": 10_000.0},
        {"timestamp": 1300, "open": 9_900.0, "close": 9_900.0},
    ])
    inventory, actions = inventory_overlay(
        frame, pd.Series([False, False]), "BTC-USDT", forced=True,
    )
    assert inventory.phase.tolist() == ["EXITING", "COOLDOWN"]
    assert len(actions) == 1
    assert actions.iloc[0].action == "MARKET_EXIT"
    assert int(actions.iloc[0].signal_ts) == 1000
    assert int(actions.iloc[0].execution_ts) == 1300


def test_generated_target_window_and_isolation() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "NO-GO"
    assert summary["package_id"] == "ethbtc-forced-exit"
    assert summary["offline_only"] is True
    assert summary["deployment_allowed"] is False
    assert summary["promotion_authorized"] is False
    actions = pd.read_csv(OUTPUT / "execution_actions.csv")
    expected = {
        ("grid", "BTC-FDUSD", epoch("2026-05-13T00:00:00Z")): epoch("2026-05-13T00:05:00Z"),
        ("grid", "ETH-FDUSD", epoch("2026-05-23T00:00:00Z")): epoch("2026-05-23T00:05:00Z"),
        ("dca", "BTC-USDT", epoch("2026-05-13T00:00:00Z")): epoch("2026-05-13T00:05:00Z"),
        ("dca", "ETH-USDT", epoch("2026-05-23T00:00:00Z")): epoch("2026-05-23T00:05:00Z"),
    }
    exits = actions[actions.action.eq("MARKET_EXIT")]
    for key, execution in expected.items():
        strategy, pair, signal = key
        row = exits[(exits.strategy == strategy) & (exits.pair == pair) & (exits.signal_ts == signal)]
        assert len(row) == 1 and int(row.iloc[0].execution_ts) == execution
        assert bool(row.iloc[0].orders_cancelled)
    reentries = actions[actions.action.eq("MARKET_REENTRY")]
    assert len(reentries[(reentries.strategy == "grid") & (reentries.pair == "BTC-FDUSD") &
                         (reentries.signal_ts == epoch("2026-06-07T12:00:00Z"))]) == 1
    assert len(reentries[(reentries.strategy == "grid") & (reentries.pair == "ETH-FDUSD") &
                         (reentries.signal_ts == epoch("2026-06-07T16:00:00Z"))]) == 1
    assert len(reentries[(reentries.strategy == "dca") & (reentries.pair == "BTC-USDT") &
                         (reentries.signal_ts == epoch("2026-06-07T12:00:00Z"))]) == 1
    assert len(reentries[(reentries.strategy == "dca") & (reentries.pair == "ETH-USDT") &
                         (reentries.signal_ts == epoch("2026-06-07T16:00:00Z"))]) == 1
    curve = pd.read_csv(OUTPUT / "grid_combined_equity.csv.gz")
    assert (curve.equity - (420 + curve.cumulative_oos_pnl)).abs().max() < 1e-9
    cash_window = curve[curve.timestamp.between(epoch("2026-05-23T00:05:00Z"), epoch("2026-06-07T12:00:00Z"))]
    assert cash_window.equity.max() - cash_window.equity.min() < .1
    html = (OUTPUT / "v22_grid_dca_forced_exit_v2.html").read_text(encoding="utf-8")
    assert "机制事件数" not in html and "<pre" not in html
    assert "BTC+ETH组合连续权益" not in html and "ETH-FDUSD 机器人阴影" in html
    assert "BTC-FDUSD 机器人连续权益" in html and "ETH-FDUSD 机器人连续权益" in html
    assert "BTC-USDT 机器人连续权益" in html and "ETH-USDT 机器人连续权益" in html
    assert "FOMC 宏观风控（无数据）" in html
    assert (OUTPUT / "v22_grid_dca_forced_exit_v2.html").stat().st_size < 15 * 1024 * 1024
    series = pd.read_csv(OUTPUT / "audit_series.csv.gz")
    for strategy in ("grid", "dca"):
        pairs = list(series[series.strategy.eq(strategy)].pair.unique())
        left = series[(series.strategy == strategy) & (series.pair == pairs[0])].set_index("timestamp").equity
        right = series[(series.strategy == strategy) & (series.pair == pairs[1])].set_index("timestamp").equity
        aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
        assert (aligned.left-aligned.right).abs().max() > 1


def test_frozen_application_hash_matches_summary() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((FROZEN / "risk_states.csv.gz").read_bytes()).hexdigest()
    assert digest == summary["frozen_inputs"]["risk_states_sha256"]
