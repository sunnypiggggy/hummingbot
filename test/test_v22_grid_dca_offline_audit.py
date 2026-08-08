from __future__ import annotations

import json
import ast
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_v22_grid_dca_offline_audit as audit  # noqa: E402
from backtest_dca_momentum_guard import gate_for_frame, run_pair_guarded  # noqa: E402
from plot_v22_grid_dca_risk import MECHANISMS, build_figure, render_dashboard  # noqa: E402


def _series() -> pd.DataFrame:
    rows = []
    for strategy, pairs in {"grid": ("BTC-FDUSD", "ETH-FDUSD"),
                            "dca": ("BTC-USDT", "ETH-USDT")}.items():
        for pair in pairs:
            for index, ts in enumerate((1000, 2000, 3000)):
                rows.append({
                    "strategy": strategy, "pair": pair, "timestamp": ts,
                    "price": 100 + index, "equity": 200 + index,
                    "peak_equity": 202, "drawdown_pct": -index / 10,
                    "probability": .1 + index / 100, "entry_threshold": .11, "fold": 1,
                })
    return pd.DataFrame(rows)


def _intervals() -> pd.DataFrame:
    return pd.DataFrame([
        {"strategy": strategy, "pair": pair, "mechanism": mechanism,
         "start_ts": 1200, "end_ts": 2200, "trigger_value": 1, "threshold": 1,
         "action": "pause_normal_buy", "reason": "test", "source": "unit-test",
         "enabled": True, "model_week": 1, "model_sha256": "m",
         "feature_schema_sha256": "f", "strategy_schema_sha256": "s"}
        for strategy, pair in (("grid", "BTC-FDUSD"), ("dca", "BTC-USDT"))
        for mechanism in MECHANISMS
    ])


def test_v22_fdusd_signal_maps_to_usdt_and_is_fail_closed_before_first_state() -> None:
    frame = pd.DataFrame({"timestamp": [500, 1500, 2500], "close": [100, 101, 102]})
    states = pd.DataFrame({
        "pair": ["BTC-FDUSD", "BTC-FDUSD"], "signal_ts": [1000, 2000],
        "recommended_buy_enabled": [False, True],
    })
    gate = gate_for_frame(frame, pd.DataFrame(), "v22", pair="BTC-USDT", v22_states=states)
    assert gate.tolist() == [False, False, True]


def test_low_level_runner_can_keep_buy_only_positions_when_requested() -> None:
    frame = pd.DataFrame([
        {"timestamp": 0, "open": 100, "high": 100, "low": 100, "close": 100},
        {"timestamp": 300, "open": 100, "high": 101, "low": 98, "close": 99},
        {"timestamp": 600, "open": 99, "high": 100, "low": 98, "close": 99},
        {"timestamp": 900, "open": 99, "high": 100, "low": 98, "close": 99},
    ])
    gate = pd.Series([True, True, False, False])
    summary, trades, _ = run_pair_guarded(
        frame, gate, "BTC-USDT", .001, 2, guarded_sides=("BUY",),
        flatten_on_risk_off=False, refresh_seconds=10_000, time_limit_seconds=10_000,
    )
    assert summary["risk_flatten_positions"] == 0
    assert trades.empty or "RISK_FLATTEN" not in set(trades.close_type)


def test_v22_ablation_uses_same_pause_and_flatten_policy_as_roc_sqz() -> None:
    expected = (("BUY", "SELL"), True)
    assert audit.dca_execution_policy("roc") == expected
    assert audit.dca_execution_policy("sqzmom") == expected
    assert audit.dca_execution_policy("combined") == expected
    assert audit.dca_execution_policy("v22") == expected
    assert audit.dca_execution_policy("v22_btc_only") == expected
    assert audit.dca_execution_policy("v22_eth_only") == expected
    assert audit.dca_scenario_gate("v22_btc_only", "BTC-USDT") == "v22"
    assert audit.dca_scenario_gate("v22_btc_only", "ETH-USDT") == "baseline"
    assert audit.dca_scenario_gate("v22_eth_only", "BTC-USDT") == "baseline"
    assert audit.dca_scenario_gate("v22_eth_only", "ETH-USDT") == "v22"


def test_dca_hard_breakers_choose_the_first_trigger_and_persist() -> None:
    index = pd.to_datetime([1000, 1300, 1600], unit="s", utc=True)
    btc = pd.DataFrame({"timestamp": [1000, 1300, 1600], "equity": [190, 180, 173]}, index=index)
    eth = pd.DataFrame({"timestamp": [1000, 1300, 1600], "equity": [190, 191, 190]}, index=index)
    cutoffs, events = audit._strategy_breakers({"BTC-USDT": btc, "ETH-USDT": eth})
    assert cutoffs == {"BTC-USDT": 1600}
    assert events[0]["mechanism"] == "strategy_loss_breaker"

    combined = pd.DataFrame({
        "timestamp": [1000, 1300, 1600], "equity": [380, 370, 347],
        "drawdown_pct": [0, -10 / 380 * 100, -33 / 380 * 100],
    })
    cutoff, portfolio_events = audit._portfolio_breaker(combined)
    assert cutoff == 1600
    assert portfolio_events[0]["mechanism"] == "portfolio_loss_breaker"
    intervals = audit.dca_breaker_intervals(portfolio_events, 9999, {
        "model_sha256": "m", "feature_schema_sha256": "f", "strategy_schema_sha256": "s",
    })
    assert intervals[0]["end_ts"] == 9999


def test_grid_v22_gate_audit_rejects_normal_buy_while_disabled(tmp_path: Path) -> None:
    trades = pd.DataFrame([
        {"timestamp": 1500, "pair": "BTC-FDUSD", "side": "BUY", "reason": "grid_fill"},
        {"timestamp": 1500, "pair": "ETH-FDUSD", "side": "SELL", "reason": "grid_fill"},
    ])
    trades.to_csv(tmp_path / "grid_trades.csv.gz", index=False)
    states = pd.DataFrame([
        {"pair": pair, "signal_ts": 1000, "risk_off_active": pair == "BTC-FDUSD",
         "recommended_buy_enabled": pair != "BTC-FDUSD"}
        for pair in audit.v22.PAIRS
    ])
    with pytest.raises(AssertionError, match="BUY gate enforcement failed"):
        audit.grid_v22_gate_enforcement_audit(tmp_path, states)


def test_frozen_v22_package_is_non_authorizing_and_integrity_checked() -> None:
    lock, bundle, states = audit.validate_frozen_package(
        ROOT / "results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d")
    assert lock["deployment_allowed"] is False
    assert lock["promotion_authorized"] is False
    assert bundle["model_version"] == audit.v22.MODEL_VERSION
    assert set(states.pair) == set(audit.v22.PAIRS)


def test_v22_runtime_isolated_from_prior_model_and_optimizer_modules() -> None:
    for name in ("xgboost_long_risk_gate_v22.py", "xgboost_long_risk_gate_v22_features.py",
                 "xgboost_v22_io.py", "build_xgboost_v22_shadow_signal.py"):
        tree = ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not [module for module in imports if any(token in module for token in (
            "v19", "v20", "v21", "legacy", "optimize_xgboost", "retrain_xgboost",
            "compare_independent",
        ))]


def test_seven_mechanisms_and_pair_subswitches_are_independent(tmp_path: Path) -> None:
    series, intervals = _series(), _intervals()
    for strategy in ("grid", "dca"):
        figure, groups = build_figure(strategy, series, intervals)
        assert set(groups) == set(MECHANISMS)
        assert all(groups[name]["shapes"] for name in MECHANISMS)
        assert len(figure.layout.shapes) == 2 * len(MECHANISMS)
        assert {shape.yref for shape in figure.layout.shapes} == {"y domain", "y3 domain"}
        if strategy == "dca":
            assert "BTC+ETH DCA 组合权益" in figure.layout.annotations[2].text
            assert any(trace.name == "DCA 组合权益" for trace in figure.data)
        else:
            assert "BTC+ETH Grid 组合权益" in figure.layout.annotations[2].text
            assert any(trace.name == "Grid 组合权益" for trace in figure.data)
        eth_pair = "ETH-FDUSD" if strategy == "grid" else "ETH-USDT"
        eth_event = intervals.iloc[[0]].copy()
        eth_event["strategy"] = strategy; eth_event["pair"] = eth_pair
        eth_figure, _ = build_figure(strategy, series, pd.concat([intervals, eth_event], ignore_index=True))
        assert eth_figure.layout.shapes[-2].xref == "x4"
        assert eth_figure.layout.shapes[-2].yref == "y5 domain"
        assert eth_figure.layout.shapes[-1].xref == "x6"
        assert eth_figure.layout.shapes[-1].yref == "y7 domain"
    target = tmp_path / "audit.html"
    render_dashboard(series, intervals, target, {"verdict": "NO-GO"})
    html = target.read_text(encoding="utf-8")
    assert "v22 Grid/DCA 离线风控审计" in html
    assert "deployment_allowed=false" in html
    assert "机制事件数" not in html
    assert "<pre>" not in html
    assert all(f"data-mechanism='{name}'" in html for name in MECHANISMS)
    assert all(f"data-bot-pair='{pair}'" in html for pair in (
        "BTC-FDUSD", "ETH-FDUSD", "BTC-USDT", "ETH-USDT"))
    assert "消融实验无数据" in html
    assert "Plotly.restyle" not in html  # controls hide shadows only, never base/event traces


def test_offline_audit_adds_no_v22_container_or_live_contract() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    source = (SCRIPTS / "build_v22_grid_dca_offline_audit.py").read_text(encoding="utf-8")
    assert "grid-xgboost-v22" not in compose
    assert "docker-compose" not in source.lower()
    assert "promotion_authorized\": True" not in source
    assert "orders_submitted\": False" in source


def test_summary_contract_keeps_no_go_flags() -> None:
    summary = json.loads((ROOT / "results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d/application_bundle/summary.json").read_text(encoding="utf-8"))
    assert summary["historical_verdict"] == "NO-GO"
    assert summary["deployment_allowed"] is False
    assert summary["promotion_authorized"] is False
