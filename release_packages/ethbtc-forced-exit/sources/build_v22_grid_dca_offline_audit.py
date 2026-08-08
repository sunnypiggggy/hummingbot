#!/usr/bin/env python3
"""Run the frozen v22 Grid/DCA offline comparison and build the audit inputs.

This command is intentionally offline-only.  It never reads credentials, writes a
controller contract, changes Compose, or converts recommended_buy_enabled into a
live authorization.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

import xgboost_long_risk_gate_v22 as v22
from backtest_dca_live_local import BAR_SECONDS, PAIRS, TOTAL_BUDGET, load_window
from backtest_dca_momentum_guard import (
    SCENARIOS,
    build_signal_bars,
    combine_scenario,
    gate_for_frame,
    run_pair_guarded,
)
from dca_live_common import LIVE_EXECUTOR_REFRESH_SECONDS, LIVE_TIME_LIMIT_SECONDS
from plot_v22_grid_dca_risk import MECHANISMS, render_dashboard
from xgboost_v22_io import atomic_json, load_candles, sha256_file


ROOT = Path("results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d")
APP = ROOT / "application_bundle"
PACKAGE = ROOT / "shadow_package"
V21_STATES = Path("results/backtests/fdusd_grid_v21_mechanisms_250d/v21_states.csv.gz")
DCA_CACHE = Path("data/backtesting_candles")
OUTPUT = Path("results/backtests/v22_grid_dca_offline_audit")
PAIR_MAP = {"BTC-USDT": "BTC-FDUSD", "ETH-USDT": "ETH-FDUSD"}
DCA_SCENARIOS = (*SCENARIOS, "v22_btc_only", "v22_eth_only")
DCA_INITIAL_PAIR_EQUITY = 190.0
DCA_INITIAL_PORTFOLIO_EQUITY = 380.0
DCA_PAIR_LOSS_LIMIT = 16.0
DCA_PORTFOLIO_LOSS_LIMIT = 32.0
DCA_PAIR_DRAWDOWN_LIMIT = .08
DCA_PORTFOLIO_DRAWDOWN_LIMIT = .08


def dca_scenario_gate(scenario: str, pair: str) -> str:
    if scenario == "v22_btc_only":
        return "v22" if pair == "BTC-USDT" else "baseline"
    if scenario == "v22_eth_only":
        return "v22" if pair == "ETH-USDT" else "baseline"
    return scenario


def dca_execution_policy(scenario: str) -> tuple[tuple[str, ...], bool]:
    """Use one pause-and-flatten action for every technical comparison."""
    del scenario
    return ("BUY", "SELL"), True


def _first_trigger(curve: pd.DataFrame, condition: pd.Series) -> tuple[int, float] | None:
    matches = curve.loc[condition]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return int(row.timestamp), float(row.equity)


def _strategy_breakers(pair_curves: Mapping[str, pd.DataFrame]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    cutoffs, events = {}, []
    for pair, curve in pair_curves.items():
        peak = curve.equity.cummax(); drawdown = curve.equity / peak - 1
        candidates = []
        loss = _first_trigger(curve, curve.equity <= DCA_INITIAL_PAIR_EQUITY - DCA_PAIR_LOSS_LIMIT)
        if loss:
            candidates.append((loss[0], "strategy_loss_breaker", loss[1]-DCA_INITIAL_PAIR_EQUITY,
                               -DCA_PAIR_LOSS_LIMIT))
        dd = _first_trigger(curve, drawdown <= -DCA_PAIR_DRAWDOWN_LIMIT)
        if dd:
            index = curve.index[curve.timestamp.eq(dd[0])][0]
            candidates.append((dd[0], "strategy_drawdown_breaker", float(drawdown.loc[index]*100),
                               -DCA_PAIR_DRAWDOWN_LIMIT*100))
        if candidates:
            trigger = min(candidates, key=lambda item: item[0]); cutoffs[pair] = trigger[0]
            events.append({"pair": pair, "timestamp": trigger[0], "mechanism": trigger[1],
                           "trigger_value": trigger[2], "threshold": trigger[3]})
    return cutoffs, events


def _portfolio_breaker(combined: pd.DataFrame) -> tuple[int | None, list[dict[str, Any]]]:
    candidates = []
    loss = _first_trigger(combined, combined.equity <= DCA_INITIAL_PORTFOLIO_EQUITY-DCA_PORTFOLIO_LOSS_LIMIT)
    if loss:
        candidates.append((loss[0], "portfolio_loss_breaker", loss[1]-DCA_INITIAL_PORTFOLIO_EQUITY,
                           -DCA_PORTFOLIO_LOSS_LIMIT))
    dd = _first_trigger(combined, combined.drawdown_pct <= -DCA_PORTFOLIO_DRAWDOWN_LIMIT*100)
    if dd:
        row = combined[combined.timestamp.eq(dd[0])].iloc[0]
        candidates.append((dd[0], "portfolio_drawdown_breaker", float(row.drawdown_pct),
                           -DCA_PORTFOLIO_DRAWDOWN_LIMIT*100))
    if not candidates:
        return None, []
    trigger = min(candidates, key=lambda item: item[0])
    return trigger[0], [{"pair": "ALL", "timestamp": trigger[0], "mechanism": trigger[1],
                         "trigger_value": trigger[2], "threshold": trigger[3]}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT)
    parser.add_argument("--dca-cache-dir", type=Path, default=DCA_CACHE)
    parser.add_argument("--v21-states", type=Path, default=V21_STATES)
    parser.add_argument("--feature-panel", type=Path,
                        default=Path("results/backtests/xgboost_grid_long_risk_gate_v19_250d/feature_panel.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--fee-rate", type=float, default=.001)
    parser.add_argument("--risk-slippage-bps", type=float, default=2.0)
    parser.add_argument("--render-only", action="store_true",
                        help="Rebuild Plotly and manifest from existing audited CSV/JSON outputs")
    return parser.parse_args()


def _epoch_bounds(path: Path) -> tuple[int, int]:
    values = pd.to_numeric(pd.read_csv(path, usecols=["timestamp"])["timestamp"], errors="raise")
    if values.max() > 10_000_000_000:
        values = values // 1000
    return int(values.min()), int(values.max()) + BAR_SECONDS


def maximal_common_window(cache_dir: Path, states: pd.DataFrame) -> tuple[datetime, datetime]:
    bounds = [_epoch_bounds(cache_dir / f"{symbol}_5m.csv") for symbol in PAIRS.values()]
    start = max([item[0] for item in bounds] + [int(states.signal_ts.min())])
    # State rows are hourly observations; the last state remains valid for one hour only.
    end = min([item[1] for item in bounds] + [int(states.signal_ts.max()) + 3600])
    start = ((start + BAR_SECONDS - 1) // BAR_SECONDS) * BAR_SECONDS
    end = (end // BAR_SECONDS) * BAR_SECONDS
    if end <= start:
        raise ValueError("FDUSD v22 states and USDT DCA candles have no common continuous window")
    return datetime.fromtimestamp(start, timezone.utc), datetime.fromtimestamp(end, timezone.utc)


def validate_frozen_package(result_dir: Path) -> tuple[dict[str, Any], Mapping[str, Any], pd.DataFrame]:
    package = result_dir / "shadow_package"
    app = result_dir / "application_bundle"
    lock = json.loads((package / "shadow_lock.json").read_text(encoding="utf-8"))
    if lock.get("deployment_allowed") is not False or lock.get("promotion_authorized") is not False:
        raise ValueError("offline audit refuses an authorizing v22 package")
    model_path = Path(lock["model_path"])
    if not model_path.exists():
        model_path = package / "models" / model_path.name
    if sha256_file(model_path) != lock["model_sha256"]:
        raise ValueError("v22 model hash mismatch")
    bundle = joblib.load(model_path)
    v22.validate_weekly_bundle(bundle)
    if bundle.get("strategy_schema_sha256") != lock["strategy_schema_sha256"]:
        raise ValueError("v22 strategy schema hash mismatch")
    if bundle.get("feature_schema_sha256") != lock["feature_schema_sha256"]:
        raise ValueError("v22 feature schema hash mismatch")
    states = pd.read_csv(app / "risk_states.csv.gz").sort_values(["pair", "signal_ts"])
    required = {"pair", "signal_ts", "fold", "probability", "entry_threshold",
                "risk_off_active", "recommended_buy_enabled", "transition", "reason"}
    if not required.issubset(states):
        raise ValueError(f"v22 application states missing {sorted(required - set(states))}")
    if states.duplicated(["pair", "signal_ts"]).any():
        raise ValueError("v22 states overlap at pair/timestamp")
    for pair in v22.PAIRS:
        pair_rows = states[states.pair.eq(pair)]
        if pair_rows.empty or not pair_rows.signal_ts.diff().dropna().eq(3600).all():
            raise ValueError(f"{pair} v22 state timeline is not continuous hourly data")
    return lock, bundle, states


def exact_state_parity(feature_panel: Path, lock: Mapping[str, Any], bundle: Mapping[str, Any],
                       expected: pd.DataFrame) -> dict[str, Any]:
    if sha256_file(feature_panel) != lock["training_panel_sha256"]:
        raise ValueError("v22 training feature panel hash mismatch")
    panel = pd.read_csv(feature_panel)
    comparisons = []
    for pair in v22.PAIRS:
        wanted = expected[expected.pair.eq(pair)].sort_values("signal_ts")
        rows = panel[(panel.pair == pair) & panel.signal_ts.isin(wanted.signal_ts)]
        actual, _ = v22.run_weekly_bundle_strategy(rows, pair=pair, pair_bundle=bundle["pairs"][pair])
        actual = actual.sort_values("signal_ts")
        if not np.array_equal(actual.signal_ts.to_numpy(), wanted.signal_ts.to_numpy()):
            raise AssertionError(f"{pair} exact replay timestamp mismatch")
        comparisons.append((actual, wanted))
    risk = sum(int((a.risk_off_active.to_numpy() != w.risk_off_active.to_numpy()).sum()) for a, w in comparisons)
    transitions = sum(int((a.transition.to_numpy() != w.transition.to_numpy()).sum()) for a, w in comparisons)
    probability = max(float(np.abs(a.probability.to_numpy()-w.probability.to_numpy()).max()) for a, w in comparisons)
    threshold = max(float(np.abs(a.entry_threshold.to_numpy()-w.entry_threshold.to_numpy()).max()) for a, w in comparisons)
    result = {"risk_state_mismatches": risk, "transition_mismatches": transitions,
              "maximum_probability_absolute_error": probability,
              "maximum_threshold_absolute_delta": threshold}
    if risk or transitions or probability > 1e-6 or threshold > 1e-8:
        raise AssertionError(f"v22 frozen application replay mismatch: {result}")
    return result


def state_intervals(states: pd.DataFrame, *, strategy: str, pair_map: Mapping[str, str],
                    start_ts: int, end_ts: int, lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_pair, source_pair in pair_map.items():
        source = states[states.pair.eq(source_pair)].sort_values("signal_ts")
        before = source[source.signal_ts <= start_ts]
        active = bool(before.iloc[-1].risk_off_active) if not before.empty else True
        active_start = start_ts if active else None
        trigger_row = before.iloc[-1] if active and not before.empty else None
        window = source[source.signal_ts.between(start_ts, end_ts, inclusive="left")]
        for event in window.itertuples(index=False):
            if event.transition == "enter" and not active:
                active, active_start, trigger_row = True, int(event.signal_ts), event
            elif event.transition == "recover" and active:
                rows.append(_v22_interval(strategy, target_pair, int(active_start), int(event.signal_ts),
                                          trigger_row, lock, "adaptive_structural_relief"))
                active, active_start, trigger_row = False, None, None
        if active and active_start is not None:
            rows.append(_v22_interval(strategy, target_pair, int(active_start), end_ts,
                                      trigger_row, lock, "signed_window_end"))
    return rows


def _v22_interval(strategy: str, pair: str, start: int, end: int, trigger: Any,
                  lock: Mapping[str, Any], reason: str) -> dict[str, Any]:
    def field(name: str) -> Any:
        if trigger is None:
            return ""
        if isinstance(trigger, pd.Series):
            return trigger.get(name, "")
        return getattr(trigger, name, "")
    return {
        "strategy": strategy, "pair": pair, "mechanism": "v22_weekly_buy_gate",
        "start_ts": start, "end_ts": end, "trigger_value": field("probability"),
        "threshold": field("entry_threshold"), "action": "pause_normal_buy",
        "reason": reason, "source": "frozen_signed_v22_weekly_application_replay",
        "enabled": True, "model_week": field("fold"), "model_sha256": lock["model_sha256"],
        "feature_schema_sha256": lock["feature_schema_sha256"],
        "strategy_schema_sha256": lock["strategy_schema_sha256"],
    }


def grid_non_model_intervals(app: Path, bundle: Mapping[str, Any], lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    trades = pd.read_csv(app / "grid_trades.csv.gz")
    weeks = bundle["pairs"]["BTC-FDUSD"]["weeks"]
    rows: list[dict[str, Any]] = []
    for event in trades[trades.reason.ne("grid_fill")].itertuples(index=False):
        timestamp = int(event.timestamp)
        week = next((item for item in weeks if int(item["test_start"]) <= timestamp < int(item["test_end"])), None)
        reset = int(week["test_end"]) if week else timestamp + BAR_SECONDS
        if event.reason == "pair_breaker_flatten":
            mechanism = "strategy_loss_breaker" if event.trigger == "pair_loss" else "strategy_drawdown_breaker"
            threshold = "-6 FDUSD" if mechanism.endswith("loss_breaker") else "-3%"
            end = reset
        elif event.reason == "portfolio_breaker":
            mechanism = "portfolio_loss_breaker" if event.trigger == "portfolio_loss" else "portfolio_drawdown_breaker"
            threshold = "-24 FDUSD" if mechanism.endswith("loss_breaker") else "-6%"
            end = reset
        elif event.reason == "max_hold_exit":
            mechanism, threshold, end = "position_protection", "48h max hold", timestamp + BAR_SECONDS
        else:
            continue
        pair = "ALL" if event.pair == "PORTFOLIO" else str(event.pair)
        value = event.trigger_pnl_quote if pd.notna(event.trigger_pnl_quote) else event.trigger_drawdown_pct
        rows.append({
            "strategy": "grid", "pair": pair, "mechanism": mechanism,
            "start_ts": timestamp, "end_ts": end, "trigger_value": value,
            "threshold": threshold, "action": str(event.reason), "reason": str(event.trigger),
            "source": "v22_exact_grid_replay_trade_audit", "enabled": True,
            "model_week": int(event.fold), "model_sha256": lock["model_sha256"],
            "feature_schema_sha256": lock["feature_schema_sha256"],
            "strategy_schema_sha256": lock["strategy_schema_sha256"],
        })
    return rows


def run_dca(args: argparse.Namespace, states: pd.DataFrame, start: datetime, end: datetime) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames = {pair: load_window(args.dca_cache_dir / f"{symbol}_5m.csv", start, end)
              for pair, symbol in PAIRS.items()}
    signals = build_signal_bars(args.dca_cache_dir / "BTCUSDT_5m.csv", "4h", 12, 20)
    v21_states = pd.read_csv(args.v21_states)
    summaries, pair_summaries, trade_parts, curves_by_scenario = [], [], [], {}
    v22_pair_curves: dict[str, pd.DataFrame] = {}
    for scenario in DCA_SCENARIOS:
        pair_curves, scenario_pairs = {}, []
        for pair, frame in frames.items():
            gate_scenario = dca_scenario_gate(scenario, pair)
            gate = gate_for_frame(frame, signals, gate_scenario, pair=pair,
                                  v21_states=v21_states, v22_states=states)
            guarded_sides, flatten_on_risk_off = dca_execution_policy(scenario)
            summary, trades, curve = run_pair_guarded(
                frame, gate, pair, args.fee_rate, args.risk_slippage_bps,
                refresh_seconds=LIVE_EXECUTOR_REFRESH_SECONDS,
                time_limit_seconds=LIVE_TIME_LIMIT_SECONDS,
                guarded_sides=guarded_sides, flatten_on_risk_off=flatten_on_risk_off,
            )
            summary["scenario"] = scenario
            if not trades.empty:
                trades.insert(0, "scenario", scenario)
                trade_parts.append(trades)
            scenario_pairs.append(summary); pair_summaries.append(summary); pair_curves[pair] = curve
            if scenario == "v22":
                v22_pair_curves[pair] = curve
        aggregate, combined = combine_scenario(pair_curves, scenario_pairs, scenario)
        aggregate["buy_disabled_pair_hours"] = sum(item["buy_disabled_hours"] for item in scenario_pairs)
        aggregate["stop_loss_positions"] = sum(item["stop_loss_positions"] for item in scenario_pairs)
        summaries.append(aggregate); curves_by_scenario[scenario] = combined

    def execute_combined(pair_cutoffs: Mapping[str, int], portfolio_cutoff: int | None,
                         scenario: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
        curves, pair_rows, trade_rows = {}, [], []
        for pair, frame in frames.items():
            gate = gate_for_frame(frame, signals, "v22", pair=pair, v22_states=states)
            cutoff_candidates = [value for value in (pair_cutoffs.get(pair), portfolio_cutoff) if value is not None]
            if cutoff_candidates:
                cutoff = min(cutoff_candidates)
                gate = gate & frame.timestamp.lt(cutoff)
            summary, trades, curve = run_pair_guarded(
                frame, gate, pair, args.fee_rate, args.risk_slippage_bps,
                refresh_seconds=LIVE_EXECUTOR_REFRESH_SECONDS,
                time_limit_seconds=LIVE_TIME_LIMIT_SECONDS,
                guarded_sides=("BUY", "SELL"), flatten_on_risk_off=True,
            )
            summary["scenario"] = scenario; pair_rows.append(summary); curves[pair] = curve
            if not trades.empty:
                trades.insert(0, "scenario", scenario); trade_rows.append(trades)
        aggregate, combined = combine_scenario(curves, pair_rows, scenario)
        aggregate["buy_disabled_pair_hours"] = sum(item["buy_disabled_hours"] for item in pair_rows)
        aggregate["stop_loss_positions"] = sum(item["stop_loss_positions"] for item in pair_rows)
        return aggregate, pair_rows, trade_rows, curves, combined

    # Stage 1: pair breakers observe the v22-only path. Stage 2 applies those
    # persistent stops, then evaluates portfolio breakers. Stage 3 is the final
    # all-mechanism path used by the DCA equity line.
    pair_cutoffs, breaker_events = _strategy_breakers(v22_pair_curves)
    _, _, _, _, pair_stopped_combined = execute_combined(pair_cutoffs, None, "v22_pair_breakers_probe")
    portfolio_cutoff, portfolio_events = _portfolio_breaker(pair_stopped_combined)
    breaker_events.extend(portfolio_events)
    aggregate, final_pairs, final_trades, final_curves, final_combined = execute_combined(
        pair_cutoffs, portfolio_cutoff, "v22_all_mechanisms")
    summaries.append(aggregate); pair_summaries.extend(final_pairs); trade_parts.extend(final_trades)
    curves_by_scenario["v22_all_mechanisms"] = final_combined
    metrics = pd.DataFrame(summaries)
    baseline_pnl = float(metrics.loc[metrics.scenario.eq("baseline"), "combined_net_pnl_quote"].iloc[0])
    metrics["missed_buy_cost_quote"] = (baseline_pnl - metrics.combined_net_pnl_quote).clip(lower=0)
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    return metrics, pd.DataFrame(pair_summaries), trades, curves_by_scenario, final_curves, breaker_events


def build_series(result_dir: Path, states: pd.DataFrame, dca_frames: Mapping[str, pd.DataFrame],
                 dca_curves: Mapping[str, pd.DataFrame], dca_combined: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    app = result_dir / "application_bundle"
    candles = load_candles(Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d"), nested=True,
                           start_ts=int(states.signal_ts.min()) - 3600,
                           end_ts=int(states.signal_ts.max()) + 3600)
    grid_equity = pd.read_csv(app / "grid_equity.csv.gz").sort_values("timestamp")
    grid_equity["timestamp"] = grid_equity.timestamp.astype("int64")
    grid_equity["peak_equity"] = grid_equity.equity.cummax()
    grid_equity["drawdown_pct"] = (grid_equity.equity / grid_equity.peak_equity - 1) * 100
    for pair in v22.PAIRS:
        state = states[states.pair.eq(pair)].sort_values("signal_ts").copy()
        price = candles[pair][["timestamp", "close"]].sort_values("timestamp")
        item = pd.merge_asof(state, price, left_on="signal_ts", right_on="timestamp", direction="backward")
        item = pd.merge_asof(item.sort_values("signal_ts"), grid_equity, left_on="signal_ts", right_on="timestamp", direction="backward", suffixes=("", "_equity"))
        parts.append(pd.DataFrame({
            "strategy": "grid", "pair": pair, "timestamp": item.signal_ts, "price": item.close,
            "equity": item.equity, "peak_equity": item.peak_equity, "drawdown_pct": item.drawdown_pct,
            "probability": item.probability, "entry_threshold": item.entry_threshold, "fold": item.fold,
        }))
    combined_hourly = dca_combined.resample("1h").last().reset_index(drop=True)
    combined_hourly["peak_equity"] = combined_hourly.equity.cummax()
    combined_hourly["drawdown_pct"] = (
        combined_hourly.equity / combined_hourly.peak_equity - 1
    ) * 100
    for pair, curve in dca_curves.items():
        frame = dca_frames[pair].copy()
        # Price and v22 probability remain bot-specific, but both DCA panels use
        # the same final BTC+ETH portfolio equity after every enabled mechanism.
        hourly = curve.resample("1h").last().reset_index(drop=True)[["timestamp"]]
        hourly = pd.merge_asof(
            hourly.sort_values("timestamp"),
            combined_hourly[["timestamp", "equity", "peak_equity", "drawdown_pct"]]
                .sort_values("timestamp"),
            on="timestamp", direction="backward",
        )
        price = frame[["timestamp", "close"]].sort_values("timestamp")
        hourly = pd.merge_asof(hourly.sort_values("timestamp"), price, on="timestamp", direction="backward")
        source_pair = PAIR_MAP[pair]
        state = states[states.pair.eq(source_pair)][["signal_ts", "probability", "entry_threshold", "fold"]].sort_values("signal_ts")
        hourly = pd.merge_asof(hourly, state, left_on="timestamp", right_on="signal_ts", direction="backward")
        parts.append(pd.DataFrame({
            "strategy": "dca", "pair": pair, "timestamp": hourly.timestamp, "price": hourly.close,
            "equity": hourly.equity, "peak_equity": hourly.peak_equity, "drawdown_pct": hourly.drawdown_pct,
            "probability": hourly.probability, "entry_threshold": hourly.entry_threshold, "fold": hourly.fold,
        }))
    return pd.concat(parts, ignore_index=True)


def dca_position_intervals(trades: pd.DataFrame, lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    selected = trades[(trades.scenario == "v22_all_mechanisms") & trades.close_type.isin(("STOP_LOSS", "TIME_LIMIT"))]
    rows = []
    for event in selected.itertuples(index=False):
        timestamp = int(pd.Timestamp(event.closed_at).timestamp())
        rows.append({
            "strategy": "dca", "pair": event.pair, "mechanism": "position_protection",
            "start_ts": timestamp, "end_ts": timestamp + BAR_SECONDS,
            "trigger_value": event.net_pnl, "threshold": "5% stop / live time limit",
            "action": event.close_type, "reason": "existing_position_exit",
            "source": "v22_dca_offline_replay", "enabled": True, "model_week": "",
            "model_sha256": lock["model_sha256"], "feature_schema_sha256": lock["feature_schema_sha256"],
            "strategy_schema_sha256": lock["strategy_schema_sha256"],
        })
    return rows


def grid_v22_gate_enforcement_audit(app: Path, states: pd.DataFrame) -> dict[str, Any]:
    """Prove that the exact Grid replay submitted no normal BUY while v22 was off."""
    trades = pd.read_csv(app / "grid_trades.csv.gz")
    pairs, total_violations = {}, 0
    for pair in v22.PAIRS:
        buys = trades[
            trades.pair.eq(pair) & trades.side.eq("BUY") & trades.reason.eq("grid_fill")
        ].sort_values("timestamp")
        schedule = states[states.pair.eq(pair)][
            ["signal_ts", "risk_off_active", "recommended_buy_enabled"]
        ].sort_values("signal_ts")
        joined = pd.merge_asof(
            buys, schedule, left_on="timestamp", right_on="signal_ts", direction="backward",
        )
        disabled = joined.risk_off_active.fillna(True) | ~joined.recommended_buy_enabled.fillna(False)
        violations = int(disabled.sum())
        total_violations += violations
        pairs[pair] = {
            "normal_buy_fills": int(len(joined)),
            "normal_buy_fills_while_v22_disabled": violations,
        }
    result = {"pairs": pairs, "total_normal_buy_violations": total_violations}
    if total_violations:
        raise AssertionError(f"Grid v22 BUY gate enforcement failed: {result}")
    return result


def dca_breaker_intervals(events: list[dict[str, Any]], end_ts: int,
                          lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{
        "strategy": "dca", "pair": event["pair"], "mechanism": event["mechanism"],
        "start_ts": int(event["timestamp"]), "end_ts": end_ts,
        "trigger_value": event["trigger_value"], "threshold": event["threshold"],
        "action": "persistent_pause_and_flatten", "reason": "hard_breaker_triggered",
        "source": "v22_dca_all_mechanisms_replay", "enabled": True, "model_week": "",
        "model_sha256": lock["model_sha256"], "feature_schema_sha256": lock["feature_schema_sha256"],
        "strategy_schema_sha256": lock["strategy_schema_sha256"],
    } for event in events]


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        series = pd.read_csv(args.output_dir / "audit_series.csv.gz")
        interval_frame = pd.read_csv(args.output_dir / "risk_intervals.csv")
        summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
        dca_metrics = pd.read_csv(args.output_dir / "dca_scenario_metrics.csv")
        render_dashboard(series, interval_frame, args.output_dir / "v22_grid_dca_risk_plotly.html",
                         {**summary, "dca_scenarios": dca_metrics.to_dict("records")})
        atomic_json(args.output_dir / "manifest.json", {
            "schema": "v22-grid-dca-offline-audit-manifest-v1", "offline_only": True,
            "deployment_allowed": False, "orders_submitted": False,
            "input_hashes": {"model": summary["model_sha256"],
                             "v22_states": sha256_file(args.result_dir / "application_bundle/risk_states.csv.gz"),
                             "feature_panel": sha256_file(args.feature_panel), "v21_states": sha256_file(args.v21_states)},
            "output_hashes": {path.name: sha256_file(path) for path in sorted(args.output_dir.iterdir())
                              if path.is_file() and path.name != "manifest.json"},
        })
        print(json.dumps({"output": str(args.output_dir), "render_only": True}, ensure_ascii=False))
        return 0
    lock, bundle, states = validate_frozen_package(args.result_dir)
    replay_parity = exact_state_parity(args.feature_panel, lock, bundle, states)
    start, end = maximal_common_window(args.dca_cache_dir, states)
    dca_metrics, dca_pairs, dca_trades, dca_combined, dca_curves, dca_breaker_events = run_dca(
        args, states, start, end)
    dca_frames = {pair: load_window(args.dca_cache_dir / f"{symbol}_5m.csv", start, end)
                  for pair, symbol in PAIRS.items()}
    start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
    intervals = state_intervals(states, strategy="grid", pair_map={pair: pair for pair in v22.PAIRS},
                                start_ts=int(states.signal_ts.min()), end_ts=int(states.signal_ts.max()) + 3600, lock=lock)
    intervals += state_intervals(states, strategy="dca", pair_map=PAIR_MAP,
                                 start_ts=start_ts, end_ts=end_ts, lock=lock)
    intervals += grid_non_model_intervals(args.result_dir / "application_bundle", bundle, lock)
    intervals += dca_position_intervals(dca_trades, lock)
    intervals += dca_breaker_intervals(dca_breaker_events, end_ts, lock)
    interval_frame = pd.DataFrame(intervals)
    series = build_series(
        args.result_dir, states, dca_frames, dca_curves,
        dca_combined["v22_all_mechanisms"],
    )
    grid_summary = json.loads((args.result_dir / "application_bundle/summary.json").read_text(encoding="utf-8"))
    grid_gate_audit = grid_v22_gate_enforcement_audit(
        args.result_dir / "application_bundle", states)
    no_data = {
        strategy: [mechanism for mechanism in MECHANISMS
                   if interval_frame[(interval_frame.strategy == strategy) & (interval_frame.mechanism == mechanism)].empty]
        for strategy in ("grid", "dca")
    }
    summary = {
        "schema": "v22-grid-dca-offline-audit-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "NO-GO", "offline_only": True, "orders_submitted": False,
        "deployment_allowed": False, "promotion_authorized": False,
        "model_version": v22.MODEL_VERSION, "model_sha256": lock["model_sha256"],
        "feature_schema_sha256": lock["feature_schema_sha256"],
        "strategy_schema_sha256": lock["strategy_schema_sha256"],
        "signed_effective_end": datetime.fromtimestamp(int(lock["effective_end"]), timezone.utc).isoformat(),
        "grid_exact_replay": grid_summary["grid_metrics"],
        "grid_exact_state_parity": replay_parity,
        "grid_v22_gate_enforcement": grid_gate_audit,
        "dca_common_window": {"start": start.isoformat(), "end_exclusive": end.isoformat(),
                              "hours": (end - start).total_seconds() / 3600},
        "dca_execution_policy": {
            "all_technical_gates": "same policy: pause BUY and SELL; flatten open executors on Risk-Off",
            "final_equity_source": "v22_all_mechanisms BTC+ETH combined portfolio (initial 380 USDT)",
            "strategy_loss_limit_quote": DCA_PAIR_LOSS_LIMIT,
            "strategy_drawdown_limit_pct": DCA_PAIR_DRAWDOWN_LIMIT * 100,
            "portfolio_loss_limit_quote": DCA_PORTFOLIO_LOSS_LIMIT,
            "portfolio_drawdown_limit_pct": DCA_PORTFOLIO_DRAWDOWN_LIMIT * 100,
            "v22_signal_scope": "each bot consumes only its mapped FDUSD pair signal",
            "bot_signal_map": PAIR_MAP,
            "fee_rate_per_fill": args.fee_rate, "risk_slippage_bps": args.risk_slippage_bps,
        },
        "mechanism_activation_counts": (
            interval_frame.groupby(["strategy", "mechanism"]).size()
            .rename("intervals").reset_index().to_dict("records")
        ),
        "grid_v22_equity_note": (
            "v22 is a normal-BUY gate: existing inventory remains marked to market, so equity may decline "
            "inside a v22 Risk-Off interval without indicating a gate bypass"
        ),
        "no_trustworthy_interval_data": no_data,
        "promotion_rule": "valid current signed week + integrity checks + explicit human approval",
    }
    dca_metrics.to_csv(args.output_dir / "dca_scenario_metrics.csv", index=False)
    dca_metrics[dca_metrics.scenario.isin(("baseline", "v22_btc_only", "v22_eth_only", "v22"))].to_csv(
        args.output_dir / "dca_v22_bot_ablation.csv", index=False)
    dca_pairs.to_csv(args.output_dir / "dca_pair_metrics.csv", index=False)
    dca_trades.to_csv(args.output_dir / "dca_positioned_executors.csv", index=False)
    # Metrics stay at 5m; the exported audit curves use hourly endpoints to keep
    # the portable HTML/CSV responsive without changing any reported result.
    pd.concat([frame.resample("1h").last().assign(scenario=scenario).reset_index()
               for scenario, frame in dca_combined.items()], ignore_index=True).to_csv(
        args.output_dir / "dca_equity_curves.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    series.to_csv(args.output_dir / "audit_series.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    interval_frame.to_csv(args.output_dir / "risk_intervals.csv", index=False)
    atomic_json(args.output_dir / "summary.json", summary)
    target = args.output_dir / "v22_grid_dca_risk_plotly.html"
    render_dashboard(series, interval_frame, target, {**summary, "dca_scenarios": dca_metrics.to_dict("records")})
    atomic_json(args.output_dir / "manifest.json", {
        "schema": "v22-grid-dca-offline-audit-manifest-v1", "offline_only": True,
        "deployment_allowed": False, "orders_submitted": False,
        "input_hashes": {"model": lock["model_sha256"], "v22_states": sha256_file(args.result_dir / "application_bundle/risk_states.csv.gz"),
                         "feature_panel": sha256_file(args.feature_panel), "v21_states": sha256_file(args.v21_states)},
        "output_hashes": {path.name: sha256_file(path) for path in sorted(args.output_dir.iterdir())
                          if path.is_file() and path.name != "manifest.json"},
    })
    print(json.dumps({"output": str(args.output_dir), "verdict": "NO-GO", "orders_submitted": False,
                      "dca": dca_metrics.to_dict("records")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
