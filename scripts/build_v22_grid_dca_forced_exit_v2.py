#!/usr/bin/env python3
"""Build the isolated v22 forced-exit-v2 offline counterfactual.

The frozen v22 package is read-only input.  This module never writes a live
contract, credentials, Compose configuration, or deployment authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backtest_dca_live_local import BAR_SECONDS, PAIRS, load_window
from backtest_dca_momentum_guard import gate_for_frame, run_pair_guarded
from build_v22_grid_dca_offline_audit import maximal_common_window, validate_frozen_package
from plot_v22_forced_exit_v2 import render_dashboard
from xgboost_v22_io import atomic_json, sha256_file


ROOT = Path("results/backtests/xgboost_grid_long_risk_gate_v22_weekly_250d")
APP = ROOT / "application_bundle"
GRID_CANDLES = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d/extended_candles")
DCA_CANDLES = Path("data/backtesting_candles")
OUTPUT = Path("results/backtests/v22_grid_dca_forced_exit_v2")
PACKAGE_ID = "ethbtc-forced-exit"
PAIR_MAP = {"BTC-USDT": "BTC-FDUSD", "ETH-USDT": "ETH-FDUSD"}
FILTERS = {
    "BTC-FDUSD": (Decimal("0.00001"), Decimal("5")),
    "ETH-FDUSD": (Decimal("0.0001"), Decimal("5")),
    "BTC-USDT": (Decimal("0.00001"), Decimal("5")),
    "ETH-USDT": (Decimal("0.0001"), Decimal("5")),
}
POLICY = {
    "package_id": PACKAGE_ID,
    "execution_policy_version": "v22-risk-off-forced-exit-v2",
    "bar_seconds": 300,
    "technical_cooldown_seconds": 0,
    "required_healthy_guard_cycles": 3,
    "guard_cycle_seconds": 2,
    "exit_fee_rate": 0.001,
    "exit_adverse_slippage_bps": 2.0,
    "grid_maker_fee_rate": 0.0,
    "grid_reentry_quote_per_pair": 100.0,
    "dca_reentry_quote_per_bot": 95.0,
    "ownership": "strategy_ledger_only",
    "risk_off_action": "cancel_orders_and_liquidate_owned_base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT)
    parser.add_argument("--grid-candles", type=Path, default=GRID_CANDLES)
    parser.add_argument("--dca-candles", type=Path, default=DCA_CANDLES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def floor_qty(pair: str, quantity: float) -> float:
    step, _ = FILTERS[pair]
    return float((Decimal(str(max(quantity, 0))) / step).to_integral_value(rounding=ROUND_DOWN) * step)


def executable_qty(pair: str, quantity: float, price: float) -> tuple[float, float]:
    qty = floor_qty(pair, quantity)
    _, minimum = FILTERS[pair]
    if Decimal(str(qty * price)) < minimum:
        return 0.0, quantity
    return qty, max(quantity - qty, 0.0)


def load_grid_candle(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "open", "close"])
    frame["timestamp"] = pd.to_numeric(frame.timestamp, errors="raise").astype("int64")
    return frame.sort_values("timestamp").drop_duplicates("timestamp")


def state_on_bars(bars: pd.DataFrame, states: pd.DataFrame, source_pair: str) -> pd.DataFrame:
    wanted = states[states.pair.eq(source_pair)].sort_values("signal_ts")[
        ["signal_ts", "risk_off_active", "probability", "entry_threshold", "fold"]
    ].copy()
    merged = pd.merge_asof(
        bars.sort_values("timestamp"), wanted, left_on="timestamp", right_on="signal_ts",
        direction="backward", tolerance=3600,
    )
    if merged.risk_off_active.isna().any():
        raise ValueError(f"{source_pair} contains bars outside the signed hourly state coverage")
    merged["risk_off_active"] = merged.risk_off_active.astype(bool)
    return merged


def _action(strategy: str, pair: str, signal_ts: int, execution_ts: int, action: str,
            quantity: float, price: float, fee: float, dust: float, quote: float,
            phase: str, baseline: float | None = None, signal_price: float | None = None,
            post_signal_additional_loss: float = 0.0) -> dict[str, Any]:
    return {
        "strategy": strategy, "pair": pair, "signal_ts": signal_ts,
        "execution_ts": execution_ts, "action": action, "phase": phase,
        "quantity": quantity, "average_price": price, "fee_quote": fee,
        "slippage_bps": POLICY["exit_adverse_slippage_bps"], "dust_base": dust,
        "quote_notional": quote,
        "signal_price": signal_price, "orders_cancelled": action == "MARKET_EXIT",
        "exit_latency_seconds": execution_ts-signal_ts if action == "MARKET_EXIT" else None,
        "retry_count": 0,
        "post_signal_additional_loss": post_signal_additional_loss,
        "risk_cycle_baseline": baseline, "execution_policy_version": POLICY["execution_policy_version"],
    }


def replay_grid(app: Path, candle_dir: Path, states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = {pair: load_grid_candle(candle_dir / f"binance_{pair}_5m.csv") for pair in ("BTC-FDUSD", "ETH-FDUSD")}
    start = max(int(states.signal_ts.min()), *(int(frame.timestamp.min()) for frame in raw.values()))
    end = min(int(states.signal_ts.max()) + 3600, *(int(frame.timestamp.max()) + BAR_SECONDS for frame in raw.values()))
    timeline = pd.DataFrame({"timestamp": range(((start + 299)//300)*300, (end//300)*300, 300)})
    bars = {pair: state_on_bars(timeline.merge(frame, on="timestamp", how="left"), states, pair) for pair, frame in raw.items()}
    for pair, frame in bars.items():
        if frame[["open", "close"]].isna().any().any():
            raise ValueError(f"{pair} grid candle coverage is not continuous")
    trades = pd.read_csv(app / "grid_trades.csv.gz")
    trades = trades[trades.reason.eq("grid_fill")].sort_values("timestamp")
    intents = {(int(ts), pair): group for (ts, pair), group in trades.groupby(["timestamp", "pair"])}
    ledgers: dict[str, dict[str, float | bool | int | str]] = {}
    actions: list[dict[str, Any]] = []
    for pair in bars:
        price = float(bars[pair].iloc[0].open)
        ledgers[pair] = {"quote": 100.0, "base": 100.0/price, "active": True,
                         "pending": "", "signal_ts": 0, "signal_price": price,
                         "signal_equity": 200.0, "baseline": 200.0}
    rows, pair_rows = [], []
    for index, timestamp in enumerate(timeline.timestamp.astype(int)):
        prices = {pair: float(frame.iloc[index].close) for pair, frame in bars.items()}
        for pair, frame in bars.items():
            bar = frame.iloc[index]; ledger = ledgers[pair]; off = bool(bar.risk_off_active)
            if off and bool(ledger["active"]) and not ledger["pending"]:
                ledger["pending"], ledger["signal_ts"] = "EXIT", timestamp
                ledger["signal_price"] = float(bar.close)
                ledger["signal_equity"] = float(ledger["quote"])+float(ledger["base"])*float(bar.close)
            if not off and not bool(ledger["active"]) and not ledger["pending"]:
                ledger["pending"], ledger["signal_ts"] = "REENTRY", timestamp
            if ledger["pending"] and timestamp >= int(ledger["signal_ts"]) + BAR_SECONDS:
                open_price = float(bar.open)
                if ledger["pending"] == "EXIT":
                    execution_price = open_price * (1 - POLICY["exit_adverse_slippage_bps"]/10_000)
                    qty, dust = executable_qty(pair, float(ledger["base"]), execution_price)
                    notional = qty * execution_price; fee = notional * POLICY["exit_fee_rate"]
                    ledger["base"] = dust; ledger["quote"] = float(ledger["quote"]) + notional - fee
                    exit_equity = float(ledger["quote"])+dust*float(bar.close)
                    ledger["active"], ledger["pending"] = False, ""
                    actions.append(_action("grid", pair, int(ledger["signal_ts"]), timestamp,
                                           "MARKET_EXIT", qty, execution_price, fee, dust, notional, "EXITING",
                                           signal_price=float(ledger["signal_price"]),
                                           post_signal_additional_loss=max(float(ledger["signal_equity"])-exit_equity, 0.0)))
                elif not off:
                    execution_price = open_price * (1 + POLICY["exit_adverse_slippage_bps"]/10_000)
                    budget = min(POLICY["grid_reentry_quote_per_pair"], float(ledger["quote"]))
                    qty = floor_qty(pair, budget/(execution_price*(1+POLICY["exit_fee_rate"])))
                    notional = qty*execution_price; fee = notional*POLICY["exit_fee_rate"]
                    if qty and notional >= float(FILTERS[pair][1]) and notional+fee <= float(ledger["quote"])+1e-9:
                        ledger["quote"] = float(ledger["quote"]) - notional - fee
                        ledger["base"] = float(ledger["base"]) + qty
                        ledger["active"], ledger["pending"] = True, ""
                        ledger["baseline"] = float(ledger["quote"]) + float(ledger["base"])*float(bar.close)
                        actions.append(_action("grid", pair, int(ledger["signal_ts"]), timestamp,
                                               "MARKET_REENTRY", qty, execution_price, fee,
                                               float(ledger["base"])-floor_qty(pair, float(ledger["base"])),
                                               notional, "REENTRY", float(ledger["baseline"])))
            if bool(ledger["active"]) and not off:
                event_rows = intents.get((timestamp, pair))
                if event_rows is not None:
                    for event in event_rows.itertuples(index=False):
                        amount, price = float(event.amount), float(event.price)
                        if event.side == "BUY" and float(ledger["quote"]) + 1e-9 >= amount*price:
                            ledger["quote"] = float(ledger["quote"]) - amount*price; ledger["base"] = float(ledger["base"]) + amount
                        elif event.side == "SELL" and float(ledger["base"]) + 1e-12 >= amount:
                            ledger["base"] = float(ledger["base"]) - amount; ledger["quote"] = float(ledger["quote"]) + amount*price
            pair_equity = float(ledger["quote"]) + float(ledger["base"])*prices[pair]
            phase = ("EXITING" if ledger["pending"] == "EXIT" else "ACTIVE" if ledger["active"]
                     else "REENTRY" if ledger["pending"] == "REENTRY" else "COOLDOWN")
            pair_rows.append({"timestamp": timestamp, "pair": pair, "equity": pair_equity,
                              "quote": ledger["quote"], "base": ledger["base"], "phase": phase,
                              "risk_off_active": off})
        equity = 20.0 + sum(float(ledgers[p]["quote"])+float(ledgers[p]["base"])*prices[p] for p in ledgers)
        rows.append({"timestamp": timestamp, "equity": equity})
    curve = pd.DataFrame(rows); curve["peak_equity"] = curve.equity.cummax()
    curve["cumulative_oos_pnl"] = curve.equity - 420.0
    curve["drawdown_pct"] = (curve.equity/curve.peak_equity-1)*100
    pair_curve = pd.DataFrame(pair_rows)
    pair_curve["peak_equity"] = pair_curve.groupby("pair").equity.cummax()
    pair_curve["drawdown_pct"] = (pair_curve.equity/pair_curve.peak_equity-1)*100
    return curve, pair_curve, pd.DataFrame(actions)


def inventory_overlay(frame: pd.DataFrame, gate: pd.Series, pair: str, forced: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_price = float(frame.iloc[0].open); quote, base = 95.0, 95.0/first_price
    active, pending, signal_ts, signal_price, signal_equity = True, "", 0, first_price, 190.0
    rows, actions = [], []
    # Treat an already Risk-Off first bar as a transition from the strategy's
    # startup ACTIVE state.  Otherwise the managed startup inventory would stay
    # exposed until a later on->off transition (or forever).
    previous = True
    for i, bar in enumerate(frame.itertuples(index=False)):
        enabled = bool(gate.iloc[i]); timestamp = int(bar.timestamp)
        if forced and previous and not enabled and active and not pending:
            pending, signal_ts = "EXIT", timestamp
            signal_price, signal_equity = float(bar.close), quote+base*float(bar.close)
        if forced and not previous and enabled and not active and not pending:
            pending, signal_ts = "REENTRY", timestamp
        if pending and timestamp >= signal_ts + BAR_SECONDS:
            if pending == "EXIT":
                price = float(bar.open)*(1-POLICY["exit_adverse_slippage_bps"]/10_000)
                qty, dust = executable_qty(pair, base, price); notional = qty*price; fee = notional*POLICY["exit_fee_rate"]
                base = dust; quote += notional-fee; active, pending = False, ""
                exit_equity = quote+base*float(bar.close)
                actions.append(_action("dca", pair, signal_ts, timestamp, "MARKET_EXIT", qty, price, fee, dust, notional, "EXITING",
                                       signal_price=signal_price, post_signal_additional_loss=max(signal_equity-exit_equity, 0.0)))
            elif enabled:
                price = float(bar.open)*(1+POLICY["exit_adverse_slippage_bps"]/10_000)
                budget = min(POLICY["dca_reentry_quote_per_bot"], quote)
                qty = floor_qty(pair, budget/(price*(1+POLICY["exit_fee_rate"])))
                notional = qty*price; fee = notional*POLICY["exit_fee_rate"]
                if qty and notional >= float(FILTERS[pair][1]) and notional+fee <= quote+1e-9:
                    quote -= notional+fee; base += qty; active, pending = True, ""
                    baseline = quote+base*float(bar.close)
                    actions.append(_action("dca", pair, signal_ts, timestamp, "MARKET_REENTRY", qty, price, fee, 0, notional, "REENTRY", baseline))
        phase = ("EXITING" if pending == "EXIT" else "ACTIVE" if active
                 else "REENTRY" if pending == "REENTRY" else "COOLDOWN")
        rows.append({"timestamp": timestamp, "managed_equity": quote+base*float(bar.close),
                     "managed_quote": quote, "managed_base": base, "phase": phase})
        previous = enabled
    return pd.DataFrame(rows), pd.DataFrame(actions)


def replay_dca(candle_dir: Path, states: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    start, end = maximal_common_window(candle_dir, states)
    curves: dict[str, pd.DataFrame] = {}
    action_parts, metric_rows, pair_curves_by_scenario = [], [], {}
    for scenario, forced in (("legacy_buy_only", False), ("forced_exit_v2", True)):
        pairs = {}
        for pair, symbol in PAIRS.items():
            frame = load_window(candle_dir/f"{symbol}_5m.csv", start, end)
            gate = gate_for_frame(frame, pd.DataFrame(), "v22", pair=pair, v22_states=states)
            summary, _, executor = run_pair_guarded(
                frame, gate, pair, POLICY["exit_fee_rate"], POLICY["exit_adverse_slippage_bps"],
                guarded_sides=("BUY", "SELL") if forced else ("BUY",), flatten_on_risk_off=forced,
            )
            inventory, actions = inventory_overlay(frame, gate, pair, forced)
            adjusted = executor.reset_index(drop=True).merge(inventory[["timestamp", "managed_equity"]], on="timestamp")
            adjusted["equity"] = adjusted.equity + adjusted.managed_equity - 190.0
            adjusted["peak_equity"] = adjusted.equity.cummax()
            adjusted["drawdown_pct"] = (adjusted.equity/adjusted.peak_equity-1)*100
            pairs[pair] = adjusted
            if not actions.empty:
                actions.insert(0, "scenario", scenario); action_parts.append(actions)
            metric_rows.append({"strategy": "dca", "scenario": scenario, "pair": pair,
                                "net_pnl_quote": float(adjusted.equity.iloc[-1]-190),
                                "max_drawdown_pct": float((adjusted.equity/adjusted.equity.cummax()-1).min()*100),
                                "fees_quote": float(summary["fees_quote"] + (actions.fee_quote.sum() if not actions.empty else 0)),
                                "forced_exits": int((actions.action == "MARKET_EXIT").sum()) if not actions.empty else 0})
        combined = pd.DataFrame({pair: curve.set_index("timestamp").equity for pair, curve in pairs.items()})
        combined["equity"] = combined.sum(axis=1); combined["timestamp"] = combined.index.astype("int64")
        combined["peak_equity"] = combined.equity.cummax(); combined["drawdown_pct"] = (combined.equity/combined.peak_equity-1)*100
        curves[scenario] = combined.reset_index(drop=True)
        pair_curves_by_scenario[scenario] = pd.concat(
            [curve.assign(pair=pair) for pair, curve in pairs.items()], ignore_index=True,
        )
        metric_rows.append({"strategy": "dca", "scenario": scenario, "pair": "ALL",
                            "net_pnl_quote": float(combined.equity.iloc[-1]-380),
                            "max_drawdown_pct": float(combined.drawdown_pct.min()),
                            "fees_quote": sum(row["fees_quote"] for row in metric_rows if row["strategy"] == "dca" and row["scenario"] == scenario and row["pair"] != "ALL"),
                            "forced_exits": sum(row["forced_exits"] for row in metric_rows if row["strategy"] == "dca" and row["scenario"] == scenario and row["pair"] != "ALL")})
    actions = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
    return curves, pair_curves_by_scenario, actions, pd.DataFrame(metric_rows)


def legacy_grid_metrics(app: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    old = pd.read_csv(app/"grid_equity.csv.gz")
    old["equity"] = 420.0 + old.cumulative_oos_pnl
    old["peak_equity"] = old.equity.cummax(); old["drawdown_pct"] = (old.equity/old.peak_equity-1)*100
    return old, {"strategy": "grid", "scenario": "legacy_buy_only", "pair": "ALL",
                 "net_pnl_quote": float(old.equity.iloc[-1]-420), "max_drawdown_pct": float(old.drawdown_pct.min())}


def intervals_from_actions(actions: pd.DataFrame, states: pd.DataFrame, lock: dict[str, Any], end_ts: int) -> pd.DataFrame:
    rows = []
    for strategy, mapping in (("grid", {"BTC-FDUSD": "BTC-FDUSD", "ETH-FDUSD": "ETH-FDUSD"}),
                              ("dca", PAIR_MAP)):
        subset = actions[actions.strategy.eq(strategy)] if not actions.empty else pd.DataFrame()
        for pair, source in mapping.items():
            source_states = states[states.pair.eq(source)].sort_values("signal_ts")
            for enter in source_states[source_states.transition.eq("enter")].itertuples(index=False):
                recover = source_states[(source_states.signal_ts > enter.signal_ts) & source_states.transition.eq("recover")]
                recovery_ts = int(recover.iloc[0].signal_ts) if not recover.empty else end_ts
                exit_action = subset[(subset.pair.eq(pair)) & (subset.action.eq("MARKET_EXIT")) & (subset.signal_ts.eq(int(enter.signal_ts)))]
                reentry = subset[(subset.pair.eq(pair)) & (subset.action.eq("MARKET_REENTRY")) & (subset.signal_ts.eq(recovery_ts))]
                exit_ts = int(exit_action.iloc[0].execution_ts) if not exit_action.empty else min(int(enter.signal_ts)+300, end_ts)
                reentry_ts = int(reentry.iloc[0].execution_ts) if not reentry.empty else recovery_ts
                common = {"strategy": strategy, "pair": pair, "mechanism": "v22_weekly_buy_gate",
                          "trigger_value": enter.probability, "threshold": enter.entry_threshold,
                          "reason": "v22_risk_off", "source": "frozen_v22_signal_plus_forced_exit_v2_overlay",
                          "enabled": True, "model_week": enter.fold, "model_sha256": lock["model_sha256"],
                          "feature_schema_sha256": lock["feature_schema_sha256"],
                          "strategy_schema_sha256": lock["strategy_schema_sha256"]}
                if exit_ts > int(enter.signal_ts): rows.append({**common, "start_ts": int(enter.signal_ts), "end_ts": exit_ts, "phase": "EXITING", "action": "cancel_and_market_exit"})
                if recovery_ts > exit_ts: rows.append({**common, "start_ts": exit_ts, "end_ts": recovery_ts, "phase": "COOLDOWN", "action": "quote_only_until_model_recovery"})
                if reentry_ts > recovery_ts: rows.append({**common, "start_ts": recovery_ts, "end_ts": reentry_ts, "phase": "REENTRY", "action": "three_healthy_cycles_then_rebuild"})
    return pd.DataFrame(rows)


def build_series(grid_pairs: pd.DataFrame, dca_pairs: pd.DataFrame, states: pd.DataFrame,
                 grid_candles: Path, dca_candles: Path) -> pd.DataFrame:
    parts = []
    configs = [("grid", "BTC-FDUSD", grid_pairs[grid_pairs.pair.eq("BTC-FDUSD")], grid_candles/"binance_BTC-FDUSD_5m.csv", "BTC-FDUSD"),
               ("grid", "ETH-FDUSD", grid_pairs[grid_pairs.pair.eq("ETH-FDUSD")], grid_candles/"binance_ETH-FDUSD_5m.csv", "ETH-FDUSD"),
               ("dca", "BTC-USDT", dca_pairs[dca_pairs.pair.eq("BTC-USDT")], dca_candles/"BTCUSDT_5m.csv", "BTC-FDUSD"),
               ("dca", "ETH-USDT", dca_pairs[dca_pairs.pair.eq("ETH-USDT")], dca_candles/"ETHUSDT_5m.csv", "ETH-FDUSD")]
    for strategy, pair, curve, path, source in configs:
        price = pd.read_csv(path, usecols=["timestamp", "close"]); price.timestamp = (price.timestamp//1000 if price.timestamp.max()>10_000_000_000 else price.timestamp).astype("int64")
        item = curve[["timestamp", "equity", "peak_equity", "drawdown_pct"]].merge(price.rename(columns={"close":"price"}), on="timestamp", how="inner")
        st = states[states.pair.eq(source)][["signal_ts", "probability", "entry_threshold", "fold"]]
        item = pd.merge_asof(item.sort_values("timestamp"), st.sort_values("signal_ts"), left_on="timestamp", right_on="signal_ts", direction="backward", tolerance=3600)
        item.insert(0, "pair", pair); item.insert(0, "strategy", strategy); parts.append(item.drop(columns=["signal_ts"]))
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    lock, _, states = validate_frozen_package(args.result_dir)
    policy_hash = canonical_hash(POLICY)
    grid, grid_pairs, grid_actions = replay_grid(args.result_dir/"application_bundle", args.grid_candles, states)
    dca_curves, dca_pair_curves, dca_actions, dca_metrics = replay_dca(args.dca_candles, states)
    legacy_grid, legacy_metric = legacy_grid_metrics(args.result_dir/"application_bundle")
    forced_grid_metric = {"strategy": "grid", "scenario": "forced_exit_v2", "pair": "ALL",
                          "net_pnl_quote": float(grid.equity.iloc[-1]-420), "max_drawdown_pct": float(grid.drawdown_pct.min()),
                          "fees_quote": float(grid_actions.fee_quote.sum()), "forced_exits": int((grid_actions.action == "MARKET_EXIT").sum())}
    metrics = pd.concat([pd.DataFrame([legacy_metric, forced_grid_metric]), dca_metrics], ignore_index=True)
    actions = pd.concat([grid_actions, dca_actions[dca_actions.scenario.eq("forced_exit_v2")].drop(columns="scenario")], ignore_index=True)
    intervals = intervals_from_actions(actions, states, lock, int(grid.timestamp.max())+300)
    series = build_series(grid_pairs, dca_pair_curves["forced_exit_v2"], states, args.grid_candles, args.dca_candles)
    grid.to_csv(args.output_dir/"grid_combined_equity.csv.gz", index=False)
    grid_pairs.to_csv(args.output_dir/"grid_pair_ledger.csv.gz", index=False)
    legacy_grid.to_csv(args.output_dir/"grid_legacy_continuous_equity.csv.gz", index=False)
    dca_curves["forced_exit_v2"].to_csv(args.output_dir/"dca_combined_equity.csv.gz", index=False)
    dca_pair_curves["forced_exit_v2"].to_csv(args.output_dir/"dca_pair_equity.csv.gz", index=False)
    actions.to_csv(args.output_dir/"execution_actions.csv", index=False)
    metrics.to_csv(args.output_dir/"ablation_metrics.csv", index=False)
    intervals.to_csv(args.output_dir/"risk_intervals.csv", index=False)
    series.to_csv(args.output_dir/"audit_series.csv.gz", index=False)
    summary = {"schema": "v22-grid-dca-forced-exit-v2", "package_id": PACKAGE_ID,
               "verdict": "NO-GO", "offline_only": True,
               "orders_submitted": False, "deployment_allowed": False, "promotion_authorized": False,
               "counterfactual_execution_overlay": True, "execution_policy_version": POLICY["execution_policy_version"],
               "execution_policy_sha256": policy_hash,
               "frozen_inputs": {"model_sha256": lock["model_sha256"], "feature_schema_sha256": lock["feature_schema_sha256"],
                                 "strategy_schema_sha256": lock["strategy_schema_sha256"],
                                 "risk_states_sha256": sha256_file(args.result_dir/"application_bundle/risk_states.csv.gz")},
               "fomc_gate": {"status": "无数据", "included_in_execution": False},
               "metrics": metrics.to_dict("records")}
    atomic_json(args.output_dir/"execution_policy.json", {**POLICY, "execution_policy_sha256": policy_hash})
    atomic_json(args.output_dir/"summary.json", summary)
    render_dashboard(series, intervals, metrics, args.output_dir/"v22_grid_dca_forced_exit_v2.html")
    print(json.dumps({"output": str(args.output_dir), "verdict": "NO-GO", "policy_sha256": policy_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
