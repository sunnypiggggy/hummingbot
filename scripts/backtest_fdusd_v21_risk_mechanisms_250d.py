#!/usr/bin/env python3
"""Evaluate the frozen v21 BUY gate and FDUSD Grid risk mechanisms 1-6.

This is a research-only entry point.  It never reads or writes OCI state and
always emits ``deployment_allowed=false`` because the frozen model was
developed with knowledge of the requested 250-day window.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compare_independent_gate_ml_stops import load_candles
from retrain_xgboost_long_risk_gate_250d_v19 import sha256_file
from validate_grid_live import (
    Candidate, ExecutionFilter, RiskMechanismConfig, simulate, slice_window,
    technical_buy_gate_timeline,
)
from xgboost_long_risk_gate_v21 import (
    FEATURES, MODEL_BUNDLE_SCHEMA, MODEL_VERSION, GateState, advance_gate,
    build_inference_panel, feature_schema_sha256,
)


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
START_TS = int(pd.Timestamp("2025-11-23T15:00:00Z").timestamp())
END_TS = int(pd.Timestamp("2026-07-31T15:00:00Z").timestamp())
FIVE_MINUTES = 300
DAY = 86_400
INITIAL_EQUITY = 420.0
BASE_CANDIDATE = Candidate(.03, .006, .006, .015, 1_800)
SOURCE_DIR = Path("results/backtests/eth_xgboost_long_risk_gate_v15_250d/extended_candles")
PACKAGE_DIR = Path("results/backtests/xgboost_grid_long_risk_gate_v21_250d/shadow_package")
OUTPUT_DIR = Path("results/backtests/fdusd_grid_v21_mechanisms_250d")
COLORS = {"BTC-FDUSD": "#0891B2", "ETH-FDUSD": "#7C3AED"}
EXECUTION_FILTERS = {
    "BTC-FDUSD": ExecutionFilter(.01, .00001, 5.0),
    "ETH-FDUSD": ExecutionFilter(.01, .0001, 5.0),
}
EXECUTION_FILTERS_OBSERVED_UTC = "2026-08-06"


@dataclass(frozen=True)
class Scenario:
    scenario: str
    label: str
    gate: str | None = None
    mechanism_2: bool = False
    mechanism_3: bool = False
    mechanism_4: bool = False
    mechanism_5: bool = False
    mechanism_6: bool = False
    category: str = "comparison"

    def mechanisms(self) -> RiskMechanismConfig:
        return RiskMechanismConfig(
            pair_loss=self.mechanism_2,
            pair_drawdown=self.mechanism_3,
            portfolio_loss=self.mechanism_4,
            portfolio_drawdown=self.mechanism_5,
            continue_after_portfolio_stop=True,
            restore_portfolio_inventory=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-stress", action="store_true")
    return parser.parse_args()


def utc(ts: int | float) -> str:
    return pd.Timestamp(int(ts), unit="s", tz="UTC").isoformat()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def validate_data(source_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    candles, quality = load_candles(source_dir)
    hashes: dict[str, str] = {}
    expected = (END_TS - START_TS) // FIVE_MINUTES
    for pair in PAIRS:
        path = source_dir / f"binance_{pair}_5m.csv"
        hashes[pair] = sha256_file(path)
        frame = candles[pair]
        evaluation = frame[(frame.timestamp >= START_TS) & (frame.timestamp < END_TS)].copy()
        if len(evaluation) != expected:
            raise AssertionError(f"{pair}: expected {expected} rows, got {len(evaluation)}")
        timestamps = evaluation.timestamp.to_numpy(np.int64)
        if len(np.unique(timestamps)) != expected or not np.all(np.diff(timestamps) == FIVE_MINUTES):
            raise AssertionError(f"{pair}: duplicate or missing 5-minute candle")
        if int(timestamps[0]) != START_TS or int(timestamps[-1]) != END_TS - FIVE_MINUTES:
            raise AssertionError(f"{pair}: evaluation boundary mismatch")
        if int(frame.timestamp.min()) > START_TS - 45 * DAY:
            raise AssertionError(f"{pair}: less than 45 days of indicator warm-up")
        invalid = (
            (evaluation.high < evaluation[["open", "close"]].max(axis=1))
            | (evaluation.low > evaluation[["open", "close"]].min(axis=1))
            | (evaluation.high < evaluation.low) | (evaluation.volume < 0)
        )
        if bool(invalid.any()):
            raise AssertionError(f"{pair}: invalid OHLCV rows: {int(invalid.sum())}")
    quality = quality.copy()
    quality["evaluation_start_utc"] = utc(START_TS)
    quality["evaluation_end_exclusive_utc"] = utc(END_TS)
    quality["evaluation_rows"] = expected
    quality["continuous_5m"] = True
    return candles, quality, hashes


def load_frozen_v21(
    candles: Mapping[str, pd.DataFrame], package_dir: Path,
) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, dict[str, Any]]:
    lock_path = package_dir / "shadow_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    model_path = Path(lock["model_path"])
    if not model_path.exists():
        model_path = package_dir / "models" / model_path.name
    checks = {
        "model_version": lock.get("model_version") == MODEL_VERSION,
        "model_sha256": sha256_file(model_path) == lock.get("model_sha256"),
        "feature_schema_sha256": feature_schema_sha256() == lock.get("feature_schema_sha256"),
        "candidate_lock_sha256": sha256_file(Path(lock["candidate_lock_path"])) == lock.get("candidate_lock_sha256"),
        "deployment_denied": lock.get("deployment_allowed") is False,
        "promotion_denied": lock.get("promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"frozen package validation failed: {checks}")
    bundle = joblib.load(model_path)
    if bundle.get("schema") != MODEL_BUNDLE_SCHEMA or bundle.get("model_version") != MODEL_VERSION:
        raise AssertionError("frozen model bundle schema/version mismatch")

    panel = build_inference_panel(candles)
    panel = panel[panel.signal_ts < END_TS].sort_values(["signal_ts", "pair"]).copy()
    states: list[dict[str, Any]] = []
    for pair in PAIRS:
        pair_bundle = bundle["pairs"][pair]
        features = list(FEATURES[pair])
        if features != list(pair_bundle["features"]) or features != lock["pairs"][pair]["features"]:
            raise AssertionError(f"{pair}: frozen feature order mismatch")
        rows = panel[panel.pair.eq(pair)].copy()
        probability = pair_bundle["model"].predict_proba(rows[features])[:, 1]
        if not np.isfinite(probability).all():
            raise AssertionError(f"{pair}: non-finite frozen probability")
        state = GateState()
        for row, value in zip(rows.itertuples(index=False), probability):
            state, snapshot = advance_gate(
                pair=pair, probability=float(value),
                entry_threshold=float(pair_bundle["entry_threshold"]),
                signal_ts=int(row.signal_ts),
                last_complete_4h_ts=int(row.last_complete_4h_ts),
                structure=(row.roc_48h_4h, row.sqzmom_pct_4h, row.di_spread,
                           row.ema20_slope_atr_12h, row.below_ema20_ratio_72h),
                state=state,
            )
            states.append({
                "pair": pair, "signal_ts": int(row.signal_ts),
                "price": float(row.close),
                "probability": float(value),
                "entry_threshold": float(pair_bundle["entry_threshold"]),
                "risk_off_active": bool(snapshot["risk_off_active"]),
                "recommended_buy_enabled": bool(snapshot["recommended_buy_enabled"]),
                "transition": snapshot["transition"], "armed": bool(snapshot["armed"]),
                "armed_until": snapshot["armed_until"],
                "risk_off_since": snapshot["risk_off_since"],
                "cooldown_until": snapshot["cooldown_until"],
                "entry_structure_confirmed": bool(snapshot["entry_structure_confirmed"]),
                "recovery_structure_confirmed": bool(snapshot["recovery_structure_confirmed"]),
            })
    state_frame = pd.DataFrame(states).sort_values(["pair", "signal_ts"]).reset_index(drop=True)
    timelines: dict[str, dict[int, bool]] = {}
    for pair in PAIRS:
        signals = state_frame[state_frame.pair.eq(pair)].sort_values("signal_ts")
        signal_times = signals.signal_ts.to_numpy(np.int64)
        enabled = signals.recommended_buy_enabled.to_numpy(bool)
        evaluation_times = candles[pair].loc[
            candles[pair].timestamp.between(START_TS, END_TS - FIVE_MINUTES), "timestamp"
        ].to_numpy(np.int64)
        indices = np.searchsorted(signal_times, evaluation_times, side="right") - 1
        values = np.where(indices >= 0, enabled[np.maximum(indices, 0)], True)
        timelines[pair] = dict(zip(evaluation_times.tolist(), values.tolist()))
    audit = {
        "lock_path": str(lock_path), "lock_sha256": sha256_file(lock_path),
        "model_path": str(model_path), "model_sha256": sha256_file(model_path),
        "feature_schema_sha256": feature_schema_sha256(), "checks": checks,
        "state_rows": len(state_frame), "state_sha256": sha256_json(states),
        "thresholds": {pair: float(bundle["pairs"][pair]["entry_threshold"]) for pair in PAIRS},
        "features": {pair: list(FEATURES[pair]) for pair in PAIRS},
    }
    return timelines, state_frame, audit


def scenario_matrix() -> list[Scenario]:
    full = dict(mechanism_2=True, mechanism_3=True, mechanism_4=True,
                mechanism_5=True, mechanism_6=True)
    scenarios = [
        Scenario("baseline", "无风控基础 Grid"),
        Scenario("roc_only", "当前 ROC/SQZMOM 门单独", gate="roc"),
        Scenario("roc_full", "当前 ROC/SQZMOM 完整栈", gate="roc", **full),
        Scenario("v21_only", "机制1：v21 独立买入门", gate="v21"),
        Scenario("mechanism_2", "机制2：单对绝对亏损", mechanism_2=True),
        Scenario("mechanism_3", "机制3：单对峰值回撤", mechanism_3=True),
        Scenario("mechanism_4", "机制4：组合绝对亏损", mechanism_4=True),
        Scenario("mechanism_5", "机制5：组合峰值回撤", mechanism_5=True),
        Scenario("mechanism_6", "机制6：移动平均成本底线", mechanism_6=True),
        Scenario("v21_full", "v21 + 机制2–6完整组合", gate="v21", **full),
    ]
    for number in range(1, 7):
        settings = full.copy()
        gate = "v21"
        if number == 1:
            gate = None
        else:
            settings[f"mechanism_{number}"] = False
        scenarios.append(Scenario(
            f"leave_out_{number}", f"完整组合移除机制{number}",
            gate=gate, category="leave_one_out", **settings,
        ))
    return scenarios


def worst_return(curve: pd.DataFrame, bars: int) -> float:
    values = curve.equity.pct_change(bars)
    return float(values.min()) if values.notna().any() else float("nan")


def stop_hours(trades: pd.DataFrame, end_ts: int) -> tuple[float, float]:
    if trades.empty:
        return 0.0, 0.0
    pair_events = trades[trades.reason.eq("pair_breaker_flatten")].drop_duplicates(["pair", "trigger"])
    pair_seconds = sum(max(0, end_ts - int(row.timestamp)) for row in pair_events.itertuples())
    portfolio = trades[trades.reason.eq("portfolio_breaker")]
    portfolio_seconds = 0 if portfolio.empty else max(0, end_ts - int(portfolio.timestamp.min()))
    return pair_seconds / 3600, portfolio_seconds / 3600


def evaluate(
    scenario: Scenario, candles: dict[str, pd.DataFrame], gates: Mapping[str, Any],
    *, taker_fee: float = .001, slippage: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    trades: list[dict[str, Any]] = []
    result, curve, pair_stats = simulate(
        candles, BASE_CANDIDATE, maker_fee=0.0, taker_fee=taker_fee,
        slippage=slippage, order_refresh_seconds=7_200,
        technical_buy_gate=gates.get(scenario.gate) if scenario.gate else None,
        trade_log=trades, risk_breakers_enabled=False,
        risk_mechanisms=scenario.mechanisms(),
        execution_filters=EXECUTION_FILTERS,
        cost_floor_enabled=scenario.mechanism_6,
        inventory_exit_policy=None, record_curve=True,
    )
    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        trade_frame = pd.DataFrame(columns=["timestamp", "pair", "side", "reason", "amount", "quote_notional"])
    pair_stop_hours, portfolio_stop_hours = stop_hours(trade_frame, END_TS)
    maker = trade_frame.reason.eq("grid_fill")
    taker = trade_frame.reason.isin(["pair_breaker_flatten", "portfolio_breaker_flatten"])
    metric = {
        "scenario": scenario.scenario, "label": scenario.label, "category": scenario.category,
        "net_pnl_fdusd": result["net_pnl_quote"], "net_return_pct": result["net_pnl_pct"] * 100,
        "max_drawdown_pct": result["max_drawdown_pct"] * 100,
        "worst_day_pct": worst_return(curve, 288) * 100,
        "worst_week_pct": worst_return(curve, 2_016) * 100,
        "trades": result["trades"], "maker_fills": int(maker.sum()),
        "taker_fills": int(taker.sum()), "fees_fdusd": result["fees_quote"],
        "forced_inventory_restore_events": int(taker.sum()),
        "pair_stop_events": int(trade_frame.reason.eq("pair_breaker_flatten").sum()),
        "portfolio_stop_events": int(trade_frame.reason.eq("portfolio_breaker").sum()),
        "pair_stop_hours": pair_stop_hours, "portfolio_stop_hours": portfolio_stop_hours,
        "effective_v21_risk_off_hours_before_stop": sum(
            v["technical_risk_off_seconds"] for v in pair_stats.values()
        ) / 3600
            if scenario.gate == "v21" else 0.0,
        "v21_risk_off_hours": 0.0,
        "taker_fee_rate": taker_fee, "slippage_rate": slippage,
        **{f"mechanism_{number}": bool(number == 1 and scenario.gate == "v21")
           if number == 1 else bool(getattr(scenario, f"mechanism_{number}")) for number in range(1, 7)},
    }
    pair_rows = []
    for pair, values in pair_stats.items():
        pair_equity = 200 + curve[f"{pair}_pnl_quote"]
        pair_rows.append({
            "scenario": scenario.scenario, "label": scenario.label, "pair": pair,
            **values,
            "max_drawdown_pct": float(values["max_drawdown_pct"]) * 100,
            "worst_day_pct": float(pair_equity.pct_change(288).min()) * 100,
            "worst_week_pct": float(pair_equity.pct_change(2_016).min()) * 100,
        })
    curve = curve.copy()
    curve["scenario"] = scenario.scenario
    curve["label"] = scenario.label
    for item in trades:
        item["scenario"] = scenario.scenario
        item["label"] = scenario.label
    return metric, pair_rows, curve, trade_frame.assign(scenario=scenario.scenario, label=scenario.label)


def build_intervals(states: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        active_start = None
        for row in states[states.pair.eq(pair)].sort_values("signal_ts").itertuples():
            if row.transition == "enter" and row.signal_ts < END_TS and active_start is None:
                active_start = max(int(row.signal_ts), START_TS)
            elif row.transition == "recover" and active_start is not None:
                if int(row.signal_ts) > START_TS:
                    rows.append({"scenario": "v21_gate", "mechanism": "v21", "pair": pair,
                                 "start_ts": active_start, "end_ts": min(int(row.signal_ts), END_TS),
                                 "reason": "risk_off", "source_event_timestamp": active_start})
                active_start = None
        if active_start is not None:
            rows.append({"scenario": "v21_gate", "mechanism": "v21", "pair": pair,
                         "start_ts": active_start, "end_ts": END_TS, "reason": "risk_off",
                         "source_event_timestamp": active_start})
    breaker_events = events[events.reason.isin(["pair_breaker_flatten", "portfolio_breaker"])].copy()
    breaker_events = breaker_events.drop_duplicates(["scenario", "pair", "reason", "timestamp"])
    for row in breaker_events.itertuples():
        rows.append({
            "scenario": row.scenario,
            "mechanism": str(row.trigger), "pair": row.pair,
            "start_ts": int(row.timestamp), "end_ts": END_TS,
            "reason": "stop_latched", "source_event_timestamp": int(row.timestamp),
        })
    return pd.DataFrame(rows)


def mechanism_impacts(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index("scenario")
    baseline = indexed.loc["baseline"]
    full = indexed.loc["v21_full"]
    rows = []
    for number in range(1, 7):
        single_key = "v21_only" if number == 1 else f"mechanism_{number}"
        single = indexed.loc[single_key]
        without = indexed.loc[f"leave_out_{number}"]
        rows.append({
            "mechanism": number,
            "independent_scenario": single_key,
            "independent_pnl_delta_vs_baseline_fdusd": single.net_pnl_fdusd - baseline.net_pnl_fdusd,
            "independent_drawdown_delta_vs_baseline_pct_point": single.max_drawdown_pct - baseline.max_drawdown_pct,
            "full_pnl_delta_from_adding_mechanism_fdusd": full.net_pnl_fdusd - without.net_pnl_fdusd,
            "full_drawdown_delta_from_adding_mechanism_pct_point": full.max_drawdown_pct - without.max_drawdown_pct,
            "full_stop_hours_delta_from_adding_mechanism": (
                full.pair_stop_hours + full.portfolio_stop_hours
                - without.pair_stop_hours - without.portfolio_stop_hours
            ),
        })
    return pd.DataFrame(rows)


def stress_candles(candles: Mapping[str, pd.DataFrame], pair: str) -> dict[str, pd.DataFrame]:
    output = {key: value.copy() for key, value in candles.items()}
    frame = output[pair]
    count = min(288, len(frame))
    multipliers = np.ones(len(frame))
    multipliers[-count:] = 1 - .15 * (np.arange(count) + 1) / count
    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].to_numpy(float) * multipliers
    return output


def build_report(
    path: Path, metrics: pd.DataFrame, pairs: pd.DataFrame, curves: pd.DataFrame,
    states: pd.DataFrame, events: pd.DataFrame, intervals: pd.DataFrame,
    stress: pd.DataFrame, impacts: pd.DataFrame, summary: Mapping[str, Any], scenarios: list[Scenario],
) -> None:
    chosen = curves[curves.scenario.isin(["baseline", "roc_full", "v21_full"])].copy()
    chosen = chosen.iloc[::12].copy()
    chosen["datetime"] = pd.to_datetime(chosen.timestamp, unit="s", utc=True)

    equity = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08,
                           subplot_titles=("权益曲线", "回撤"))
    palette = {"baseline": "#64748B", "roc_full": "#F59E0B", "v21_full": "#2563EB"}
    for key, group in chosen.groupby("scenario", sort=False):
        label = group.label.iloc[0]
        equity.add_trace(go.Scatter(x=group.datetime, y=group.equity, name=label,
                                    line={"color": palette[key]}), row=1, col=1)
        equity.add_trace(go.Scatter(x=group.datetime, y=group.drawdown_pct * 100,
                                    name=f"{label} 回撤", legendgroup=key, showlegend=False,
                                    line={"color": palette[key]}), row=2, col=1)
    equity.update_yaxes(title_text="FDUSD", row=1, col=1)
    equity.update_yaxes(title_text="%", row=2, col=1)
    equity.update_layout(height=700, template="plotly_white", hovermode="x unified")

    signal = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                           subplot_titles=("BTC-FDUSD：价格、v21概率与进出点",
                                           "ETH-FDUSD：价格、v21概率与进出点"))
    for row_number, pair in enumerate(PAIRS, 1):
        item = states[(states.pair.eq(pair)) & states.signal_ts.between(START_TS, END_TS - 1)].copy()
        item["datetime"] = pd.to_datetime(item.signal_ts, unit="s", utc=True)
        signal.add_trace(go.Scatter(x=item.datetime, y=item.probability, name=f"{pair} 概率",
                                    line={"color": COLORS[pair]}), row=row_number, col=1, secondary_y=False)
        signal.add_trace(go.Scatter(x=item.datetime, y=item.entry_threshold, name=f"{pair} 阈值",
                                    line={"color": "#DC2626", "dash": "dash"}),
                         row=row_number, col=1, secondary_y=False)
        signal.add_trace(go.Scatter(x=item.datetime, y=item.price, name=f"{pair} 价格",
                                    line={"color": "#475569", "width": 1}, opacity=.6),
                         row=row_number, col=1, secondary_y=True)
        enters = item[item.transition.eq("enter")]
        exits = item[item.transition.eq("recover")]
        signal.add_trace(go.Scatter(x=enters.datetime, y=enters.probability, mode="markers",
                                    name=f"{pair} 进入", marker={"symbol": "triangle-down", "size": 11,
                                    "color": "#DC2626"}), row=row_number, col=1, secondary_y=False)
        signal.add_trace(go.Scatter(x=exits.datetime, y=exits.probability, mode="markers",
                                    name=f"{pair} 退出", marker={"symbol": "triangle-up", "size": 11,
                                    "color": "#16A34A"}), row=row_number, col=1, secondary_y=False)
        for interval in intervals[(intervals.scenario.eq("v21_gate")) & intervals.pair.eq(pair)].itertuples():
            signal.add_vrect(x0=pd.to_datetime(interval.start_ts, unit="s", utc=True),
                             x1=pd.to_datetime(interval.end_ts, unit="s", utc=True),
                             fillcolor="#FCA5A5", opacity=.18, line_width=0,
                             row=row_number, col=1)
        signal.update_yaxes(title_text="Risk-Off 概率", row=row_number, col=1, secondary_y=False)
        signal.update_yaxes(title_text="价格", row=row_number, col=1, secondary_y=True)
    signal.update_layout(height=850, template="plotly_white", hovermode="x unified")

    timing = go.Figure()
    timing_rows = intervals[
        intervals.scenario.isin(["v21_gate", "v21_full"])
    ].copy()
    for index, row in enumerate(timing_rows.itertuples()):
        label = f"{row.mechanism} · {row.pair}"
        timing.add_trace(go.Scatter(
            x=[pd.to_datetime(row.start_ts, unit="s", utc=True),
               pd.to_datetime(row.end_ts, unit="s", utc=True)],
            y=[label, label], mode="lines+markers", showlegend=False,
            line={"width": 8, "color": "#DC2626" if row.mechanism != "v21" else "#F59E0B"},
            marker={"size": [9, 5], "symbol": ["diamond", "circle"]},
            hovertemplate=("%{y}<br>开始/结束 %{x|%Y-%m-%d %H:%M UTC}<extra></extra>"),
        ))
    timing.update_layout(
        height=650, template="plotly_white", title="v21进入/退出与完整组合熔断停止区间",
        xaxis_title="UTC 时间", yaxis_title="机制 / 交易对",
    )

    independent = metrics[metrics.scenario.isin(["baseline", "v21_only", "mechanism_2",
                                                  "mechanism_3", "mechanism_4", "mechanism_5",
                                                  "mechanism_6", "v21_full"])]
    comparison = make_subplots(rows=1, cols=3,
        subplot_titles=("净收益 FDUSD", "最大回撤 %", "停止时长（小时）"))
    comparison.add_trace(go.Bar(x=independent.label, y=independent.net_pnl_fdusd,
                                marker_color="#2563EB", name="收益"), row=1, col=1)
    comparison.add_trace(go.Bar(x=independent.label, y=independent.max_drawdown_pct,
                                marker_color="#DC2626", name="回撤"), row=1, col=2)
    comparison.add_trace(go.Bar(x=independent.label,
                                y=independent.pair_stop_hours + independent.portfolio_stop_hours,
                                marker_color="#F59E0B", name="停止"), row=1, col=3)
    comparison.update_xaxes(tickangle=-35)
    comparison.update_layout(height=620, template="plotly_white", showlegend=False)

    pair_plot = pairs[pairs.scenario.isin(["baseline", "roc_full", "v21_full"])].copy()
    contribution = go.Figure()
    for pair in PAIRS:
        item = pair_plot[pair_plot.pair.eq(pair)]
        contribution.add_bar(x=item.label, y=item.net_pnl_quote, name=pair,
                             marker_color=COLORS[pair], customdata=item[["buys", "sells"]],
                             hovertemplate="%{x}<br>收益 %{y:.3f}<br>BUY %{customdata[0]} / SELL %{customdata[1]}<extra></extra>")
    contribution.update_layout(barmode="group", height=500, template="plotly_white",
                               title="BTC/ETH 收益贡献与成交数量")

    matrix_rows = []
    for scenario in scenarios:
        matrix_rows.append({"方案": scenario.label, **{
            f"机制{i}": "✓" if (i == 1 and scenario.gate == "v21") or
            (i > 1 and getattr(scenario, f"mechanism_{i}")) else "—" for i in range(1, 7)
        }})
    matrix = pd.DataFrame(matrix_rows)
    verdict = "GO：可进入新样本影子验证" if summary["eligible_for_new_shadow"] else "NO-GO"
    reasons = "；".join(summary["failed_acceptance_reasons"]) or "全部历史诊断门槛通过"
    css = """
    body{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f8fafc;color:#172033;margin:0}
    main{max-width:1500px;margin:auto;padding:24px}.banner{padding:18px;background:#fff7ed;border-left:6px solid #ea580c}
    section{background:white;margin:18px 0;padding:18px;border:1px solid #e2e8f0;border-radius:8px}
    table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;border-bottom:1px solid #e2e8f0;text-align:right}
    th{background:#f1f5f9;position:sticky;top:0}.scroll{overflow:auto}.note{color:#475569}.bad{color:#b91c1c}
    """
    figures = [equity, signal, timing, comparison, contribution]
    blocks = []
    for index, figure in enumerate(figures):
        blocks.append(figure.to_html(
            full_html=False, include_plotlyjs="inline" if index == 0 else False,
            config={"responsive": True, "displaylogo": False}, div_id=f"fdusd-v21-figure-{index}",
        ))
    html_text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
    <title>FDUSD Grid v21 250天机制1–6评估</title><style>{css}</style></head><body><main>
    <h1>FDUSD Grid v21 替换技术门：250天机制1–6评估</h1>
    <div class='banner'><h2>{verdict}</h2><p>{html.escape(reasons)}</p>
    <p><strong>部署权限始终为 false。</strong> 本窗口已参与 v21 开发，只能用于联合诊断，不能作为全新样本外证据。</p></div>
    <section><h2>评估口径</h2><p>2025-11-23 15:00 UTC 至 2026-07-31 15:00 UTC（右端不含），
    BTC/ETH 各 200 FDUSD + 20 FDUSD 储备；Maker 0%，Taker 0.1%，挂单 2 小时；固定 6% 全区间、10 层、
    0.6% 止盈、1.5% 移动阈值、30 分钟冷却。宏观/FOMC 与机制7关闭。</p></section>
    <section>{blocks[0]}</section><section>{blocks[1]}</section><section>{blocks[2]}</section>
    <section>{blocks[3]}</section><section>{blocks[4]}</section>
    <section><h2>机制矩阵</h2><div class='scroll'>{matrix.to_html(index=False, escape=False)}</div></section>
    <section><h2>全部方案指标</h2><div class='scroll'>{metrics.round(6).to_html(index=False)}</div></section>
    <section><h2>独立贡献与 Leave-one-out 边际影响</h2><div class='scroll'>{impacts.round(6).to_html(index=False)}</div></section>
    <section><h2>压力测试</h2><div class='scroll'>{stress.round(6).to_html(index=False)}</div></section>
    <section><h2>机制解释</h2><ol>
    <li>v21：BTC/ETH 各自模型只关闭本交易对普通 BUY；SELL 与风险恢复订单不受影响。</li>
    <li>单对绝对亏损：相对 200 FDUSD 启动基准亏损达到 6 FDUSD 后锁存停止并恢复启动库存。</li>
    <li>单对峰值回撤：单对风险周期峰值回撤达到 3% 后锁存停止。</li>
    <li>组合绝对亏损：组合 420 FDUSD 权益亏损达到 24 FDUSD 后停止两对。</li>
    <li>组合峰值回撤：组合峰值回撤达到 6% 后停止两对。</li>
    <li>成本底线：SELL 价格不低于 max(网格价, 当前价×1.006, 移动平均成本×1.006)。</li>
    </ol></section></main></body></html>"""
    path.write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    np.random.seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_candles, quality, data_hashes = validate_data(args.source_dir)
    evaluation = slice_window(all_candles, START_TS, END_TS)
    v21_gate, states, model_audit = load_frozen_v21(all_candles, args.package_dir)
    roc_gate = technical_buy_gate_timeline(evaluation["BTC-FDUSD"])
    gates = {"v21": v21_gate, "roc": roc_gate}
    scenarios = scenario_matrix()

    metrics_rows, pair_rows, curve_frames, event_frames = [], [], [], []
    for scenario in scenarios:
        print(f"running {scenario.scenario}", flush=True)
        metric, pairs, curve, events = evaluate(scenario, evaluation, gates)
        metrics_rows.append(metric); pair_rows.extend(pairs)
        curve_frames.append(curve); event_frames.append(events)
    metrics = pd.DataFrame(metrics_rows)
    total_v21_risk_off_hours = float(
        states[states.signal_ts.between(START_TS, END_TS - 1)].risk_off_active.sum()
    )
    v21_scenarios = {item.scenario for item in scenarios if item.gate == "v21"}
    metrics.loc[metrics.scenario.isin(v21_scenarios), "v21_risk_off_hours"] = total_v21_risk_off_hours
    pair_metrics = pd.DataFrame(pair_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    intervals = build_intervals(states, events)
    impacts = mechanism_impacts(metrics)

    full = scenarios[[item.scenario for item in scenarios].index("v21_full")]
    stress_rows = []
    stress_specs = [
        ("base", evaluation, gates, .001, 0.0),
        ("taker_fee_1_5x", evaluation, gates, .0015, 0.0),
        ("slippage_0_05pct", evaluation, gates, .001, .0005),
        ("slippage_0_1pct", evaluation, gates, .001, .001),
    ]
    if not args.skip_stress:
        btc_crash_all = stress_candles(all_candles, "BTC-FDUSD")
        eth_crash_all = stress_candles(all_candles, "ETH-FDUSD")
        btc_crash_gate, _, _ = load_frozen_v21(btc_crash_all, args.package_dir)
        eth_crash_gate, _, _ = load_frozen_v21(eth_crash_all, args.package_dir)
        stress_specs.extend([
            ("btc_single_day_drop_15pct", slice_window(btc_crash_all, START_TS, END_TS),
             {"v21": btc_crash_gate, "roc": roc_gate}, .001, 0.0),
            ("eth_single_day_drop_15pct", slice_window(eth_crash_all, START_TS, END_TS),
             {"v21": eth_crash_gate, "roc": roc_gate}, .001, 0.0),
        ])
    for name, stressed, stress_gates, fee, slippage in stress_specs:
        print(f"running stress {name}", flush=True)
        item, stress_pairs, _, _ = evaluate(
            full, stressed, stress_gates, taker_fee=fee, slippage=slippage,
        )
        item["stress"] = name
        stress_pair_map = {row["pair"]: row["net_pnl_quote"] for row in stress_pairs}
        item["btc_pnl_fdusd"] = stress_pair_map["BTC-FDUSD"]
        item["eth_pnl_fdusd"] = stress_pair_map["ETH-FDUSD"]
        item["both_pairs_nonnegative"] = all(row["net_pnl_quote"] >= 0 for row in stress_pairs)
        item["passed"] = bool(
            item["net_pnl_fdusd"] > 0 and item["both_pairs_nonnegative"]
            and item["pair_stop_events"] == 0 and item["portfolio_stop_events"] == 0
            and item["max_drawdown_pct"] >= -6
        )
        stress_rows.append(item)
    stress = pd.DataFrame(stress_rows)

    full_metric = metrics.set_index("scenario").loc["v21_full"]
    roc_metric = metrics.set_index("scenario").loc["roc_full"]
    full_pairs = pair_metrics[pair_metrics.scenario.eq("v21_full")]
    acceptance = {
        "positive_net_profit": bool(full_metric.net_pnl_fdusd > 0),
        "both_pairs_nonnegative": bool((full_pairs.net_pnl_quote >= 0).all()),
        "zero_portfolio_stops": bool(full_metric.portfolio_stop_events == 0),
        "zero_pair_stops": bool(full_metric.pair_stop_events == 0),
        "max_drawdown_within_6pct": bool(full_metric.max_drawdown_pct >= -6),
        "profit_better_than_roc_full": bool(full_metric.net_pnl_fdusd > roc_metric.net_pnl_fdusd),
        "drawdown_not_worse_than_roc_full": bool(full_metric.max_drawdown_pct >= roc_metric.max_drawdown_pct),
        "all_stress_pass": bool(not stress.empty and stress.passed.all()),
    }
    reason_names = {
        "positive_net_profit": "250天净收益不为正",
        "both_pairs_nonnegative": "BTC或ETH单对亏损",
        "zero_portfolio_stops": "发生组合熔断",
        "zero_pair_stops": "发生单对熔断",
        "max_drawdown_within_6pct": "最大回撤超过6%",
        "profit_better_than_roc_full": "收益未超过ROC/SQZMOM完整栈",
        "drawdown_not_worse_than_roc_full": "回撤劣于ROC/SQZMOM完整栈",
        "all_stress_pass": "费用、滑点或暴跌压力测试未全部通过",
    }
    failed = [reason_names[key] for key, value in acceptance.items() if not value]
    summary = {
        "schema": "fdusd-grid-v21-mechanisms-250d-evaluation-v1",
        "evidence_status": "known_250d_window_joint_diagnostic",
        "period": {"start_utc": utc(START_TS), "end_exclusive_utc": utc(END_TS), "days": 250},
        "grid": {**asdict(BASE_CANDIDATE), "levels": BASE_CANDIDATE.levels,
                 "maker_fee": 0.0, "taker_fee": .001, "order_refresh_seconds": 7_200,
                 "pair_budget_fdusd": 200, "reserve_fdusd": 20,
                 "initial_quote_per_pair": 100, "initial_base_value_per_pair": 100},
        "execution_filters": {
            "observed_utc": EXECUTION_FILTERS_OBSERVED_UTC,
            "source": "Binance /api/v3/exchangeInfo reproducibility snapshot",
            "pairs": {pair: asdict(value) for pair, value in EXECUTION_FILTERS.items()},
        },
        "acceptance": acceptance, "eligible_for_new_shadow": all(acceptance.values()),
        "failed_acceptance_reasons": failed, "deployment_allowed": False,
        "oci_modified": False, "model_audit": model_audit,
        "input_hashes": data_hashes,
        "v21_state_summary": {
            pair: {
                "entries": int(((states.pair == pair) & (states.transition == "enter")
                                & states.signal_ts.between(START_TS, END_TS - 1)).sum()),
                "recoveries": int(((states.pair == pair) & (states.transition == "recover")
                                   & states.signal_ts.between(START_TS, END_TS - 1)).sum()),
                "risk_off_hours": int(((states.pair == pair) & states.risk_off_active
                                       & states.signal_ts.between(START_TS, END_TS - 1)).sum()),
                "cooldown_hours": int(((states.pair == pair)
                                       & (states.cooldown_until.fillna(-1) > states.signal_ts)
                                       & ~states.risk_off_active
                                       & states.signal_ts.between(START_TS, END_TS - 1)).sum()),
            } for pair in PAIRS
        },
    }

    metrics.to_csv(args.output_dir / "scenario_metrics.csv", index=False)
    pair_metrics.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    curves.to_csv(args.output_dir / "equity_curves.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0})
    events.to_csv(args.output_dir / "events.csv", index=False)
    intervals.to_csv(args.output_dir / "risk_intervals.csv", index=False)
    impacts.to_csv(args.output_dir / "mechanism_impacts.csv", index=False)
    states.to_csv(args.output_dir / "v21_states.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0})
    stress.to_csv(args.output_dir / "stress_tests.csv", index=False)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    atomic_json(args.output_dir / "summary.json", summary)
    build_report(args.output_dir / "fdusd_grid_v21_mechanisms_250d_plotly.html",
                 metrics, pair_metrics, curves, states, events, intervals, stress, impacts,
                 summary, scenarios)
    output_hashes = {
        path.name: sha256_file(path) for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    atomic_json(args.output_dir / "manifest.json", {
        "schema": "fdusd-grid-v21-mechanisms-250d-manifest-v1",
        "random_seed": 42, "input_hashes": data_hashes,
        "model_sha256": model_audit["model_sha256"], "config_sha256": sha256_json(summary["grid"]),
        "output_hashes": output_hashes, "deployment_allowed": False,
    })
    print(json.dumps({"eligible_for_new_shadow": summary["eligible_for_new_shadow"],
                      "deployment_allowed": False, "failed": failed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
