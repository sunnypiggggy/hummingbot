#!/usr/bin/env python3
"""Add persistent probability/ROC/SQZ evidence to the locked v9 long gate.

This is a focused state-machine refinement: models, predictions, thresholds,
short-spike gates and Grid accounting remain locked to v9.  A long entry is
eligible only when either probability rises for three complete hourly bars or
both 4h ROC and SQZMOM deteriorate across two complete 4h observations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine


v5 = engine.v5
dual = engine.dual
HOUR = engine.HOUR
FIVE_MINUTES = engine.FIVE_MINUTES
PAIRS = engine.PAIRS

MODEL_VERSION = "xgboost-v9-long-entry-persistence-v14"
V9_DIR = Path("results/backtests/xgboost_regime_spike_pair_risk_gate_v9")
OUTPUT_DIR = Path("results/backtests/xgboost_v9_long_entry_persistence_v14")
CONTEXT_FEATURES = ("roc_48h_4h", "sqzmom_pct_4h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-dir", type=Path, default=V9_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    return parser.parse_args()


def load_locked_inputs(v9_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    summary = json.loads((v9_dir / "summary.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(v9_dir / "feature_panel.csv.gz", usecols=["pair", "signal_ts", *CONTEXT_FEATURES])
    predictions: dict[str, pd.DataFrame] = {}
    for pair, row in summary["pair_winners"].items():
        for channel in ("long", "short"):
            key = str(row[f"{channel}_model_key"])
            path = v9_dir / "prediction_cache" / "weekly" / f"{key.replace('|', '__')}.csv.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            predictions[key] = pd.read_csv(path)
    return summary, panel, predictions


def attach_entry_evidence(prediction: pd.DataFrame, panel: pd.DataFrame, pair: str) -> pd.DataFrame:
    context = panel[panel.pair.eq(pair)][["pair", "signal_ts", *CONTEXT_FEATURES]]
    output = prediction.merge(context, on=["pair", "signal_ts"], how="left", validate="one_to_one")
    output = output.sort_values("signal_ts").reset_index(drop=True)
    output["probability_lag_1h"] = output.probability.shift(1)
    output["probability_lag_2h"] = output.probability.shift(2)
    output["roc_lag_4h"] = output.roc_48h_4h.shift(4)
    output["roc_lag_8h"] = output.roc_48h_4h.shift(8)
    output["sqz_lag_4h"] = output.sqzmom_pct_4h.shift(4)
    output["sqz_lag_8h"] = output.sqzmom_pct_4h.shift(8)
    output["probability_rising_3h"] = (
        (output.probability > output.probability_lag_1h)
        & (output.probability_lag_1h > output.probability_lag_2h)
    )
    output["roc_sqz_worsening_8h"] = (
        (output.roc_48h_4h < output.roc_lag_4h)
        & (output.roc_lag_4h <= output.roc_lag_8h)
        & (output.sqzmom_pct_4h < output.sqz_lag_4h)
        & (output.sqz_lag_4h <= output.sqz_lag_8h)
        & (output.roc_48h_4h < 0.0)
        & (output.sqzmom_pct_4h < 0.0)
    )
    return output


def build_long_gate(
    prediction: pd.DataFrame, panel: pd.DataFrame, pair: str, target: str,
    gate: v5.GateParameters,
) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Large parameter searches reuse one prediction across many state
    # machines. An already enriched immutable frame avoids repeating the
    # context merge and lag construction for every gate.
    required_evidence = {
        "probability_lag_2h", "probability_rising_3h", "roc_sqz_worsening_8h",
    }
    enriched = prediction if required_evidence.issubset(prediction.columns) else attach_entry_evidence(
        prediction, panel, pair
    )
    rows = list(enriched.itertuples(index=False))
    timeline = {item: {} for item in PAIRS}
    states, events, intervals = [], [], []
    state, interval_start = v5.GateState(), None
    for index, row in enumerate(rows):
        entry = float(getattr(row, v5.quantile_column(gate.entry_quantile)))
        recovery = float(getattr(row, v5.quantile_column(gate.recovery_quantile)))
        probability_rise = float(row.probability) - float(row.probability_lag_2h)
        minimum_rise = max(1e-4, 0.25 * max(entry - recovery, 0.0))
        probability_condition = bool(row.probability_rising_3h and probability_rise >= minimum_rise)
        technical_condition = bool(row.roc_sqz_worsening_8h)
        entry_condition = probability_condition or technical_condition
        effective_probability = float(row.probability)
        if not state.active and not entry_condition:
            effective_probability = min(effective_probability, float(np.nextafter(entry, -np.inf)))
        state, transition, reason = v5.step_gate(
            effective_probability, entry, recovery, int(row.signal_ts), state, gate
        )
        if not state.active and transition == "clear" and float(row.probability) >= entry and not entry_condition:
            reason = "entry_blocked_no_persistent_probability_or_roc_sqz_deterioration"
        right = min(
            int(rows[index + 1].signal_ts) if index + 1 < len(rows) else engine.END_TS,
            engine.END_TS,
        )
        for timestamp in range(max(engine.START_TS, int(row.signal_ts)), right, FIVE_MINUTES):
            timeline[pair][timestamp] = not state.active
        states.append({
            "strategy": engine.strategy_name(pair, "long", target), "channel": "long",
            "target": target, "pair": pair, "signal_ts": int(row.signal_ts),
            "probability": float(row.probability), "entry_threshold": entry,
            "recovery_threshold": recovery, "risk_off_active": bool(state.active),
            "buy_enabled": not bool(state.active), "transition": transition, "reason": reason,
            "probability_rising_3h": probability_condition,
            "roc_sqz_worsening_8h": technical_condition,
            "entry_condition_pass": entry_condition,
        })
        if transition == "enter":
            interval_start = int(row.signal_ts)
        elif transition == "recover" and interval_start is not None:
            intervals.append({
                "strategy": engine.strategy_name(pair, "long", target), "channel": "long",
                "target": target, "pair": pair, "start_ts": interval_start,
                "end_ts": int(row.signal_ts),
                "duration_hours": (int(row.signal_ts) - interval_start) / HOUR,
                "end_reason": reason,
            })
            interval_start = None
        if transition in {"enter", "recover"}:
            events.append({
                "strategy": engine.strategy_name(pair, "long", target), "channel": "long",
                "target": target, "pair": pair, "timestamp": int(row.signal_ts),
                "event": transition, "probability": float(row.probability),
                "entry_threshold": entry, "recovery_threshold": recovery,
                "event_id": f"{MODEL_VERSION}-{pair}-long-{int(row.signal_ts)}-{transition}",
            })
    if interval_start is not None:
        intervals.append({
            "strategy": engine.strategy_name(pair, "long", target), "channel": "long",
            "target": target, "pair": pair, "start_ts": interval_start,
            "end_ts": engine.END_TS, "duration_hours": (engine.END_TS - interval_start) / HOUR,
            "end_reason": "research_period_end",
        })
    return timeline, pd.DataFrame(states), pd.DataFrame(events), pd.DataFrame(intervals)


def filtered_combiner(panel: pd.DataFrame):
    def combine(
        specifications: Sequence[tuple[pd.DataFrame, str, str, str, v5.GateParameters]],
    ) -> tuple[dict[str, dict[int, bool]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        timelines, states, events, intervals = [], [], [], []
        for prediction, pair, channel, target, gate in specifications:
            result = (
                build_long_gate(prediction, panel, pair, target, gate)
                if channel == "long"
                else engine.build_pair_gate(prediction, pair, channel, target, gate)
            )
            timeline, state, event, interval = result
            timelines.append(timeline); states.append(state)
            if not event.empty: events.append(event)
            if not interval.empty: intervals.append(interval)
        return (
            dual.combine_channel_gates(timelines, engine.START_TS, engine.END_TS),
            pd.concat(states, ignore_index=True),
            pd.concat(events, ignore_index=True) if events else pd.DataFrame(),
            pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame(),
        )
    return combine


def specifications(
    summary: Mapping[str, Any], predictions: Mapping[str, pd.DataFrame],
) -> list[tuple[pd.DataFrame, str, str, str, v5.GateParameters]]:
    output = []
    for pair, row in summary["pair_winners"].items():
        for channel in ("long", "short"):
            key = str(row[f"{channel}_model_key"])
            output.append((predictions[key], pair, channel, str(row[f"{channel}_target"]), engine.gate_from_row(row, f"{channel}_")))
    return output


def build_plot(args: argparse.Namespace, metrics: pd.DataFrame) -> Path:
    states = pd.read_csv(args.output_dir / "final_risk_states.csv.gz")
    events = pd.read_csv(args.output_dir / "final_risk_events.csv")
    intervals = pd.read_csv(args.output_dir / "final_risk_intervals.csv")
    mapping = {"long": "long_persistent_72h", "short": "short_spike_1h_24h"}
    labels = {"long": "长期Risk-off", "short": "短期插针Risk-off"}
    for frame in (states, events, intervals):
        frame["strategy"] = frame.channel.map(mapping)
        frame["strategy_label"] = frame.channel.map(labels)
    original = {key: value.copy() for key, value in dual.STRATEGIES.items()}
    try:
        dual.STRATEGIES["long_persistent_72h"]["label"] = labels["long"]
        dual.STRATEGIES["short_spike_1h_24h"]["label"] = labels["short"]
        source = dual.build_plotly(
            args.cache_dir, args.output_dir, states, events, intervals, metrics,
            pd.DataFrame(), anchor_windows=engine.ANCHOR_WINDOWS,
        )
    finally:
        dual.STRATEGIES.clear(); dual.STRATEGIES.update(original)
    page = source.read_text(encoding="utf-8")
    title = "XGBoost v14：持续概率或ROC/SQZ恶化长期Risk-off"
    page = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", page, count=1, flags=re.S)
    page = re.sub(r"<h1>.*?</h1>", f"<h1>{title}</h1>", page, count=1, flags=re.S)
    note = (
        '<div class="note"><b>v14进入过滤：</b>超过模型阈值后，还必须满足“概率连续3小时上升”'
        '或“ROC48与SQZMOM%连续两个完整4小时周期同时恶化”。橙色长期与蓝色短期阴影可独立开关；'
        '模型只暂停普通BUY，不触发卖出。</div>'
    )
    page = page.replace(f"<h1>{title}</h1>", f"<h1>{title}</h1>{note}", 1)
    target = args.output_dir / "xgboost_v14_long_entry_persistence_plotly.html"
    target.write_text(page, encoding="utf-8")
    return target


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, panel, predictions = load_locked_inputs(args.v9_dir)
    candles, _ = engine.load_candles(args.cache_dir)
    selections = pd.read_csv(args.v9_dir / "grid_selections.csv")
    specs = specifications(summary, predictions)
    original_combiner = engine.combine_pair_gates
    try:
        engine.combine_pair_gates = filtered_combiner(panel)
        detailed = engine.detailed_replay(candles, selections, specs, MODEL_VERSION)
    finally:
        engine.combine_pair_gates = original_combiner
    metrics = detailed["summary"]
    original_metrics = summary["winner_metrics"]
    intervals = detailed["intervals"]
    long_intervals = intervals[intervals.channel.eq("long")]
    frequency_rows = []
    for pair in PAIRS:
        group = long_intervals[long_intervals.pair.eq(pair)]
        frequency_rows.append({
            "pair": pair, "interval_count": len(group),
            "active_hours": float(group.duration_hours.sum()),
            "median_duration_hours": float(group.duration_hours.median()) if len(group) else 0.0,
            **engine.pair_anchor_metrics(group, pair),
        })
    frequency = pd.DataFrame(frequency_rows)
    detailed["states"].to_csv(args.output_dir / "final_risk_states.csv.gz", index=False, compression="gzip")
    detailed["events"].to_csv(args.output_dir / "final_risk_events.csv", index=False)
    intervals.to_csv(args.output_dir / "final_risk_intervals.csv", index=False)
    detailed["equity"].to_csv(args.output_dir / "final_equity_curve.csv.gz", index=False, compression="gzip")
    frequency.to_csv(args.output_dir / "long_frequency_metrics.csv", index=False)
    mechanism = pd.read_csv(args.v9_dir / "final_metrics.csv").iloc[0].to_dict()
    comparison = pd.DataFrame([
        {"scenario": str(mechanism.pop("scenario")), **mechanism},
        {"scenario": "XGBoost v9", **original_metrics},
        {"scenario": "XGBoost v14 persistence filter", **metrics},
    ])
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    comparison.to_csv(args.output_dir / "final_metrics.csv", index=False)
    result = {
        "model_version": MODEL_VERSION, "deployment_allowed": False,
        "evidence_status": "same_180d_targeted_state_machine_refinement",
        "entry_rule": {
            "probability": "three strictly rising 1h probabilities with net rise >= 25% of hysteresis gap",
            "technical": "ROC48 and SQZMOM% both worsen over two complete 4h steps and are below zero",
            "combination": "probability OR technical; model threshold remains mandatory",
        },
        "original_metrics": original_metrics, "refined_metrics": metrics,
        "frequency": frequency.to_dict("records"),
        "verdict": "DIAGNOSTIC_ONLY",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot = build_plot(args, comparison)
    result["plotly"] = plot.as_posix()
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
