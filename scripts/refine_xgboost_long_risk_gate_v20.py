#!/usr/bin/env python3
"""Refine v19 long-only entry/holding logic without retraining its models."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import retrain_xgboost_long_risk_gate_250d_v19 as v19


MODEL_VERSION = "xgboost-grid-long-risk-gate-v20-250d"
OUTPUT_DIR = Path("results/backtests/xgboost_grid_long_risk_gate_v20_250d")
V19_DIR = v19.OUTPUT_DIR


@dataclass(frozen=True)
class Gate:
    entry_quantile: float
    entry_bars: int
    arm_hours: int
    minimum_hours: int
    cooldown_hours: int
    recovery_4h_bars: int
    confirmation_mode: str
    recovery_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "finalize", "plot", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--v19-dir", type=Path, default=V19_DIR)
    parser.add_argument("--source-dir", type=Path, default=v19.SOURCE_DIR)
    return parser.parse_args()


def candidates() -> list[Gate]:
    result = [Gate(q, entry, arm, minimum, cooldown, recovery, confirmation, recovery_mode)
              for q in (.90, .925, .95, .97, .98)
              for entry in (1, 2)
              for arm in (48, 72)
              for minimum in (48, 72)
              for cooldown in (24, 48)
              for recovery in (3, 4)
              for confirmation in ("directional_relaxed", "persistent_bearish")
              for recovery_mode in ("structural_relief", "regime_exit")]
    if len(result) != 640 or len({tuple(asdict(item).values()) for item in result}) != 640:
        raise AssertionError("v20 state search must contain 640 deterministic gates")
    return result


def _entry_confirm(current: tuple[float, ...], previous: tuple[float, ...] | None, mode: str) -> bool:
    roc, sqz, di, slope, below = current
    bearish_votes = int(di < 0) + int(slope < 0) + int(below >= .50)
    if not (roc < 0 and sqz < 0 and bearish_votes >= 2):
        return False
    if mode == "persistent_bearish":
        return below >= .55 or (di < 0 and slope < 0)
    if previous is None:
        return False
    return bool(roc < previous[0] or sqz < previous[1])


def _recovery_confirm(current: tuple[float, ...], previous: tuple[float, ...] | None, mode: str) -> bool:
    if previous is None:
        return False
    roc, sqz, di, slope, below = current
    relief_votes = int(di > 0) + int(slope >= 0) + int(below < .50)
    improving = roc > previous[0] and sqz > previous[1] and relief_votes >= 2
    if mode == "regime_exit":
        improving = improving and (roc >= 0 or sqz >= 0)
    return bool(improving)


def build_state(prediction: pd.DataFrame, pair: str, gate: Gate,
                include_timeline: bool = True, include_states: bool = True
                ) -> tuple[dict[int, bool], pd.DataFrame, pd.DataFrame]:
    rows = prediction.sort_values("signal_ts").reset_index(drop=True)
    qcol = v19.legacy.v5.quantile_column(gate.entry_quantile)
    ts = rows.signal_ts.to_numpy(np.int64); probability = rows.probability.to_numpy(float)
    threshold = rows[qcol].to_numpy(float); complete = rows.last_complete_4h_ts.to_numpy(np.int64)
    roc = rows.roc_48h_4h.to_numpy(float); sqz = rows.sqzmom_pct_4h.to_numpy(float)
    di = rows.di_spread.to_numpy(float); slope = rows.ema20_slope_atr_12h.to_numpy(float)
    below = rows.below_ema20_ratio_72h.to_numpy(float)
    active = False; above = 0; armed_until = -1; cooldown_until = -1
    recovery_count = 0; start = None; last_complete = None; previous = None
    timeline: dict[int, bool] = {}; states = []; intervals = []
    for index in range(len(rows)):
        timestamp = int(ts[index]); current = (roc[index], sqz[index], di[index], slope[index], below[index])
        above = above + 1 if probability[index] >= threshold[index] else 0
        if not active and above >= gate.entry_bars:
            armed_until = max(armed_until, timestamp + gate.arm_hours * v19.HOUR)
        new4h = last_complete != int(complete[index])
        entry_ok = new4h and _entry_confirm(current, previous, gate.confirmation_mode)
        recovery_ok = new4h and _recovery_confirm(current, previous, gate.recovery_mode)
        transition = "hold" if active else "clear"
        if not active and timestamp >= cooldown_until and timestamp <= armed_until and entry_ok:
            active = True; start = timestamp; recovery_count = 0; transition = "enter"
        elif active and new4h:
            recovery_count = recovery_count + 1 if recovery_ok else 0
            strong_relief = (current[0] >= 0 or current[1] >= 0) and current[2] > 0 and current[3] >= 0 and current[4] < .50
            required_recovery = 2 if gate.recovery_mode == "adaptive_relief" and strong_relief else gate.recovery_4h_bars
            if start is not None and timestamp - start >= gate.minimum_hours * v19.HOUR and recovery_count >= required_recovery:
                active = False; transition = "recover"; cooldown_until = timestamp + gate.cooldown_hours * v19.HOUR
                intervals.append({"pair": pair, "start_ts": start, "end_ts": timestamp,
                                  "duration_hours": (timestamp - start) / v19.HOUR,
                                  "end_reason": f"{gate.recovery_4h_bars}_4h_{gate.recovery_mode}"})
                start = None; above = 0; armed_until = -1
        if new4h: previous = current; last_complete = int(complete[index])
        right = min(int(ts[index + 1]) if index + 1 < len(rows) else v19.END_TS, v19.END_TS)
        if include_timeline:
            for timestamp5m in range(max(timestamp, v19.START_TS), right, 300): timeline[timestamp5m] = not active
        if include_states:
            states.append({"pair": pair, "signal_ts": timestamp, "probability": probability[index],
                           "entry_threshold": threshold[index], "risk_off_active": active,
                           "buy_enabled": not active, "armed": timestamp <= armed_until,
                           "entry_structure_confirmed": entry_ok, "recovery_structure_confirmed": recovery_ok,
                           "structure_recovery_count": recovery_count, "transition": transition})
    if start is not None:
        intervals.append({"pair": pair, "start_ts": start, "end_ts": v19.END_TS,
                          "duration_hours": (v19.END_TS - start) / v19.HOUR, "end_reason": "research_period_end"})
    return timeline, pd.DataFrame(states), pd.DataFrame(intervals)


def load_predictions(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    specs = json.loads((args.v19_dir / "weekly_finalists.json").read_text(encoding="utf-8"))
    predictions = {}
    proxy = type("Args", (), {"output_dir": args.v19_dir})()
    for spec in specs:
        predictions[str(spec["model_key"])] = pd.read_csv(v19.cache_path(proxy, "weekly", spec))
    return specs, predictions


def structural_search(args: argparse.Namespace, specs: list[dict[str, Any]],
                      predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        prediction = predictions[str(spec["model_key"])]
        for gate in candidates():
            _, _, intervals = build_state(prediction, str(spec["pair"]), gate, False, False)
            metrics = v19.anchor_metrics(intervals, str(spec["pair"]))
            gate_hash = hashlib.sha256(json.dumps(asdict(gate), sort_keys=True).encode()).hexdigest()[:12]
            interval_payload = intervals[["start_ts", "end_ts"]].to_dict("records") if not intervals.empty else []
            interval_hash = hashlib.sha256(json.dumps(interval_payload, sort_keys=True).encode()).hexdigest()
            rows.append({"candidate_id": f"{spec['model_key']}|{gate_hash}",
                         "model_key": spec["model_key"], "pair": spec["pair"], "target": spec["target"],
                         "feature_id": spec["feature_id"], "interval_sha256": interval_hash,
                         **asdict(gate), **metrics,
                         "active_hours": float(intervals.duration_hours.sum()) if not intervals.empty else 0.0})
    frame = pd.DataFrame(rows); frame["structure_pass"] = frame.anchor_pass.astype(bool)
    frame["minimum_anchor_coverage"] = frame[[f"{name}_coverage" for name, _, _ in v19.ANCHOR_WINDOWS]].min(axis=1)
    frame = frame.sort_values(["pair", "structure_pass", "minimum_anchor_coverage", "interval_count",
                               "outside_anchor_share", "active_hours"], ascending=[True, False, False, True, True, True])
    frame.to_csv(args.output_dir / "structural_search.csv", index=False)
    return frame


def gate_from_row(row: Any) -> Gate:
    return Gate(float(row.entry_quantile), int(row.entry_bars), int(row.arm_hours), int(row.minimum_hours),
                int(row.cooldown_hours), int(row.recovery_4h_bars), str(row.confirmation_mode), str(row.recovery_mode))


def materialize(row: Any, predictions: Mapping[str, pd.DataFrame], states: bool = False):
    return build_state(predictions[str(row.model_key)], str(row.pair), gate_from_row(row), True, states)


def write_states(args: argparse.Namespace, rows: list[Any], predictions: Mapping[str, pd.DataFrame]) -> None:
    state_parts, interval_parts = [], []
    for row in rows:
        _, state, interval = materialize(row, predictions, True)
        state_parts.append(state); interval_parts.append(interval)
    states = pd.concat(state_parts, ignore_index=True); intervals = pd.concat(interval_parts, ignore_index=True)
    states.to_csv(args.output_dir / "final_risk_states.csv.gz", index=False, compression="gzip")
    states[states.transition.isin(["enter", "recover"])].to_csv(args.output_dir / "final_risk_events.csv", index=False)
    intervals.to_csv(args.output_dir / "final_risk_intervals.csv", index=False)


def diagnostic(args: argparse.Namespace, structure: pd.DataFrame,
               predictions: Mapping[str, pd.DataFrame], candles: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    chosen = [group.iloc[0] for _, group in structure.groupby("pair", sort=False)]
    write_states(args, chosen, predictions)
    payload = {"model_version": MODEL_VERSION, "deployment_allowed": False, "verdict": "NO-GO",
               "reason": "no_structurally_eligible_candidate", "grid_search_executed": False,
               "diagnostic_pairs": {str(row.pair): row.to_dict() for row in chosen}}
    v19.atomic_json(args.output_dir / "locked_configuration.json", payload)
    v19.atomic_json(args.output_dir / "summary.json", payload)
    plot_args = type("PlotArgs", (), {"output_dir": args.output_dir})()
    source = v19.build_plot(plot_args, candles)
    target = args.output_dir / "xgboost_v20_long_only_250d_plotly.html"
    source.replace(target)
    payload["plotly"] = target.as_posix(); v19.atomic_json(args.output_dir / "summary.json", payload)
    return payload


def search_grid(args: argparse.Namespace, structure: pd.DataFrame,
                predictions: Mapping[str, pd.DataFrame], candles: Mapping[str, pd.DataFrame],
                selections: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pools = {pair: structure[(structure.pair == pair) & structure.structure_pass]
             .drop_duplicates("interval_sha256").copy() for pair in v19.PAIRS}
    references = {pair: pools[pair].iloc[0] for pair in v19.PAIRS}
    reference_timelines = {pair: materialize(row, predictions)[0] for pair, row in references.items()}
    pair_results = []
    for pair in v19.PAIRS:
        other = next(item for item in v19.PAIRS if item != pair)
        for row in pools[pair].itertuples(index=False):
            candidate_timeline = materialize(row, predictions)[0]
            gate = v19.combine_timelines({pair: candidate_timeline, other: reference_timelines[other]})
            pair_results.append({"pair": pair, "candidate_id": row.candidate_id,
                                 **v19.exact_replay(candles, selections, gate)})
    pair_frame = pd.DataFrame(pair_results)
    ranked_parts = []
    for pair, group in pair_frame.groupby("pair"):
        item = group.copy(); item["profit_percentile"] = item.oos_pnl_fdusd.rank(pct=True)
        item["drawdown_percentile"] = item.stitched_max_drawdown_pct.rank(pct=True)
        item["objective_score"] = .5 * item.profit_percentile + .5 * item.drawdown_percentile
        ranked_parts.append(item.sort_values(["objective_score", "portfolio_stop_events", "pair_stop_events"],
                                              ascending=[False, True, True]))
    pair_frame = pd.concat(ranked_parts, ignore_index=True)
    pair_frame.to_csv(args.output_dir / "pair_grid_search.csv", index=False)
    finalists = {pair: pair_frame[pair_frame.pair.eq(pair)].head(8).candidate_id.tolist() for pair in v19.PAIRS}
    row_lookup = {str(row.candidate_id): row for row in structure.itertuples(index=False)}
    timeline_cache = {candidate_id: materialize(row_lookup[candidate_id], predictions)[0]
                      for pair in v19.PAIRS for candidate_id in finalists[pair]}
    results = []
    for btc_id in finalists["BTC-FDUSD"]:
        for eth_id in finalists["ETH-FDUSD"]:
            gate = v19.combine_timelines({"BTC-FDUSD": timeline_cache[btc_id],
                                          "ETH-FDUSD": timeline_cache[eth_id]})
            results.append({"candidate_id": f"{btc_id}||{eth_id}",
                            "BTC_candidate_id": btc_id, "ETH_candidate_id": eth_id,
                            **v19.exact_replay(candles, selections, gate)})
    frame = pd.DataFrame(results)
    frame["profit_percentile"] = frame.oos_pnl_fdusd.rank(pct=True)
    frame["drawdown_percentile"] = frame.stitched_max_drawdown_pct.rank(pct=True)
    frame["objective_score"] = .5 * frame.profit_percentile + .5 * frame.drawdown_percentile
    baseline = v19.baseline_reference(args)
    frame["eligible"] = ((frame.oos_pnl_fdusd > 0) & (frame.oos_pnl_fdusd > float(baseline["oos_pnl_fdusd"]))
        & (frame.stitched_max_drawdown_pct >= float(baseline["stitched_max_drawdown_pct"]))
        & (frame.btc_pnl_fdusd >= 0) & (frame.eth_pnl_fdusd >= 0)
        & (frame.portfolio_stop_events == 0) & (frame.pair_stop_events < 7))
    frame = frame.sort_values(["eligible", "objective_score", "portfolio_stop_events", "pair_stop_events"],
                              ascending=[False, False, True, True]).reset_index(drop=True)
    frame.to_csv(args.output_dir / "grid_search.csv", index=False)
    winner = frame.iloc[0].to_dict()
    v19.atomic_json(args.output_dir / "locked_configuration.json", {
        "model_version": MODEL_VERSION, "deployment_allowed": False, "shadow_mode": True,
        "short_spike_enabled": False, "market_sell_action": False,
        "evidence_status": "250d_known_window_state_machine_refinement",
        "verdict": "SEARCH_LOCKED" if bool(winner["eligible"]) else "DIAGNOSTIC_ONLY",
        "candidate": winner})
    return frame, winner


def finalize_result(args: argparse.Namespace, winner: Mapping[str, Any], structure: pd.DataFrame,
                    predictions: Mapping[str, pd.DataFrame], candles: Mapping[str, pd.DataFrame],
                    selections: pd.DataFrame) -> dict[str, Any]:
    chosen = [structure[structure.candidate_id.eq(winner[f"{pair[:3]}_candidate_id"])].iloc[0]
              for pair in v19.PAIRS]
    write_states(args, chosen, predictions)
    timelines = {str(row.pair): materialize(row, predictions)[0] for row in chosen}
    detail = v19.exact_replay(candles, selections, v19.combine_timelines(timelines), return_details=True)
    for name in ("weekly", "pairs", "equity", "trades"):
        detail[name].to_csv(args.output_dir / f"final_{name}.csv.gz", index=False, compression="gzip")
    acceptance = {
        "positive_net_profit": float(detail["summary"]["oos_pnl_fdusd"]) > 0,
        "both_pair_pnl_nonnegative": float(detail["summary"]["btc_pnl_fdusd"]) >= 0 and float(detail["summary"]["eth_pnl_fdusd"]) >= 0,
        "zero_portfolio_stops": int(detail["summary"]["portfolio_stop_events"]) == 0,
        "fewer_than_7_pair_stops": int(detail["summary"]["pair_stop_events"]) < 7,
        "both_pair_structure_pass": all(bool(row.structure_pass) for row in chosen),
    }
    eligible = all(acceptance.values())
    payload = {
        "model_version": MODEL_VERSION, "deployment_allowed": False, "shadow_mode": True,
        "short_spike_enabled": False, "market_sell_action": False,
        "evidence_status": "250d_known_window_state_machine_refinement",
        "verdict": "NEXT_STAGE_JOINT_VALIDATION" if eligible else "NO-GO",
        "baseline": v19.baseline_reference(args), "metrics": detail["summary"],
        "acceptance": acceptance, "stress_tests_executed": False,
        "stress_tests_reason": None if eligible else "base_acceptance_failed",
        "selected_structure": {str(row.pair): row.to_dict() for row in chosen},
    }
    v19.atomic_json(args.output_dir / "summary.json", payload)
    comparison = pd.DataFrame([
        {"version": "Mechanism 1 250d", **payload["baseline"], "evidence_valid": True},
        {"version": "v16 diagnostic", "oos_pnl_fdusd": -12.1484398278,
         "stitched_max_drawdown_pct": -18.3555190824, "pair_stop_events": 25,
         "portfolio_stop_events": 3, "evidence_valid": False},
        {"version": "v21 corrected structural gate", **detail["summary"], "evidence_valid": True},
    ])
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    plot_args = type("PlotArgs", (), {"output_dir": args.output_dir})()
    source = v19.build_plot(plot_args, candles)
    target = args.output_dir / f"{MODEL_VERSION}_plotly.html"; source.replace(target)
    payload["plotly"] = target.as_posix(); v19.atomic_json(args.output_dir / "summary.json", payload)
    return payload


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    specs, predictions = load_predictions(args)
    candles = v19.load_candles(args.source_dir); selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    structure = structural_search(args, specs, predictions)
    missing = [pair for pair in v19.PAIRS if structure[(structure.pair == pair) & structure.structure_pass].empty]
    if missing:
        print(json.dumps(diagnostic(args, structure, predictions, candles), ensure_ascii=False, indent=2, default=str))
        return 2
    if args.stage in {"search", "all"}:
        _, winner = search_grid(args, structure, predictions, candles, selections)
    else:
        winner = json.loads((args.output_dir / "locked_configuration.json").read_text(encoding="utf-8"))["candidate"]
    result = finalize_result(args, winner, structure, predictions, candles, selections)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
