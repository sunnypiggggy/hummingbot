#!/usr/bin/env python3
"""Compare maker-only DCA configurations that increase live fill frequency."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest_dca_live_local import PAIRS, floor_five_minutes, load_window, run_pair


SPREAD_SETS = {
    "current_1_2_4_8": (0.01, 0.02, 0.04, 0.08),
    "near_0.9_1.8_3.6_7.2": (0.009, 0.018, 0.036, 0.072),
    "near_0.8_1.6_3.2_6.4": (0.008, 0.016, 0.032, 0.064),
    "near_0.75_1.5_3_6": (0.0075, 0.015, 0.03, 0.06),
    "near_0.67_1.33_2.67_5.33": (0.0067, 0.0133, 0.0267, 0.0533),
    "near_0.5_1_2_4": (0.005, 0.01, 0.02, 0.04),
    "near_0.3_0.8_2_4": (0.003, 0.008, 0.02, 0.04),
    "near_0.2_0.5_1_2": (0.002, 0.005, 0.01, 0.02),
    "near_0.1_0.3_0.8_2": (0.001, 0.003, 0.008, 0.02),
}
REFRESH_VALUES = (300, 900, 2700)


def aggregate(name: str, spreads: tuple[float, ...], refresh_seconds: int,
              fee_rate: float, pair_results: list[dict]) -> dict:
    return {
        "scenario": f"{name}_refresh_{refresh_seconds // 60}m",
        "spread_name": name,
        "spreads": spreads,
        "refresh_seconds": refresh_seconds,
        "fee_rate": fee_rate,
        "combined_net_pnl_quote": sum(item["net_pnl_quote"] for item in pair_results),
        "return_on_380": sum(item["net_pnl_quote"] for item in pair_results) / 380.0,
        "combined_filled_executors": sum(item["executors_with_position"] for item in pair_results),
        "combined_positioned_per_day": sum(item["positioned_executors_per_day"] for item in pair_results),
        "combined_fill_events": sum(item["fill_events"] for item in pair_results),
        "combined_fees_quote": sum(item["fees_quote"] for item in pair_results),
        "worst_pair_drawdown_pct": min(item["max_drawdown_pct_mtm"] for item in pair_results),
        "worst_pair_max_fill_gap_hours": max(item["max_fill_gap_hours"] for item in pair_results),
        "btc_days_with_fills": next(item["days_with_fills"] for item in pair_results if item["pair"] == "BTC-USDT"),
        "eth_days_with_fills": next(item["days_with_fills"] for item in pair_results if item["pair"] == "ETH-USDT"),
        "pairs": pair_results,
    }


def choose_recommendation(rows: list[dict]) -> dict:
    standard = [row for row in rows if row["fee_rate"] == 0.001]
    baseline = next(
        row for row in standard
        if row["spread_name"] == "current_1_2_4_8" and row["refresh_seconds"] == 300
    )
    eligible = [
        row for row in standard
        if row["combined_net_pnl_quote"] >= 0
        and row["combined_positioned_per_day"] >= baseline["combined_positioned_per_day"] * 3
        and row["worst_pair_drawdown_pct"] >= -0.10
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (row["combined_positioned_per_day"], row["combined_net_pnl_quote"]),
        )
        reason = "positive at 0.10% fee, at least 3x baseline fills, and pair drawdown no worse than 10%"
    else:
        improved = [
            row for row in standard
            if row["combined_positioned_per_day"] > baseline["combined_positioned_per_day"]
            and row["worst_pair_drawdown_pct"] >= -0.10
        ]
        selected = max(
            improved,
            key=lambda row: (row["combined_net_pnl_quote"], row["combined_positioned_per_day"]),
        )
        reason = "no candidate passed every guardrail; selected the highest 0.10% fee PnL among higher-fill candidates"
    return {
        "scenario": selected["scenario"],
        "reason": reason,
        "baseline_scenario": baseline["scenario"],
        "fill_multiple_vs_baseline": (
            selected["combined_positioned_per_day"] / baseline["combined_positioned_per_day"]
        ),
        "standard_fee_result": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize current live DCA fill frequency")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end", default="2026-07-27T02:00:00Z")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/backtests/dca_live_fill_optimization_180d"))
    parser.add_argument(
        "--spread-names",
        default=",".join(SPREAD_SETS),
        help="Comma-separated spread-set names; current_1_2_4_8 must be included",
    )
    parser.add_argument(
        "--refresh-values",
        default=",".join(str(item) for item in REFRESH_VALUES),
        help="Comma-separated refresh intervals in seconds",
    )
    args = parser.parse_args()
    if args.days != 180:
        raise ValueError("optimization comparison is fixed to 180 days")
    end = floor_five_minutes(datetime.fromisoformat(args.end.replace("Z", "+00:00")))
    start = end - timedelta(days=args.days)
    spread_names = tuple(item.strip() for item in args.spread_names.split(",") if item.strip())
    unknown_spreads = sorted(set(spread_names) - set(SPREAD_SETS))
    if unknown_spreads:
        raise ValueError(f"unknown spread sets: {', '.join(unknown_spreads)}")
    if "current_1_2_4_8" not in spread_names:
        raise ValueError("current_1_2_4_8 is required as the baseline")
    refresh_values = tuple(int(item.strip()) for item in args.refresh_values.split(",") if item.strip())
    if not refresh_values or any(item <= 0 or item % 300 for item in refresh_values):
        raise ValueError("refresh values must be positive multiples of 300 seconds")
    frames: dict[str, pd.DataFrame] = {
        pair: load_window(args.cache_dir / f"{symbol}_5m.csv", start, end)
        for pair, symbol in PAIRS.items()
    }
    rows: list[dict] = []
    for spread_name in spread_names:
        spreads = SPREAD_SETS[spread_name]
        for refresh_seconds in refresh_values:
            for fee_rate in (0.00075, 0.001):
                pair_results = []
                for pair, frame in frames.items():
                    result, records = run_pair(frame, pair, fee_rate, spreads, refresh_seconds)
                    pair_results.append(result)
                    del records
                    gc.collect()
                row = aggregate(spread_name, spreads, refresh_seconds, fee_rate, pair_results)
                rows.append(row)
                print(
                    f"{row['scenario']} fee={fee_rate:.3%} "
                    f"positions/day={row['combined_positioned_per_day']:.2f} "
                    f"pnl={row['combined_net_pnl_quote']:+.2f}",
                    flush=True,
                )

    recommendation = choose_recommendation(rows)
    output = args.output_dir / end.strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": args.days},
        "selection_guardrails": {
            "fee_rate": 0.001,
            "minimum_fill_multiple": 3,
            "minimum_combined_net_pnl_quote": 0,
            "minimum_worst_pair_drawdown_pct": -0.10,
        },
        "recommendation": recommendation,
        "scenarios": rows,
    }
    (output / "optimization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ranked = sorted(
        (row for row in rows if row["fee_rate"] == 0.001),
        key=lambda row: row["combined_positioned_per_day"],
        reverse=True,
    )
    lines = [
        "# DCA Fill Optimization - 180 Day Backtest",
        "",
        f"Recommended: `{recommendation['scenario']}`",
        f"Selection: {recommendation['reason']}",
        "",
        "| Scenario | Positions/day | Net PnL @0.10% | Return/380 | Worst pair DD | Worst fill gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['scenario']} | {row['combined_positioned_per_day']:.2f} | "
            f"{row['combined_net_pnl_quote']:+.2f} | {row['return_on_380']:+.2%} | "
            f"{row['worst_pair_drawdown_pct']:.2%} | {row['worst_pair_max_fill_gap_hours']:.1f}h |"
        )
    lines.extend([
        "",
        "Maker-only caveat: a closer limit grid can raise historical fill frequency but cannot guarantee a future fill.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "recommendation": recommendation["scenario"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
