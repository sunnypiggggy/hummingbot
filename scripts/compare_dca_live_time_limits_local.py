#!/usr/bin/env python3
"""Compare the current live DCA across time limits from 45 minutes to 16 hours."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest_dca_live_local import (
    PAIRS,
    REFRESH_SECONDS,
    SPREADS,
    floor_five_minutes,
    load_window,
    run_pair,
)


TIME_LIMITS = (2700, 3600, 7200, 10800, 14400, 18000, 21600, 28800, 57600)
FEE_RATES = (0.0002, 0.00075, 0.001)


def aggregate(time_limit_seconds: int, fee_rate: float, pair_results: list[dict]) -> dict:
    return {
        "time_limit_seconds": time_limit_seconds,
        "time_limit_hours": time_limit_seconds / 3600,
        "fee_rate": fee_rate,
        "combined_net_pnl_quote": sum(item["net_pnl_quote"] for item in pair_results),
        "return_on_380": sum(item["net_pnl_quote"] for item in pair_results) / 380.0,
        "combined_gross_pnl_quote": sum(item["gross_pnl_quote"] for item in pair_results),
        "combined_fees_quote": sum(item["fees_quote"] for item in pair_results),
        "combined_positioned_executors": sum(item["executors_with_position"] for item in pair_results),
        "combined_fill_events": sum(item["fill_events"] for item in pair_results),
        "combined_take_profits": sum(item["close_types"].get("TAKE_PROFIT", 0) for item in pair_results),
        "combined_time_limits": sum(item["close_types"].get("TIME_LIMIT", 0) for item in pair_results),
        "worst_pair_drawdown_pct": min(item["max_drawdown_pct_mtm"] for item in pair_results),
        "pairs": pair_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure-Python live DCA time-limit comparison")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end", default="2026-07-27T02:00:00Z")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/backtests/dca_live_time_limits_180d"),
    )
    args = parser.parse_args()
    if args.days != 180:
        raise ValueError("comparison is fixed to the requested 180-day window")

    end = floor_five_minutes(datetime.fromisoformat(args.end.replace("Z", "+00:00")))
    start = end - timedelta(days=args.days)
    frames: dict[str, pd.DataFrame] = {
        pair: load_window(args.cache_dir / f"{symbol}_5m.csv", start, end)
        for pair, symbol in PAIRS.items()
    }
    for pair, frame in frames.items():
        print(f"{pair}: validated {len(frame)} contiguous candles", flush=True)

    rows: list[dict] = []
    for time_limit_seconds in TIME_LIMITS:
        for fee_rate in FEE_RATES:
            pair_results = []
            for pair, frame in frames.items():
                result, records = run_pair(
                    frame, pair, fee_rate, SPREADS, REFRESH_SECONDS, time_limit_seconds
                )
                pair_results.append(result)
                del records
                gc.collect()
            row = aggregate(time_limit_seconds, fee_rate, pair_results)
            rows.append(row)
            print(
                f"limit={time_limit_seconds / 3600:g}h fee={fee_rate:.3%} "
                f"pnl={row['combined_net_pnl_quote']:+.4f} "
                f"TP={row['combined_take_profits']} time_limit={row['combined_time_limits']}",
                flush=True,
            )

    output = args.output_dir / end.strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": args.days},
        "fixed_parameters": {
            "pairs": list(PAIRS),
            "budget_per_pair": 190.0,
            "spreads": SPREADS,
            "weights": (0.10, 0.20, 0.30, 0.40),
            "executor_refresh_seconds": REFRESH_SECONDS,
            "take_profit": 0.02,
            "stop_loss": 0.05,
        },
        "time_limits_seconds": TIME_LIMITS,
        "results": rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Live DCA Time-Limit Comparison - 180 Days",
        "",
        f"- Window (UTC): `{start.isoformat()}` to `{end.isoformat()}`",
        "- Fixed: BTC/ETH, 190 USDT per pair, 1/2/4/8% levels, 5-minute unfilled refresh, 2% TP, 5% SL",
        "- Local deterministic 5-minute OHLC simulation; no Docker or order submission",
        "",
        "| Limit | Fee | Net PnL | Return / 380 | Worst pair DD | Filled executors | TP exits | Time-limit exits |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = "45m" if row["time_limit_seconds"] == 2700 else f"{row['time_limit_hours']:g}h"
        lines.append(
            f"| {label} | {row['fee_rate']:.3%} | {row['combined_net_pnl_quote']:+.4f} | "
            f"{row['return_on_380']:+.2%} | {row['worst_pair_drawdown_pct']:.2%} | "
            f"{row['combined_positioned_executors']} | {row['combined_take_profits']} | "
            f"{row['combined_time_limits']} |"
        )
    lines.extend([
        "",
        "## Caveats",
        "",
        "- Entries are considered filled when a 5-minute candle touches the limit price; queue position is not modeled.",
        "- Barriers are evaluated at the 5-minute close after entry fills, with market-exit fees and no slippage model.",
        "- The time limit is measured from executor creation, matching the controller configuration.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "docker_used": False, "orders_submitted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
