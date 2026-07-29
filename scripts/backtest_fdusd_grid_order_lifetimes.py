#!/usr/bin/env python3
"""Compare FDUSD Grid maker-order lifetimes without changing live settings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fdusd_live_grid_optimizer import rolling_validation_windows
from validate_grid_live import (
    Candidate,
    crash_candles,
    read_cache,
    simulate,
    slice_window,
    technical_buy_gate_timeline,
)


LIFETIMES = {
    "1m-baseline": 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "3h": 3 * 60 * 60,
    "4h": 4 * 60 * 60,
    "5h": 5 * 60 * 60,
    "6h": 6 * 60 * 60,
}


def load_candidate(path: Path) -> Candidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload["parameters"]
    return Candidate(
        half_range=float(params["half_range"]),
        min_spread=float(params["minimum_spread"]),
        take_profit=float(params["take_profit"]),
        move_threshold=float(params["move_threshold"]),
        move_cooldown_seconds=int(params["min_grid_move_seconds"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/backtests/grid_live_fdusd_400_walk_forward/active_selection.json"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("results/backtests/grid_live_fdusd_400_walk_forward/validation_result.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/backtests/grid_live_fdusd_order_lifetimes"),
    )
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument("--taker-fee", type=float, default=0.001)
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    start_ts = int(validation["period"]["start_ts"])
    end_ts = int(validation["period"]["end_ts"])
    candidate = load_candidate(args.selection)
    candles = {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        frame = read_cache(args.cache_dir / f"binance_{pair}_5m.csv")
        frame = frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)].reset_index(drop=True)
        if len(frame) < 1:
            raise RuntimeError(f"No cached candles for {pair} in requested period")
        candles[pair] = frame

    windows = rolling_validation_windows(start_ts, end_ts)
    technical_gate = technical_buy_gate_timeline(candles["BTC-FDUSD"])
    stress_candles = slice_window(candles, end_ts - 7 * 86400, end_ts)
    crashed_stress_candles = crash_candles(stress_candles)
    crashed_history = {}
    stress_start = end_ts - 7 * 86400
    for pair, frame in candles.items():
        crashed_history[pair] = pd.concat(
            [frame[frame.timestamp < stress_start], crashed_stress_candles[pair]],
            ignore_index=True,
        )
    crash_technical_gate = technical_buy_gate_timeline(crashed_history["BTC-FDUSD"])
    rows, weekly_rows = [], []
    for label, seconds in LIFETIMES.items():
        full, _, full_pairs = simulate(
            candles,
            candidate,
            args.maker_fee,
            taker_fee=args.taker_fee,
            order_refresh_seconds=seconds,
            technical_buy_gate=technical_gate,
        )
        folds = []
        pair_stops = 0
        for fold, (_, _, test_start, test_end) in enumerate(windows, 1):
            testing = slice_window(candles, test_start, test_end)
            result, _, pair_stats = simulate(
                testing,
                candidate,
                args.maker_fee,
                taker_fee=args.taker_fee,
                order_refresh_seconds=seconds,
                technical_buy_gate=technical_gate,
            )
            stopped_pairs = sum(int(value["liquidations"] > 0) for value in pair_stats.values())
            pair_stops += stopped_pairs
            folds.append(result)
            weekly_rows.append({
                "lifetime": label,
                "order_refresh_seconds": seconds,
                "fold": fold,
                "test_start": test_start,
                "test_end": test_end,
                "pair_stops": stopped_pairs,
                **result,
            })

        base_stress, _, base_pairs = simulate(
            stress_candles,
            candidate,
            args.maker_fee,
            taker_fee=args.taker_fee,
            order_refresh_seconds=seconds,
            technical_buy_gate=technical_gate,
        )
        crash_stress, _, crash_pairs = simulate(
            crashed_stress_candles,
            candidate,
            args.maker_fee,
            taker_fee=args.taker_fee,
            slippage=0.001,
            order_refresh_seconds=seconds,
            technical_buy_gate=crash_technical_gate,
        )
        oos_pnl = sum(float(value["net_pnl_quote"]) for value in folds)
        worst_oos_drawdown = min(float(value["max_drawdown_pct"]) for value in folds)
        oos_portfolio_stops = sum(int(value["liquidated"]) for value in folds)
        base_pair_stop = any(value["liquidations"] for value in base_pairs.values())
        crash_pair_stop = any(value["liquidations"] for value in crash_pairs.values())
        qualified = (
            oos_pnl > 0
            and worst_oos_drawdown >= -0.06
            and oos_portfolio_stops == 0
            and pair_stops == 0
            and not base_stress["liquidated"]
            and not base_pair_stop
            and not crash_stress["liquidated"]
            and not crash_pair_stop
        )
        rows.append({
            "lifetime": label,
            "order_refresh_seconds": seconds,
            "qualified": qualified,
            "oos_pnl_fdusd": oos_pnl,
            "worst_oos_drawdown_pct": worst_oos_drawdown,
            "oos_trades": sum(int(value["trades"]) for value in folds),
            "oos_portfolio_stops": oos_portfolio_stops,
            "oos_pair_stops": pair_stops,
            "full_period_pnl_fdusd": full["net_pnl_quote"],
            "full_period_drawdown_pct": full["max_drawdown_pct"],
            "full_period_trades": full["trades"],
            "full_period_portfolio_stop": full["liquidated"],
            "full_period_pair_stops": sum(
                int(value["liquidations"] > 0) for value in full_pairs.values()
            ),
            "base_stress_pnl_fdusd": base_stress["net_pnl_quote"],
            "base_stress_drawdown_pct": base_stress["max_drawdown_pct"],
            "crash_stress_pnl_fdusd": crash_stress["net_pnl_quote"],
            "crash_stress_drawdown_pct": crash_stress["max_drawdown_pct"],
            "crash_stress_portfolio_stop": crash_stress["liquidated"],
            "crash_stress_pair_stop": crash_pair_stop,
        })

    summary = pd.DataFrame(rows).sort_values("order_refresh_seconds")
    qualified = summary[summary.qualified]
    recommended = None
    if not qualified.empty:
        recommended = str(
            qualified.sort_values(
                ["oos_pnl_fdusd", "worst_oos_drawdown_pct"], ascending=False
            ).iloc[0].lifetime
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(weekly_rows).to_csv(args.output_dir / "weekly_oos.csv", index=False)
    payload = {
        "period": {"start_ts": start_ts, "end_ts": end_ts, "days": validation["period"]["days"]},
        "pairs": ["BTC-FDUSD", "ETH-FDUSD"],
        "candidate": {**asdict(candidate), "levels": candidate.levels},
        "maker_fee": args.maker_fee,
        "taker_fee": args.taker_fee,
        "technical_buy_gate": (
            "ROC48 <= -8% and SQZMOM <= -3% disables BUY; first maroon "
            "SQZMOM bar while active restores BUY; SELL preserved"
        ),
        "lifetimes_seconds": LIFETIMES,
        "recommended_lifetime": recommended,
        "deployment_allowed": False,
        "trading_settings_changed": False,
        "results": summary.to_dict(orient="records"),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_rows = [
        "# FDUSD Grid 挂单时间对比",
        "",
        f"- 样本：{validation['period']['days']} 天，BTC-FDUSD + ETH-FDUSD，5分钟K线",
        f"- Maker费率：{args.maker_fee:.4%}；Taker费率：{args.taker_fee:.4%}",
        f"- 推荐：**{recommended or '无合格时长，维持现有实盘参数不变'}**",
        "- 本次仅回测，未修改实盘配置。",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report_rows), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
