#!/usr/bin/env python3
"""360-day ablation for the live DCA SELL trend/recovery protection.

The run is offline-only. It compares the legacy symmetric DCA lifecycle with
the new SELL-only strong-uptrend gate and 30m/2h/6h post-stop recovery holds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest_dca_live_local import (
    BAR_SECONDS,
    SPREADS,
    STOP_LOSS,
    TAKE_PROFIT,
    TOTAL_BUDGET,
    Executor,
    Fill,
    close_executor,
    create_executor,
    load_window,
    mark_to_market,
)


PAIRS = {"BTC-USDT": "BTCUSDT", "ETH-USDT": "ETHUSDT"}
REFRESH_SECONDS = 18000
TIME_LIMIT_SECONDS = 18000
COOLDOWNS = (1800, 7200, 21600)


def trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].astype(float)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=48, adjust=False).mean()
    result = pd.DataFrame(index=frame.index)
    result["roc"] = close / close.shift(12) - 1
    result["ema_gap"] = fast / slow - 1
    result["trigger"] = (
        (close > slow)
        & (result["ema_gap"] >= 0.002)
        & (result["roc"] >= 0.006)
    ).fillna(False)
    result["recovery"] = ((fast <= slow) | (result["roc"] <= 0.002)).fillna(False)
    return result


def run_pair(
    frame: pd.DataFrame,
    pair: str,
    *,
    fee_rate: float,
    trend_gate: bool,
    sell_stop_cooldown_seconds: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    features = trend_features(frame)
    active: dict[str, Executor | None] = {"BUY": None, "SELL": None}
    next_id = 1
    realized = 0.0
    records: list[dict] = []
    curve_rows: list[dict] = []
    counters = Counter()
    sell_blocked = False
    sell_block_reason = "not_triggered"
    sell_cooldown_until = 0
    recovery_count = 0

    def ensure(timestamp: int, reference: float) -> None:
        nonlocal next_id
        for side in ("BUY", "SELL"):
            if active[side] is not None:
                continue
            if side == "SELL" and trend_gate and sell_blocked:
                continue
            active[side] = create_executor(next_id, side, timestamp, reference, SPREADS)
            next_id += 1

    first = frame.iloc[0]
    ensure(int(first.timestamp), float(first.close))
    peak = TOTAL_BUDGET

    for offset, row in enumerate(frame.iloc[1:].itertuples(index=False), start=1):
        timestamp = int(row.timestamp)
        close = float(row.close)
        high = float(row.high)
        low = float(row.low)

        if trend_gate:
            if bool(features.trigger.iloc[offset]):
                if not sell_blocked:
                    counters["trend_blocks"] += 1
                sell_blocked = True
                sell_block_reason = "strong_uptrend"
                recovery_count = 0
            elif sell_blocked:
                recovery_count = recovery_count + 1 if bool(features.recovery.iloc[offset]) else 0
                if recovery_count >= 3 and timestamp >= sell_cooldown_until:
                    sell_blocked = False
                    sell_block_reason = "trend_recovered"
                    recovery_count = 0
                    counters["trend_recoveries"] += 1

        for side in ("BUY", "SELL"):
            executor = active[side]
            if executor is None:
                continue
            for level, (price, quote) in enumerate(zip(executor.prices, executor.amounts_quote)):
                if level in executor.filled_levels:
                    continue
                touched = low <= price if side == "BUY" else high >= price
                if touched:
                    executor.fills.append(Fill(level, price, quote, timestamp))
                    executor.filled_levels.add(level)

            close_type = None
            if executor.has_position:
                net_pnl_pct = (
                    executor.gross_pnl(close) - executor.open_quote * fee_rate
                ) / executor.open_quote
                age_from_fill = timestamp - executor.fills[0].timestamp
                if net_pnl_pct <= -STOP_LOSS:
                    close_type = "STOP_LOSS"
                elif net_pnl_pct > TAKE_PROFIT:
                    close_type = "TAKE_PROFIT"
                elif age_from_fill >= TIME_LIMIT_SECONDS:
                    close_type = "TIME_LIMIT"
            elif timestamp - executor.created_at >= REFRESH_SECONDS:
                close_type = "REFRESH"

            if close_type:
                record = close_executor(executor, timestamp, close, fee_rate, close_type)
                record.update({"pair": pair, "trend_gate": trend_gate,
                               "sell_stop_cooldown_seconds": sell_stop_cooldown_seconds})
                records.append(record)
                realized += record["net_pnl"]
                active[side] = None
                counters[f"{side}_{close_type}"] += 1
                if side == "SELL" and close_type == "STOP_LOSS" and trend_gate:
                    sell_blocked = True
                    sell_block_reason = "sell_stop_loss_recovery"
                    recovery_count = 0
                    sell_cooldown_until = max(
                        sell_cooldown_until,
                        timestamp + sell_stop_cooldown_seconds,
                    )

        ensure(timestamp, close)
        unrealized = sum(
            mark_to_market(executor, close, fee_rate)
            for executor in active.values()
            if executor is not None
        )
        equity = TOTAL_BUDGET + realized + unrealized
        peak = max(peak, equity)
        if trend_gate and sell_blocked:
            counters["sell_blocked_bars"] += 1
        curve_rows.append({
            "timestamp": timestamp,
            "equity": equity,
            "drawdown_pct": (equity / peak - 1) * 100,
            "sell_blocked": sell_blocked,
            "sell_block_reason": sell_block_reason,
        })

    final = frame.iloc[-1]
    for side, executor in active.items():
        if executor is None:
            continue
        record = close_executor(
            executor, int(final.timestamp), float(final.close), fee_rate, "END_OF_BACKTEST"
        )
        record.update({"pair": pair, "trend_gate": trend_gate,
                       "sell_stop_cooldown_seconds": sell_stop_cooldown_seconds})
        records.append(record)
        realized += record["net_pnl"]

    trades = pd.DataFrame(records)
    curve = pd.DataFrame(curve_rows)
    positioned = trades[trades.levels_filled > 0] if not trades.empty else trades
    sell_stops = positioned[(positioned.side == "SELL") & (positioned.close_type == "STOP_LOSS")]
    summary = {
        "pair": pair,
        "trend_gate": trend_gate,
        "sell_stop_cooldown_seconds": sell_stop_cooldown_seconds,
        "net_pnl_quote": realized,
        "return_pct": realized / TOTAL_BUDGET * 100,
        "max_drawdown_pct": float(curve.drawdown_pct.min()),
        "positioned_executors": int(len(positioned)),
        "sell_stop_losses": int(len(sell_stops)),
        "sell_stop_loss_pnl": float(sell_stops.net_pnl.sum()) if len(sell_stops) else 0.0,
        "fees_quote": float(positioned.fees.sum()) if len(positioned) else 0.0,
        "sell_blocked_hours": counters["sell_blocked_bars"] * BAR_SECONDS / 3600,
        "trend_blocks": int(counters["trend_blocks"]),
        "trend_recoveries": int(counters["trend_recoveries"]),
    }
    return summary, trades, curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default="2026-08-20T00:00:00Z")
    parser.add_argument("--days", type=int, default=360)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles/dca_sell_gate_360d"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/backtests/dca_sell_trend_recovery_360d"))
    parser.add_argument("--fee-rate", type=float, default=0.001)
    args = parser.parse_args()
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    start = end - timedelta(days=args.days)
    frames = {
        pair: load_window(args.cache_dir / f"{symbol}_5m.csv", start, end)
        for pair, symbol in PAIRS.items()
    }
    scenarios = [("legacy_15s", False, 0)] + [
        (f"sell_trend_{seconds // 3600 if seconds >= 3600 else 0.5:g}h", True, seconds)
        for seconds in COOLDOWNS
    ]
    summaries, trades, curves = [], [], []
    for scenario, enabled, cooldown in scenarios:
        for pair, frame in frames.items():
            summary, pair_trades, curve = run_pair(
                frame, pair, fee_rate=args.fee_rate, trend_gate=enabled,
                sell_stop_cooldown_seconds=cooldown,
            )
            summary["scenario"] = scenario
            pair_trades.insert(0, "scenario", scenario)
            curve.insert(0, "pair", pair)
            curve.insert(0, "scenario", scenario)
            summaries.append(summary)
            trades.append(pair_trades)
            curves.append(curve)
    summary_frame = pd.DataFrame(summaries)
    aggregate = summary_frame.groupby("scenario", sort=False).agg(
        net_pnl_quote=("net_pnl_quote", "sum"),
        worst_pair_drawdown_pct=("max_drawdown_pct", "min"),
        positioned_executors=("positioned_executors", "sum"),
        sell_stop_losses=("sell_stop_losses", "sum"),
        sell_stop_loss_pnl=("sell_stop_loss_pnl", "sum"),
        fees_quote=("fees_quote", "sum"),
        sell_blocked_hours=("sell_blocked_hours", "sum"),
    ).reset_index()
    aggregate["return_on_380_pct"] = aggregate.net_pnl_quote / (2 * TOTAL_BUDGET) * 100
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(output / "pair_summary.csv", index=False)
    aggregate.to_csv(output / "aggregate_summary.csv", index=False)
    pd.concat(trades, ignore_index=True).to_csv(output / "executors.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output / "equity.csv", index=False)
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": args.days},
        "execution": {
            "fee_asset": "USDT", "fee_rate": args.fee_rate,
            "entry": "maker level touched by 5m high/low",
            "exit": "5m close; stop protects partial fills",
            "sell_trend_trigger": "EMA12>EMA48, EMA gap>=0.2%, ROC(12)>=0.6%",
            "sell_trend_recovery": "3 completed 5m bars with EMA12<=EMA48 or ROC(12)<=0.2%, after cooldown",
        },
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(aggregate.round(4).to_string(index=False))
    print(json.dumps({"output": str(output.resolve()), "orders_submitted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
