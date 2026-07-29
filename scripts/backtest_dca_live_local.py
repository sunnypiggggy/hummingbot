#!/usr/bin/env python3
"""Pure-Python backtest for the current BTC/ETH live DCA configuration.

The simulator intentionally mirrors the live controller's important lifecycle:

* one BUY and one SELL executor per pair, each with 95 USDT side budget;
* maker levels at 1%, 2%, 4%, and 8%, weighted 10/20/30/40;
* an executor with no fills is refreshed after the configured refresh interval;
* after the first fill it remains active until TP, SL, or the configured time limit;
* all exits are valued as taker/market exits.

This script does not connect to a trading account and cannot place orders.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from cache_binance_klines import update_cache


PAIRS = {"BTC-USDT": "BTCUSDT", "ETH-USDT": "ETHUSDT"}
TOTAL_BUDGET = 190.0
SIDE_BUDGET = TOTAL_BUDGET / 2
SPREADS = (0.01, 0.02, 0.04, 0.08)
WEIGHTS = (0.10, 0.20, 0.30, 0.40)
REFRESH_SECONDS = 300
TIME_LIMIT_SECONDS = 2700
TAKE_PROFIT = 0.02
STOP_LOSS = 0.05
BAR_SECONDS = 300


@dataclass
class Fill:
    level: int
    price: float
    quote: float
    timestamp: int

    @property
    def base(self) -> float:
        return self.quote / self.price


@dataclass
class Executor:
    executor_id: int
    side: str
    created_at: int
    reference_price: float
    prices: tuple[float, ...]
    amounts_quote: tuple[float, ...]
    fills: list[Fill] = field(default_factory=list)
    filled_levels: set[int] = field(default_factory=set)

    @property
    def has_position(self) -> bool:
        return bool(self.fills)

    @property
    def all_levels_filled(self) -> bool:
        return len(self.filled_levels) == len(self.prices)

    @property
    def open_quote(self) -> float:
        return sum(fill.quote for fill in self.fills)

    @property
    def open_base(self) -> float:
        return sum(fill.base for fill in self.fills)

    @property
    def average_price(self) -> float:
        return self.open_quote / self.open_base if self.open_base else 0.0

    def gross_pnl(self, mark: float) -> float:
        direction = 1.0 if self.side == "BUY" else -1.0
        return direction * self.open_base * (mark - self.average_price)


def utc_iso(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat()


def floor_five_minutes(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return value - timedelta(minutes=value.minute % 5)


def load_window(path: Path, start: datetime, end: datetime) -> pd.DataFrame:
    frame = pd.read_csv(path)
    timestamp = pd.to_numeric(frame["timestamp"], errors="raise")
    if timestamp.max() > 10_000_000_000:
        timestamp = timestamp // 1000
    frame["timestamp"] = timestamp.astype("int64")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    start_epoch = int(start.timestamp())
    end_epoch = int(end.timestamp())
    frame = frame[(frame["timestamp"] >= start_epoch) & (frame["timestamp"] < end_epoch)]
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    expected = int((end - start).total_seconds() // BAR_SECONDS)
    if len(frame) != expected:
        raise ValueError(f"{path} has {len(frame)} rows in the requested window; expected {expected}")
    gaps = frame["timestamp"].diff().dropna()
    if not gaps.eq(BAR_SECONDS).all():
        samples = frame.loc[gaps[gaps.ne(BAR_SECONDS)].index[:5], "timestamp"].tolist()
        raise ValueError(f"{path} contains non-5m gaps near {samples}")
    return frame


def create_executor(executor_id: int, side: str, timestamp: int, reference: float,
                    spreads: tuple[float, ...] = SPREADS) -> Executor:
    multiplier = -1.0 if side == "BUY" else 1.0
    prices = tuple(reference * (1 + multiplier * spread) for spread in spreads)
    amounts = tuple(SIDE_BUDGET * weight for weight in WEIGHTS)
    return Executor(executor_id, side, timestamp, reference, prices, amounts)


def mark_to_market(executor: Executor, mark: float, fee_rate: float) -> float:
    if not executor.has_position:
        return 0.0
    entry_fees = executor.open_quote * fee_rate
    return executor.gross_pnl(mark) - entry_fees


def close_executor(executor: Executor, timestamp: int, mark: float, fee_rate: float,
                   close_type: str) -> dict:
    entry_fees = executor.open_quote * fee_rate
    close_notional = executor.open_base * mark
    exit_fees = close_notional * fee_rate
    gross = executor.gross_pnl(mark)
    return {
        "executor_id": executor.executor_id,
        "side": executor.side,
        "created_at": utc_iso(executor.created_at),
        "closed_at": utc_iso(timestamp),
        "first_fill_at": utc_iso(executor.fills[0].timestamp) if executor.fills else None,
        "last_fill_at": utc_iso(executor.fills[-1].timestamp) if executor.fills else None,
        "lifetime_minutes": (timestamp - executor.created_at) / 60,
        "reference_price": executor.reference_price,
        "average_entry_price": executor.average_price if executor.has_position else None,
        "close_price": mark if executor.has_position else None,
        "levels_filled": len(executor.fills),
        "filled_level_ids": ",".join(str(fill.level + 1) for fill in executor.fills),
        "open_quote": executor.open_quote,
        "close_notional": close_notional,
        "gross_pnl": gross,
        "fees": entry_fees + exit_fees,
        "net_pnl": gross - entry_fees - exit_fees,
        "close_type": close_type,
    }


def run_pair(frame: pd.DataFrame, pair: str, fee_rate: float,
             spreads: tuple[float, ...] = SPREADS,
             refresh_seconds: int = REFRESH_SECONDS,
             time_limit_seconds: int = TIME_LIMIT_SECONDS) -> tuple[dict, list[dict]]:
    if len(spreads) != len(WEIGHTS) or tuple(sorted(spreads)) != spreads:
        raise ValueError("spreads must contain four ascending values")
    if time_limit_seconds <= 0 or time_limit_seconds % BAR_SECONDS:
        raise ValueError("time_limit_seconds must be a positive multiple of 300")
    if refresh_seconds <= 0 or refresh_seconds > time_limit_seconds:
        raise ValueError("refresh_seconds must be between 1 and the time limit")
    active: dict[str, Executor | None] = {"BUY": None, "SELL": None}
    next_id = 1
    records: list[dict] = []
    realized_pnl = 0.0
    peak_equity = TOTAL_BUDGET
    max_drawdown = 0.0
    layer_fills = {"BUY": Counter(), "SELL": Counter()}

    def ensure_executors(timestamp: int, reference: float) -> None:
        nonlocal next_id
        for side in ("BUY", "SELL"):
            if active[side] is None:
                active[side] = create_executor(next_id, side, timestamp, reference, spreads)
                next_id += 1

    first = frame.iloc[0]
    ensure_executors(int(first["timestamp"]), float(first["close"]))

    for row in frame.iloc[1:].itertuples(index=False):
        timestamp = int(row.timestamp)
        high = float(row.high)
        low = float(row.low)
        close = float(row.close)
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
                    layer_fills[side][level + 1] += 1

            age = timestamp - executor.created_at
            close_type = None
            if executor.has_position:
                entry_fees = executor.open_quote * fee_rate
                net_pnl_pct = (executor.gross_pnl(close) - entry_fees) / executor.open_quote
                if executor.all_levels_filled and net_pnl_pct <= -STOP_LOSS:
                    close_type = "STOP_LOSS"
                elif net_pnl_pct > TAKE_PROFIT:
                    close_type = "TAKE_PROFIT"
                elif age >= time_limit_seconds:
                    close_type = "TIME_LIMIT"
            elif age >= refresh_seconds:
                close_type = "REFRESH"

            if close_type:
                record = close_executor(executor, timestamp, close, fee_rate, close_type)
                records.append(record)
                realized_pnl += record["net_pnl"]
                active[side] = None

        ensure_executors(timestamp, close)
        unrealized = sum(
            mark_to_market(executor, close, fee_rate)
            for executor in active.values()
            if executor is not None
        )
        equity = TOTAL_BUDGET + realized_pnl + unrealized
        peak_equity = max(peak_equity, equity)
        max_drawdown = min(max_drawdown, equity / peak_equity - 1)

    final = frame.iloc[-1]
    final_timestamp = int(final["timestamp"])
    final_close = float(final["close"])
    for side in ("BUY", "SELL"):
        executor = active[side]
        if executor is None:
            continue
        close_type = "END_OF_BACKTEST" if executor.has_position else "REFRESH"
        record = close_executor(executor, final_timestamp, final_close, fee_rate, close_type)
        records.append(record)
        realized_pnl += record["net_pnl"]

    positioned = [record for record in records if record["levels_filled"] > 0]
    winning = [record for record in positioned if record["net_pnl"] > 0]
    close_types = Counter(record["close_type"] for record in records)
    total_fees = sum(record["fees"] for record in records)
    gross_pnl = sum(record["gross_pnl"] for record in records)
    turnover = sum(record["open_quote"] + record["close_notional"] for record in records)
    first_fill_epochs = sorted(
        int(datetime.fromisoformat(record["first_fill_at"]).timestamp())
        for record in positioned
        if record["first_fill_at"]
    )
    window_start = int(frame.iloc[0]["timestamp"])
    window_end = int(frame.iloc[-1]["timestamp"])
    fill_gap_points = [window_start, *first_fill_epochs, window_end]
    max_fill_gap_hours = max(
        (right - left) / 3600 for left, right in zip(fill_gap_points, fill_gap_points[1:])
    ) if len(fill_gap_points) > 1 else 0.0
    days = (window_end - window_start + BAR_SECONDS) / 86400
    summary = {
        "pair": pair,
        "fee_rate": fee_rate,
        "spreads": spreads,
        "refresh_seconds": refresh_seconds,
        "time_limit_seconds": time_limit_seconds,
        "initial_strategy_equity": TOTAL_BUDGET,
        "net_pnl_quote": realized_pnl,
        "return_on_190": realized_pnl / TOTAL_BUDGET,
        "gross_pnl_quote": gross_pnl,
        "fees_quote": total_fees,
        "turnover_quote": turnover,
        "max_drawdown_pct_mtm": max_drawdown,
        "executors_total": len(records),
        "executors_with_position": len(positioned),
        "executor_fill_rate": len(positioned) / len(records) if records else 0.0,
        "positioned_executors_per_day": len(positioned) / days,
        "days_with_fills": len({datetime.fromtimestamp(epoch, timezone.utc).date() for epoch in first_fill_epochs}),
        "max_fill_gap_hours": max_fill_gap_hours,
        "profitable_executors": len(winning),
        "win_rate": len(winning) / len(positioned) if positioned else 0.0,
        "fill_events": sum(record["levels_filled"] for record in positioned),
        "layer_fills": {
            side: {str(level): layer_fills[side][level] for level in range(1, 5)}
            for side in ("BUY", "SELL")
        },
        "close_types": dict(sorted(close_types.items())),
    }
    return summary, records


def write_outputs(output: Path, start: datetime, end: datetime, summaries: list[dict],
                  records: Iterable[dict], spreads: tuple[float, ...] = SPREADS,
                  refresh_seconds: int = REFRESH_SECONDS,
                  time_limit_seconds: int = TIME_LIMIT_SECONDS) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = list(records)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat(), "days": 180},
        "data_source": "Binance spot 5m klines",
        "simulation": {
            "budget_per_pair": TOTAL_BUDGET,
            "side_budget": SIDE_BUDGET,
            "spreads": spreads,
            "weights": WEIGHTS,
            "refresh_seconds": refresh_seconds,
            "time_limit_seconds": time_limit_seconds,
            "take_profit": TAKE_PROFIT,
            "stop_loss": STOP_LOSS,
            "barrier_evaluation": "5m close after limit fills; exits at close",
            "intrabar_assumption": "all limit levels touched by the 5m high/low fill before close barriers",
        },
        "results": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if records:
        with (output / "executors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    lines = [
        "# DCA Candidate - 180 Day Local Backtest",
        "",
        f"- Window (UTC): `{start.isoformat()}` to `{end.isoformat()}`",
        "- Source: Binance spot 5-minute klines",
        f"- Spreads: `{', '.join(f'{value:.2%}' for value in spreads)}`",
        f"- Unfilled-order refresh: `{refresh_seconds} seconds`",
        f"- Executor time limit: `{time_limit_seconds} seconds`",
        "- No Docker, Hummingbot API, account credentials, or order submission used",
        "",
        "| Pair | Fee | Net PnL | Return / 190 | MTM max DD | Positions/day | Fill days | Max fill gap | Fill events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['pair']} | {item['fee_rate']:.3%} | {item['net_pnl_quote']:+.4f} | "
            f"{item['return_on_190']:+.2%} | {item['max_drawdown_pct_mtm']:.2%} | "
            f"{item['positioned_executors_per_day']:.2f} | {item['days_with_fills']} | "
            f"{item['max_fill_gap_hours']:.1f}h | {item['fill_events']} |"
        )
    lines.extend([
        "",
        "## Method caveats",
        "",
        "- This reproduces the current controller lifecycle in a deterministic 5-minute OHLC simulator.",
        "- When a candle touches an entry and an exit threshold in the same bar, entries are processed first and barriers at the close.",
        "- Drawdown is marked to the 5-minute close and does not model order-book slippage; fee scenarios provide cost sensitivity.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pure-Python current live DCA backtest")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--fee-rates", default="0.0002,0.001")
    parser.add_argument("--spreads", default=",".join(str(value) for value in SPREADS))
    parser.add_argument("--refresh-seconds", type=int, default=REFRESH_SECONDS)
    parser.add_argument("--time-limit-seconds", type=int, default=TIME_LIMIT_SECONDS)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/backtests/dca_live_local_180d"))
    parser.add_argument("--refresh-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--end", help="Optional fixed UTC end time for reproducible reruns")
    args = parser.parse_args()
    if args.days != 180:
        raise ValueError("this audit run is fixed to the requested 180-day window")

    if args.end:
        parsed_end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        if parsed_end.tzinfo is None:
            parsed_end = parsed_end.replace(tzinfo=timezone.utc)
        end = floor_five_minutes(parsed_end)
    else:
        end = floor_five_minutes(datetime.now(timezone.utc))
    start = end - timedelta(days=args.days)
    if args.refresh_data:
        for symbol in PAIRS.values():
            path = args.cache_dir / f"{symbol}_5m.csv"
            added = update_cache(path, symbol, start, end)
            print(f"{symbol}: added {added} rows", flush=True)

    all_summaries: list[dict] = []
    all_records: list[dict] = []
    spreads = tuple(float(value.strip()) for value in args.spreads.split(",") if value.strip())
    fee_rates = [float(value.strip()) for value in args.fee_rates.split(",") if value.strip()]
    for pair, symbol in PAIRS.items():
        frame = load_window(args.cache_dir / f"{symbol}_5m.csv", start, end)
        print(f"{pair}: validated {len(frame)} contiguous candles", flush=True)
        for fee_rate in fee_rates:
            summary, records = run_pair(
                frame, pair, fee_rate, spreads, args.refresh_seconds, args.time_limit_seconds
            )
            all_summaries.append(summary)
            for record in records:
                all_records.append({"pair": pair, "fee_rate": fee_rate, **record})
            print(
                f"{pair} fee={fee_rate:.4%}: pnl={summary['net_pnl_quote']:+.4f}, "
                f"filled_executors={summary['executors_with_position']}",
                flush=True,
            )

    run_output = args.output_dir / end.strftime("%Y%m%dT%H%M%SZ")
    write_outputs(
        run_output, start, end, all_summaries, all_records, spreads,
        args.refresh_seconds, args.time_limit_seconds,
    )
    print(json.dumps({"output": str(run_output), "docker_used": False, "orders_submitted": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
