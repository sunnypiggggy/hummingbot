"""Pure-Pandas replay helpers for the portfolio moving-grid reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd


TOP10_PAIRS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT", "LINK-USDT", "AVAX-USDT", "TRX-USDT"]


@dataclass
class GridState:
    quote: float
    base: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    levels: list[float] = field(default_factory=list)
    lots: list[dict[str, float]] = field(default_factory=list)
    active_buy_levels: set[float] = field(default_factory=set)
    last_move_ts: float = 0.0


def levels(center: float, grid_range: float, grid_levels: int) -> tuple[float, float, list[float]]:
    lower = center * (1 - grid_range / 2)
    upper = center * (1 + grid_range / 2)
    if grid_levels <= 1:
        return lower, upper, [center]
    step = (upper - lower) / (grid_levels - 1)
    return lower, upper, [lower + step * index for index in range(grid_levels)]


def read_candles(cache_dir: Path, pair: str, start: float, end: float) -> pd.DataFrame:
    path = cache_dir / f"binance_{pair}_5m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candle cache: {path}")
    candles = pd.read_csv(path)
    candles = candles[(candles["timestamp"] >= start) & (candles["timestamp"] <= end)].copy()
    if candles.empty:
        raise RuntimeError(f"No candles for {pair} in the requested window.")
    return candles.reset_index(drop=True)


def _new_state(quote: float, price: float, timestamp: float, params: dict[str, float]) -> GridState:
    lower, upper, grid = levels(price, params["grid_range"], int(params["grid_levels"]))
    return GridState(quote=quote, lower=lower, upper=upper, levels=grid, last_move_ts=timestamp)


def _equity(states: Dict[str, GridState], prices: dict[str, float]) -> float:
    return sum(state.quote + state.base * prices[pair] for pair, state in states.items())


def _empty_stats() -> dict[str, object]:
    return {"buy_entries": 0, "take_profit_sells": 0, "liquidation_sells": 0, "completed_cycles": 0, "grid_moves": 0, "liquidations": 0, "liquidation_times": [], "trades": 0}


def replay_portfolio(candles_by_pair: dict[str, pd.DataFrame], params: dict[str, float], initial_quote: float,
                     fee_rate: float, portfolio_stop_loss: float, cooldown_hours: float,
                     min_grid_move_seconds: float, states: Dict[str, GridState] | None = None,
                     peak_equity: float | None = None, cooldown_until: float = -1.0,
                     regrid_at_start: bool = False, dynamic_allocation: bool = True,
                     terminate_on_liquidation: bool = False) -> tuple[dict[str, object], Dict[str, GridState], float, float]:
    pairs = list(candles_by_pair)
    steps = min(len(candles) for candles in candles_by_pair.values())
    if steps < 2:
        raise RuntimeError("At least two aligned candles are required.")
    # Avoid DataFrame scalar access inside the multi-month replay loop.
    arrays = {
        pair: candles_by_pair[pair].loc[:steps - 1, ["timestamp", "close", "low", "high"]].to_numpy(dtype=float)
        for pair in pairs
    }
    if states is None:
        allocation = initial_quote / len(pairs)
        states = {pair: _new_state(allocation, arrays[pair][0, 1], arrays[pair][0, 0], params) for pair in pairs}
    if regrid_at_start:
        for pair, state in states.items():
            price = arrays[pair][0, 1]
            state.lower, state.upper, state.levels = levels(price, params["grid_range"], int(params["grid_levels"]))
            state.active_buy_levels.clear()

    stats = _empty_stats()
    peak_equity = peak_equity if peak_equity is not None else initial_quote
    step_seconds = arrays[pairs[0]][1, 0] - arrays[pairs[0]][0, 0]
    cooldown_seconds = max(cooldown_hours * 3600, step_seconds)
    initial_allocation = initial_quote / len(pairs)
    terminated = False

    for index in range(steps):
        timestamp = arrays[pairs[0]][index, 0]
        prices = {pair: arrays[pair][index, 1] for pair in pairs}
        lows = {pair: arrays[pair][index, 2] for pair in pairs}
        highs = {pair: arrays[pair][index, 3] for pair in pairs}
        if timestamp >= cooldown_until and not terminated:
            allocation = max(_equity(states, prices) / len(pairs), 0.0) if dynamic_allocation else initial_allocation
            for pair, state in states.items():
                price = prices[pair]
                trigger = price > state.upper * (1 + params["move_threshold"]) or price < state.lower * (1 - params["move_threshold"])
                if trigger and timestamp - state.last_move_ts >= min_grid_move_seconds:
                    state.lower, state.upper, state.levels = levels(price, params["grid_range"], int(params["grid_levels"]))
                    state.active_buy_levels.clear()
                    state.last_move_ts = timestamp
                    stats["grid_moves"] += 1

                order_quote = allocation * params["order_quote_pct"]
                for level in state.levels:
                    if level >= price or level in state.active_buy_levels or lows[pair] > level or state.quote < order_quote or order_quote <= 0:
                        continue
                    base = order_quote * (1 - fee_rate) / level
                    state.quote -= order_quote
                    state.base += base
                    state.lots.append({"entry": level, "base": base, "tp": level * (1 + params["take_profit"])})
                    state.active_buy_levels.add(level)
                    stats["buy_entries"] += 1
                    stats["trades"] += 1

                remaining = []
                for lot in state.lots:
                    if highs[pair] >= lot["tp"]:
                        state.quote += lot["base"] * lot["tp"] * (1 - fee_rate)
                        state.base -= lot["base"]
                        state.active_buy_levels.discard(lot["entry"])
                        stats["take_profit_sells"] += 1
                        stats["completed_cycles"] += 1
                        stats["trades"] += 1
                    else:
                        remaining.append(lot)
                state.lots = remaining

        equity = _equity(states, prices)
        peak_equity = max(peak_equity, equity)
        drawdown = (equity - peak_equity) / peak_equity if peak_equity else 0.0
        if drawdown <= -portfolio_stop_loss:
            sells = 0
            for pair, state in states.items():
                if state.base > 0:
                    state.quote += state.base * prices[pair] * (1 - fee_rate)
                    state.base = 0.0
                    state.lots.clear()
                    state.active_buy_levels.clear()
                    sells += 1
            stats["liquidation_sells"] += sells
            stats["trades"] += sells
            stats["liquidations"] += 1
            stats["liquidation_times"].append(timestamp)
            cooldown_until = timestamp + cooldown_seconds
            peak_equity = _equity(states, prices)
            if terminate_on_liquidation:
                terminated = True

    return stats, states, peak_equity, cooldown_until


def replay_fixed(cache_dir: Path, params: dict[str, float], end_ts: float = 1783566900, days: int = 182) -> dict[str, object]:
    start_ts = end_ts - days * 24 * 3600
    candles = {pair: read_candles(cache_dir, pair, start_ts, end_ts) for pair in TOP10_PAIRS}
    stats, _, _, _ = replay_portfolio(
        candles, params, 10000.0, 0.0002, params["portfolio_stop_loss"], 24.0, params["min_grid_move_seconds"],
        dynamic_allocation=False, terminate_on_liquidation=True,
    )
    stats["source_window"] = f"{pd.to_datetime(start_ts, unit='s', utc=True):%Y-%m-%d} to {pd.to_datetime(end_ts, unit='s', utc=True):%Y-%m-%d}"
    return stats


def replay_walk_forward(cache_dir: Path, summary_path: Path) -> dict[str, object]:
    summary = pd.read_csv(summary_path)
    states: Dict[str, GridState] | None = None
    peak, cooldown_until = 10000.0, -1.0
    aggregate = _empty_stats()
    weekly: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        start = pd.Timestamp(row.trade_start, tz="UTC").timestamp()
        end = pd.Timestamp(row.trade_end, tz="UTC").timestamp()
        params = {"grid_range": float(row.grid_range), "grid_levels": int(row.grid_levels), "order_quote_pct": float(row.order_quote_pct), "take_profit": float(row.take_profit), "move_threshold": float(row.move_threshold), "portfolio_stop_loss": float(row.selected_portfolio_stop_loss), "min_grid_move_seconds": float(row.min_grid_move_seconds)}
        candles = {pair: read_candles(cache_dir, pair, start, end) for pair in TOP10_PAIRS}
        stats, states, peak, cooldown_until = replay_portfolio(candles, params, 10000.0, 0.0002, params["portfolio_stop_loss"], 24.0, params["min_grid_move_seconds"], states, peak, cooldown_until, regrid_at_start=states is not None)
        for key in ("buy_entries", "take_profit_sells", "liquidation_sells", "completed_cycles", "grid_moves", "liquidations", "trades"):
            aggregate[key] += stats[key]
        aggregate["liquidation_times"].extend(stats["liquidation_times"])
        weekly.append({"week_index": int(row.week_index), "pnl_pct": float(row.net_pnl_pct), "drawdown_pct": float(row.max_drawdown_pct), "grid_range": params["grid_range"], "grid_levels": params["grid_levels"], "move_threshold": params["move_threshold"], "buy_entries": stats["buy_entries"], "grid_moves": stats["grid_moves"]})
    aggregate["weekly"] = weekly
    return aggregate
