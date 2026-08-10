"""Pure-Python grid simulation shared by scheduled selection and backtests."""

import concurrent.futures
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class GridParams:
    grid_range: float
    grid_levels: int
    order_quote_pct: float
    take_profit: float
    move_threshold: float
    stop_loss: float = 0.04


def default_search_space() -> List[GridParams]:
    return [
        GridParams(grid_range, levels, order_pct, profit, move)
        for grid_range in (0.04, 0.06, 0.08, 0.10)
        for levels in (8, 12, 16, 24)
        for order_pct in (0.01, 0.015, 0.02)
        for profit in (0.003, 0.005, 0.008, 0.01)
        for move in (0.005, 0.01, 0.015)
    ]


def make_levels(price: float, params: GridParams) -> Tuple[float, float, List[float]]:
    lower = price * (1 - params.grid_range / 2)
    upper = price * (1 + params.grid_range / 2)
    step = (upper - lower) / max(params.grid_levels - 1, 1)
    return lower, upper, [lower + step * index for index in range(params.grid_levels)]


def simulate_portfolio(candles_by_pair: Dict[str, pd.DataFrame], params: GridParams, initial_quote: float,
                       fee_rate: float, portfolio_stop_loss: float, cooldown_hours: float,
                       min_grid_move_seconds: float = 0, maker_fee_rate: float | None = None,
                       taker_fee_rate: float | None = None) -> Dict[str, float]:
    """A deterministic 5m OHLC simulation used only for parameter ranking."""
    maker_fee = fee_rate if maker_fee_rate is None else maker_fee_rate
    taker_fee = fee_rate if taker_fee_rate is None else taker_fee_rate
    pairs = list(candles_by_pair)
    steps = min(len(frame) for frame in candles_by_pair.values())
    if steps < 2:
        raise ValueError("At least two candles are required for every trading pair.")
    # Pandas iloc in the candidate/time/pair inner loop dominates runtime on
    # small OCI instances. Materialize numeric arrays once per simulation;
    # this preserves the deterministic OHLC rules while avoiding millions of
    # Series allocations.
    market = {
        pair: {
            column: frame[column].to_numpy(dtype=float, copy=False)
            for column in ("timestamp", "high", "low", "close")
        }
        for pair, frame in candles_by_pair.items()
    }
    allocation = initial_quote / len(pairs)
    states = {}
    for pair in pairs:
        lower, upper, levels = make_levels(float(market[pair]["close"][0]), params)
        states[pair] = {"quote": allocation, "base": 0.0, "lower": lower, "upper": upper, "levels": levels,
                        "lots": [], "active": set(), "moves": 0, "trades": 0, "cycles": 0, "last_move": 0.0}
    peak = initial_quote
    max_drawdown = 0.0
    liquidated = False
    cooldown_until = -1.0
    for index in range(steps):
        now = float(market[pairs[0]]["timestamp"][index])
        prices = {pair: float(market[pair]["close"][index]) for pair in pairs}
        if now >= cooldown_until and not liquidated:
            for pair, state in states.items():
                candle_low = float(market[pair]["low"][index])
                candle_high = float(market[pair]["high"][index])
                price = prices[pair]
                if now - state["last_move"] >= min_grid_move_seconds and (
                    price > state["upper"] * (1 + params.move_threshold) or price < state["lower"] * (1 - params.move_threshold)
                ):
                    state["lower"], state["upper"], state["levels"] = make_levels(price, params)
                    state["active"].clear()
                    state["moves"] += 1
                    state["last_move"] = now
                order_quote = allocation * params.order_quote_pct
                for level in state["levels"]:
                    if level >= price or level in state["active"] or candle_low > level or state["quote"] < order_quote:
                        continue
                    base = order_quote * (1 - maker_fee) / level
                    state["quote"] -= order_quote
                    state["base"] += base
                    state["lots"].append((level, base, level * (1 + params.take_profit)))
                    state["active"].add(level)
                    state["trades"] += 1
                open_lots = []
                for entry, base, target in state["lots"]:
                    if candle_high >= target:
                        state["quote"] += base * target * (1 - maker_fee)
                        state["base"] -= base
                        state["active"].discard(entry)
                        state["trades"] += 1
                        state["cycles"] += 1
                    else:
                        open_lots.append((entry, base, target))
                state["lots"] = open_lots
        equity = sum(state["quote"] + state["base"] * prices[pair] for pair, state in states.items())
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak if peak else 0.0
        max_drawdown = min(max_drawdown, drawdown)
        if drawdown <= -portfolio_stop_loss and not liquidated:
            for pair, state in states.items():
                state["quote"] += state["base"] * prices[pair] * (1 - taker_fee)
                state["base"] = 0.0
                state["lots"] = []
                state["active"].clear()
            liquidated = True
            cooldown_until = now + cooldown_hours * 3600
            break
    final_equity = sum(state["quote"] + state["base"] * float(market[pair]["close"][-1])
                       for pair, state in states.items())
    return {"net_pnl_quote": final_equity - initial_quote, "net_pnl_pct": (final_equity - initial_quote) / initial_quote,
            "max_drawdown_pct": max_drawdown, "liquidated": liquidated,
            "trades": sum(state["trades"] for state in states.values()),
            "completed_cycles": sum(state["cycles"] for state in states.values()),
            "grid_moves": sum(state["moves"] for state in states.values())}


def select_params(candles_by_pair: Dict[str, pd.DataFrame], candidates: Iterable[GridParams], initial_quote: float,
                  fee_rate: float, portfolio_stop_loss: float, cooldown_hours: float,
                  drawdown_penalty: float = 1.5, liquidation_penalty: float = 0.25,
                  min_grid_move_seconds: float = 0, maker_fee_rate: float | None = None,
                  taker_fee_rate: float | None = None) -> Tuple[GridParams, Dict[str, float], float, pd.DataFrame]:
    rows = []
    best = None
    for params in candidates:
        result = simulate_portfolio(
            candles_by_pair, params, initial_quote, fee_rate, portfolio_stop_loss,
            cooldown_hours, min_grid_move_seconds, maker_fee_rate, taker_fee_rate,
        )
        score = result["net_pnl_pct"] - abs(result["max_drawdown_pct"]) * drawdown_penalty
        if result["liquidated"]:
            score -= liquidation_penalty
        row = {**asdict(params), **{f"selection_{key}": value for key, value in result.items()}, "selection_score": score}
        rows.append(row)
        if best is None or score > best[2]:
            best = (params, result, score)
    if best is None:
        raise ValueError("No candidates supplied")
    return best[0], best[1], best[2], pd.DataFrame(rows)


def _evaluate_chunk(args):
    (candles, candidates, initial_quote, fee_rate, stop_loss, cooldown_hours, drawdown_penalty,
     liquidation_penalty, move_seconds, maker_fee_rate, taker_fee_rate) = args
    rows = []
    for params in candidates:
        result = simulate_portfolio(
            candles, params, initial_quote, fee_rate, stop_loss, cooldown_hours, move_seconds,
            maker_fee_rate, taker_fee_rate,
        )
        score = result["net_pnl_pct"] - abs(result["max_drawdown_pct"]) * drawdown_penalty
        if result["liquidated"]:
            score -= liquidation_penalty
        rows.append((params, result, score))
    return rows


def select_params_parallel(candles_by_pair: Dict[str, pd.DataFrame], candidates: Iterable[GridParams], initial_quote: float,
                           fee_rate: float, portfolio_stop_loss: float, cooldown_hours: float, workers: int,
                           drawdown_penalty: float = 1.5, liquidation_penalty: float = 0.25,
                           min_grid_move_seconds: float = 0, maker_fee_rate: float | None = None,
                           taker_fee_rate: float | None = None):
    candidate_list = list(candidates)
    if workers <= 1:
        return select_params(
            candles_by_pair, candidate_list, initial_quote, fee_rate, portfolio_stop_loss, cooldown_hours,
            drawdown_penalty, liquidation_penalty, min_grid_move_seconds, maker_fee_rate, taker_fee_rate,
        )
    worker_count = min(workers, len(candidate_list))
    chunks = [candidate_list[index::worker_count] for index in range(worker_count)]
    args = [(candles_by_pair, chunk, initial_quote, fee_rate, portfolio_stop_loss, cooldown_hours,
             drawdown_penalty, liquidation_penalty, min_grid_move_seconds, maker_fee_rate, taker_fee_rate)
            for chunk in chunks if chunk]
    values = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        for rows in executor.map(_evaluate_chunk, args):
            values.extend(rows)
    best = max(values, key=lambda value: value[2])
    rows = [{**asdict(params), **{f"selection_{key}": value for key, value in result.items()}, "selection_score": score}
            for params, result, score in values]
    return best[0], best[1], best[2], pd.DataFrame(rows)
