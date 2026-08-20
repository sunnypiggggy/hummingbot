#!/usr/bin/env python3
"""Offline 360-day comparison of bidirectional and long-only spot grids.

The script is deliberately isolated from live configuration.  It compares the
current fixed grid with the Binance short-sideways mapping, enforces a 10 FDUSD
post-quantisation minimum order, consumes the frozen v22 gate-state evidence,
and writes a self-contained Plotly audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/backtests/binance_short_sideways_10usd_long_vs_bidirectional_360d"
START_TS = 1_755_993_600  # 2025-08-24 00:00:00 UTC
END_TS = 1_787_097_600  # 2026-08-19 00:00:00 UTC, exclusive
ROWS_PER_PAIR = 103_680
PAIR_CAPITAL = 200.0
SIDE_BUDGET = 100.0
MINIMUM_ORDER = 10.0
TAKER_FEE = 0.001
TAKER_SLIPPAGE = 0.0002
REFRESH_BARS = 24  # 2 hours on 5-minute candles
MOVE_COOLDOWN_BARS = 6
HEALTHY_REENTRY_BARS = 3
PAIR_COOLDOWN_BARS = 72  # 6 hours
PORTFOLIO_COOLDOWN_BARS = 144  # 12 hours
POSITION_PROTECTION_BARS = 576  # 48 hours


@dataclass(frozen=True)
class ExchangeFilter:
    tick_size: float
    step_size: float
    minimum_notional: float


@dataclass(frozen=True)
class Preset:
    preset_id: str
    label: str
    total_range: float
    levels: int
    take_profit: float
    move_threshold: float = 0.015

    @property
    def side_levels(self) -> int:
        return self.levels // 2

    @property
    def actual_step(self) -> float:
        return self.total_range / (self.levels - 1)


PRESETS = {
    "fixed_current": Preset("fixed_current", "当前固定参数", 0.06, 10, 0.006),
    "short_sideways_10usd": Preset(
        "short_sideways_10usd", "Binance短期横盘（10 FDUSD）", 0.07695669969152726, 18, 0.004
    ),
}


@dataclass
class Order:
    order_id: str
    side: str
    price: float
    quantity: float
    kind: str
    created_bar: int

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class Lot:
    lot_id: str
    quantity: float
    entry_price: float
    target_price: float
    created_bar: int


@dataclass
class PairState:
    pair: str
    mode: str
    preset: Preset
    exchange_filter: ExchangeFilter
    quote: float
    base: float
    initial_base: float
    center: float
    minimum_order: float = MINIMUM_ORDER
    orders: list[Order] = field(default_factory=list)
    lots: list[Lot] = field(default_factory=list)
    last_refresh_bar: int = -REFRESH_BARS
    last_move_bar: int = -MOVE_COOLDOWN_BARS
    peak_equity: float = PAIR_CAPITAL
    cycle_baseline: float = PAIR_CAPITAL
    pair_halted_until: int = -1
    healthy_cycles: int = 0
    active: bool = True
    last_gate: str = "RISK_ON"
    maker_orders: int = 0
    maker_buys: int = 0
    normal_sells: int = 0
    take_profit_sells: int = 0
    maker_notional: float = 0.0
    min_order_notional: float = math.inf
    max_order_notional: float = 0.0
    clipped_levels: int = 0
    grid_moves: int = 0
    forced_exits: int = 0
    reentries: int = 0
    fees: float = 0.0
    risk_execution_cost: float = 0.0
    reentry_cost: float = 0.0
    blocked_bars: int = 0
    exposure_sum: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    equity_rows: list[dict[str, Any]] = field(default_factory=list)

    def equity(self, price: float) -> float:
        return self.quote + self.base * price


def floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor((value + 1e-12) / step) * step


def ceil_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil((value - 1e-12) / step) * step


def quantize_price(price: float, side: str, exchange_filter: ExchangeFilter) -> float:
    fn = floor_to_step if side == "BUY" else ceil_to_step
    return fn(price, exchange_filter.tick_size)


def build_grid_orders(
    *,
    pair: str,
    mode: str,
    preset: Preset,
    exchange_filter: ExchangeFilter,
    center: float,
    quote_available: float,
    base_available: float,
    bar: int,
    minimum_order: float = MINIMUM_ORDER,
) -> tuple[list[Order], int]:
    """Build post-quantisation-valid grid orders.

    Levels are removed from the far edge until every submitted order meets the
    configured 10 FDUSD floor after both price and quantity quantisation.
    """

    minimum = max(minimum_order, exchange_filter.minimum_notional)
    half = preset.total_range / 2.0
    raw_prices = [center * (1.0 - half + preset.actual_step * i) for i in range(preset.levels)]
    buy_prices = [p for p in raw_prices if p < center]
    sell_prices = [p for p in raw_prices if p > center]
    clipped = 0

    def valid_side(side: str, prices: list[float], budget: float) -> list[Order]:
        nonlocal clipped
        candidates = list(prices)
        while candidates:
            per_order = budget / len(candidates)
            built: list[Order] = []
            valid = True
            for index, raw_price in enumerate(candidates):
                price = quantize_price(raw_price, side, exchange_filter)
                quantity = floor_to_step(per_order / price, exchange_filter.step_size)
                if quantity <= 0 or quantity * price + 1e-9 < minimum:
                    valid = False
                    break
                built.append(Order(f"{pair}-{bar}-{side}-{index}", side, price, quantity, "GRID", bar))
            required = sum(order.quantity * order.price for order in built)
            if valid and required <= budget + 1e-8:
                return built
            candidates.pop(0 if side == "BUY" else -1)
            clipped += 1
        return []

    buy_budget = min(SIDE_BUDGET, max(0.0, quote_available))
    buy_orders = valid_side("BUY", buy_prices, buy_budget) if buy_budget >= minimum else []
    sell_orders: list[Order] = []
    if mode == "bidirectional" and base_available > 0:
        sell_budget = min(SIDE_BUDGET, base_available * center)
        sell_orders = valid_side("SELL", sell_prices, sell_budget) if sell_budget >= minimum else []
        total_quantity = sum(order.quantity for order in sell_orders)
        while sell_orders and total_quantity > base_available + 1e-12:
            clipped += 1
            sell_orders.pop()
            total_quantity = sum(order.quantity for order in sell_orders)
    return buy_orders + sell_orders, clipped


def load_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, ExchangeFilter], dict[str, Any]]:
    candle_frames: dict[str, pd.DataFrame] = {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        path = ROOT / f"data/backtesting_candles/binance_{pair}_5m.csv"
        frame = pd.read_csv(path)
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame = frame[(frame.timestamp >= START_TS) & (frame.timestamp < END_TS)].copy()
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        validate_candles(pair, frame)
        candle_frames[pair] = frame

    gate_path = ROOT / "results/backtests/binance_ai_grid_presets_360d/v22_gate_states.csv.gz"
    gate = pd.read_csv(gate_path)
    gate["signal_ts"] = gate["signal_ts"].astype("int64")

    filter_path = ROOT / "results/backtests/binance_ai_grid_presets_360d/exchange_filters.json"
    raw_filters = json.loads(filter_path.read_text(encoding="utf-8"))
    filters = {
        pair: ExchangeFilter(
            tick_size=float(payload["tick_size"]),
            step_size=float(payload["step_size"]),
            minimum_notional=float(payload["minimum_notional"]),
        )
        for pair, payload in raw_filters["pairs"].items()
    }
    evidence = {
        "candles": {pair: str(ROOT / f"data/backtesting_candles/binance_{pair}_5m.csv") for pair in candle_frames},
        "v22_gate_states": str(gate_path),
        "exchange_filters": str(filter_path),
    }
    return candle_frames, gate, filters, evidence


def validate_candles(pair: str, frame: pd.DataFrame) -> None:
    if len(frame) != ROWS_PER_PAIR:
        raise ValueError(f"{pair}: expected {ROWS_PER_PAIR} rows, got {len(frame)}")
    if frame.timestamp.duplicated().any():
        raise ValueError(f"{pair}: duplicated timestamps")
    if not np.all(np.diff(frame.timestamp.to_numpy()) == 300):
        raise ValueError(f"{pair}: candle continuity failure")
    if not ((frame.high >= frame[["open", "close"]].max(axis=1)) & (frame.low <= frame[["open", "close"]].min(axis=1))).all():
        raise ValueError(f"{pair}: OHLC integrity failure")
    if (frame[["open", "high", "low", "close", "volume"]].isna().any()).any():
        raise ValueError(f"{pair}: missing OHLCV values")


def expand_v22_gate(frame: pd.DataFrame, gate: pd.DataFrame, pair: str) -> np.ndarray:
    pair_gate = gate[gate.pair == pair].sort_values("signal_ts")
    coverage_start = int(pair_gate.signal_ts.min())
    coverage_end = int(pair_gate.signal_ts.max()) + 3600
    lookup = pair_gate.set_index("signal_ts")["risk_off_active"]
    hour_ts = (frame.timestamp.to_numpy(dtype=np.int64) // 3600) * 3600
    values = np.full(len(frame), "UNAVAILABLE", dtype=object)
    covered = (hour_ts >= coverage_start) & (hour_ts < coverage_end)
    mapped = pd.Series(hour_ts[covered]).map(lookup)
    if mapped.isna().any():
        raise ValueError(f"{pair}: signed v22 coverage has an internal hourly gap")
    values[covered] = np.where(mapped.to_numpy(dtype=bool), "RISK_OFF", "RISK_ON")
    return values


def initialise_state(pair: str, mode: str, preset: Preset, exchange_filter: ExchangeFilter,
                     first_price: float, minimum_order: float = MINIMUM_ORDER) -> PairState:
    if mode == "bidirectional":
        base = floor_to_step(SIDE_BUDGET / first_price, exchange_filter.step_size)
        quote = PAIR_CAPITAL - base * first_price
    else:
        base = 0.0
        quote = PAIR_CAPITAL
    return PairState(
        pair, mode, preset, exchange_filter, quote, base, base, first_price,
        minimum_order=minimum_order,
    )


def place_grid(state: PairState, bar: int, center: float) -> None:
    state.orders = [order for order in state.orders if order.kind == "TAKE_PROFIT"]
    quote_reserved = sum(order.notional for order in state.orders if order.side == "BUY")
    base_reserved = sum(order.quantity for order in state.orders if order.side == "SELL")
    orders, clipped = build_grid_orders(
        pair=state.pair,
        mode=state.mode,
        preset=state.preset,
        exchange_filter=state.exchange_filter,
        center=center,
        quote_available=max(0.0, state.quote - quote_reserved - (SIDE_BUDGET if state.mode == "long_only" else 0.0)),
        base_available=max(0.0, state.base - base_reserved),
        bar=bar,
        minimum_order=state.minimum_order,
    )
    if state.mode == "long_only":
        # The first 100 FDUSD is active capital; the second 100 remains reserve.
        orders, clipped = build_grid_orders(
            pair=state.pair,
            mode=state.mode,
            preset=state.preset,
            exchange_filter=state.exchange_filter,
            center=center,
            quote_available=min(SIDE_BUDGET, max(0.0, state.quote - SIDE_BUDGET)),
            base_available=0.0,
            bar=bar,
            minimum_order=state.minimum_order,
        )
    state.orders.extend(orders)
    state.maker_orders += len(orders)
    state.clipped_levels += clipped
    for order in orders:
        state.min_order_notional = min(state.min_order_notional, order.notional)
        state.max_order_notional = max(state.max_order_notional, order.notional)
        event(state, START_TS + bar * 300, "ORDER_PLACED", side=order.side, price=order.price,
              quantity=order.quantity, notional=order.notional, kind=order.kind)
    state.center = center
    state.last_refresh_bar = bar


def event(state: PairState, timestamp: int, event_type: str, **payload: Any) -> None:
    state.events.append({"timestamp": timestamp, "pair": state.pair, "mode": state.mode, "preset": state.preset.preset_id,
                         "event_type": event_type, **payload})


def fill_order(state: PairState, order: Order, timestamp: int, bar: int) -> None:
    notional = order.notional
    if order.side == "BUY":
        if notional > state.quote + 1e-8:
            return
        state.quote -= notional
        state.base += order.quantity
        state.maker_buys += 1
        if state.mode == "long_only":
            target = quantize_price(order.price * (1.0 + state.preset.take_profit), "SELL", state.exchange_filter)
            lot = Lot(f"{order.order_id}-lot", order.quantity, order.price, target, bar)
            state.lots.append(lot)
            tp = Order(f"{order.order_id}-tp", "SELL", target, order.quantity, "TAKE_PROFIT", bar)
            state.orders.append(tp)
            state.maker_orders += 1
            state.min_order_notional = min(state.min_order_notional, tp.notional)
            state.max_order_notional = max(state.max_order_notional, tp.notional)
            event(state, timestamp, "ORDER_PLACED", side=tp.side, price=tp.price, quantity=tp.quantity,
                  notional=tp.notional, kind=tp.kind)
        event(state, timestamp, "BUY", price=order.price, quantity=order.quantity, notional=notional, kind=order.kind)
    else:
        quantity = min(order.quantity, state.base)
        if quantity <= 0:
            return
        notional = quantity * order.price
        state.base -= quantity
        state.quote += notional
        if order.kind == "TAKE_PROFIT":
            state.take_profit_sells += 1
            remaining = quantity
            new_lots: list[Lot] = []
            for lot in state.lots:
                if lot.lot_id == order.order_id.removesuffix("-tp") + "-lot":
                    lot.quantity -= remaining
                    remaining = 0.0
                if lot.quantity > 1e-12:
                    new_lots.append(lot)
            state.lots = new_lots
        else:
            state.normal_sells += 1
        event(state, timestamp, "TAKE_PROFIT_SELL" if order.kind == "TAKE_PROFIT" else "NORMAL_SELL",
              price=order.price, quantity=quantity, notional=notional, kind=order.kind)
    state.maker_notional += notional


def process_maker_fills(state: PairState, candle: pd.Series, bar: int) -> None:
    if not state.active:
        return
    timestamp = int(candle.timestamp)
    orders = list(state.orders)
    state.orders = []
    if candle.close >= candle.open:
        ordered = sorted(orders, key=lambda order: 0 if order.side == "BUY" else 1)
    else:
        ordered = sorted(orders, key=lambda order: 0 if order.side == "SELL" else 1)
    filled_ids: set[str] = set()
    for order in ordered:
        eligible = bar > order.created_bar if order.kind == "TAKE_PROFIT" else True
        touched = candle.low <= order.price if order.side == "BUY" else candle.high >= order.price
        if eligible and touched:
            before = state.base if order.side == "SELL" else state.quote
            fill_order(state, order, timestamp, bar)
            after = state.base if order.side == "SELL" else state.quote
            if abs(after - before) > 1e-12:
                filled_ids.add(order.order_id)
        if order.order_id not in filled_ids:
            state.orders.append(order)


def force_exit(state: PairState, price: float, timestamp: int, reason: str) -> None:
    state.orders.clear()
    state.lots.clear()
    quantity = floor_to_step(state.base, state.exchange_filter.step_size)
    if quantity * price < state.exchange_filter.minimum_notional or quantity <= 0:
        state.active = False
        return
    execution_price = price * (1.0 - TAKER_SLIPPAGE)
    gross = quantity * execution_price
    fee = gross * TAKER_FEE
    reference = quantity * price
    state.base -= quantity
    state.quote += gross - fee
    state.fees += fee
    state.risk_execution_cost += reference - (gross - fee)
    state.forced_exits += 1
    state.active = False
    event(state, timestamp, "FORCED_EXIT", price=execution_price, quantity=quantity, notional=gross, fee=fee, reason=reason)


def reenter(state: PairState, price: float, timestamp: int, bar: int) -> None:
    if state.mode == "bidirectional":
        spend = min(SIDE_BUDGET, state.quote)
        execution_price = price * (1.0 + TAKER_SLIPPAGE)
        quantity = floor_to_step((spend / (1.0 + TAKER_FEE)) / execution_price, state.exchange_filter.step_size)
        gross = quantity * execution_price
        fee = gross * TAKER_FEE
        if gross + fee >= max(state.minimum_order, state.exchange_filter.minimum_notional) and gross + fee <= state.quote + 1e-8:
            state.quote -= gross + fee
            state.base += quantity
            state.fees += fee
            state.reentry_cost += quantity * (execution_price - price) + fee
            event(state, timestamp, "REENTRY_BUY", price=execution_price, quantity=quantity, notional=gross, fee=fee)
    state.active = True
    state.reentries += 1
    state.cycle_baseline = state.equity(price)
    state.peak_equity = state.cycle_baseline
    state.last_refresh_bar = bar - REFRESH_BARS


def position_protection(state: PairState, price: float, timestamp: int, bar: int) -> None:
    if state.mode != "long_only" or not state.lots:
        return
    expired = [lot for lot in state.lots if bar - lot.created_bar >= POSITION_PROTECTION_BARS]
    quantity = floor_to_step(sum(lot.quantity for lot in expired), state.exchange_filter.step_size)
    if quantity * price < state.exchange_filter.minimum_notional or quantity <= 0:
        return
    execution_price = price * (1.0 - TAKER_SLIPPAGE)
    gross = quantity * execution_price
    fee = gross * TAKER_FEE
    state.base -= quantity
    state.quote += gross - fee
    state.fees += fee
    state.risk_execution_cost += quantity * price - (gross - fee)
    expired_ids = {lot.lot_id for lot in expired}
    state.lots = [lot for lot in state.lots if lot.lot_id not in expired_ids]
    state.orders = [order for order in state.orders if order.order_id.removesuffix("-tp") + "-lot" not in expired_ids]
    event(state, timestamp, "POSITION_PROTECTION_SELL", price=execution_price, quantity=quantity, notional=gross, fee=fee)


def simulate_arm(
    candle_frames: dict[str, pd.DataFrame],
    gate_arrays: dict[str, np.ndarray],
    filters: dict[str, ExchangeFilter],
    mode: str,
    preset: Preset | dict[str, Preset],
    scope: str,
    minimum_order_by_pair: dict[str, float] | None = None,
    movement_semantics: str = "center_threshold",
    portfolio_reserve: float = 0.0,
) -> tuple[dict[str, PairState], pd.DataFrame, pd.DataFrame]:
    pairs = ("BTC-FDUSD", "ETH-FDUSD")
    pair_presets = preset if isinstance(preset, dict) else {pair: preset for pair in pairs}
    minimums = minimum_order_by_pair or {pair: MINIMUM_ORDER for pair in pairs}
    states = {
        pair: initialise_state(
            pair, mode, pair_presets[pair], filters[pair],
            float(candle_frames[pair].iloc[0].close), minimums[pair],
        ) for pair in pairs
    }
    portfolio_baseline = PAIR_CAPITAL * 2 + portfolio_reserve
    portfolio_peak = portfolio_baseline
    portfolio_halted_until = -1

    for bar in range(ROWS_PER_PAIR):
        prices = {pair: float(candle_frames[pair].iloc[bar].close) for pair in pairs}
        timestamp = int(candle_frames[pairs[0]].iloc[bar].timestamp)

        for pair in pairs:
            state = states[pair]
            gate = gate_arrays[pair][bar] if scope == "protected_v22" else "RISK_ON"
            state.last_gate = gate
            blocked = scope == "protected_v22" and (gate != "RISK_ON" or bar < state.pair_halted_until or bar < portfolio_halted_until)
            if blocked:
                state.healthy_cycles = 0
                state.blocked_bars += 1
                if state.active or state.orders:
                    force_exit(state, prices[pair], timestamp, f"gate={gate};pair_until={state.pair_halted_until};portfolio_until={portfolio_halted_until}")
            else:
                state.healthy_cycles += 1
                if not state.active and state.healthy_cycles >= HEALTHY_REENTRY_BARS:
                    reenter(state, prices[pair], timestamp, bar)

            if state.active:
                process_maker_fills(state, candle_frames[pair].iloc[bar], bar)
                if scope == "protected_v22":
                    position_protection(state, prices[pair], timestamp, bar)
                if movement_semantics == "boundary_plus_threshold":
                    half = state.preset.total_range / 2.0
                    moved = (
                        prices[pair] > state.center * (1.0 + half) * (1.0 + state.preset.move_threshold)
                        or prices[pair] < state.center * (1.0 - half) * (1.0 - state.preset.move_threshold)
                    ) and bar - state.last_move_bar >= MOVE_COOLDOWN_BARS
                else:
                    moved = (
                        abs(prices[pair] / state.center - 1.0) >= state.preset.move_threshold
                        and bar - state.last_move_bar >= MOVE_COOLDOWN_BARS
                    )
                refreshed = bar - state.last_refresh_bar >= REFRESH_BARS
                if moved or refreshed:
                    place_grid(state, bar, prices[pair])
                    if moved:
                        state.grid_moves += 1
                        state.last_move_bar = bar

        equities = {pair: states[pair].equity(prices[pair]) for pair in pairs}
        if scope == "protected_v22":
            for pair in pairs:
                state = states[pair]
                state.peak_equity = max(state.peak_equity, equities[pair])
                loss = state.cycle_baseline - equities[pair]
                drawdown = equities[pair] / state.peak_equity - 1.0
                if state.active and (loss >= 6.0 or drawdown <= -0.03):
                    state.pair_halted_until = max(state.pair_halted_until, bar + PAIR_COOLDOWN_BARS)
                    force_exit(state, prices[pair], timestamp, "strategy_loss_or_drawdown_breaker")
            portfolio_equity = sum(states[pair].equity(prices[pair]) for pair in pairs) + portfolio_reserve
            portfolio_peak = max(portfolio_peak, portfolio_equity)
            if bar >= portfolio_halted_until and (portfolio_baseline - portfolio_equity >= 24.0 or portfolio_equity / portfolio_peak - 1.0 <= -0.06):
                portfolio_halted_until = bar + PORTFOLIO_COOLDOWN_BARS
                for pair in pairs:
                    force_exit(states[pair], prices[pair], timestamp, "portfolio_loss_or_drawdown_breaker")
                portfolio_baseline = sum(states[pair].equity(prices[pair]) for pair in pairs) + portfolio_reserve
                portfolio_peak = portfolio_baseline

        for pair in pairs:
            state = states[pair]
            equity = state.equity(prices[pair])
            state.exposure_sum += state.base * prices[pair]
            state.equity_rows.append({
                "timestamp": timestamp,
                "pair": pair,
                "mode": mode,
                "preset": state.preset.preset_id,
                "scope": scope,
                "price": prices[pair],
                "equity": equity,
                "quote": state.quote,
                "base": state.base,
                "base_value": state.base * prices[pair],
                "active": state.active,
                "v22_state": state.last_gate,
            })

    equity_frame = pd.DataFrame([row for state in states.values() for row in state.equity_rows])
    event_frame = pd.DataFrame([row for state in states.values() for row in state.events])
    return states, equity_frame, event_frame


def summarise(states: dict[str, PairState], equity_frame: pd.DataFrame, scope: str, mode: str, preset: Preset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, state in states.items():
        pair_equity = equity_frame[equity_frame.pair == pair].equity.to_numpy()
        running_peak = np.maximum.accumulate(pair_equity)
        drawdown = pair_equity / running_peak - 1.0
        total_maker_fills = state.maker_buys + state.normal_sells + state.take_profit_sells
        final_price = float(equity_frame[equity_frame.pair == pair].price.iloc[-1])
        rows.append({
            "scope": scope,
            "mode": mode,
            "preset": preset.preset_id,
            "label": preset.label,
            "pair": pair,
            "initial_equity_fdusd": PAIR_CAPITAL,
            "final_equity_fdusd": float(pair_equity[-1]),
            "net_pnl_fdusd": float(pair_equity[-1] - PAIR_CAPITAL),
            "net_return_pct": float((pair_equity[-1] / PAIR_CAPITAL - 1.0) * 100.0),
            "active_capital_return_pct": (
                float((pair_equity[-1] - PAIR_CAPITAL) / SIDE_BUDGET * 100.0) if mode == "long_only" else None
            ),
            "max_drawdown_pct": float(drawdown.min() * 100.0),
            "maker_buys": state.maker_buys,
            "normal_sells": state.normal_sells,
            "take_profit_sells": state.take_profit_sells,
            "maker_fills": total_maker_fills,
            "maker_orders": state.maker_orders,
            "maker_fill_rate_pct": float(total_maker_fills / state.maker_orders * 100.0) if state.maker_orders else 0.0,
            "maker_notional_fdusd": state.maker_notional,
            "minimum_placed_order_fdusd": None if math.isinf(state.min_order_notional) else state.min_order_notional,
            "maximum_placed_order_fdusd": state.max_order_notional,
            "clipped_levels": state.clipped_levels,
            "grid_moves": state.grid_moves,
            "forced_exits": state.forced_exits,
            "reentries": state.reentries,
            "fees_fdusd": state.fees,
            "risk_execution_cost_fdusd": state.risk_execution_cost,
            "reentry_cost_fdusd": state.reentry_cost,
            "blocked_hours": state.blocked_bars * 5.0 / 60.0,
            "mean_market_exposure_fdusd": state.exposure_sum / ROWS_PER_PAIR,
            "capital_utilisation_pct": state.exposure_sum / ROWS_PER_PAIR / SIDE_BUDGET * 100.0,
            "ending_base": state.base,
            "ending_base_value_fdusd": state.base * final_price,
            "ending_quote_fdusd": state.quote,
            "negative_inventory_observations": int((equity_frame[equity_frame.pair == pair].base < -1e-12).sum()),
        })
    return rows


def risk_intervals(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    data = frame[["timestamp", "v22_state"]].drop_duplicates().sort_values("timestamp")
    state = data.v22_state.to_numpy()
    ts = pd.to_datetime(data.timestamp, unit="s", utc=True).to_numpy()
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    start = 0
    for index in range(1, len(data) + 1):
        if index == len(data) or state[index] != state[start]:
            if state[start] != "RISK_ON":
                end_index = min(index, len(data) - 1)
                intervals.append((pd.Timestamp(ts[start]), pd.Timestamp(ts[end_index]), str(state[start])))
            start = index
    return intervals


def hourly(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data.timestamp, unit="s", utc=True)
    return data.set_index("datetime").resample("1h").agg({
        "timestamp": "last", "price": "last", "equity": "last", "quote": "last", "base": "last",
        "base_value": "last", "active": "last", "v22_state": "last",
    }).dropna().reset_index()


def aggregate_events(events: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    data = events[events.event_type.isin(["BUY", "NORMAL_SELL", "TAKE_PROFIT_SELL"])].copy()
    if data.empty:
        return data
    data["datetime"] = pd.to_datetime(data.timestamp, unit="s", utc=True).dt.floor("1h")
    grouped = data.groupby(["datetime", "event_type"], as_index=False).agg(count=("event_type", "size"), notional=("notional", "sum"))
    price_hour = hourly(price)[["datetime", "price"]]
    return grouped.merge(price_hour, on="datetime", how="left")


def add_v22_shapes(fig: go.Figure, intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp, str]], rows: int) -> None:
    first = True
    for start, end, state in intervals:
        color = "rgba(214,69,65,0.10)" if state == "RISK_OFF" else "rgba(120,120,120,0.08)"
        border = "#c0392b" if state == "RISK_OFF" else "#6b7280"
        for row in range(1, rows):
            fig.add_vrect(x0=start, x1=end, fillcolor=color, line_color=border, line_width=1,
                          layer="below", row=row, col=1)
        if first:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=10, color=color,
                                      line=dict(color=border, width=1)), name="v22 Risk-Off/不可用阴影", legendgroup="v22"), row=1, col=1)
            first = False


def strategy_figure(pair: str, mode: str, equity: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    mode_label = "现货双向Grid" if mode == "bidirectional" else "现货只做多（买入后止盈）"
    subset = equity[(equity.pair == pair) & (equity["mode"] == mode)]
    figure = make_subplots(rows=6, cols=1, shared_xaxes=False, vertical_spacing=0.035,
                           row_heights=[0.21, 0.22, 0.14, 0.14, 0.17, 0.12],
                           subplot_titles=("标的价格", "单交易对连续权益", "回撤", "短期横盘相对固定参数",
                                           "短期横盘资金与库存", "实际挂单金额分布"))
    colors = {("fixed_current", "parameter_only"): "#7f8c8d", ("short_sideways_10usd", "parameter_only"): "#d4a017",
              ("fixed_current", "protected_v22"): "#3465a4", ("short_sideways_10usd", "protected_v22"): "#d97706"}
    dashes = {"parameter_only": "dot", "protected_v22": "solid"}
    price_source = subset[(subset.preset == "short_sideways_10usd") & (subset.scope == "protected_v22")]
    price_hour = hourly(price_source)
    figure.add_trace(go.Scatter(x=price_hour.datetime, y=price_hour.price, name=f"{pair}价格", line=dict(color="#273746", width=1.2)), row=1, col=1)

    curves: dict[tuple[str, str], pd.DataFrame] = {}
    for preset_id in PRESETS:
        for scope in ("parameter_only", "protected_v22"):
            data = hourly(subset[(subset.preset == preset_id) & (subset.scope == scope)])
            curves[(preset_id, scope)] = data
            label = f"{PRESETS[preset_id].label} / {'纯参数' if scope == 'parameter_only' else 'v22风控'}"
            figure.add_trace(go.Scatter(x=data.datetime, y=data.equity, name=label,
                                        line=dict(color=colors[(preset_id, scope)], dash=dashes[scope], width=1.5)), row=2, col=1)
            peak = data.equity.cummax()
            drawdown = (data.equity / peak - 1.0) * 100.0
            figure.add_trace(go.Scatter(x=data.datetime, y=drawdown, name=f"{label}回撤", showlegend=False,
                                        line=dict(color=colors[(preset_id, scope)], dash=dashes[scope], width=1.2)), row=3, col=1)

    for scope, color in (("parameter_only", "#8c6d1f"), ("protected_v22", "#b45309")):
        short = curves[("short_sideways_10usd", scope)]
        fixed = curves[("fixed_current", scope)]
        delta = short.equity.to_numpy() - fixed.equity.to_numpy()
        figure.add_trace(go.Scatter(x=short.datetime, y=delta, name=f"短期横盘-固定 / {'纯参数' if scope == 'parameter_only' else 'v22风控'}",
                                    line=dict(color=color, dash=dashes[scope], width=1.3)), row=4, col=1)

    candidate = curves[("short_sideways_10usd", "protected_v22")]
    figure.add_trace(go.Scatter(x=candidate.datetime, y=candidate.quote, name="FDUSD现金", line=dict(color="#3465a4")), row=5, col=1)
    figure.add_trace(go.Scatter(x=candidate.datetime, y=candidate.base_value, name="基础币市值", line=dict(color="#d97706")), row=5, col=1)

    pair_events = events[(events.pair == pair) & (events["mode"] == mode) & (events.preset == "short_sideways_10usd") &
                         (events.scope == "protected_v22")]
    event_hour = aggregate_events(pair_events, price_source)
    marker_styles = {"BUY": ("triangle-up", "#3465a4", "BUY"),
                     "NORMAL_SELL": ("triangle-down", "#d97706", "上方SELL/启动库存"),
                     "TAKE_PROFIT_SELL": ("diamond", "#7a5195", "买入后止盈SELL")}
    for event_type, (symbol, color, display_name) in marker_styles.items():
        rows = event_hour[event_hour.event_type == event_type] if not event_hour.empty else event_hour
        if not rows.empty:
            figure.add_trace(go.Scatter(x=rows.datetime, y=rows.price, mode="markers", name=display_name,
                                        marker=dict(symbol=symbol, color=color, size=np.clip(5 + rows["count"], 6, 14)),
                                        customdata=np.c_[rows["count"], rows.notional],
                                        hovertemplate="%{x}<br>价格=%{y:.4f}<br>成交数=%{customdata[0]}<br>金额=%{customdata[1]:.2f}<extra></extra>"), row=1, col=1)

    notionals = pair_events[pair_events.event_type == "ORDER_PLACED"].notional if not pair_events.empty else pd.Series(dtype=float)
    figure.add_trace(go.Histogram(x=notionals, nbinsx=35, name="实际挂单金额分布", marker_color="#d4a017", opacity=0.8), row=6, col=1)
    figure.add_vline(x=MINIMUM_ORDER, line_color="#b91c1c", line_dash="dash", row=6, col=1)
    add_v22_shapes(figure, risk_intervals(price_source), 6)
    figure.update_layout(title=f"{pair} · {mode_label} · 360天离线回测", height=1450, template="plotly_white",
                         hovermode="x unified", legend=dict(orientation="h", y=1.03, x=0), margin=dict(l=65, r=30, t=110, b=50))
    for row in range(1, 6):
        figure.update_xaxes(rangeslider_visible=False, rangeselector=dict(buttons=[
            dict(count=360, label="360天", step="day", stepmode="backward"),
            dict(label="全部", step="all"),
        ]) if row == 1 else None, row=row, col=1)
    figure.update_yaxes(title_text="FDUSD", row=2, col=1)
    figure.update_yaxes(title_text="%", row=3, col=1)
    figure.update_yaxes(title_text="FDUSD", row=4, col=1)
    figure.update_yaxes(title_text="FDUSD", row=5, col=1)
    figure.update_xaxes(title_text="订单/成交金额（FDUSD）", row=6, col=1)
    return figure


def comparison_figure(pair: str, equity: pd.DataFrame) -> go.Figure:
    data = equity[(equity.pair == pair) & (equity.preset == "short_sideways_10usd") & (equity.scope == "protected_v22")]
    bidirectional = hourly(data[data["mode"] == "bidirectional"])
    long_only = hourly(data[data["mode"] == "long_only"])
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                           subplot_titles=("短期横盘权益", "回撤", "基础币市场敞口"))
    for label, frame, color, dash in (("现货双向", bidirectional, "#3465a4", "solid"),
                                      ("只做多", long_only, "#d97706", "dash")):
        figure.add_trace(go.Scatter(x=frame.datetime, y=frame.equity, name=label, line=dict(color=color, dash=dash)), row=1, col=1)
        dd = (frame.equity / frame.equity.cummax() - 1.0) * 100.0
        figure.add_trace(go.Scatter(x=frame.datetime, y=dd, name=f"{label}回撤", showlegend=False,
                                    line=dict(color=color, dash=dash)), row=2, col=1)
        figure.add_trace(go.Scatter(x=frame.datetime, y=frame.base_value, name=f"{label}敞口", showlegend=False,
                                    line=dict(color=color, dash=dash)), row=3, col=1)
    add_v22_shapes(figure, risk_intervals(data[data["mode"] == "bidirectional"]), 4)
    figure.update_layout(title=f"{pair} · Binance短期横盘 · 双向与只做多", height=900, template="plotly_white",
                         hovermode="x unified", legend=dict(orientation="h", y=1.05), margin=dict(l=65, r=30, t=100, b=45))
    figure.update_yaxes(title_text="FDUSD", row=1, col=1)
    figure.update_yaxes(title_text="%", row=2, col=1)
    figure.update_yaxes(title_text="FDUSD", row=3, col=1)
    return figure


def write_plotly(output: Path, equity: pd.DataFrame, events: pd.DataFrame) -> Path:
    tabs: list[tuple[str, go.Figure]] = []
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        for mode in ("bidirectional", "long_only"):
            label = f"{pair.split('-')[0]} {'双向' if mode == 'bidirectional' else '只做多'}"
            tabs.append((label, strategy_figure(pair, mode, equity, events)))
    tabs.extend((f"{pair.split('-')[0]} 模式对比", comparison_figure(pair, equity)) for pair in ("BTC-FDUSD", "ETH-FDUSD"))
    plot_blocks: list[str] = []
    for index, (label, figure) in enumerate(tabs):
        payload = figure.to_json(pretty=False, remove_uids=True).replace("</", "<\\/")
        plot_blocks.append(
            f'<section id="tab-{index}" class="plot-tab {"active" if index == 0 else ""}">'
            f'<div id="plot-{index}" class="plot-target"></div><script id="spec-{index}" type="application/json">{payload}</script></section>'
        )
    buttons = "".join(f'<button class="tab-button {"active" if i == 0 else ""}" onclick="showTab({i},this)">{label}</button>'
                      for i, (label, _) in enumerate(tabs))
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Binance短期横盘：双向与只做多</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f6f7f9;color:#1f2937}}header{{padding:18px 24px;background:white;border-bottom:1px solid #ddd}}header h1{{font-size:22px;margin:0 0 8px}}header p{{margin:4px 0;color:#52606d}}nav{{position:sticky;top:0;z-index:5;background:white;padding:10px 18px;border-bottom:1px solid #ddd;overflow-x:auto;white-space:nowrap}}button{{padding:8px 13px;margin:2px;border:1px solid #cbd5e1;background:#fff;border-radius:7px;cursor:pointer}}button.active{{background:#273746;color:white}}.plot-tab{{display:none;background:white;margin:14px;box-shadow:0 1px 4px #ddd;min-height:700px}}.plot-tab.active{{display:block}}.plot-target{{width:100%}}@media(max-width:700px){{header{{padding:14px}}.plot-tab{{margin:4px}}}}</style><script>{get_plotlyjs()}</script></head><body>
<header><h1>Binance短期横盘 Grid：现货双向与只做多360天对照</h1><p>区间：2025-08-24 00:00—2026-08-19 00:00 UTC；每对200 FDUSD；Maker 0%；实际订单下限10 FDUSD。</p><p>SELL仅出售已有库存；不使用合约、借币、BNB手续费或FOMC门控。本报告仅供离线验证，不构成实盘授权。</p></header><nav>{buttons}</nav>{''.join(plot_blocks)}
<div style="position:fixed;right:12px;bottom:12px;z-index:8;background:white;padding:7px;border:1px solid #ddd;border-radius:8px"><button onclick="setWindow('2025-08-24','2026-08-19')">360天</button><button onclick="setWindow('2026-01-01','2026-03-01')">2026年1–2月</button><button onclick="setWindow('2026-05-01','2026-07-01')">2026年5–6月</button><button onclick="toggleV22()">v22阴影</button></div>
<script>const rendered=new Set();function renderTab(i){{if(rendered.has(i))return;const s=JSON.parse(document.getElementById('spec-'+i).textContent);Plotly.newPlot('plot-'+i,s.data,s.layout,{{responsive:true,displaylogo:false}});rendered.add(i);}}function showTab(i,b){{document.querySelectorAll('.plot-tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab-button').forEach(x=>x.classList.remove('active'));document.getElementById('tab-'+i).classList.add('active');b.classList.add('active');renderTab(i);setTimeout(()=>window.dispatchEvent(new Event('resize')),50);}}function setWindow(a,b){{const p=document.querySelector('.plot-tab.active .js-plotly-plot');if(!p)return;const u={{}};for(let i=1;i<=5;i++){{const k=i===1?'xaxis':'xaxis'+i;u[k+'.range']=[a,b];}}Plotly.relayout(p,u);}}function toggleV22(){{const p=document.querySelector('.plot-tab.active .js-plotly-plot');if(!p)return;p.__v22Visible=p.__v22Visible===false;const u={{}};(p.layout.shapes||[]).forEach((_,i)=>u['shapes['+i+'].visible']=p.__v22Visible);Plotly.relayout(p,u);}}renderTab(0);</script></body></html>"""
    path = output / "grid_short_sideways_long_vs_bidirectional_plotly.html"
    path.write_text(html, encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_simulation(equity: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    orders = events[events.event_type == "ORDER_PLACED"]
    minimum_placed = float(orders.notional.min())
    under_minimum = int((orders.notional < MINIMUM_ORDER - 1e-9).sum())
    negative_inventory = int((equity.base < -1e-12).sum())
    long_normal_sells = int(((events["mode"] == "long_only") & (events.event_type == "NORMAL_SELL")).sum())

    keys = ["timestamp", "pair", "mode", "preset", "scope"]
    buy_events = events[events.event_type == "BUY"][keys]
    gate_lookup = equity[keys + ["v22_state"]].drop_duplicates(keys)
    protected_buys = buy_events.merge(gate_lookup, on=keys, how="left")
    blocked_buys = int(((protected_buys.scope == "protected_v22") & (protected_buys.v22_state != "RISK_ON")).sum())

    unbacked_tp = 0
    long_trades = events[(events["mode"] == "long_only") & events.event_type.isin(["BUY", "TAKE_PROFIT_SELL"])].copy()
    for _, group in long_trades.groupby(["pair", "preset", "scope"]):
        group = group.sort_values(["timestamp", "event_type"])
        signed = np.where(group.event_type == "BUY", group.quantity, -group.quantity)
        unbacked_tp += int((np.cumsum(signed) < -1e-10).sum())

    checks = {
        "order_rows": int(len(orders)),
        "minimum_placed_order_fdusd": minimum_placed,
        "orders_below_10_fdusd": under_minimum,
        "negative_inventory_observations": negative_inventory,
        "long_only_normal_grid_sells": long_normal_sells,
        "long_only_unbacked_take_profit_sells": unbacked_tp,
        "protected_buys_during_v22_block": blocked_buys,
        "fomc_enabled": False,
        "passed": all(value == 0 for value in (under_minimum, negative_inventory, long_normal_sells, unbacked_tp, blocked_buys)),
    }
    if not checks["passed"]:
        raise AssertionError(f"simulation acceptance failed: {checks}")
    return checks


def write_outputs(output: Path, summaries: list[dict[str, Any]], equity: pd.DataFrame, events: pd.DataFrame,
                  evidence: dict[str, Any], validation: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    equity.to_csv(output / "continuous_equity.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    events.to_csv(output / "trade_and_risk_events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    parameters = pd.DataFrame([{
        "preset_id": preset.preset_id, "label": preset.label, "total_range_pct": preset.total_range * 100,
        "levels": preset.levels, "side_levels": preset.side_levels, "actual_step_pct": preset.actual_step * 100,
        "take_profit_pct": preset.take_profit * 100, "minimum_order_fdusd": MINIMUM_ORDER,
        "move_threshold_pct": preset.move_threshold * 100, "refresh_hours": 2,
    } for preset in PRESETS.values()])
    parameters.to_csv(output / "parameter_mapping.csv", index=False, encoding="utf-8-sig")
    report = {
        "schema": "binance-short-sideways-long-vs-bidirectional-360d-v1",
        "offline_only": True,
        "deployment_allowed": False,
        "oci_modified": False,
        "window": {"start": START_TS, "end_exclusive": END_TS, "rows_per_pair": ROWS_PER_PAIR},
        "execution": {"pair_capital_fdusd": PAIR_CAPITAL, "active_side_budget_fdusd": SIDE_BUDGET,
                      "long_only_cash_reserve_fdusd": SIDE_BUDGET, "minimum_order_fdusd": MINIMUM_ORDER,
                      "maker_fee": 0.0, "taker_fee": TAKER_FEE, "taker_slippage": TAKER_SLIPPAGE,
                      "fomc_enabled": False, "bnb_fee_used": False, "momentum_maker_tp_enabled": False},
        "parameters": [asdict(preset) | {"actual_step": preset.actual_step} for preset in PRESETS.values()],
        "summaries": summaries,
        "evidence": evidence,
        "validation": validation,
    }
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_analysis(output, summary)
    write_plotly(output, equity, events)
    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"schema": "binance-short-sideways-long-vs-bidirectional-manifest-v1", "artifacts": artifacts}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_analysis(output: Path, summary: pd.DataFrame) -> None:
    protected = summary[(summary.scope == "protected_v22") & (summary.preset == "short_sideways_10usd")]
    lines = [
        "# Binance短期横盘：双向与只做多影响分析",
        "",
        "本报告是离线反事实。现货双向的SELL只出售已有基础币，不建立负仓，也不是合约做空。",
        "",
        "## v22风控口径结果",
        "",
        "| 交易对 | 模式 | 净收益(FDUSD) | 最大回撤 | BUY | 普通SELL | 止盈SELL | 平均敞口(FDUSD) | 阻止时长(h) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in protected.itertuples(index=False):
        mode = "双向" if row.mode == "bidirectional" else "只做多"
        lines.append(f"| {row.pair} | {mode} | {row.net_pnl_fdusd:.4f} | {row.max_drawdown_pct:.4f}% | "
                     f"{row.maker_buys} | {row.normal_sells} | {row.take_profit_sells} | "
                     f"{row.mean_market_exposure_fdusd:.4f} | {row.blocked_hours:.2f} |")
    lines.extend(["", "## 模式差异", ""])
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        pair_rows = protected[protected.pair == pair].set_index("mode")
        if {"bidirectional", "long_only"}.issubset(pair_rows.index):
            pnl_delta = pair_rows.loc["long_only", "net_pnl_fdusd"] - pair_rows.loc["bidirectional", "net_pnl_fdusd"]
            dd_delta = pair_rows.loc["long_only", "max_drawdown_pct"] - pair_rows.loc["bidirectional", "max_drawdown_pct"]
            exposure_delta = pair_rows.loc["long_only", "mean_market_exposure_fdusd"] - pair_rows.loc["bidirectional", "mean_market_exposure_fdusd"]
            lines.append(f"- **{pair}**：只做多相对双向净收益差 `{pnl_delta:+.4f} FDUSD`，最大回撤差 "
                         f"`{dd_delta:+.4f} 个百分点`，平均市场敞口差 `{exposure_delta:+.4f} FDUSD`。")
    lines.extend([
        "",
        "## 阅读提示",
        "",
        "- 横盘期重点看BUY、普通SELL/止盈SELL成交与资金利用率；持续上涨期重点看双向启动库存是否过早卖出。",
        "- 持续下跌期只做多初始无币，通常减少启动库存浮亏，但逐级BUY后仍会承受下跌敞口。",
        "- 只做多总资本收益按200 FDUSD计算，同时另报100 FDUSD活跃资金收益；不能用闲置现金掩盖策略效率。",
        "- v22覆盖外为UNAVAILABLE并Fail-Closed；它不会被标记为正常模型Risk-Off。FOMC未参与。",
    ])
    (output / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> None:
    candles, gate, filters, evidence = load_inputs()
    gate_arrays = {pair: expand_v22_gate(candles[pair], gate, pair) for pair in candles}
    summaries: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for mode in ("bidirectional", "long_only"):
        for preset in PRESETS.values():
            for scope in ("parameter_only", "protected_v22"):
                states, equity, events = simulate_arm(candles, gate_arrays, filters, mode, preset, scope)
                summaries.extend(summarise(states, equity, scope, mode, preset))
                equity_frames.append(equity)
                if not events.empty:
                    events["scope"] = scope
                    event_frames.append(events)
    all_equity = pd.concat(equity_frames, ignore_index=True)
    all_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    validation = validate_simulation(all_equity, all_events)
    write_outputs(output, summaries, all_equity, all_events, evidence, validation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
