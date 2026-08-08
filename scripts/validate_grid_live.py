#!/usr/bin/env python3
"""Cache candles and validate the two budget-isolated live moving grids."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
import requests
import yaml

from grid_live_common import (
    BOT_LOSS_LIMIT,
    CAPITAL_LIMIT,
    MIN_ORDER_QUOTE,
    ORDER_REFRESH_SECONDS,
    PAIR_BUDGET,
    PAIR_DRAWDOWN_LIMIT_PCT,
    PAIR_LOSS_LIMIT,
    PORTFOLIOS,
    PORTFOLIO_DRAWDOWN_LIMIT_PCT,
    RESERVE_QUOTE,
    SIDE_BUDGET,
    budget_for_pair,
    budget_for_quote,
    build_live_config,
    effective_take_profit,
    required_balances,
    validate_exchange_filters,
    validate_live_config,
)
from grid_technical_gate import (
    DEFAULT_ROC_RISK_OFF_PCT,
    DEFAULT_SQZMOM_RISK_OFF_PCT,
    build_technical_buy_gate,
    roc_sqz_signal_from_klines,
)


BINANCE_APIS = ["https://api.binance.com", "https://api-gcp.binance.com",
                "https://api1.binance.com", "https://api2.binance.com"]
INTERVAL_SECONDS = 300
KLINE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_ARRAY_CACHE: dict[int, tuple[tuple[int, float, float], dict[str, object]]] = {}


@dataclass(frozen=True)
class Candidate:
    half_range: float
    min_spread: float
    take_profit: float
    move_threshold: float
    move_cooldown_seconds: int = 1800

    @property
    def levels(self) -> int:
        per_side = max(2, min(int(SIDE_BUDGET / MIN_ORDER_QUOTE), int(self.half_range / self.min_spread)))
        return per_side * 2


@dataclass(frozen=True)
class InventoryExitPolicy:
    """Bound excess inventory and progressively relax its cost floor."""

    max_extra_inventory_quote: float
    max_hold_seconds: int
    stage_one_profit_rate: float = 0.002
    stage_two_profit_rate: float = 0.0
    stage_one_fraction: float = 1 / 3
    stage_two_fraction: float = 2 / 3

    def __post_init__(self) -> None:
        if self.max_extra_inventory_quote <= 0:
            raise ValueError("max_extra_inventory_quote must be positive")
        if self.max_hold_seconds <= 0:
            raise ValueError("max_hold_seconds must be positive")
        if not 0 < self.stage_one_fraction < self.stage_two_fraction < 1:
            raise ValueError("inventory exit stage fractions must satisfy 0 < first < second < 1")
        if self.stage_one_profit_rate < 0 or self.stage_two_profit_rate < 0:
            raise ValueError("inventory exit profit rates cannot be negative")


@dataclass(frozen=True)
class PairBreakerPolicy:
    """Offline-only pair breaker with a deterministic recovery cooldown."""

    trigger: str
    cooldown_seconds: int
    reset_baseline: bool
    pairs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.trigger not in {"loss", "drawdown"}:
            raise ValueError("pair breaker trigger must be loss or drawdown")
        if self.cooldown_seconds <= 0:
            raise ValueError("pair breaker cooldown_seconds must be positive")
        if not self.pairs:
            raise ValueError("pair breaker policy requires at least one pair")


@dataclass(frozen=True)
class RiskMechanismConfig:
    """Independent offline switches for the four loss/drawdown breakers.

    ``simulate`` keeps its historical all-on/all-off behaviour when this
    object is omitted.  Supplying it is therefore explicit and does not
    change existing live validation or parameter-search callers.
    """

    pair_loss: bool = True
    pair_drawdown: bool = True
    portfolio_loss: bool = True
    portfolio_drawdown: bool = True
    continue_after_portfolio_stop: bool = False
    restore_portfolio_inventory: bool = False


@dataclass(frozen=True)
class ExecutionFilter:
    """Reproducible exchange precision snapshot for offline order placement."""

    tick_size: float
    step_size: float
    min_notional: float

    def __post_init__(self) -> None:
        if self.tick_size <= 0 or self.step_size <= 0 or self.min_notional <= 0:
            raise ValueError("execution filter values must be positive")


def _quantize_down(value: float, step: float) -> float:
    return math.floor((value + step * 1e-9) / step) * step


def _quantize_up(value: float, step: float) -> float:
    return math.ceil((value - step * 1e-9) / step) * step


def inventory_profit_floor_rate(
    state: Mapping[str, Any], now: float, base_rate: float,
    policy: InventoryExitPolicy | None,
) -> float:
    if policy is None or state.get("excess_inventory_started_at") is None:
        return base_rate
    age_fraction = max(
        0.0,
        (now - float(state["excess_inventory_started_at"])) / policy.max_hold_seconds,
    )
    if age_fraction < policy.stage_one_fraction:
        return base_rate
    if age_fraction < policy.stage_two_fraction:
        return min(base_rate, policy.stage_one_profit_rate)
    return min(base_rate, policy.stage_two_profit_rate)


def update_excess_inventory_timer(state: dict, now: float) -> None:
    if state["base"] > state["initial_base"] + 1e-12:
        if state["excess_inventory_started_at"] is None:
            state["excess_inventory_started_at"] = now
    else:
        state["excess_inventory_started_at"] = None


def candidates() -> list[Candidate]:
    return [Candidate(*values) for values in itertools.product(
        (0.03, 0.04, 0.05), (0.006, 0.008, 0.01), (0.006, 0.008, 0.01), (0.015, 0.02, 0.03)
    )]


def cache_path(cache_dir: Path, pair: str) -> Path:
    return cache_dir / f"binance_{pair}_5m.csv"


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS)
    frame = pd.read_csv(path, usecols=lambda column: column in KLINE_COLUMNS)
    for column in KLINE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def request_json(session: requests.Session, path: str, params: Mapping[str, object]) -> object:
    errors = []
    for attempt in range(8):
        base = BINANCE_APIS[attempt % len(BINANCE_APIS)]
        try:
            response = session.get(f"{base}{path}", params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            errors.append(f"{base}: {exc}")
            time.sleep(min(0.5 * (attempt + 1), 3))
    raise RuntimeError("Binance public request failed after endpoint rotation: " + " | ".join(errors[-4:]))


def download_klines(pair: str, start_ts: int, end_ts: int, session: requests.Session) -> pd.DataFrame:
    rows = []
    cursor_ms = start_ts * 1000
    end_ms = end_ts * 1000
    while cursor_ms < end_ms:
        batch = request_json(session, "/api/v3/klines", {
            "symbol": pair.replace("-", ""), "interval": "5m", "startTime": cursor_ms,
            "endTime": end_ms - 1, "limit": 1000,
        })
        if not batch:
            break
        rows.extend({
            "timestamp": item[0] / 1000, "open": item[1], "high": item[2],
            "low": item[3], "close": item[4], "volume": item[5],
        } for item in batch)
        next_ms = int(batch[-1][0]) + INTERVAL_SECONDS * 1000
        if next_ms <= cursor_ms:
            break
        cursor_ms = next_ms
        time.sleep(0.03)
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    for column in KLINE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna()


def load_candles(pair: str, start_ts: int, end_ts: int, cache_dir: Path, allow_download: bool) -> pd.DataFrame:
    path = cache_path(cache_dir, pair)
    cached = read_cache(path)
    chunks = [cached]
    if cached.empty:
        missing = [(start_ts, end_ts)]
    else:
        missing = []
        first = int(cached.timestamp.min())
        last = int(cached.timestamp.max()) + INTERVAL_SECONDS
        if start_ts < first:
            missing.append((start_ts, first))
        if last < end_ts:
            missing.append((last, end_ts))
    if missing and not allow_download:
        raise RuntimeError(f"Missing cached candles for {pair}: {missing}")
    session = requests.Session()
    for missing_start, missing_end in missing:
        chunks.append(download_klines(pair, missing_start, missing_end, session))
    merged = pd.concat(chunks, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    window = merged[(merged.timestamp >= start_ts) & (merged.timestamp < end_ts)].reset_index(drop=True)
    expected = max(1, math.floor((end_ts - start_ts) / INTERVAL_SECONDS))
    if len(window) < expected * 0.98:
        raise RuntimeError(f"{pair} candle coverage is only {len(window)}/{expected}.")
    return window


def public_market_state(pairs: Iterable[str]) -> tuple[Dict[str, float], Dict[str, Mapping]]:
    prices, symbols = {}, {}
    session = requests.Session()
    for pair in pairs:
        symbol = pair.replace("-", "")
        info = request_json(session, "/api/v3/exchangeInfo", {"symbol": symbol})
        ticker = request_json(session, "/api/v3/ticker/price", {"symbol": symbol})
        symbols[pair] = info["symbols"][0]
        prices[pair] = float(ticker["price"])
        validate_exchange_filters(symbols[pair], budget_for_pair(pair).side_budget / 5)
    return prices, symbols


def _levels(center: float, candidate: Candidate) -> tuple[float, float, list[float]]:
    lower, upper = center * (1 - candidate.half_range), center * (1 + candidate.half_range)
    step = (upper - lower) / (candidate.levels - 1)
    return lower, upper, [lower + step * index for index in range(candidate.levels)]


def candle_arrays(frame: pd.DataFrame) -> dict[str, object]:
    key = id(frame)
    arrays = {column: frame[column].to_numpy(dtype=float, copy=False)
              for column in ("timestamp", "high", "low", "close")}
    signature = (
        len(frame),
        float(frame.timestamp.iloc[0]),
        float(frame.timestamp.iloc[-1]),
        *(int(array.ctypes.data) for array in arrays.values()),
    )
    entry = _ARRAY_CACHE.get(key)
    if entry is None or entry[0] != signature:
        _ARRAY_CACHE[key] = (signature, arrays)
        return arrays
    return entry[1]


def technical_buy_gate_timeline(
    frame: pd.DataFrame, *, roc_risk_off_pct: float = DEFAULT_ROC_RISK_OFF_PCT,
    sqzmom_risk_off_pct: float = DEFAULT_SQZMOM_RISK_OFF_PCT,
) -> Dict[int, bool]:
    """Build a no-lookahead 4h ROC/SQZMOM gate for each 5m timestamp."""
    source = frame.sort_values("timestamp").copy()
    source["bucket"] = (source.timestamp.astype("int64") // 14400) * 14400
    bars = source.groupby("bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), rows=("close", "size"),
    ).reset_index()
    bars = bars[bars.rows == 48]
    klines: list[list[Any]] = []
    transitions: list[tuple[int, bool]] = []
    active = False
    previous_bar_close_time = None
    previous_sqzmom_color = None
    for row in bars.itertuples(index=False):
        close_time = int(row.bucket) + 14_400 - 1
        klines.append([
            int(row.bucket) * 1000, row.open, row.high, row.low, row.close, 0,
            close_time * 1000,
        ])
        if len(klines) < 40:
            continue
        signal = roc_sqz_signal_from_klines(klines[-64:])
        gate = build_technical_buy_gate(
            signal,
            previously_active=active,
            previous_bar_close_time=previous_bar_close_time,
            previous_sqzmom_color=previous_sqzmom_color,
            roc_risk_off_pct=roc_risk_off_pct,
            sqzmom_risk_off_pct=sqzmom_risk_off_pct,
        )
        active = bool(gate["risk_off_active"])
        previous_bar_close_time = int(gate["last_evaluated_bar_close_time"])
        previous_sqzmom_color = str(gate["last_sqzmom_color"])
        transitions.append((close_time + 1, bool(gate["buy_enabled"])))
    timeline: Dict[int, bool] = {}
    pointer = 0
    enabled = False
    for raw_timestamp in source.timestamp:
        timestamp = int(raw_timestamp)
        while pointer < len(transitions) and transitions[pointer][0] <= timestamp:
            enabled = transitions[pointer][1]
            pointer += 1
        timeline[timestamp] = enabled
    return timeline


def simulate(candles: Dict[str, pd.DataFrame], candidate: Candidate, maker_fee: float,
             taker_fee: float | None = None, slippage: float = 0.0,
             order_refresh_seconds: int = ORDER_REFRESH_SECONDS,
             technical_buy_gate: Dict[int, bool] | Dict[str, Dict[int, bool]] | None = None,
             momentum_stop_timeline: Dict[str, Dict[int, float | bool]] | None = None,
             momentum_stop_threshold: float = 0.5,
             trade_log: list[dict] | None = None,
             risk_breakers_enabled: bool = True,
             cost_floor_enabled: bool = True,
             inventory_exit_policy: InventoryExitPolicy | None = None,
             pair_breaker_policy: PairBreakerPolicy | None = None,
             risk_mechanisms: RiskMechanismConfig | None = None,
             execution_filters: Mapping[str, ExecutionFilter] | None = None,
             record_curve: bool = True,
             ) -> tuple[dict, pd.DataFrame, dict]:
    if order_refresh_seconds <= 0:
        raise ValueError("order_refresh_seconds must be positive")
    if not 0 <= momentum_stop_threshold <= 1:
        raise ValueError("momentum_stop_threshold must be between zero and one")
    taker_fee = maker_fee if taker_fee is None else taker_fee
    pairs = list(candles)
    quotes = {pair.rsplit("-", 1)[-1].upper() for pair in pairs}
    if len(quotes) != 1:
        raise ValueError("A simulation must contain pairs with one quote asset.")
    budget = budget_for_quote(quotes.pop())
    if pair_breaker_policy is not None:
        unknown_pairs = set(pair_breaker_policy.pairs) - set(pairs)
        if unknown_pairs:
            raise ValueError(f"pair breaker policy contains unknown pairs: {sorted(unknown_pairs)}")
    if pair_breaker_policy is not None and risk_mechanisms is not None:
        raise ValueError("pair_breaker_policy and risk_mechanisms cannot be combined")
    mechanisms = risk_mechanisms or RiskMechanismConfig(
        pair_loss=risk_breakers_enabled,
        pair_drawdown=risk_breakers_enabled,
        portfolio_loss=risk_breakers_enabled,
        portfolio_drawdown=risk_breakers_enabled,
    )
    if execution_filters is not None:
        missing_filters = set(pairs) - set(execution_filters)
        if missing_filters:
            raise ValueError(f"execution filters missing pairs: {sorted(missing_filters)}")
    arrays = {pair: candle_arrays(frame) for pair, frame in candles.items()}
    steps = min(len(frame) for frame in candles.values())
    technical_gate_is_pair_mapping = bool(
        technical_buy_gate is not None
        and any(isinstance(value, Mapping) for value in technical_buy_gate.values())
    )
    states = {}
    for pair, frame in candles.items():
        price = float(arrays[pair]["close"][0])
        lower, upper, levels = _levels(price, candidate)
        states[pair] = {
            "quote": float(budget.side_budget), "base": float(budget.side_budget) / price,
            "base_cost_quote": float(budget.side_budget),
            "initial_base": float(budget.side_budget) / price, "lower": lower, "upper": upper,
            "levels": levels, "last_move": 0.0, "moves": 0, "buys": 0, "sells": 0,
            "fees": 0.0, "halted": False, "liquidations": 0,
            "excess_inventory_started_at": None, "forced_inventory_exits": 0,
            "max_extra_inventory_quote_observed": 0.0,
            "peak_equity": float(budget.pair_budget),
            "loss_reference_equity": float(budget.pair_budget),
            "max_drawdown_pct": 0.0, "halted_bars": 0,
            "halted_until": None, "cooldown_expiry_logged": False,
            "next_recovery_check": None,
            "technical_enabled_previous": True, "technical_risk_off_bars": 0,
            "momentum_risk_off_previous": False, "momentum_risk_off_bars": 0,
            "momentum_stop_exits": 0,
            "next_order_refresh": 0.0, "active_buys": [], "active_sells": [],
        }
    rows = []
    peak = float(budget.capital_limit)
    max_drawdown = 0.0
    portfolio_liquidated = False
    last_processed_index = 0
    for index in range(1, steps):
        last_processed_index = index
        now = float(arrays[pairs[0]]["timestamp"][index])
        for pair, state in states.items():
            highs = arrays[pair]["high"]
            lows = arrays[pair]["low"]
            reference = float(arrays[pair]["close"][index - 1])
            current_price = float(arrays[pair]["close"][index])
            if state["halted"]:
                state["halted_bars"] += 1
                policy_applies = (
                    pair_breaker_policy is not None
                    and pair in pair_breaker_policy.pairs
                )
                if (
                    policy_applies
                    and now >= float(state["halted_until"])
                    and now >= float(state["next_recovery_check"])
                ):
                    current_equity = state["quote"] + state["base"] * current_price
                    episode_loss = current_equity - state["loss_reference_equity"]
                    episode_drawdown = (
                        (state["peak_equity"] - current_equity) / state["peak_equity"]
                        if state["peak_equity"] > 0 else 0.0
                    )
                    if not state["cooldown_expiry_logged"] and trade_log is not None:
                        trade_log.append({
                            "timestamp": int(now), "pair": pair, "side": "PAUSE",
                            "price": current_price, "amount": 0.0, "quote_notional": 0.0,
                            "reason": "pair_breaker_cooldown_expired",
                            "trigger": f"pair_{pair_breaker_policy.trigger}",
                            "reset_baseline": pair_breaker_policy.reset_baseline,
                        })
                    state["cooldown_expiry_logged"] = True
                    safe = (
                        episode_loss > -float(budget.pair_loss_limit)
                        if pair_breaker_policy.trigger == "loss"
                        else episode_drawdown < float(PAIR_DRAWDOWN_LIMIT_PCT)
                    )
                    if pair_breaker_policy.reset_baseline or safe:
                        if pair_breaker_policy.reset_baseline:
                            if pair_breaker_policy.trigger == "loss":
                                state["loss_reference_equity"] = current_equity
                            else:
                                state["peak_equity"] = current_equity
                        state["halted"] = False
                        state["halted_until"] = None
                        state["next_recovery_check"] = None
                        state["cooldown_expiry_logged"] = False
                        state["next_order_refresh"] = 0.0
                        if trade_log is not None:
                            trade_log.append({
                                "timestamp": int(now), "pair": pair, "side": "RESUME",
                                "price": current_price, "amount": 0.0, "quote_notional": 0.0,
                                "reason": "pair_breaker_recovered",
                                "trigger": f"pair_{pair_breaker_policy.trigger}",
                                "reset_baseline": pair_breaker_policy.reset_baseline,
                                "original_pnl_quote": current_equity - float(budget.pair_budget),
                            })
                    else:
                        state["next_recovery_check"] = now + 24 * 60 * 60
                if state["halted"]:
                    continue
            if technical_buy_gate is None:
                technical_enabled = True
            elif pair in technical_buy_gate and isinstance(technical_buy_gate[pair], Mapping):
                technical_enabled = bool(technical_buy_gate[pair].get(int(now), False))
            elif technical_gate_is_pair_mapping:
                technical_enabled = True
            else:
                technical_enabled = bool(technical_buy_gate.get(int(now), False))
            if not technical_enabled:
                state["technical_risk_off_bars"] += 1
                state["active_buys"] = []
            elif not state["technical_enabled_previous"]:
                state["next_order_refresh"] = 0.0
            state["technical_enabled_previous"] = technical_enabled
            momentum_score = 0.0
            if momentum_stop_timeline is not None:
                pair_timeline = momentum_stop_timeline.get(pair, {})
                raw_score = pair_timeline.get(int(now), 0.0)
                momentum_score = float(raw_score)
            momentum_risk_off = momentum_score >= momentum_stop_threshold
            if momentum_risk_off:
                state["momentum_risk_off_bars"] += 1
                state["active_buys"] = []
                delta = max(state["base"] - state["initial_base"], 0.0)
                if delta > 1e-12:
                    execution = reference * (1 - slippage)
                    proceeds = delta * execution
                    if state["base"] > 0:
                        state["base_cost_quote"] *= (state["base"] - delta) / state["base"]
                    state["base"] -= delta
                    state["quote"] += proceeds * (1 - taker_fee)
                    state["fees"] += proceeds * taker_fee
                    state["sells"] += 1
                    state["forced_inventory_exits"] += 1
                    state["momentum_stop_exits"] += 1
                    if trade_log is not None:
                        trade_log.append({
                            "timestamp": int(now), "pair": pair, "side": "SELL",
                            "price": execution, "amount": delta,
                            "quote_notional": proceeds, "reason": "momentum_stop_exit",
                            "momentum_score": momentum_score,
                            "momentum_stop_threshold": momentum_stop_threshold,
                        })
                    update_excess_inventory_timer(state, now)
                if not state["momentum_risk_off_previous"] and trade_log is not None:
                    trade_log.append({
                        "timestamp": int(now), "pair": pair, "side": "PAUSE",
                        "price": reference, "amount": 0.0, "quote_notional": 0.0,
                        "reason": "momentum_stop_active",
                        "momentum_score": momentum_score,
                        "momentum_stop_threshold": momentum_stop_threshold,
                    })
            elif state["momentum_risk_off_previous"]:
                state["next_order_refresh"] = 0.0
                if trade_log is not None:
                    trade_log.append({
                        "timestamp": int(now), "pair": pair, "side": "RESUME",
                        "price": reference, "amount": 0.0, "quote_notional": 0.0,
                        "reason": "momentum_stop_recovered",
                        "momentum_score": momentum_score,
                        "momentum_stop_threshold": momentum_stop_threshold,
                    })
            state["momentum_risk_off_previous"] = momentum_risk_off
            update_excess_inventory_timer(state, now)
            inventory_age = (
                now - state["excess_inventory_started_at"]
                if state["excess_inventory_started_at"] is not None else 0.0
            )
            if (
                inventory_exit_policy is not None
                and inventory_age >= inventory_exit_policy.max_hold_seconds
            ):
                delta = max(state["base"] - state["initial_base"], 0.0)
                if delta > 1e-12:
                    execution = reference * (1 - slippage)
                    proceeds = delta * execution
                    if state["base"] > 0:
                        state["base_cost_quote"] *= (state["base"] - delta) / state["base"]
                    state["base"] -= delta
                    state["quote"] += proceeds * (1 - taker_fee)
                    state["fees"] += proceeds * taker_fee
                    state["sells"] += 1
                    state["forced_inventory_exits"] += 1
                    if trade_log is not None:
                        trade_log.append({
                            "timestamp": int(now), "pair": pair, "side": "SELL",
                            "price": execution, "amount": delta,
                            "quote_notional": proceeds, "reason": "max_hold_exit",
                        })
                state["active_buys"], state["active_sells"] = [], []
                state["next_order_refresh"] = now + order_refresh_seconds
                update_excess_inventory_timer(state, now)
                continue
            if now >= state["next_order_refresh"]:
                if now - state["last_move"] >= candidate.move_cooldown_seconds and (
                    reference > state["upper"] * (1 + candidate.move_threshold)
                    or reference < state["lower"] * (1 - candidate.move_threshold)
                ):
                    state["lower"], state["upper"], state["levels"] = _levels(reference, candidate)
                    state["last_move"], state["moves"] = now, state["moves"] + 1
                lower = [value for value in state["levels"] if value < reference]
                upper = [value for value in state["levels"] if value > reference]
                buy_budget = min(max(state["quote"], 0.0), float(budget.side_budget))
                if inventory_exit_policy is not None:
                    maximum_base = state["initial_base"] + (
                        inventory_exit_policy.max_extra_inventory_quote / reference
                    )
                    inventory_capacity = max((maximum_base - state["base"]) * reference, 0.0)
                    buy_budget = min(buy_budget, inventory_capacity)
                buy_quote = buy_budget / max(len(lower), 1)
                if execution_filters is None:
                    state["active_buys"] = [
                        {"price": level, "quote": buy_quote}
                        for level in lower
                        if technical_enabled and not momentum_risk_off
                        and buy_quote >= float(MIN_ORDER_QUOTE)
                    ]
                else:
                    filters = execution_filters[pair]
                    active_buys = []
                    for level in lower:
                        price = _quantize_down(level, filters.tick_size)
                        amount = _quantize_down(buy_quote / price, filters.step_size)
                        quote = amount * price
                        if (
                            technical_enabled and not momentum_risk_off and amount > 0
                            and quote >= max(float(MIN_ORDER_QUOTE), filters.min_notional)
                        ):
                            active_buys.append({"price": price, "quote": quote, "amount": amount})
                    state["active_buys"] = active_buys
                sell_budget = min(max(state["base"], 0.0), state["initial_base"])
                sell_amount = sell_budget / max(len(upper), 1)
                target_floor = reference * (
                    1 + max(candidate.take_profit, 2 * maker_fee + 0.004)
                )
                average_cost = (
                    state["base_cost_quote"] / state["base"]
                    if state["base"] > 0 else 0.0
                )
                base_profit_rate = max(candidate.take_profit, 2 * maker_fee + 0.004)
                inventory_profit_rate = inventory_profit_floor_rate(
                    state, now, base_profit_rate, inventory_exit_policy,
                )
                cost_floor = average_cost * (
                    1 + inventory_profit_rate
                ) if cost_floor_enabled else 0.0
                active_sells = []
                for level in upper:
                    price = max(level, target_floor, cost_floor)
                    amount = sell_amount
                    minimum = float(MIN_ORDER_QUOTE)
                    if execution_filters is not None:
                        filters = execution_filters[pair]
                        price = _quantize_up(price, filters.tick_size)
                        amount = _quantize_down(amount, filters.step_size)
                        minimum = max(minimum, filters.min_notional)
                    if amount > 0 and amount * price >= minimum:
                        active_sells.append({
                            "price": price, "amount": amount,
                            "grid_level": level, "target_floor": target_floor,
                            "cost_floor": cost_floor, "average_cost": average_cost,
                        })
                state["active_sells"] = active_sells
                state["next_order_refresh"] = now + order_refresh_seconds

            remaining_buys = []
            for order in state["active_buys"]:
                execution = order["price"] * (1 + slippage)
                amount = float(order.get("amount", order["quote"] / execution))
                quote_notional = amount * execution
                cost = quote_notional * (1 + maker_fee)
                if float(lows[index]) <= order["price"] and state["quote"] >= cost:
                    state["quote"] -= cost
                    state["base"] += amount
                    state["base_cost_quote"] += cost
                    state["fees"] += quote_notional * maker_fee
                    state["buys"] += 1
                    if trade_log is not None:
                        trade_log.append({
                            "timestamp": int(now), "pair": pair, "side": "BUY",
                            "price": execution, "amount": amount,
                            "quote_notional": quote_notional, "reason": "grid_fill",
                        })
                else:
                    remaining_buys.append(order)
            state["active_buys"] = remaining_buys

            remaining_sells = []
            for order in state["active_sells"]:
                amount = min(order["amount"], state["base"])
                if amount > 0 and float(highs[index]) >= order["price"]:
                    execution = order["price"] * (1 - slippage)
                    proceeds = amount * execution
                    if state["base"] > 0:
                        state["base_cost_quote"] *= max(state["base"] - amount, 0.0) / state["base"]
                    state["base"] -= amount
                    state["quote"] += proceeds * (1 - maker_fee)
                    state["fees"] += proceeds * maker_fee
                    state["sells"] += 1
                    if trade_log is not None:
                        trade_log.append({
                            "timestamp": int(now), "pair": pair, "side": "SELL",
                            "price": execution, "amount": amount,
                            "quote_notional": proceeds, "reason": "grid_fill",
                            "grid_level": order.get("grid_level"),
                            "target_floor": order.get("target_floor"),
                            "cost_floor": order.get("cost_floor"),
                            "average_cost": order.get("average_cost"),
                        })
                else:
                    remaining_sells.append(order)
            state["active_sells"] = remaining_sells
            update_excess_inventory_timer(state, now)
            state["max_extra_inventory_quote_observed"] = max(
                state["max_extra_inventory_quote_observed"],
                max(state["base"] - state["initial_base"], 0.0) * reference,
            )
        prices = {pair: float(arrays[pair]["close"][index]) for pair in pairs}
        pair_risk_snapshot = {}
        for pair, state in states.items():
            pair_equity = state["quote"] + state["base"] * prices[pair]
            state["peak_equity"] = max(state["peak_equity"], pair_equity)
            pair_drawdown = (
                (state["peak_equity"] - pair_equity) / state["peak_equity"]
                if state["peak_equity"] > 0 else 0.0
            )
            pair_pnl = pair_equity - float(budget.pair_budget)
            episode_pnl = pair_equity - state["loss_reference_equity"]
            pair_loss_tripped = episode_pnl <= -float(budget.pair_loss_limit)
            pair_drawdown_tripped = pair_drawdown >= float(PAIR_DRAWDOWN_LIMIT_PCT)
            state["max_drawdown_pct"] = min(
                state["max_drawdown_pct"], -pair_drawdown,
            )
            pair_risk_snapshot[pair] = {
                "pnl": pair_pnl,
                "drawdown_pct": -pair_drawdown * 100,
                "halted": bool(state["halted"]),
            }
            policy_applies = (
                pair_breaker_policy is not None
                and pair in pair_breaker_policy.pairs
            )
            policy_tripped = bool(
                policy_applies
                and (
                    pair_loss_tripped
                    if pair_breaker_policy.trigger == "loss"
                    else pair_drawdown_tripped
                )
            )
            legacy_tripped = bool(
                pair_breaker_policy is None
                and (
                    (mechanisms.pair_loss and pair_loss_tripped)
                    or (mechanisms.pair_drawdown and pair_drawdown_tripped)
                )
            )
            if (
                not state["halted"]
                and (policy_tripped or legacy_tripped)
            ):
                delta = state["base"] - state["initial_base"]
                notional = abs(delta) * prices[pair]
                state["fees"] += notional * taker_fee
                if delta > 0:
                    if state["base"] > 0:
                        state["base_cost_quote"] *= max(state["base"] - delta, 0.0) / state["base"]
                    state["quote"] += delta * prices[pair] * (1 - slippage) * (1 - taker_fee)
                elif delta < 0:
                    restore_cost = abs(delta) * prices[pair] * (1 + slippage) * (1 + taker_fee)
                    state["quote"] -= restore_cost
                    state["base_cost_quote"] += restore_cost
                if trade_log is not None:
                    trade_log.append({
                        "timestamp": int(now), "pair": pair,
                        "side": "SELL" if delta > 0 else "BUY",
                        "price": prices[pair], "amount": abs(delta),
                        "quote_notional": notional, "reason": "pair_breaker_flatten",
                        "trigger": (
                            f"pair_{pair_breaker_policy.trigger}"
                            if policy_applies
                            else "pair_loss"
                            if mechanisms.pair_loss and pair_loss_tripped
                            else "pair_drawdown"
                        ),
                        "trigger_pnl_quote": episode_pnl,
                        "original_pnl_quote": pair_pnl,
                        "trigger_drawdown_pct": -pair_drawdown * 100,
                        "cooldown_until": (
                            int(now + pair_breaker_policy.cooldown_seconds)
                            if policy_applies else None
                        ),
                        "reset_baseline": (
                            pair_breaker_policy.reset_baseline if policy_applies else False
                        ),
                    })
                state["base"] = state["initial_base"]
                state["halted"], state["liquidations"] = True, state["liquidations"] + 1
                state["halted_until"] = (
                    now + pair_breaker_policy.cooldown_seconds
                    if policy_applies else None
                )
                state["next_recovery_check"] = state["halted_until"]
                state["cooldown_expiry_logged"] = False
                state["active_buys"], state["active_sells"] = [], []
                pair_risk_snapshot[pair]["halted"] = True
        equity = float(budget.reserve_quote) + sum(
            state["quote"] + state["base"] * prices[pair] for pair, state in states.items()
        )
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak
        max_drawdown = min(max_drawdown, drawdown)
        curve_row = {
            "timestamp": now, "equity": equity, "peak_equity": peak,
            "drawdown_pct": drawdown, "portfolio_pnl_quote": equity - float(budget.capital_limit),
        }
        for pair, snapshot in pair_risk_snapshot.items():
            curve_row[f"{pair}_pnl_quote"] = snapshot["pnl"]
            curve_row[f"{pair}_drawdown_pct"] = snapshot["drawdown_pct"]
            curve_row[f"{pair}_halted"] = snapshot["halted"]
        if record_curve:
            rows.append(curve_row)
        portfolio_loss_tripped = (
            equity - float(budget.capital_limit) <= -float(budget.portfolio_loss_limit)
        )
        portfolio_drawdown_tripped = -drawdown >= float(PORTFOLIO_DRAWDOWN_LIMIT_PCT)
        if (
            not portfolio_liquidated
            and (
                (mechanisms.portfolio_loss and portfolio_loss_tripped)
                or (mechanisms.portfolio_drawdown and portfolio_drawdown_tripped)
            )
        ):
            portfolio_liquidated = True
            if trade_log is not None:
                trade_log.append({
                    "timestamp": int(now), "pair": "PORTFOLIO", "side": "STOP",
                    "price": 0.0, "amount": 0.0, "quote_notional": 0.0,
                    "reason": "portfolio_breaker",
                    "trigger": (
                        "portfolio_loss"
                        if mechanisms.portfolio_loss and portfolio_loss_tripped
                        else "portfolio_drawdown"
                    ),
                    "trigger_pnl_quote": equity - float(budget.capital_limit),
                    "trigger_drawdown_pct": drawdown * 100,
                })
            for pair, state in states.items():
                if mechanisms.restore_portfolio_inventory:
                    delta = state["base"] - state["initial_base"]
                    notional = abs(delta) * prices[pair]
                    state["fees"] += notional * taker_fee
                    if delta > 0:
                        if state["base"] > 0:
                            state["base_cost_quote"] *= max(
                                state["base"] - delta, 0.0
                            ) / state["base"]
                        state["quote"] += (
                            delta * prices[pair] * (1 - slippage) * (1 - taker_fee)
                        )
                    elif delta < 0:
                        restore_cost = (
                            abs(delta) * prices[pair] * (1 + slippage) * (1 + taker_fee)
                        )
                        state["quote"] -= restore_cost
                        state["base_cost_quote"] += restore_cost
                    if trade_log is not None and abs(delta) > 1e-12:
                        trade_log.append({
                            "timestamp": int(now), "pair": pair,
                            "side": "SELL" if delta > 0 else "BUY",
                            "price": prices[pair], "amount": abs(delta),
                            "quote_notional": notional,
                            "reason": "portfolio_breaker_flatten",
                            "trigger": (
                                "portfolio_loss"
                                if mechanisms.portfolio_loss and portfolio_loss_tripped
                                else "portfolio_drawdown"
                            ),
                        })
                    state["base"] = state["initial_base"]
                    state["active_buys"], state["active_sells"] = [], []
                state["halted"] = True
            if not mechanisms.continue_after_portfolio_stop:
                break
    last_index = last_processed_index
    final_prices = {pair: float(arrays[pair]["close"][last_index]) for pair in pairs}
    final_equity = float(budget.reserve_quote) + sum(
        state["quote"] + state["base"] * final_prices[pair] for pair, state in states.items()
    )
    per_pair = {
        pair: {
            "net_pnl_quote": state["quote"] + state["base"] * final_prices[pair] - float(budget.pair_budget),
            "fees_quote": state["fees"], "buys": state["buys"], "sells": state["sells"],
            "moves": state["moves"], "inventory_delta": state["base"] - state["initial_base"],
            "average_base_cost": (
                state["base_cost_quote"] / state["base"] if state["base"] > 0 else 0.0
            ),
            "liquidations": state["liquidations"],
            "forced_inventory_exits": state["forced_inventory_exits"],
            "max_extra_inventory_quote_observed": state["max_extra_inventory_quote_observed"],
            "max_drawdown_pct": state["max_drawdown_pct"],
            "halted_seconds": state["halted_bars"] * INTERVAL_SECONDS,
            "technical_risk_off_seconds": state["technical_risk_off_bars"] * INTERVAL_SECONDS,
            "momentum_risk_off_seconds": state["momentum_risk_off_bars"] * INTERVAL_SECONDS,
            "momentum_stop_exits": state["momentum_stop_exits"],
            "loss_reference_equity": state["loss_reference_equity"],
        } for pair, state in states.items()
    }
    result = {
        "initial_equity": float(budget.capital_limit), "final_equity": final_equity,
        "net_pnl_quote": final_equity - float(budget.capital_limit),
        "net_pnl_pct": (final_equity - float(budget.capital_limit)) / float(budget.capital_limit),
        "max_drawdown_pct": max_drawdown, "liquidated": portfolio_liquidated,
        "trades": sum(state["buys"] + state["sells"] for state in states.values()),
        "fees_quote": sum(state["fees"] for state in states.values()),
        "grid_moves": sum(state["moves"] for state in states.values()),
        "order_refresh_seconds": order_refresh_seconds,
        "cost_floor_enabled": cost_floor_enabled,
        "inventory_exit_policy": (
            asdict(inventory_exit_policy) if inventory_exit_policy is not None else None
        ),
        "forced_inventory_exits": sum(
            state["forced_inventory_exits"] for state in states.values()
        ),
        "risk_breakers_enabled": risk_breakers_enabled,
        "risk_mechanisms": asdict(mechanisms),
        "execution_filters": (
            {pair: asdict(value) for pair, value in execution_filters.items()}
            if execution_filters is not None else None
        ),
        "pair_breaker_policy": (
            asdict(pair_breaker_policy) if pair_breaker_policy is not None else None
        ),
        "technical_buy_gate_enabled": technical_buy_gate is not None,
        "momentum_stop_enabled": momentum_stop_timeline is not None,
        "momentum_stop_threshold": momentum_stop_threshold,
        "momentum_stop_exits": sum(
            state["momentum_stop_exits"] for state in states.values()
        ),
        "technical_risk_off_bars": max(
            (state["technical_risk_off_bars"] for state in states.values()), default=0,
        ),
    }
    return result, pd.DataFrame(rows), per_pair


def windows(start_ts: int) -> list[tuple[int, int, int, int]]:
    day = 86400
    return [(start_ts + offset * day, start_ts + (offset + 60) * day,
             start_ts + (offset + 60) * day, start_ts + (offset + 90) * day)
            for offset in (0, 30, 60, 90)]


def slice_window(candles: Dict[str, pd.DataFrame], start: int, end: int) -> Dict[str, pd.DataFrame]:
    return {pair: frame[(frame.timestamp >= start) & (frame.timestamp < end)].reset_index(drop=True)
            for pair, frame in candles.items()}


def validate_portfolio(portfolio_key: str, candles: Dict[str, pd.DataFrame], fee: float,
                       start_ts: int) -> tuple[pd.DataFrame, pd.DataFrame, Candidate, dict, pd.DataFrame, dict]:
    evaluation_rows, summary_rows = [], []
    latest_candidate = candidates()[0]
    latest_result, latest_curve, latest_pairs = {}, pd.DataFrame(), {}
    for fold, (train_start, train_end, test_start, test_end) in enumerate(windows(start_ts), 1):
        training, testing = slice_window(candles, train_start, train_end), slice_window(candles, test_start, test_end)
        scored = []
        for candidate in candidates():
            result, _, _ = simulate(training, candidate, fee)
            score = result["net_pnl_pct"] - 1.5 * abs(result["max_drawdown_pct"])
            if result["liquidated"]:
                score -= 0.25
            row = {"portfolio": portfolio_key, "fold": fold, **asdict(candidate), "levels": candidate.levels,
                   "train_score": score, **{f"train_{key}": value for key, value in result.items()}}
            evaluation_rows.append(row)
            scored.append((score, not result["liquidated"], candidate))
        eligible = [item for item in scored if item[1]]
        _, _, selected = max(eligible or scored, key=lambda item: item[0])
        result, curve, pair_stats = simulate(testing, selected, fee)
        summary_rows.append({"portfolio": portfolio_key, "fold": fold, "train_start": train_start,
                             "train_end": train_end, "test_start": test_start, "test_end": test_end,
                             **asdict(selected), "levels": selected.levels, **result})
        latest_candidate, latest_result, latest_curve, latest_pairs = selected, result, curve, pair_stats
    return (pd.DataFrame(evaluation_rows), pd.DataFrame(summary_rows), latest_candidate,
            latest_result, latest_curve, latest_pairs)


def crash_candles(candles: Dict[str, pd.DataFrame], drop: float = 0.15) -> Dict[str, pd.DataFrame]:
    """Apply a linear 15% decline to the final day without changing timestamps."""
    stressed = {}
    for pair, frame in candles.items():
        copy = frame.copy()
        count = min(288, len(copy))
        multipliers = [1 - drop * (index + 1) / count for index in range(count)]
        for column in ("open", "high", "low", "close"):
            copy.loc[copy.index[-count:], column] = copy.loc[copy.index[-count:], column].to_numpy() * multipliers
        stressed[pair] = copy
    return stressed


def html_report(path: Path, payload: dict, summaries: pd.DataFrame, stress: pd.DataFrame,
                pair_rows: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, border=0, classes="data") if not frame.empty else "<p>No data</p>"
    cards = "".join(
        f"<div class='metric'><span>{key}</span><strong>{value}</strong></div>"
        for key, value in payload["overview"].items()
    )
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>BTC/ETH Live Grid Validation</title><style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f6f8fa;color:#17202a}}main{{max-width:1320px;margin:auto}}
h1,h2{{letter-spacing:0}}.banner{{padding:14px;border-left:5px solid #b54708;background:#fff4e5}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin:18px 0}}
.metric{{background:white;border:1px solid #d9e0e7;border-radius:6px;padding:12px}}.metric span{{display:block;color:#59636e}}
.metric strong{{font-size:20px}}section{{margin:20px 0}}.table-wrap{{overflow:auto;background:white;border:1px solid #d9e0e7}}
table.data{{border-collapse:collapse;width:100%;font-size:12px}}.data th,.data td{{padding:7px;border-bottom:1px solid #e5e9ed;text-align:right;white-space:nowrap}}
.data th{{background:#eef2f5;position:sticky;top:0}}code{{background:#eef2f5;padding:2px 5px}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><h1>BTC/ETH 双组合 Grid 实盘验证</h1>
<div class='banner'><strong>{payload['decision']}</strong><p>{payload['decision_reason']}</p></div>
<div class='metrics'>{cards}</div><section><h2>滚动样本外结果</h2><div class='table-wrap'>{table(summaries)}</div></section>
<section><h2>逐交易对指标</h2><div class='table-wrap'>{table(pair_rows)}</div></section>
<section><h2>压力测试</h2><div class='table-wrap'>{table(stress)}</div></section>
<section><h2>上线门禁</h2><ul><li>实盘开关保持 <code>false</code></li><li>私有账户费率、余额、签名和 test-order 尚待验证</li>
<li>不得修改或复用线上 paper 实例目录</li><li>USDT 通过 24 小时 canary 后才允许 FDUSD</li></ul></section>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=182)
    parser.add_argument("--end-ts", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/backtests/grid_live_validation_500"))
    parser.add_argument("--usdt-maker-fee", type=float, default=0.001)
    parser.add_argument("--fdusd-maker-fee", type=float, default=0.0)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    end_ts = int(args.end_ts or time.time()) // INTERVAL_SECONDS * INTERVAL_SECONDS
    start_ts = end_ts - args.days * 86400
    pairs = [pair for portfolio in PORTFOLIOS.values() for pair in portfolio.pairs]
    market_prices, symbols = public_market_state(pairs)
    candle_data = {pair: load_candles(pair, start_ts, end_ts, args.cache_dir, not args.no_download) for pair in pairs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_evaluations, all_summaries, all_stress, all_pairs = [], [], [], []
    recommendations = {}
    config_records = {}
    for key, portfolio in PORTFOLIOS.items():
        fee = args.usdt_maker_fee if key == "USDT" else args.fdusd_maker_fee
        subset = {pair: candle_data[pair] for pair in portfolio.pairs}
        evaluations, summaries, selected, result, curve, pair_stats = validate_portfolio(key, subset, fee, start_ts)
        all_evaluations.append(evaluations)
        all_summaries.append(summaries)
        recommendations[key] = {**asdict(selected), "levels": selected.levels, "latest_oos": result}
        curve.to_csv(args.output_dir / f"{key.lower()}_latest_oos_equity.csv", index=False)
        for pair, metrics in pair_stats.items():
            all_pairs.append({"portfolio": key, "pair": pair, **metrics})
        for label, fee_multiplier, slippage in (("base", 1, 0), ("fee_150pct", 1.5, 0),
                                                ("slippage_005pct", 1, 0.0005),
                                                ("slippage_010pct", 1, 0.001)):
            stress_result, _, _ = simulate(subset, selected, fee * fee_multiplier, slippage=slippage)
            all_stress.append({"portfolio": key, "scenario": label, **stress_result})
        crash_result, _, _ = simulate(crash_candles(subset), selected, fee, slippage=0.001)
        all_stress.append({"portfolio": key, "scenario": "15pct_one_day_drop", **crash_result})
        config = build_live_config(portfolio, {pair: Decimal(str(market_prices[pair])) for pair in portfolio.pairs},
                                   Decimal(str(fee)), trading_enabled=False)
        config["grid_range"] = selected.half_range * 2
        config["grid_levels"] = selected.levels
        config["take_profit"] = float(effective_take_profit(Decimal(str(fee)), Decimal(str(selected.take_profit))))
        config["move_threshold"] = selected.move_threshold
        validate_live_config(config)
        config_text = yaml.safe_dump(config, sort_keys=False)
        config_path = args.output_dir / portfolio.config_name
        config_path.write_text(config_text, encoding="utf-8")
        config_records[key] = {"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    evaluations_df, summaries_df = pd.concat(all_evaluations), pd.concat(all_summaries)
    stress_df, pair_df = pd.DataFrame(all_stress), pd.DataFrame(all_pairs)
    evaluations_df.to_csv(args.output_dir / "candidate_evaluations.csv", index=False)
    summaries_df.to_csv(args.output_dir / "walk_forward_summary.csv", index=False)
    stress_df.to_csv(args.output_dir / "stress_tests.csv", index=False)
    pair_df.to_csv(args.output_dir / "pair_metrics.csv", index=False)
    pair_ok = pair_df.net_pnl_quote.min() >= -float(PAIR_BUDGET) * 0.03
    oos_ok = (summaries_df.net_pnl_quote.sum() > 0 and summaries_df.max_drawdown_pct.min() >= -0.06
              and not summaries_df.liquidated.any() and pair_ok)
    stress_ok = not stress_df.liquidated.any()
    decision = "CONDITIONAL GO" if oos_ok and stress_ok else "NO-GO"
    reason = ("公共行情、规则与离线风险门槛通过；仍需私有账户费率、余额、签名及test-order预检。"
              if decision == "CONDITIONAL GO" else "离线收益或风险门槛未通过，不应进入实盘预检。")
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(), "decision": decision, "decision_reason": reason,
        "period": {"start_ts": start_ts, "end_ts": end_ts}, "recommendations": recommendations,
        "required_balances": {key: str(value) for key, value in required_balances({pair: Decimal(str(price)) for pair, price in market_prices.items()}).items()},
        "private_preflight_complete": False,
        "overview": {"Decision": decision, "OOS PnL": f"{summaries_df.net_pnl_quote.sum():+.2f}",
                     "Worst DD": f"{summaries_df.max_drawdown_pct.min():.2%}",
                     "Stress liquidations": int(stress_df.liquidated.sum())},
    }
    (args.output_dir / "validation_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reservations = {
        "version": 1, "generated_at": payload["generated_at"], "deployment_allowed": False,
        "trading_enabled": False, "shared_main_account": True,
        "profiles": {key: portfolio.profile_name for key, portfolio in PORTFOLIOS.items()},
        "bots": {key: portfolio.bot_name for key, portfolio in PORTFOLIOS.items()},
        "configs": config_records,
        "prices": {pair: str(price) for pair, price in market_prices.items()},
        "global_required_balances": payload["required_balances"],
        "reservations": {
            key: {"quote": str(SIDE_BUDGET * 2 + RESERVE_QUOTE),
                  "base": {pair.split("-")[0]: str(SIDE_BUDGET / Decimal(str(market_prices[pair])))
                           for pair in portfolio.pairs}}
            for key, portfolio in PORTFOLIOS.items()
        },
        "validation_decision": decision, "private_preflight_complete": False,
    }
    (args.output_dir / "capital_reservations.json").write_text(json.dumps(reservations, indent=2), encoding="utf-8")
    (args.output_dir / "validation_result.md").write_text(
        f"# Live Grid Validation\n\n- decision: **{decision}**\n- reason: {reason}\n"
        f"- OOS PnL: {summaries_df.net_pnl_quote.sum():+.2f}\n- worst drawdown: {summaries_df.max_drawdown_pct.min():.2%}\n"
        "- live deployment: disabled\n- private account preflight: pending\n", encoding="utf-8")
    html_report(args.output_dir / "report.html", payload, summaries_df, stress_df, pair_df)
    print(json.dumps(payload, indent=2))
    return 0 if decision == "CONDITIONAL GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
