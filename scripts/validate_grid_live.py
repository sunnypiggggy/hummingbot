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
from typing import Dict, Iterable, Mapping

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
from grid_technical_gate import build_technical_buy_gate, roc_sqz_signal_from_klines


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
    frame: pd.DataFrame, *, roc_risk_off_pct: float = -8.0,
    sqzmom_risk_off_pct: float = -3.0,
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
             technical_buy_gate: Dict[int, bool] | None = None,
             trade_log: list[dict] | None = None,
             risk_breakers_enabled: bool = True,
             ) -> tuple[dict, pd.DataFrame, dict]:
    if order_refresh_seconds <= 0:
        raise ValueError("order_refresh_seconds must be positive")
    taker_fee = maker_fee if taker_fee is None else taker_fee
    pairs = list(candles)
    quotes = {pair.rsplit("-", 1)[-1].upper() for pair in pairs}
    if len(quotes) != 1:
        raise ValueError("A simulation must contain pairs with one quote asset.")
    budget = budget_for_quote(quotes.pop())
    arrays = {pair: candle_arrays(frame) for pair, frame in candles.items()}
    steps = min(len(frame) for frame in candles.values())
    states = {}
    for pair, frame in candles.items():
        price = float(arrays[pair]["close"][0])
        lower, upper, levels = _levels(price, candidate)
        states[pair] = {
            "quote": float(budget.side_budget), "base": float(budget.side_budget) / price,
            "initial_base": float(budget.side_budget) / price, "lower": lower, "upper": upper,
            "levels": levels, "last_move": 0.0, "moves": 0, "buys": 0, "sells": 0,
            "fees": 0.0, "halted": False, "liquidations": 0,
            "peak_equity": float(budget.pair_budget),
            "next_order_refresh": 0.0, "active_buys": [], "active_sells": [],
        }
    rows = []
    peak = float(budget.capital_limit)
    max_drawdown = 0.0
    portfolio_liquidated = False
    technical_enabled_previous = True
    technical_risk_off_bars = 0
    for index in range(1, steps):
        now = float(arrays[pairs[0]]["timestamp"][index])
        technical_enabled = (
            True if technical_buy_gate is None
            else bool(technical_buy_gate.get(int(now), False))
        )
        if not technical_enabled:
            technical_risk_off_bars += 1
            for state in states.values():
                state["active_buys"] = []
        elif not technical_enabled_previous:
            for state in states.values():
                state["next_order_refresh"] = 0.0
        technical_enabled_previous = technical_enabled
        for pair, state in states.items():
            if state["halted"]:
                continue
            highs = arrays[pair]["high"]
            lows = arrays[pair]["low"]
            reference = float(arrays[pair]["close"][index - 1])
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
                buy_quote = buy_budget / max(len(lower), 1)
                state["active_buys"] = [
                    {"price": level, "quote": buy_quote}
                    for level in lower
                    if technical_enabled and buy_quote >= float(MIN_ORDER_QUOTE)
                ]
                sell_budget = min(max(state["base"], 0.0), state["initial_base"])
                sell_amount = sell_budget / max(len(upper), 1)
                target_floor = reference * (
                    1 + max(candidate.take_profit, 2 * maker_fee + 0.004)
                )
                state["active_sells"] = [
                    {"price": max(level, target_floor), "amount": sell_amount}
                    for level in upper
                    if sell_amount > 0
                    and sell_amount * max(level, target_floor) >= float(MIN_ORDER_QUOTE)
                ]
                state["next_order_refresh"] = now + order_refresh_seconds

            remaining_buys = []
            for order in state["active_buys"]:
                execution = order["price"] * (1 + slippage)
                cost = order["quote"] * (1 + maker_fee)
                if float(lows[index]) <= order["price"] and state["quote"] >= cost:
                    state["quote"] -= cost
                    amount = order["quote"] / execution
                    state["base"] += amount
                    state["fees"] += order["quote"] * maker_fee
                    state["buys"] += 1
                    if trade_log is not None:
                        trade_log.append({
                            "timestamp": int(now), "pair": pair, "side": "BUY",
                            "price": execution, "amount": amount,
                            "quote_notional": order["quote"], "reason": "grid_fill",
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
                    state["base"] -= amount
                    state["quote"] += proceeds * (1 - maker_fee)
                    state["fees"] += proceeds * maker_fee
                    state["sells"] += 1
                    if trade_log is not None:
                        trade_log.append({
                            "timestamp": int(now), "pair": pair, "side": "SELL",
                            "price": execution, "amount": amount,
                            "quote_notional": proceeds, "reason": "grid_fill",
                        })
                else:
                    remaining_sells.append(order)
            state["active_sells"] = remaining_sells
        prices = {pair: float(arrays[pair]["close"][index]) for pair in pairs}
        for pair, state in states.items():
            pair_equity = state["quote"] + state["base"] * prices[pair]
            state["peak_equity"] = max(state["peak_equity"], pair_equity)
            pair_drawdown = (
                (state["peak_equity"] - pair_equity) / state["peak_equity"]
                if state["peak_equity"] > 0 else 0.0
            )
            if (
                risk_breakers_enabled
                and not state["halted"]
                and pair_drawdown >= float(PAIR_DRAWDOWN_LIMIT_PCT)
            ):
                delta = state["base"] - state["initial_base"]
                notional = abs(delta) * prices[pair]
                state["fees"] += notional * taker_fee
                if delta > 0:
                    state["quote"] += delta * prices[pair] * (1 - slippage) * (1 - taker_fee)
                elif delta < 0:
                    state["quote"] -= abs(delta) * prices[pair] * (1 + slippage) * (1 + taker_fee)
                if trade_log is not None and notional > 0:
                    trade_log.append({
                        "timestamp": int(now), "pair": pair,
                        "side": "SELL" if delta > 0 else "BUY",
                        "price": prices[pair], "amount": abs(delta),
                        "quote_notional": notional, "reason": "pair_breaker_flatten",
                    })
                state["base"] = state["initial_base"]
                state["halted"], state["liquidations"] = True, state["liquidations"] + 1
                state["active_buys"], state["active_sells"] = [], []
        equity = float(budget.reserve_quote) + sum(
            state["quote"] + state["base"] * prices[pair] for pair, state in states.items()
        )
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak
        max_drawdown = min(max_drawdown, drawdown)
        rows.append({"timestamp": now, "equity": equity, "peak_equity": peak, "drawdown_pct": drawdown})
        if (
            risk_breakers_enabled
            and not portfolio_liquidated
            and -drawdown >= float(PORTFOLIO_DRAWDOWN_LIMIT_PCT)
        ):
            portfolio_liquidated = True
            for pair, state in states.items():
                state["halted"] = True
            break
    last_index = min(len(rows), steps - 1)
    final_prices = {pair: float(arrays[pair]["close"][last_index]) for pair in pairs}
    final_equity = float(budget.reserve_quote) + sum(
        state["quote"] + state["base"] * final_prices[pair] for pair, state in states.items()
    )
    per_pair = {
        pair: {
            "net_pnl_quote": state["quote"] + state["base"] * final_prices[pair] - float(budget.pair_budget),
            "fees_quote": state["fees"], "buys": state["buys"], "sells": state["sells"],
            "moves": state["moves"], "inventory_delta": state["base"] - state["initial_base"],
            "liquidations": state["liquidations"],
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
        "risk_breakers_enabled": risk_breakers_enabled,
        "technical_buy_gate_enabled": technical_buy_gate is not None,
        "technical_risk_off_bars": technical_risk_off_bars,
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
