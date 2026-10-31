"""Offline, cash-backed spot simulations for the strategy comparison suite.

Minute candles are labelled by their UTC opening time. Aggregated candles and
signals are labelled by the *following* boundary, when all inputs are known.
No connector, credential, trading configuration, or live state is accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import talib


STRATEGY_IDS = ("donchian", "cluc", "rsi2", "macd_ema")
OHLCV = ["open", "high", "low", "close", "volume"]
FILL_COLUMNS = ["timestamp", "symbol", "side", "qty", "price", "fee", "reason"]
TRADE_COLUMNS = [
    "entry_time", "exit_time", "symbol", "qty", "entry_price", "exit_price",
    "pnl", "return_pct", "holding_minutes", "exit_reason",
]


def _utc_time(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Candles require a timezone-aware DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Candle timestamps must be unique and increasing")
    missing = set(OHLCV) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    result = frame.loc[:, OHLCV].astype(float).copy()
    result.index = result.index.tz_convert("UTC")
    values = result.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Candles contain non-finite values")
    if not result.empty and (
        (result[["open", "high", "low", "close"]] <= 0).any().any()
        or (result.volume < 0).any()
        or (result.high < result[["open", "close", "low"]].max(axis=1)).any()
        or (result.low > result[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise ValueError("Invalid OHLCV candle")
    return result


def aggregate_candles(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate complete UTC buckets; gaps or partial edge buckets are omitted."""
    if not isinstance(minutes, int) or minutes < 1:
        raise ValueError("minutes must be a positive integer")
    candles = _frame(frame)
    if (candles.index != candles.index.floor("min")).any():
        raise ValueError("Minute candles must be aligned to minute opens")
    rule = f"{minutes}min"
    groups = candles.resample(rule, origin="epoch", closed="left", label="right")
    result = groups.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    # Unique, minute-aligned timestamps plus an exact bucket count establish
    # continuity within each bucket, without assuming missing minutes exist.
    result = result.loc[groups.size().eq(minutes)].dropna(subset=OHLCV)
    result.index.name = "timestamp"
    return result


def _talib(function, series: pd.Series, **kwargs) -> pd.Series:
    return pd.Series(function(series.to_numpy(dtype=float), **kwargs), index=series.index)


def generate_signals(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Evaluate the frozen rules on already aggregated close-boundary bars."""
    if strategy not in STRATEGY_IDS:
        raise ValueError(f"Unknown strategy: {strategy}")
    candles = _frame(frame)
    close = candles.close
    if candles.empty:
        return pd.DataFrame({"enter": pd.Series(dtype=bool), "exit": pd.Series(dtype=bool)}, index=candles.index)
    if strategy == "donchian":
        upper = candles.high.rolling(20, min_periods=20).max().shift(1)
        lower = candles.low.rolling(20, min_periods=20).min().shift(1)
        slow = _talib(talib.SMA, close, timeperiod=200)
        enter = (close > upper) & (close > slow)
        leave = close < lower
    elif strategy == "cluc":
        typical = (candles.high + candles.low + close) / 3.0
        middle = typical.rolling(20, min_periods=20).mean()
        lower = middle - 2.0 * typical.rolling(20, min_periods=20).std(ddof=1)
        ema = _talib(talib.EMA, close, timeperiod=50)
        average_volume = candles.volume.rolling(30, min_periods=30).mean().shift(1)
        enter = (close < ema) & (close < 0.985 * lower) & (candles.volume < 20.0 * average_volume)
        leave = close > middle
    elif strategy == "rsi2":
        slow = _talib(talib.SMA, close, timeperiod=200)
        fast = _talib(talib.SMA, close, timeperiod=5)
        rsi = _talib(talib.RSI, close, timeperiod=2)
        enter = (close > slow) & (rsi <= 10.0)
        leave = close > fast
    else:
        ema = _talib(talib.EMA, close, timeperiod=100)
        main, signal, _ = talib.MACD(close.to_numpy(dtype=float), fastperiod=12, slowperiod=26, signalperiod=9)
        main = pd.Series(main, index=candles.index)
        signal = pd.Series(signal, index=candles.index)
        enter = (close > ema) & (main > signal)
        leave = (close < ema) & (main < signal)
    return pd.DataFrame({"enter": enter.fillna(False).astype(bool), "exit": leave.fillna(False).astype(bool)})


def rotation_targets(frames: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Select one coin or cash using complete daily close-boundary candles."""
    btc, eth = _frame(frames["BTCUSDT"]), _frame(frames["ETHUSDT"])
    if not btc.index.equals(eth.index):
        raise ValueError("Rotation daily candles must share exactly the same timestamps")
    ratio = eth.close / btc.close
    ema = _talib(talib.EMA, ratio, timeperiod=20)
    btc_sma = _talib(talib.SMA, btc.close, timeperiod=200)
    eth_sma = _talib(talib.SMA, eth.close, timeperiod=200)
    result = pd.Series("USDT", index=btc.index, name="target", dtype=object)
    valid = ema.notna() & btc_sma.notna() & eth_sma.notna()
    selected_eth = ratio > ema
    result.loc[valid & selected_eth & (eth.close > eth_sma)] = "ETHUSDT"
    result.loc[valid & ~selected_eth & (btc.close > btc_sma)] = "BTCUSDT"
    return result


@dataclass
class _Position:
    symbol: str
    quantity: float
    entry_time: pd.Timestamp
    entry_price: float
    entry_fee: float

    @property
    def entry_cost(self) -> float:
        return self.quantity * self.entry_price + self.entry_fee


class _SpotLedger:
    def __init__(self, capital: float, fee: float, slippage: float, filters: Mapping[str, Any]):
        if capital <= 0 or not np.isfinite(capital):
            raise ValueError("capital must be positive and finite")
        if not 0 <= fee < 1 or not 0 <= slippage < 1:
            raise ValueError("fee and slippage must be fractions in [0, 1)")
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.fee_rate = float(fee)
        self.slippage = float(slippage)
        self.filters = filters
        self.position: _Position | None = None
        self.fills: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.slippage_cost = 0.0
        self.rejected_orders = 0

    def _quantity(self, symbol: str, quantity: float) -> float:
        spec = self.filters.get(symbol, {}) or {}
        maximum = float(spec.get("max_qty", spec.get("maximum_quantity", np.inf)))
        quantity = min(quantity, maximum)
        step = float(spec.get("step_size", 0) or 0)
        if step > 0:
            quantity = np.floor(quantity / step + 1e-10) * step
        return float(quantity)

    def _valid_order(self, symbol: str, quantity: float, price: float) -> bool:
        spec = self.filters.get(symbol, {}) or {}
        minimum = float(spec.get("min_qty", spec.get("minimum_quantity", 0)) or 0)
        notional = float(spec.get("minimum_notional", spec.get("min_notional", 0)) or 0)
        valid = quantity > 0 and quantity + 1e-12 >= minimum and quantity * price + 1e-8 >= notional
        if not valid:
            self.rejected_orders += 1
        return valid

    def buy(self, symbol: str, reference: float, timestamp: pd.Timestamp, reason: str) -> bool:
        assert self.position is None, "Cannot add to an existing position"
        price = reference * (1.0 + self.slippage)
        quantity = self._quantity(symbol, self.cash * 0.95 / ((1.0 + self.fee_rate) * price))
        if not self._valid_order(symbol, quantity, price):
            return False
        notional = quantity * price
        fee = notional * self.fee_rate
        assert notional + fee <= self.cash + 1e-8
        self.cash -= notional + fee
        self.position = _Position(symbol, quantity, timestamp, price, fee)
        self.slippage_cost += quantity * (price - reference)
        self.fills.append(dict(timestamp=timestamp, symbol=symbol, side="BUY", qty=quantity, price=price, fee=fee, reason=reason))
        return True

    def sell(self, reference: float, timestamp: pd.Timestamp, reason: str) -> bool:
        position = self.position
        assert position is not None
        price = reference * (1.0 - self.slippage)
        quantity = position.quantity
        if not self._valid_order(position.symbol, quantity, price):
            return False
        notional = quantity * price
        fee = notional * self.fee_rate
        self.cash += notional - fee
        self.slippage_cost += quantity * (reference - price)
        pnl = notional - fee - position.entry_cost
        self.fills.append(dict(timestamp=timestamp, symbol=position.symbol, side="SELL", qty=quantity, price=price, fee=fee, reason=reason))
        self.trades.append(dict(
            entry_time=position.entry_time, exit_time=timestamp, symbol=position.symbol, qty=quantity,
            entry_price=position.entry_price, exit_price=price, pnl=pnl,
            return_pct=pnl / position.entry_cost * 100.0,
            holding_minutes=(timestamp - position.entry_time).total_seconds() / 60.0,
            exit_reason=reason,
        ))
        self.position = None
        return True

    def barriers(self, candle: np.ndarray, timestamp: pd.Timestamp, stop: float | None, roi: float | None) -> bool:
        if self.position is None:
            return False
        opening, high, low = candle[:3]
        position = self.position
        # Price stops use the actual entry fill. ROI targets additionally
        # compensate entry/exit fees and the adverse execution on the sale.
        stop_price = position.entry_price * (1.0 - stop) if stop is not None else None
        roi_price = (
            position.entry_cost * (1.0 + roi)
            / (position.quantity * (1.0 - self.slippage) * (1.0 - self.fee_rate))
            if roi is not None else None
        )
        if stop_price is not None and low <= stop_price:
            return self.sell(min(opening, stop_price), timestamp, "stop_loss")
        if roi_price is not None and high >= roi_price:
            return self.sell(max(opening, roi_price), timestamp, "roi")
        return False

    def check(self) -> None:
        assert np.isfinite(self.cash) and self.cash >= -1e-8, "Negative or invalid cash"
        if self.position is not None:
            assert np.isfinite(self.position.quantity) and self.position.quantity > 0, "Invalid spot inventory"

    def finish(self, equity: pd.DataFrame, last_close: Mapping[str, float], **metadata) -> dict[str, Any]:
        position = self.position
        base_value = position.quantity * last_close[position.symbol] if position else 0.0
        liquidation = self.cash + base_value * (1.0 - self.slippage) * (1.0 - self.fee_rate)
        fills = pd.DataFrame(self.fills, columns=FILL_COLUMNS)
        trades = pd.DataFrame(self.trades, columns=TRADE_COLUMNS)
        info = dict(
            initial_capital=self.initial_capital, final_cash=self.cash,
            final_positions={position.symbol: position.quantity} if position else {},
            fees=float(fills.fee.sum()) if len(fills) else 0.0,
            total_notional=float((fills.qty * fills.price).sum()) if len(fills) else 0.0,
            slippage_cost=self.slippage_cost, final_liquidation_equity=liquidation,
            unrealized_pnl=base_value - position.entry_cost if position else 0.0,
            rejected_orders=self.rejected_orders, final_equity=self.cash + base_value,
            equity_timestamp="minute_open", valuation="minute_close_mark_to_market",
            fee_asset="USDT", entry_cash_fraction=0.95, return_pct_unit="percent",
            fee_rate=self.fee_rate, slippage=self.slippage, forced_final_liquidation=False,
            **metadata,
        )
        return {"equity": equity, "fills": fills, "trades": trades, "metadata": info}


def _window(frame: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    start, end = _utc_time(start), _utc_time(end)
    if start >= end or start != start.floor("min") or end != end.floor("min"):
        raise ValueError("Require a non-empty, minute-aligned [start, end) window")
    candles = _frame(frame)
    candles = candles.loc[(candles.index >= start) & (candles.index < end)]
    expected = pd.date_range(start, end, freq="min", inclusive="left")
    if not candles.index.equals(expected):
        raise ValueError("Simulation requires every minute in [start, end), without gaps")
    return candles


def _signal_rows(signals: pd.DataFrame) -> dict[pd.Timestamp, tuple[bool, bool]]:
    if not isinstance(signals.index, pd.DatetimeIndex) or signals.index.tz is None or signals.index.has_duplicates:
        raise ValueError("Signals need unique, timezone-aware boundary timestamps")
    if not {"enter", "exit"}.issubset(signals.columns):
        raise ValueError("Signals need enter and exit columns")
    rows = {}
    for timestamp, enter, leave in zip(signals.index, signals.enter, signals.exit):
        if not isinstance(enter, (bool, np.bool_)) or not isinstance(leave, (bool, np.bool_)):
            continue
        rows[timestamp.tz_convert("UTC")] = (bool(enter), bool(leave))
    return rows


def _stop(protective_stop: float | None, strategy: str) -> float | None:
    if protective_stop is not None and not 0 < protective_stop < 1:
        raise ValueError("protective_stop must be a fractional loss between 0 and 1")
    if strategy == "cluc":
        return min(0.05, protective_stop) if protective_stop is not None else 0.05
    return protective_stop


def simulate_single(
    frame: pd.DataFrame, signals: pd.DataFrame, *, symbol: str, strategy: str,
    start: Any, end: Any, capital: float = 5000.0, fee: float = 0.001,
    slippage: float = 0.0002, protective_stop: float | None = None,
    filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One long position, no pyramiding; boundary signals fill at minute open."""
    if strategy not in (*STRATEGY_IDS, "buy_hold"):
        raise ValueError(f"Unknown strategy: {strategy}")
    candles = _window(frame, start, end)
    signals_by_time = _signal_rows(signals)
    ledger = _SpotLedger(capital, fee, slippage, {symbol: filters or {}})
    stop = _stop(protective_stop, strategy)
    roi = 0.01 if strategy == "cluc" else None
    blocked_until: pd.Timestamp | None = None
    rows = np.empty((len(candles), 4), dtype=float)
    for i, (timestamp, candle) in enumerate(zip(candles.index, candles.to_numpy())):
        signal = signals_by_time.get(timestamp)
        if signal is not None:
            enter, leave = signal
            if ledger.position is not None:
                if leave:
                    ledger.sell(float(candle[0]), timestamp, "signal_exit")
            elif enter and not leave and (blocked_until is None or timestamp > blocked_until):
                ledger.buy(symbol, float(candle[0]), timestamp, "signal_entry")
        if ledger.barriers(candle, timestamp, stop, roi):
            blocked_until = timestamp
        ledger.check()
        base_value = ledger.position.quantity * float(candle[3]) if ledger.position else 0.0
        equity = ledger.cash + base_value
        assert equity > 0 and np.isfinite(equity)
        rows[i] = (equity, ledger.cash, base_value, base_value / equity)
    equity = pd.DataFrame(rows, index=candles.index, columns=["equity", "cash", "base_value", "exposure"])
    return ledger.finish(equity, {symbol: float(candles.close.iloc[-1])}, strategy=strategy, protective_stop=stop, roi=roi)


def simulate_buy_hold(
    frame: pd.DataFrame, *, symbol: str, start: Any, end: Any, capital: float = 5000.0,
    fee: float = 0.001, slippage: float = 0.0002, filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    signals = pd.DataFrame({"enter": [True], "exit": [False]}, index=pd.DatetimeIndex([_utc_time(start)]))
    return simulate_single(frame, signals, symbol=symbol, strategy="buy_hold", start=start, end=end,
                           capital=capital, fee=fee, slippage=slippage, filters=filters)


def simulate_rotation(
    frames: Mapping[str, pd.DataFrame], targets: pd.Series, *, start: Any, end: Any,
    capital: float = 10000.0, fee: float = 0.001, slippage: float = 0.0002,
    protective_stop: float | None = None, filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Daily spot rotation; switch sales and purchases are one minute apart."""
    symbols = ("BTCUSDT", "ETHUSDT")
    candles = {symbol: _window(frames[symbol], start, end) for symbol in symbols}
    index = candles[symbols[0]].index
    if not index.equals(candles[symbols[1]].index):
        raise ValueError("Rotation minute candles must be synchronized")
    if not isinstance(targets.index, pd.DatetimeIndex) or targets.index.tz is None or targets.index.has_duplicates:
        raise ValueError("Targets need unique, timezone-aware boundary timestamps")
    valid_targets = {
        timestamp.tz_convert("UTC"): target for timestamp, target in targets.items()
        if isinstance(target, str) and target in (*symbols, "USDT")
    }
    arrays = {symbol: candles[symbol].to_numpy() for symbol in symbols}
    ledger = _SpotLedger(capital, fee, slippage, filters or {})
    stop = _stop(protective_stop, "rotation")
    pending: tuple[pd.Timestamp, str] | None = None
    blocked_until: pd.Timestamp | None = None
    rows = np.empty((len(index), 4), dtype=float)
    for i, timestamp in enumerate(index):
        if pending is not None and timestamp >= pending[0]:
            _, symbol = pending
            if ledger.position is None and (blocked_until is None or timestamp > blocked_until):
                ledger.buy(symbol, float(arrays[symbol][i, 0]), timestamp, "rotation_entry")
            pending = None
        target = valid_targets.get(timestamp)
        if target is not None:
            current = ledger.position.symbol if ledger.position else "USDT"
            if target != current:
                if ledger.position is not None:
                    if ledger.sell(float(arrays[current][i, 0]), timestamp, "rotation_exit"):
                        pending = (timestamp + pd.Timedelta(minutes=1), target) if target != "USDT" else None
                elif target != "USDT" and (blocked_until is None or timestamp > blocked_until):
                    ledger.buy(target, float(arrays[target][i, 0]), timestamp, "rotation_entry")
        if ledger.position is not None:
            symbol = ledger.position.symbol
            if ledger.barriers(arrays[symbol][i], timestamp, stop, None):
                blocked_until = timestamp
                pending = None
        ledger.check()
        position = ledger.position
        base_value = position.quantity * float(arrays[position.symbol][i, 3]) if position else 0.0
        equity = ledger.cash + base_value
        assert equity > 0 and np.isfinite(equity)
        rows[i] = (equity, ledger.cash, base_value, base_value / equity)
    equity = pd.DataFrame(rows, index=index, columns=["equity", "cash", "base_value", "exposure"])
    last_close = {symbol: float(candles[symbol].close.iloc[-1]) for symbol in symbols}
    return ledger.finish(equity, last_close, strategy="rotation", protective_stop=stop, roi=None)
