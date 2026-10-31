"""Frozen second-round spot hypotheses, independent of production controllers.

These are long/cash adaptations of public indicator ideas, not claims that the
source authors published these complete crypto systems. Parameters are selected
before this round's numerical run and never optimized against its year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from scripts import spot_strategy_suite_engine as engine

IDS = ("supertrend", "squeeze", "ibs", "mfi", "absolute_momentum")
PERIODS = {"supertrend": 240, "squeeze": 240, "ibs": 1440, "mfi": 240, "absolute_momentum": 1440}
LABELS = {
    "supertrend": "Supertrend 波动跟踪", "squeeze": "波动压缩释放",
    "ibs": "IBS 日线反转", "mfi": "MFI 资金流反转",
    "absolute_momentum": "180日绝对动量",
}
RULES = {
    "supertrend": "4h; ATR(10) Wilder via TA-Lib, factor 3; TradingView recursive bands; up state enter, down state exit; initial direction down after ATR ready",
    "squeeze": "4h; BB SMA20 +/- 2 population std(ddof0); KC EMA20 +/- 1.5 ATR20; previous bar BB strictly within KC, current bar released AND close>BBupper AND MOM20>0 enter; close<SMA20 OR MOM20<=0 exit",
    "ibs": "daily; IBS=(close-low)/(high-low), flat bar IBS=.5; enter IBS<=.2, exit IBS>=.8; no trend filter",
    "mfi": "4h; TA-Lib MFI14; enter previous>=20 AND current<20; exit previous<=80 AND current>80; crossings not persistent states",
    "absolute_momentum": "daily ROC180=close/close.shift(180)-1; evaluate only Monday 00:00 UTC at closed daily boundary; ROC>0 enter, ROC<=0 exit; no rebalance while held; USDT cash has zero yield",
}


def _indicator(function, frame, *columns, **kwargs):
    return pd.Series(function(*(frame[c].to_numpy(dtype=float) for c in columns), **kwargs), index=frame.index)


def supertrend_bands(frame: pd.DataFrame, period: int = 10, multiplier: float = 3.) -> pd.DataFrame:
    """TradingView's causal recursive bands, with Wilder ATR seeded by TA-Lib."""
    candles = engine._frame(frame)
    atr = _indicator(talib.ATR, candles, "high", "low", "close", timeperiod=period)
    mid = (candles.high + candles.low) / 2
    raw_upper, raw_lower = (mid + multiplier * atr).to_numpy(), (mid - multiplier * atr).to_numpy()
    upper, lower, line = (np.full(len(candles), np.nan) for _ in range(3))
    direction = np.zeros(len(candles), dtype=np.int8)
    close = candles.close.to_numpy()
    for i in range(len(candles)):
        if not np.isfinite(raw_upper[i]):
            continue
        if i == 0 or not np.isfinite(upper[i - 1]):
            upper[i], lower[i], direction[i] = raw_upper[i], raw_lower[i], -1
        else:
            upper[i] = raw_upper[i] if raw_upper[i] < upper[i - 1] or close[i - 1] > upper[i - 1] else upper[i - 1]
            lower[i] = raw_lower[i] if raw_lower[i] > lower[i - 1] or close[i - 1] < lower[i - 1] else lower[i - 1]
            if direction[i - 1] == -1:
                direction[i] = 1 if close[i] > upper[i] else -1
            else:
                direction[i] = -1 if close[i] < lower[i] else 1
        line[i] = lower[i] if direction[i] == 1 else upper[i]
    return pd.DataFrame({"atr": atr, "upper": upper, "lower": lower, "direction": direction, "line": line}, index=candles.index)


def generate_signals(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Input is complete bars labelled by their following UTC close boundary."""
    if strategy not in IDS:
        raise ValueError(f"Unknown round-two strategy: {strategy}")
    candles = engine._frame(frame)
    close = candles.close
    if candles.empty:
        return pd.DataFrame({"enter": pd.Series(dtype=bool), "exit": pd.Series(dtype=bool)}, index=candles.index)
    if strategy == "supertrend":
        bands = supertrend_bands(candles)
        enter, leave = bands.direction.eq(1), bands.direction.eq(-1)
    elif strategy == "squeeze":
        middle = close.rolling(20, min_periods=20).mean()
        deviation = close.rolling(20, min_periods=20).std(ddof=0)
        bb_upper, bb_lower = middle + 2 * deviation, middle - 2 * deviation
        ema = _indicator(talib.EMA, candles, "close", timeperiod=20)
        atr = _indicator(talib.ATR, candles, "high", "low", "close", timeperiod=20)
        compressed = (bb_upper < ema + 1.5 * atr) & (bb_lower > ema - 1.5 * atr)
        momentum = close - close.shift(20)
        enter = compressed.shift(1, fill_value=False) & ~compressed & (close > bb_upper) & (momentum > 0)
        leave = (close < middle) | (momentum <= 0)
    elif strategy == "ibs":
        spread = candles.high - candles.low
        value = ((close - candles.low) / spread.where(spread.ne(0))).where(spread.ne(0), .5)
        enter, leave = value.le(.2), value.ge(.8)
    elif strategy == "mfi":
        value = _indicator(talib.MFI, candles, "high", "low", "close", "volume", timeperiod=14)
        enter = value.shift(1).ge(20) & value.lt(20)
        leave = value.shift(1).le(80) & value.gt(80)
    else:
        roc = close / close.shift(180) - 1
        monday = (candles.index.dayofweek == 0) & (candles.index.hour == 0) & (candles.index.minute == 0)
        enter, leave = roc.gt(0) & monday, roc.le(0) & monday
    return pd.DataFrame({"enter": enter.fillna(False).astype(bool), "exit": leave.fillna(False).astype(bool)}, index=candles.index)


def simulate_strategy(frame: pd.DataFrame, signals: pd.DataFrame, *, strategy: str, **kwargs) -> dict:
    """Use the unchanged first-round generic external-signal spot ledger.

    The buy_hold dispatch has no special entry or exit logic: provided signals
    determine every trade and no ROI target/native stop is added. Only Cluc has
    special economics in the original engine. Relabel the returned metadata,
    keeping the selected external-signal dispatch explicit for auditability.
    """
    if strategy not in IDS:
        raise ValueError(f"Unknown round-two strategy: {strategy}")
    result = engine.simulate_single(frame, signals, strategy="buy_hold", **kwargs)
    result["metadata"].update(strategy=strategy, execution_adapter="unchanged generic buy_hold dispatch with external signals; no ROI target")
    return result
