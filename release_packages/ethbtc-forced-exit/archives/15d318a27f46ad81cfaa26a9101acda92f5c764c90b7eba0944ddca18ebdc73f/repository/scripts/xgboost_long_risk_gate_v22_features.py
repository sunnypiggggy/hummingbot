"""Self-contained feature engineering for the v22 weekly risk gate.

The formulas are frozen copies of the research feature contract.  This module
deliberately imports no v1-v21 model, optimizer, live guard, or ROC/SQZ gate.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
HOUR = 3600
CORE_FEATURES = (
    "roc_5", "roc_20", "return_1", "return_5", "return_20", "rsi_14", "rsi_slope_3",
    "stoch_rsi_k_minus_d", "ppo_hist", "ppo_hist_slope", "tsi", "adx_14", "di_spread",
    "sqzmom_value", "sqzmom_slope", "atr_pct", "volume_zscore", "mfi_14", "obv_slope",
    "price_to_ema20_atr", "btc_return_1", "btc_volatility_20", "btc_corr_48", "hour_sin",
    "hour_cos", "dow_sin", "dow_cos", "pair_is_eth",
)
FOUR_HOUR_FEATURES = (
    "roc_48h_4h", "sqzmom_pct_4h", "sqzmom_value_4h", "sqzmom_slope_4h",
    "sqzmom_improving_4h", "roc_to_entry_4h", "sqz_to_entry_4h",
    "roc_to_recovery_4h", "sqz_to_recovery_4h",
)
ALL_FEATURES = CORE_FEATURES + FOUR_HOUR_FEATURES
TECHNICAL_PARAMS = {
    "BTC-FDUSD": (-7.0, -4.0, 1.0, -3.0),
    "ETH-FDUSD": (-9.0, -5.0, 3.0, -3.0),
}


def _wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _linear_regression_last(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float); centered = x - x.mean()
    slope = np.dot(centered, values - values.mean()) / np.dot(centered, centered)
    return float(values.mean() + slope * (x[-1] - x.mean()))


def add_momentum_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy(); close, high, low, volume = out.close, out.high, out.low, out.volume
    for length in (1, 5, 20):
        out[f"return_{length}"] = close.pct_change(length)
    out["roc_5"] = close.pct_change(5) * 100; out["roc_20"] = close.pct_change(20) * 100
    delta = close.diff(); gain = _wilder(delta.clip(lower=0), 14); loss = _wilder(-delta.clip(upper=0), 14)
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan)); out["rsi_slope_3"] = out.rsi_14.diff(3) / 3
    rsi_low = out.rsi_14.rolling(14).min(); rsi_high = out.rsi_14.rolling(14).max()
    stoch = 100 * (out.rsi_14 - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan); stoch_k = stoch.rolling(3).mean()
    out["stoch_rsi_k_minus_d"] = stoch_k - stoch_k.rolling(3).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean(); ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ppo = 100 * (ema12 - ema26) / ema26.replace(0, np.nan)
    out["ppo_hist"] = ppo - ppo.ewm(span=9, adjust=False, min_periods=9).mean(); out["ppo_hist_slope"] = out.ppo_hist.diff(3) / 3
    tsi_num = delta.ewm(span=25, adjust=False, min_periods=25).mean().ewm(span=13, adjust=False, min_periods=13).mean()
    tsi_den = delta.abs().ewm(span=25, adjust=False, min_periods=25).mean().ewm(span=13, adjust=False, min_periods=13).mean()
    out["tsi"] = 100 * tsi_num / tsi_den.replace(0, np.nan)
    previous_close = close.shift(1)
    true_range = pd.concat([high-low, (high-previous_close).abs(), (low-previous_close).abs()], axis=1).max(axis=1)
    atr = _wilder(true_range, 14); out["atr_pct"] = atr / close.replace(0, np.nan)
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    plus_di = 100 * _wilder(plus_dm, 14) / atr.replace(0, np.nan); minus_di = 100 * _wilder(minus_dm, 14) / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    out["adx_14"] = _wilder(dx, 14); out["di_spread"] = plus_di-minus_di
    basis = close.rolling(20).mean(); midpoint = ((high.rolling(20).max()+low.rolling(20).min())/2+basis)/2
    out["sqzmom_value"] = (close-midpoint).rolling(20).apply(_linear_regression_last, raw=True)
    out["sqzmom_slope"] = out.sqzmom_value.diff(3) / 3
    volume_mean = volume.rolling(20).mean(); out["volume_zscore"] = (volume-volume_mean) / volume.rolling(20).std(ddof=0).replace(0, np.nan)
    typical = (high+low+close)/3; money = typical*volume
    positive = money.where(typical.diff()>0, 0.0); negative = money.where(typical.diff()<0, 0.0)
    out["mfi_14"] = 100 - 100 / (1 + positive.rolling(14).sum()/negative.rolling(14).sum().replace(0, np.nan))
    obv = (np.sign(close.diff()).fillna(0)*volume).cumsum(); out["obv_slope"] = obv.diff(5)/volume.rolling(20).sum().replace(0, np.nan)
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean(); out["price_to_ema20_atr"] = (close-ema20)/atr.replace(0, np.nan)
    return out


def _linreg_endpoint(values: list[float]) -> float:
    length = len(values); x_mean = (length - 1) / 2; y_mean = sum(values) / length
    denominator = sum((index - x_mean) ** 2 for index in range(length))
    slope = sum((index-x_mean)*(value-y_mean) for index, value in enumerate(values)) / denominator
    return y_mean - slope*x_mean + slope*(length-1)


def roc_sqz_signal_from_klines(klines: list[list[Any]], roc_length: int = 12, sqz_length: int = 20) -> dict[str, float]:
    highs = [float(item[2]) for item in klines]; lows = [float(item[3]) for item in klines]; closes = [float(item[4]) for item in klines]
    sources = []
    for index in range(sqz_length-1, len(klines)):
        start = index-sqz_length+1; midpoint = ((max(highs[start:index+1])+min(lows[start:index+1]))/2 + sum(closes[start:index+1])/sqz_length)/2
        sources.append(closes[index]-midpoint)
    current = _linreg_endpoint(sources[-sqz_length:]); previous = _linreg_endpoint(sources[-sqz_length-1:-1]); close = closes[-1]
    return {"close": close, "roc_48h_pct": (close/closes[-1-roc_length]-1)*100,
            "sqzmom": current, "sqzmom_previous": previous, "sqzmom_pct": current/close*100}


def hourly_bars(candles: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    output = {}
    for pair, raw in candles.items():
        item = raw.copy(); item["datetime"] = pd.to_datetime(item.timestamp, unit="s", utc=True)
        bars = item.set_index("datetime").resample("1h", label="left", closed="left", origin="epoch").agg(
            open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last"), volume=("volume","sum"), rows=("close","size"))
        output[pair] = bars[bars.rows == 12].drop(columns="rows")
    return output


def four_hour_frame(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    source = frame.sort_values("timestamp").assign(bucket=lambda x: (x.timestamp.astype("int64")//14400)*14400)
    bars = source.groupby("bucket", sort=True).agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last"), rows=("close","size")).reset_index()
    bars = bars[bars.rows == 48]; klines, observations = [], []
    for row in bars.itertuples(index=False):
        close_time = int(row.bucket)+14399; klines.append([int(row.bucket)*1000,row.open,row.high,row.low,row.close,0,close_time*1000])
        if len(klines) >= 40:
            observations.append({"last_complete_4h_ts": close_time+1, **roc_sqz_signal_from_klines(klines[-64:])})
    out = pd.DataFrame(observations); trigger_roc, trigger_sqz, recovery_roc, recovery_sqz = TECHNICAL_PARAMS[pair]
    out = out.rename(columns={"roc_48h_pct":"roc_48h_4h","sqzmom_pct":"sqzmom_pct_4h","sqzmom":"sqzmom_value_4h"})
    out["sqzmom_slope_4h"] = (out.sqzmom_value_4h-out.sqzmom_previous)/out.close.replace(0,np.nan)*100
    out["sqzmom_improving_4h"] = (out.sqzmom_value_4h>out.sqzmom_previous).astype(float)
    out["roc_to_entry_4h"] = out.roc_48h_4h-trigger_roc; out["sqz_to_entry_4h"] = out.sqzmom_pct_4h-trigger_sqz
    out["roc_to_recovery_4h"] = out.roc_48h_4h-recovery_roc; out["sqz_to_recovery_4h"] = out.sqzmom_pct_4h-recovery_sqz
    return out[["last_complete_4h_ts", *FOUR_HOUR_FEATURES]].sort_values("last_complete_4h_ts")


def rolling_duration_from_high(close: pd.Series, window: int) -> pd.Series:
    values = close.to_numpy(float); result = np.full(len(values), np.nan)
    for index in range(window-1, len(values)):
        sample = values[index-window+1:index+1]; result[index] = len(sample)-1-int(np.nanargmax(sample))
    return pd.Series(result, index=close.index)


def rolling_percentile(series: pd.Series, window: int = 720, minimum: int = 240) -> pd.Series:
    return series.rolling(window, min_periods=minimum).apply(lambda x: float((x[:-1] <= x[-1]).mean()) if len(x)>1 else np.nan, raw=True)


def expected_shortfall(values: np.ndarray) -> float:
    return float(np.mean(np.sort(values)[:max(1, int(np.ceil(.05*len(values))))]))


def add_structure_features(panel: pd.DataFrame) -> pd.DataFrame:
    parts, returns = {}, {}
    for pair in PAIRS:
        item = panel[panel.pair.eq(pair)].sort_values("signal_ts").copy(); close, low = item.close.astype(float), item.low.astype(float)
        r1 = np.log(close).diff(); returns[pair] = pd.Series(r1.to_numpy(), index=item.signal_ts)
        negative, total = r1.clip(upper=0).pow(2), r1.pow(2); ema20 = close.ewm(span=20, adjust=False).mean(); rv24 = total.rolling(24,min_periods=12).sum().pow(.5)
        item["drawdown_from_high_72h"] = close/close.rolling(72,min_periods=36).max()-1
        item["drawdown_from_high_168h"] = close/close.rolling(168,min_periods=84).max()-1
        item["drawdown_duration_168h"] = rolling_duration_from_high(close,168)
        item["below_ema20_ratio_72h"] = close.lt(ema20).astype(float).rolling(72,min_periods=36).mean()
        item["lower_low_ratio_72h"] = low.lt(low.shift()).astype(float).rolling(72,min_periods=36).mean()
        for hours in (24,72):
            item[f"downside_semivariance_ratio_{hours}h"] = negative.rolling(hours,min_periods=hours//2).sum()/total.rolling(hours,min_periods=hours//2).sum().replace(0,np.nan)
        item["rv_24h_percentile_30d"] = rolling_percentile(rv24)
        item["vol_of_vol_72h"] = rv24.rolling(72,min_periods=36).std(ddof=0)/rv24.rolling(72,min_periods=36).mean().replace(0,np.nan)
        item["trend_efficiency_72h"] = (close-close.shift(72)).abs()/close.diff().abs().rolling(72,min_periods=36).sum().replace(0,np.nan)
        item["ema20_slope_atr_12h"] = (ema20-ema20.shift(12))/(item.atr_pct.astype(float)*close).replace(0,np.nan)
        item["historical_var_72h"] = r1.rolling(72,min_periods=36).quantile(.05)
        item["expected_shortfall_72h"] = r1.rolling(72,min_periods=36).apply(expected_shortfall,raw=True)
        item["negative_skew_72h"] = -r1.rolling(72,min_periods=36).skew(); parts[pair] = item
    btc, eth = returns["BTC-FDUSD"], returns["ETH-FDUSD"]
    for pair, own, other in (("BTC-FDUSD",btc,eth),("ETH-FDUSD",eth,btc)):
        item = parts[pair].set_index("signal_ts"); down = other.where(other<0)
        item["cross_pair_downside_beta_72h"] = own.where(other<0).rolling(72,min_periods=24).cov(down)/down.rolling(72,min_periods=24).var().replace(0,np.nan)
        other_dd = other.add(1).rolling(72,min_periods=36).apply(lambda x:x.prod(),raw=True)-1
        own_dd = own.add(1).rolling(72,min_periods=36).apply(lambda x:x.prod(),raw=True)-1
        item["relative_drawdown_72h"] = own_dd-other_dd; parts[pair] = item.reset_index()
    return pd.concat(parts.values(),ignore_index=True).sort_values(["signal_ts","pair"]).reset_index(drop=True)


def build_inference_panel(candles: Mapping[str, pd.DataFrame], features: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    hourly = hourly_bars(candles); featured = {pair:add_momentum_features(frame) for pair,frame in hourly.items()}
    btc = featured["BTC-FDUSD"]; btc_return = btc.return_1.rename("btc_return_1"); btc_volatility = btc.return_1.rolling(20).std(ddof=0).rename("btc_volatility_20")
    rows = []
    for pair,item in featured.items():
        item = item.copy(); item["btc_return_1"] = btc_return.reindex(item.index); item["btc_volatility_20"] = btc_volatility.reindex(item.index)
        item["btc_corr_48"] = item.return_1.rolling(48,min_periods=24).corr(btc_return)
        item["hour_sin"] = np.sin(2*np.pi*item.index.hour/24); item["hour_cos"] = np.cos(2*np.pi*item.index.hour/24)
        item["dow_sin"] = np.sin(2*np.pi*item.index.dayofweek/7); item["dow_cos"] = np.cos(2*np.pi*item.index.dayofweek/7); item["pair_is_eth"] = float(pair=="ETH-FDUSD")
        item["bar_open_ts"] = item.index.astype("int64")//10**9; item["signal_ts"] = item.bar_open_ts+HOUR; item["last_complete_1h_ts"] = item.signal_ts; item["pair"] = pair
        base = item.reset_index(names="bar_open_utc").sort_values("signal_ts")
        rows.append(pd.merge_asof(base,four_hour_frame(candles[pair],pair),left_on="signal_ts",right_on="last_complete_4h_ts",direction="backward"))
    panel = pd.concat(rows,ignore_index=True); panel[list(ALL_FEATURES)] = panel[list(ALL_FEATURES)].replace([np.inf,-np.inf],np.nan)
    panel = panel.dropna(subset=list(ALL_FEATURES)).sort_values(["signal_ts","pair"]).reset_index(drop=True)
    panel = add_structure_features(panel); required = sorted(set().union(*features.values()))
    panel[required] = panel[required].replace([np.inf,-np.inf],np.nan)
    return pd.concat([panel[panel.pair.eq(pair)].dropna(subset=list(features[pair])) for pair in PAIRS],ignore_index=True).sort_values(["signal_ts","pair"]).reset_index(drop=True)
