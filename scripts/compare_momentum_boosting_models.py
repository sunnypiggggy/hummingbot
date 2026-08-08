#!/usr/bin/env python3
"""Compare boosting models for a multi-asset crypto momentum strategy.

The experiment uses closed hourly bars to create signals, executes model and
momentum-reversal orders at the next bar open, and applies a momentum-adaptive
ATR stop carried forward from the previous close.  Train, validation, and test
sets are separated chronologically with a purge equal to the label horizon.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


DEFAULT_SYMBOLS = (
    "BTC",
    "ETH",
    "BNB",
    "SOL",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "LINK",
    "TRX",
)

BASE_FEATURES = (
    "roc_5",
    "roc_20",
    "return_1",
    "return_5",
    "return_20",
    "rsi_14",
    "rsi_slope_3",
    "stoch_rsi_k_minus_d",
    "ppo_hist",
    "ppo_hist_slope",
    "tsi",
    "adx_14",
    "di_spread",
    "sqzmom_value",
    "sqzmom_slope",
    "atr_pct",
    "volume_zscore",
    "mfi_14",
    "obv_slope",
    "price_to_ema20_atr",
    "btc_return_1",
    "btc_volatility_20",
    "btc_corr_48",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

MODEL_COLORS = {
    "LightGBM": "#2563eb",
    "XGBoost": "#d97706",
    "CatBoost": "#1d4ed8",
    "Gradient Boosting Tree": "#64748b",
    "AdaBoost": "#94a3b8",
}

MODEL_DASHES = {
    "LightGBM": "solid",
    "XGBoost": "dash",
    "CatBoost": "dot",
    "Gradient Boosting Tree": "dashdot",
    "AdaBoost": "longdash",
}


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("data/backtesting_candles")
    output_dir: Path = Path("results/backtests/momentum_boosting_comparison")
    timeframe: str = "1h"
    horizon_bars: int = 6
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    initial_capital: float = 10_000.0
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    normal_stop_atr: float = 2.50
    weak_stop_atr: float = 1.25
    random_seed: int = 42
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.fee_bps + self.slippage_bps) / 10_000


@dataclass
class StrategyMetrics:
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    trades: int
    win_rate_pct: float
    profit_factor: float
    exposure_pct: float
    stop_exits: int
    stop_exit_rate_pct: float


@dataclass
class BacktestResult:
    equity: pd.Series
    exposure: pd.Series
    stop_events: pd.Series
    trades: pd.DataFrame
    metrics: StrategyMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ExperimentConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    parser.add_argument("--timeframe", default=ExperimentConfig.timeframe)
    parser.add_argument("--horizon-bars", type=int, default=ExperimentConfig.horizon_bars)
    parser.add_argument("--fee-bps", type=float, default=ExperimentConfig.fee_bps)
    parser.add_argument("--slippage-bps", type=float, default=ExperimentConfig.slippage_bps)
    parser.add_argument("--initial-capital", type=float, default=ExperimentConfig.initial_capital)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    return parser.parse_args()


def _timestamp_unit(values: pd.Series) -> str:
    return "ms" if values.abs().median() > 10_000_000_000 else "s"


def load_hourly_bars(
    data_dir: Path,
    symbols: tuple[str, ...],
    timeframe: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, Any]] = []
    target_delta = pd.Timedelta(timeframe)

    for symbol in symbols:
        path = data_dir / f"binance_{symbol}-USDT_5m.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required candle file: {path}")
        raw = pd.read_csv(path)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        duplicate_count = int(raw["timestamp"].duplicated().sum())
        unit = _timestamp_unit(raw["timestamp"])
        raw["datetime"] = pd.to_datetime(raw["timestamp"], unit=unit, utc=True)
        raw = raw.sort_values("datetime").drop_duplicates("datetime", keep="last")
        raw = raw.set_index("datetime")
        source_delta = raw.index.to_series().diff().dropna().median()
        expected_rows = int(round(target_delta / source_delta))
        gap_units = raw.index.to_series().diff().dropna() / source_delta
        missing_source_bars = int(np.maximum(gap_units.to_numpy() - 1, 0).sum())
        invalid_ohlc = int(
            (
                (raw["high"] < raw[["open", "close"]].max(axis=1))
                | (raw["low"] > raw[["open", "close"]].min(axis=1))
                | (raw["high"] < raw["low"])
                | (raw["volume"] < 0)
            ).sum()
        )

        hourly = raw.resample(
            timeframe,
            label="left",
            closed="left",
            origin="epoch",
        ).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source_rows=("close", "count"),
        )
        incomplete_bars = int((hourly["source_rows"] != expected_rows).sum())
        hourly = hourly.loc[hourly["source_rows"] == expected_rows].drop(columns="source_rows")
        bars_by_symbol[symbol] = hourly
        quality_rows.append(
            {
                "symbol": symbol,
                "source_path": str(path),
                "raw_rows": len(raw),
                "duplicate_timestamps_removed": duplicate_count,
                "missing_source_bars": missing_source_bars,
                "invalid_ohlcv_rows": invalid_ohlc,
                "source_interval_minutes": source_delta.total_seconds() / 60,
                "expected_source_rows_per_bar": expected_rows,
                "incomplete_resampled_bars_removed": incomplete_bars,
                "complete_resampled_bars": len(hourly),
                "raw_start_utc": str(raw.index.min()),
                "raw_end_utc": str(raw.index.max()),
            }
        )

    common_start = max(frame.index.min() for frame in bars_by_symbol.values())
    common_end = min(frame.index.max() for frame in bars_by_symbol.values())
    for symbol, frame in bars_by_symbol.items():
        bars_by_symbol[symbol] = frame.loc[common_start:common_end].copy()
    quality = pd.DataFrame(quality_rows)
    quality["common_start_utc"] = str(common_start)
    quality["common_end_utc"] = str(common_end)
    quality["common_bars"] = [len(bars_by_symbol[s]) for s in quality["symbol"]]
    return bars_by_symbol, quality


def _wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _linear_regression_last(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    x_centered = x - x.mean()
    denominator = np.dot(x_centered, x_centered)
    slope = np.dot(x_centered, values - values.mean()) / denominator
    return float(values.mean() + slope * (x[-1] - x.mean()))


def add_momentum_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    for length in (1, 5, 20):
        out[f"return_{length}"] = close.pct_change(length)
    out["roc_5"] = close.pct_change(5) * 100
    out["roc_20"] = close.pct_change(20) * 100

    delta = close.diff()
    average_gain = _wilder(delta.clip(lower=0), 14)
    average_loss = _wilder(-delta.clip(upper=0), 14)
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + relative_strength)
    out["rsi_slope_3"] = out["rsi_14"].diff(3) / 3
    rsi_low = out["rsi_14"].rolling(14).min()
    rsi_high = out["rsi_14"].rolling(14).max()
    stoch_rsi = 100 * (out["rsi_14"] - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
    stoch_k = stoch_rsi.rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()
    out["stoch_rsi_k_minus_d"] = stoch_k - stoch_d

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ppo = 100 * (ema_12 - ema_26) / ema_26.replace(0, np.nan)
    ppo_signal = ppo.ewm(span=9, adjust=False, min_periods=9).mean()
    out["ppo_hist"] = ppo - ppo_signal
    out["ppo_hist_slope"] = out["ppo_hist"].diff(3) / 3

    momentum = close.diff()
    tsi_numerator = momentum.ewm(span=25, adjust=False, min_periods=25).mean()
    tsi_numerator = tsi_numerator.ewm(span=13, adjust=False, min_periods=13).mean()
    tsi_denominator = momentum.abs().ewm(span=25, adjust=False, min_periods=25).mean()
    tsi_denominator = tsi_denominator.ewm(span=13, adjust=False, min_periods=13).mean()
    out["tsi"] = 100 * tsi_numerator / tsi_denominator.replace(0, np.nan)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = _wilder(true_range, 14)
    out["atr"] = atr
    out["atr_pct"] = atr / close.replace(0, np.nan)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    plus_di = 100 * _wilder(plus_dm, 14) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, 14) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx_14"] = _wilder(dx, 14)
    out["di_spread"] = plus_di - minus_di

    basis = close.rolling(20).mean()
    midpoint = ((high.rolling(20).max() + low.rolling(20).min()) / 2 + basis) / 2
    sqz_source = close - midpoint
    out["sqzmom_value"] = sqz_source.rolling(20).apply(_linear_regression_last, raw=True)
    out["sqzmom_slope"] = out["sqzmom_value"].diff(3) / 3

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std(ddof=0)
    out["volume_zscore"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0.0)
    negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0.0)
    money_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum().replace(0, np.nan)
    out["mfi_14"] = 100 - 100 / (1 + money_ratio)
    signed_volume = np.sign(close.diff()).fillna(0) * volume
    obv = signed_volume.cumsum()
    out["obv_slope"] = obv.diff(5) / volume.rolling(20).sum().replace(0, np.nan)

    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    out["price_to_ema20_atr"] = (close - ema_20) / atr.replace(0, np.nan)
    return out


def build_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, list[str]]:
    featured = {symbol: add_momentum_features(bars) for symbol, bars in bars_by_symbol.items()}
    btc = featured["BTC"]
    btc_return = btc["return_1"].rename("btc_return_1")
    btc_volatility = btc["return_1"].rolling(20).std(ddof=0).rename("btc_volatility_20")
    rows: list[pd.DataFrame] = []

    for symbol, frame in featured.items():
        item = frame.copy()
        item["btc_return_1"] = btc_return.reindex(item.index)
        item["btc_volatility_20"] = btc_volatility.reindex(item.index)
        item["btc_corr_48"] = item["return_1"].rolling(48, min_periods=24).corr(btc_return)
        item["hour_sin"] = np.sin(2 * np.pi * item.index.hour / 24)
        item["hour_cos"] = np.cos(2 * np.pi * item.index.hour / 24)
        item["dow_sin"] = np.sin(2 * np.pi * item.index.dayofweek / 7)
        item["dow_cos"] = np.cos(2 * np.pi * item.index.dayofweek / 7)
        item["forward_return"] = item["close"].shift(-config.horizon_bars) / item["open"].shift(-1) - 1
        item["target"] = (item["forward_return"] > config.round_trip_cost).astype(float)
        item.loc[item["forward_return"].isna(), "target"] = np.nan
        item["momentum_score"] = (
            np.where(item["roc_5"] >= 0, 1, -1)
            + np.where(item["ppo_hist"] >= 0, 1, -1)
            + np.where(item["sqzmom_slope"] >= 0, 1, -1)
            + np.where(item["rsi_14"] >= 50, 1, -1)
        )
        item["symbol"] = symbol
        item["datetime"] = item.index
        rows.append(item.reset_index(drop=True))

    panel = pd.concat(rows, ignore_index=True)
    symbol_dummies = pd.get_dummies(panel["symbol"], prefix="symbol", dtype=float)
    panel = pd.concat([panel, symbol_dummies], axis=1)
    feature_columns = list(BASE_FEATURES) + sorted(symbol_dummies.columns)
    panel[feature_columns] = panel[feature_columns].replace([np.inf, -np.inf], np.nan)
    required = feature_columns + [
        "target",
        "forward_return",
        "open",
        "high",
        "low",
        "close",
        "atr",
        "momentum_score",
    ]
    panel = panel.dropna(subset=required).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    panel["target"] = panel["target"].astype(int)
    return panel, feature_columns


def chronological_split(
    panel: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    timestamps = pd.Index(sorted(panel["datetime"].unique()))
    train_position = int(len(timestamps) * config.train_fraction)
    validation_position = int(len(timestamps) * (config.train_fraction + config.validation_fraction))
    train_cut = pd.Timestamp(timestamps[train_position])
    validation_cut = pd.Timestamp(timestamps[validation_position])
    purge = pd.Timedelta(config.timeframe) * config.horizon_bars

    train = panel.loc[panel["datetime"] < train_cut - purge].copy()
    validation = panel.loc[
        (panel["datetime"] >= train_cut) & (panel["datetime"] < validation_cut - purge)
    ].copy()
    test = panel.loc[panel["datetime"] >= validation_cut].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition")

    split_info = {
        "train_cut_utc": str(train_cut),
        "validation_cut_utc": str(validation_cut),
        "purge_hours": purge.total_seconds() / 3600,
        "train_start_utc": str(train["datetime"].min()),
        "train_end_utc": str(train["datetime"].max()),
        "validation_start_utc": str(validation["datetime"].min()),
        "validation_end_utc": str(validation["datetime"].max()),
        "test_start_utc": str(test["datetime"].min()),
        "test_end_utc": str(test["datetime"].max()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_positive_rate": float(train["target"].mean()),
        "validation_positive_rate": float(validation["target"].mean()),
        "test_positive_rate": float(test["target"].mean()),
    }
    return {"train": train, "validation": validation, "test": test}, split_info


def build_models(seed: int) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "This comparison requires lightgbm, xgboost, and catboost. "
            "Install them with: python -m pip install lightgbm xgboost catboost"
        ) from exc

    return {
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        ),
        "CatBoost": CatBoostClassifier(
            iterations=400,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            random_seed=seed,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        ),
        "Gradient Boosting Tree": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=50,
            subsample=0.85,
            random_state=seed,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=2,
                min_samples_leaf=50,
                random_state=seed,
            ),
            n_estimators=250,
            learning_rate=0.04,
            random_state=seed,
        ),
    }


def classification_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    return {
        "roc_auc": float(roc_auc_score(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "balanced_accuracy_0_50": float(balanced_accuracy_score(y_true, clipped >= 0.50)),
    }


def _close_trade(
    *,
    cash: float,
    quantity: float,
    fill: float,
    fee: float,
    entry: dict[str, Any],
    timestamp: pd.Timestamp,
    reason: str,
    symbol: str,
) -> tuple[float, dict[str, Any]]:
    exit_equity = quantity * fill * (1 - fee)
    trade = {
        "symbol": symbol,
        **entry,
        "exit_time": timestamp,
        "exit_price": fill,
        "exit_reason": reason,
        "return_pct": (exit_equity / entry["entry_equity"] - 1) * 100,
        "holding_hours": (timestamp - entry["entry_time"]).total_seconds() / 3600,
    }
    return exit_equity, trade


def simulate_symbol(
    frame: pd.DataFrame,
    symbol: str,
    entry_threshold: float,
    exit_threshold: float,
    config: ExperimentConfig,
    enable_momentum_stop: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fee = config.fee_bps / 10_000
    slip = config.slippage_bps / 10_000
    cash = 1.0
    quantity = 0.0
    entry: dict[str, Any] | None = None
    pending_action: str | None = None
    pending_reason: str | None = None
    pending_atr: float | None = None
    stop_price: float | None = None
    peak_close: float | None = None
    records: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    ordered = frame.sort_values("datetime")
    for row in ordered.itertuples(index=False):
        timestamp = pd.Timestamp(row.datetime)
        stop_event = 0

        if pending_action == "buy" and entry is None:
            fill = float(row.open) * (1 + slip)
            entry_equity = cash
            quantity = cash * (1 - fee) / fill
            cash = 0.0
            entry = {
                "entry_time": timestamp,
                "entry_price": fill,
                "entry_equity": entry_equity,
            }
            peak_close = max(fill, float(row.open))
            initial_atr = float(pending_atr if pending_atr is not None else row.atr)
            stop_price = fill - config.normal_stop_atr * initial_atr if enable_momentum_stop else None
        elif pending_action == "sell" and entry is not None:
            fill = float(row.open) * (1 - slip)
            cash, trade = _close_trade(
                cash=cash,
                quantity=quantity,
                fill=fill,
                fee=fee,
                entry=entry,
                timestamp=timestamp,
                reason=str(pending_reason),
                symbol=symbol,
            )
            trades.append(trade)
            quantity = 0.0
            entry = None
            stop_price = None
            peak_close = None
        pending_action = None
        pending_reason = None
        pending_atr = None

        if enable_momentum_stop and entry is not None and stop_price is not None and float(row.low) <= stop_price:
            raw_fill = min(float(row.open), stop_price)
            fill = raw_fill * (1 - slip)
            cash, trade = _close_trade(
                cash=cash,
                quantity=quantity,
                fill=fill,
                fee=fee,
                entry=entry,
                timestamp=timestamp,
                reason="momentum_atr_stop",
                symbol=symbol,
            )
            trades.append(trade)
            quantity = 0.0
            entry = None
            stop_price = None
            peak_close = None
            stop_event = 1

        equity = cash if entry is None else quantity * float(row.close)
        position = int(entry is not None)
        records.append(
            {
                "datetime": timestamp,
                "symbol": symbol,
                "equity": equity,
                "position": position,
                "stop_event": stop_event,
                "probability": float(row.probability),
            }
        )

        if entry is not None:
            peak_close = max(float(peak_close), float(row.close))
            if enable_momentum_stop:
                weak_momentum = int(row.momentum_score) <= 0 or float(row.probability) < entry_threshold
                stop_multiple = config.weak_stop_atr if weak_momentum else config.normal_stop_atr
                candidate_stop = peak_close - stop_multiple * float(row.atr)
                stop_price = max(float(stop_price), candidate_stop)
            if enable_momentum_stop and int(row.momentum_score) <= -4:
                pending_action = "sell"
                pending_reason = "momentum_reversal"
            elif float(row.probability) < exit_threshold:
                pending_action = "sell"
                pending_reason = "model_exit"
        elif float(row.probability) >= entry_threshold and int(row.momentum_score) > 0:
            pending_action = "buy"
            pending_reason = "model_entry"
            pending_atr = float(row.atr)

    if entry is not None and records:
        final_row = ordered.iloc[-1]
        timestamp = pd.Timestamp(final_row["datetime"])
        fill = float(final_row["close"]) * (1 - slip)
        cash, trade = _close_trade(
            cash=cash,
            quantity=quantity,
            fill=fill,
            fee=fee,
            entry=entry,
            timestamp=timestamp,
            reason="end_of_test",
            symbol=symbol,
        )
        trades.append(trade)
        records[-1]["equity"] = cash
        records[-1]["position"] = 0

    return pd.DataFrame(records), pd.DataFrame(trades)


def strategy_metrics(
    equity: pd.Series,
    exposure: pd.Series,
    trades: pd.DataFrame,
    stop_events: pd.Series,
    timeframe: str,
) -> StrategyMetrics:
    baseline_time = equity.index.min() - pd.Timedelta(timeframe)
    equity_with_baseline = pd.concat(
        [pd.Series([1.0], index=pd.DatetimeIndex([baseline_time])), equity]
    )
    returns = equity_with_baseline.pct_change().dropna()
    periods_per_year = pd.Timedelta(days=365.25) / pd.Timedelta(timeframe)
    years = max(
        (equity.index.max() - baseline_time).total_seconds() / (365.25 * 86400),
        1 / 365.25,
    )
    total_return = equity.iloc[-1] - 1
    annualized = equity.iloc[-1] ** (1 / years) - 1
    drawdown = equity_with_baseline / equity_with_baseline.cummax() - 1
    volatility = returns.std(ddof=0)
    sharpe = returns.mean() / volatility * math.sqrt(periods_per_year) if volatility else 0.0
    downside = returns.where(returns < 0, 0).std(ddof=0)
    sortino = returns.mean() / downside * math.sqrt(periods_per_year) if downside else 0.0
    max_drawdown = float(drawdown.min())
    calmar = annualized / abs(max_drawdown) if max_drawdown else 0.0
    if trades.empty:
        win_rate = 0.0
        profit_factor = 0.0
    else:
        wins = trades["return_pct"] > 0
        win_rate = float(wins.mean())
        gross_profit = float(trades.loc[trades["return_pct"] > 0, "return_pct"].sum())
        gross_loss = float(-trades.loc[trades["return_pct"] < 0, "return_pct"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else 0.0)
    stop_count = int(stop_events.sum())
    return StrategyMetrics(
        total_return_pct=float(total_return * 100),
        annualized_return_pct=float(annualized * 100),
        max_drawdown_pct=float(max_drawdown * 100),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        trades=int(len(trades)),
        win_rate_pct=win_rate * 100,
        profit_factor=float(profit_factor),
        exposure_pct=float(exposure.mean() * 100),
        stop_exits=stop_count,
        stop_exit_rate_pct=float(stop_count / len(trades) * 100) if len(trades) else 0.0,
    )


def backtest_panel(
    frame: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    config: ExperimentConfig,
    enable_momentum_stop: bool = True,
) -> BacktestResult:
    paths: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        path, symbol_trades = simulate_symbol(
            symbol_frame,
            str(symbol),
            entry_threshold,
            exit_threshold,
            config,
            enable_momentum_stop,
        )
        paths.append(path)
        if not symbol_trades.empty:
            trades.append(symbol_trades)

    combined = pd.concat(paths, ignore_index=True)
    equity_wide = combined.pivot(index="datetime", columns="symbol", values="equity").sort_index()
    position_wide = combined.pivot(index="datetime", columns="symbol", values="position").sort_index()
    stop_wide = combined.pivot(index="datetime", columns="symbol", values="stop_event").sort_index()
    portfolio_equity = equity_wide.mean(axis=1)
    exposure = position_wide.mean(axis=1)
    stop_events = stop_wide.sum(axis=1)
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    metrics = strategy_metrics(portfolio_equity, exposure, all_trades, stop_events, config.timeframe)
    return BacktestResult(portfolio_equity, exposure, stop_events, all_trades, metrics)


def buy_and_hold(frame: pd.DataFrame, config: ExperimentConfig) -> BacktestResult:
    fee = config.fee_bps / 10_000
    slip = config.slippage_bps / 10_000
    equities: list[pd.Series] = []
    for _, symbol_frame in frame.groupby("symbol", sort=True):
        ordered = symbol_frame.sort_values("datetime").set_index("datetime")
        quantity = (1 - fee) / (float(ordered.iloc[0]["open"]) * (1 + slip))
        equity = quantity * ordered["close"]
        equity.iloc[-1] = equity.iloc[-1] * (1 - fee) * (1 - slip)
        equities.append(equity)
    portfolio = pd.concat(equities, axis=1).mean(axis=1)
    exposure = pd.Series(1.0, index=portfolio.index)
    stops = pd.Series(0, index=portfolio.index, dtype=int)
    metrics = strategy_metrics(portfolio, exposure, pd.DataFrame(), stops, config.timeframe)
    return BacktestResult(portfolio, exposure, stops, pd.DataFrame(), metrics)


def choose_threshold(
    validation: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[float, float, pd.DataFrame]:
    candidates: list[dict[str, float]] = []
    probabilities = validation["probability"]
    for entry_quantile in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        exit_quantile = max(0.30, entry_quantile - 0.30)
        entry_threshold = float(probabilities.quantile(entry_quantile))
        exit_threshold = float(probabilities.quantile(exit_quantile))
        if exit_threshold >= entry_threshold:
            exit_threshold = float(np.nextafter(entry_threshold, -np.inf))
        result = backtest_panel(validation, entry_threshold, exit_threshold, config)
        row = {
            "entry_quantile": entry_quantile,
            "exit_quantile": exit_quantile,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            **asdict(result.metrics),
        }
        candidates.append(row)
    table = pd.DataFrame(candidates)
    eligible = table.loc[(table["trades"] >= 10) & (table["exposure_pct"] >= 5)]
    if eligible.empty:
        eligible = table
    selected = eligible.sort_values(
        ["sharpe", "total_return_pct", "max_drawdown_pct"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(selected["entry_threshold"]), float(selected["exit_threshold"]), table


def extract_feature_importance(model: Any, feature_columns: list[str]) -> pd.Series:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return pd.Series(dtype=float)
    importance = pd.Series(np.asarray(values, dtype=float), index=feature_columns)
    total = importance.sum()
    if total > 0:
        importance = importance / total
    return importance.sort_values(ascending=False)


def create_interactive_chart(
    summary: pd.DataFrame,
    equity_curves: dict[str, pd.Series],
    benchmark: BacktestResult,
    best_name: str,
    best_result: BacktestResult,
    best_without_stop: BacktestResult,
    feature_importance: pd.Series,
    split_info: dict[str, Any],
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    fig = make_subplots(
        rows=5,
        cols=1,
        specs=[[{}], [{}], [{"secondary_y": True}], [{"type": "table"}], [{}]],
        row_heights=[0.31, 0.16, 0.14, 0.20, 0.19],
        vertical_spacing=0.055,
        subplot_titles=(
            "锁定测试集：模型组合净值",
            f"最佳模型回撤：{best_name}",
            "最佳模型持仓暴露与动量 ATR 止损事件",
            "模型对比（锁定测试集）",
            f"{best_name} 特征重要性（Top 15）",
        ),
    )

    ranked_models = summary.sort_values(["test_sharpe", "test_total_return_pct"], ascending=False).index
    for model_name in ranked_models:
        curve = equity_curves[model_name] * config.initial_capital
        is_best = model_name == best_name
        line_color = "#2563eb" if is_best else "#94a3b8"
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve,
                name=model_name,
                mode="lines",
                line={
                    "color": line_color,
                    "width": 3.2 if is_best else 1.2,
                    "dash": MODEL_DASHES.get(model_name, "solid"),
                },
                opacity=1.0 if is_best else 0.7,
                hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>净值=%{y:,.2f} USDT<extra>" + model_name + "</extra>",
            ),
            row=1,
            col=1,
        )
    benchmark_equity = benchmark.equity * config.initial_capital
    no_stop_equity = best_without_stop.equity * config.initial_capital
    fig.add_trace(
        go.Scatter(
            x=no_stop_equity.index,
            y=no_stop_equity,
            name=f"{best_name}（关闭动量止损）",
            mode="lines",
            line={"color": "#d97706", "width": 1.8, "dash": "dash"},
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>净值=%{y:,.2f} USDT<extra>关闭动量止损</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity,
            name="等权买入持有",
            mode="lines",
            line={"color": "#334155", "width": 1.8, "dash": "dot"},
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>净值=%{y:,.2f} USDT<extra>等权买入持有</extra>",
        ),
        row=1,
        col=1,
    )

    best_drawdown = (best_result.equity / best_result.equity.cummax() - 1) * 100
    benchmark_drawdown = (benchmark.equity / benchmark.equity.cummax() - 1) * 100
    no_stop_drawdown = (best_without_stop.equity / best_without_stop.equity.cummax() - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=no_stop_drawdown.index,
            y=no_stop_drawdown,
            name="关闭动量止损回撤",
            line={"color": "#d97706", "width": 1.4, "dash": "dash"},
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>回撤=%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=best_drawdown.index,
            y=best_drawdown,
            name=f"{best_name} 回撤",
            line={"color": "#2563eb", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.12)",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>回撤=%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_drawdown.index,
            y=benchmark_drawdown,
            name="买入持有回撤",
            line={"color": "#475569", "width": 1.4, "dash": "dot"},
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>回撤=%{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=best_result.exposure.index,
            y=best_result.exposure * 100,
            name="持仓暴露",
            line={"color": "#2563eb", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.12)",
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>暴露=%{y:.0f}%<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    stop_events = best_result.stop_events.loc[best_result.stop_events > 0]
    fig.add_trace(
        go.Bar(
            x=stop_events.index,
            y=stop_events,
            name="止损退出数",
            marker={"color": "#d97706", "line": {"color": "#92400e", "width": 0.5}},
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>止损退出=%{y:.0f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    display = summary.sort_values(["test_sharpe", "test_total_return_pct"], ascending=False)
    best_marker = ["★ " if name == best_name else "" for name in display.index]
    fig.add_trace(
        go.Table(
            header={
                "values": ["模型", "入场阈值", "收益", "Sharpe", "最大回撤", "交易", "止损退出", "ROC AUC"],
                "fill_color": "#e2e8f0",
                "font": {"color": "#0f172a", "size": 12},
                "align": ["left", "right", "right", "right", "right", "right", "right", "right"],
                "height": 28,
            },
            cells={
                "values": [
                    [prefix + name for prefix, name in zip(best_marker, display.index)],
                    [f"{v:.2f}" for v in display["entry_threshold"]],
                    [f"{v:.2f}%" for v in display["test_total_return_pct"]],
                    [f"{v:.2f}" for v in display["test_sharpe"]],
                    [f"{v:.2f}%" for v in display["test_max_drawdown_pct"]],
                    [f"{int(v)}" for v in display["test_trades"]],
                    [f"{int(v)}" for v in display["test_stop_exits"]],
                    [f"{v:.3f}" for v in display["test_roc_auc"]],
                ],
                "fill_color": [["#eff6ff" if name == best_name else "#ffffff" for name in display.index]],
                "font": {"color": "#0f172a", "size": 11},
                "align": ["left", "right", "right", "right", "right", "right", "right", "right"],
                "height": 25,
            },
        ),
        row=4,
        col=1,
    )

    top_importance = feature_importance.head(15).sort_values()
    fig.add_trace(
        go.Bar(
            x=top_importance.values * 100,
            y=top_importance.index,
            orientation="h",
            name="特征重要性",
            marker={"color": "#2563eb", "line": {"color": "#1e3a8a", "width": 0.6}},
            text=[f"{value * 100:.1f}%" for value in top_importance.values],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{y}<br>归一化重要性=%{x:.2f}%<extra></extra>",
        ),
        row=5,
        col=1,
    )

    test_start = pd.Timestamp(split_info["test_start_utc"]).strftime("%Y-%m-%d")
    test_end = pd.Timestamp(split_info["test_end_utc"]).strftime("%Y-%m-%d")
    subtitle = (
        f"10 币种等权组合｜{config.timeframe} K 线｜测试期 {test_start} 至 {test_end} UTC｜"
        f"信号收盘生成、下一开盘成交｜单边费用 {config.fee_bps:.0f} bps + 滑点 {config.slippage_bps:.0f} bps"
    )
    fig.update_layout(
        title={
            "text": f"动量提升模型比较：最佳模型 {best_name}<br><sup>{subtitle}</sup>",
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        height=1520,
        hovermode="x unified",
        barmode="overlay",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        margin={"l": 85, "r": 70, "t": 135, "b": 60},
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#0f172a"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.update_yaxes(title_text="组合净值（USDT）", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="回撤（%）", ticksuffix="%", row=2, col=1)
    fig.update_yaxes(title_text="暴露（%）", range=[0, 105], row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="事件数", rangemode="tozero", row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="归一化重要性（%）", rangemode="tozero", row=5, col=1)
    for row in (1, 2, 3):
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1, row=row, col=1)
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)
    fig.update_annotations(font={"size": 13, "color": "#334155"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )
    fig.write_json(output_path.with_suffix(".plotly.json"), pretty=True)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    np.random.seed(config.random_seed)
    warnings.filterwarnings("ignore", category=FutureWarning)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    bars_by_symbol, data_quality = load_hourly_bars(config.data_dir, config.symbols, config.timeframe)
    panel, feature_columns = build_panel(bars_by_symbol, config)
    splits, split_info = chronological_split(panel, config)
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    models = build_models(config.random_seed)
    summary_rows: list[dict[str, Any]] = []
    threshold_tables: list[pd.DataFrame] = []
    equity_curves: dict[str, pd.Series] = {}
    backtests: dict[str, BacktestResult] = {}
    importance_tables: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    trade_tables: list[pd.DataFrame] = []
    scored_tests: dict[str, pd.DataFrame] = {}

    x_train = train[feature_columns]
    y_train = train["target"]
    x_validation = validation[feature_columns]
    y_validation = validation["target"]
    x_test = test[feature_columns]
    y_test = test["target"]

    for model_name, model in models.items():
        print(f"Training {model_name} on {len(train):,} panel rows...")
        model.fit(x_train, y_train)
        validation_probability = model.predict_proba(x_validation)[:, 1]
        test_probability = model.predict_proba(x_test)[:, 1]
        validation_scored = validation.copy()
        validation_scored["probability"] = validation_probability
        test_scored = test.copy()
        test_scored["probability"] = test_probability
        scored_tests[model_name] = test_scored

        entry_threshold, exit_threshold, threshold_table = choose_threshold(validation_scored, config)
        threshold_table.insert(0, "model", model_name)
        threshold_tables.append(threshold_table)
        validation_result = backtest_panel(validation_scored, entry_threshold, exit_threshold, config)
        test_result = backtest_panel(test_scored, entry_threshold, exit_threshold, config)
        test_classification = classification_metrics(y_test, test_probability)
        validation_classification = classification_metrics(y_validation, validation_probability)
        importance = extract_feature_importance(model, feature_columns)
        importance_tables.append(
            importance.rename("importance").rename_axis("feature").reset_index().assign(model=model_name)
        )

        prediction_rows.append(
            test_scored[["datetime", "symbol", "target", "forward_return"]]
            .assign(model=model_name, probability=test_probability)
        )
        if not test_result.trades.empty:
            trade_tables.append(test_result.trades.assign(model=model_name))
        equity_curves[model_name] = test_result.equity
        backtests[model_name] = test_result
        summary_rows.append(
            {
                "model": model_name,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                **{f"validation_{key}": value for key, value in asdict(validation_result.metrics).items()},
                **{f"validation_{key}": value for key, value in validation_classification.items()},
                **{f"test_{key}": value for key, value in asdict(test_result.metrics).items()},
                **{f"test_{key}": value for key, value in test_classification.items()},
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("model")
    best_name = str(
        summary.sort_values(["test_sharpe", "test_total_return_pct"], ascending=False).index[0]
    )
    best_result = backtests[best_name]
    best_without_stop = backtest_panel(
        scored_tests[best_name],
        float(summary.loc[best_name, "entry_threshold"]),
        float(summary.loc[best_name, "exit_threshold"]),
        config,
        enable_momentum_stop=False,
    )
    benchmark = buy_and_hold(test, config)
    summary["rank_by_test_sharpe"] = summary["test_sharpe"].rank(ascending=False, method="min").astype(int)
    summary["test_excess_return_vs_buy_hold_pct"] = (
        summary["test_total_return_pct"] - benchmark.metrics.total_return_pct
    )

    all_importance = pd.concat(importance_tables, ignore_index=True)
    best_importance = (
        all_importance.loc[all_importance["model"] == best_name]
        .set_index("feature")["importance"]
        .sort_values(ascending=False)
    )
    thresholds = pd.concat(threshold_tables, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    trades = pd.concat(trade_tables, ignore_index=True) if trade_tables else pd.DataFrame()

    summary.sort_values("rank_by_test_sharpe").to_csv(config.output_dir / "model_summary.csv")
    thresholds.to_csv(config.output_dir / "validation_threshold_search.csv", index=False)
    predictions.to_csv(config.output_dir / "test_predictions.csv", index=False)
    trades.to_csv(config.output_dir / "test_trades.csv", index=False)
    all_importance.to_csv(config.output_dir / "feature_importance.csv", index=False)
    data_quality.to_csv(config.output_dir / "data_quality.csv", index=False)
    equity_table = pd.concat(equity_curves, axis=1)
    equity_table["Equal Weight Buy & Hold"] = benchmark.equity
    equity_table.to_csv(config.output_dir / "test_equity_curves.csv", index_label="datetime")
    pd.DataFrame(
        {
            "best_equity": best_result.equity,
            "best_exposure": best_result.exposure,
            "stop_events": best_result.stop_events,
            "no_stop_equity": best_without_stop.equity,
            "buy_hold_equity": benchmark.equity,
        }
    ).to_csv(config.output_dir / "best_model_path.csv", index_label="datetime")
    pd.DataFrame(
        [
            {"variant": f"{best_name} + momentum stop", **asdict(best_result.metrics)},
            {"variant": f"{best_name} without momentum stop", **asdict(best_without_stop.metrics)},
            {"variant": "Equal Weight Buy & Hold", **asdict(benchmark.metrics)},
        ]
    ).to_csv(config.output_dir / "stop_ablation.csv", index=False)

    chart_path = config.output_dir / "best_model_interactive.html"
    create_interactive_chart(
        summary,
        equity_curves,
        benchmark,
        best_name,
        best_result,
        best_without_stop,
        best_importance,
        split_info,
        config,
        chart_path,
    )
    chart_map = {
        "surface": str(chart_path),
        "palette_policy": "hard two-root cap (blue/orange plus neutrals), with line styles for model identity",
        "panels": [
            {
                "panel": "Locked-test portfolio equity",
                "question": "Which boosting model produced the strongest out-of-sample equity path, and what changed without the momentum stop?",
                "family": "Trend",
                "type": "highlighted multi-series line",
                "fields": ["datetime", "model", "equity"],
            },
            {
                "panel": "Drawdown",
                "question": "How did the best model's downside path compare with buy-and-hold?",
                "family": "Trend",
                "type": "highlighted line with open fill",
                "fields": ["datetime", "drawdown_pct"],
            },
            {
                "panel": "Exposure and stops",
                "question": "When was capital exposed and when did the momentum-adaptive ATR stop fire?",
                "family": "Trend",
                "type": "area plus event bars",
                "fields": ["datetime", "exposure_pct", "stop_events"],
            },
            {
                "panel": "Model comparison",
                "question": "What are the exact locked-test metrics for each model?",
                "family": "Tables & Scorecards",
                "type": "compact table",
                "fields": ["return", "sharpe", "max_drawdown", "trades", "stops", "roc_auc"],
            },
            {
                "panel": "Feature importance",
                "question": "Which features contributed most to the best fitted model?",
                "family": "Comparison & Ranking",
                "type": "horizontal bar",
                "fields": ["feature", "normalized_importance"],
            },
        ],
    }
    (config.output_dir / "chart_map.json").write_text(
        json.dumps(chart_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    audit = {
        "experiment": "multi_asset_momentum_boosting_comparison",
        "config": {
            **asdict(config),
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "symbols": list(config.symbols),
            "round_trip_cost_pct": config.round_trip_cost * 100,
        },
        "features": feature_columns,
        "target": (
            f"1 when close[t+{config.horizon_bars}] / open[t+1] - 1 exceeds "
            f"round-trip fee plus slippage ({config.round_trip_cost:.6f})"
        ),
        "execution": "closed-bar signal; next-bar-open entry/exit; previous-close ATR stop applied to next bar",
        "stop_rule": {
            "normal_trailing_distance_atr": config.normal_stop_atr,
            "weak_momentum_trailing_distance_atr": config.weak_stop_atr,
            "weak_momentum_definition": "momentum_score <= 0 or probability below entry threshold",
            "momentum_reversal_exit": "momentum_score == -4 (all four momentum votes negative), executed next open",
            "gap_fill": "sell stop fills at min(next open, carried stop), then slippage and fee",
        },
        "model_selection": {
            "threshold_selection": (
                "model-specific validation probability quantiles; highest validation Sharpe "
                "with >=10 trades and >=5% exposure"
            ),
            "reported_best_model": "highest locked-test Sharpe; test set not used for fitting or threshold tuning",
            "best_model": best_name,
        },
        "split": split_info,
        "benchmark": asdict(benchmark.metrics),
        "best_model_stop_ablation": {
            "with_momentum_stop": asdict(best_result.metrics),
            "without_momentum_stop": asdict(best_without_stop.metrics),
            "comparison_uses_same_model_and_validation_selected_thresholds": True,
        },
        "excluded_optional_features": {
            "funding_rate": "not present in local spot OHLCV cache",
            "open_interest_change": "not present in local spot OHLCV cache",
            "taker_buy_ratio": "not present in six-column local OHLCV cache",
        },
        "caveats": [
            "This is a single historical holdout, not a live or paper-trading result.",
            "Selecting the displayed winner among several models creates model-selection uncertainty.",
            "Feature importance is model-specific association, not causality.",
            "Funding, open interest, and active-buy-volume features were unavailable.",
        ],
    }
    (config.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Best locked-test model: {best_name}")
    print(summary.sort_values("rank_by_test_sharpe")[["test_total_return_pct", "test_sharpe", "test_max_drawdown_pct"]])
    print(f"Interactive chart: {chart_path}")
    return {
        "summary": summary,
        "best_model": best_name,
        "best_result": best_result,
        "benchmark": benchmark,
        "best_without_stop": best_without_stop,
        "feature_importance": best_importance,
        "split_info": split_info,
        "chart_path": chart_path,
        "output_dir": config.output_dir,
    }


def main() -> int:
    args = parse_args()
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
        horizon_bars=args.horizon_bars,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        initial_capital=args.initial_capital,
        symbols=tuple(value.strip().upper() for value in args.symbols.split(",") if value.strip()),
    )
    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
