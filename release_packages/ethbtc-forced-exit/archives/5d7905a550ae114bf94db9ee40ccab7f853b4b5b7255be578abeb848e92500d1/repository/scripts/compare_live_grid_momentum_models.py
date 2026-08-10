#!/usr/bin/env python3
"""Compare boosting-based momentum stops on the project's online FDUSD Grid.

The online baseline is replayed with its frozen weekly candidate selections,
technical BUY gate, moving-average cost floor, and existing pair/portfolio
breakers.  Models only add a six-hour risk-off overlay: cancel new BUY orders
and Taker-sell excess grid inventory while preserving the bootstrap inventory.

Signals are calculated from completed hourly bars and become actionable at the
bar close.  Labels look six hours forward, and training rows are purged until
their full label horizon is known at each weekly cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from validate_grid_live import Candidate, read_cache, simulate, technical_buy_gate_timeline


PAIRS = ("BTC-FDUSD", "ETH-FDUSD")
BAR_SECONDS = 3_600
FIVE_MINUTES = 300
INITIAL_EQUITY = 420.0
TAKER_FEE = 0.001
ORDER_REFRESH_SECONDS = 7_200
DEFAULT_HORIZON_HOURS = 6
DEFAULT_SEED = 42

CORE_FEATURES = (
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
    "pair_is_eth",
)

MODEL_NAMES = (
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "Gradient Boosting Tree",
    "AdaBoost",
)

MODEL_COLORS = {
    "Online Grid": "#111827",
    "LightGBM": "#2563EB",
    "XGBoost": "#D97706",
    "CatBoost": "#7C3AED",
    "Gradient Boosting Tree": "#059669",
    "AdaBoost": "#64748B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/backtesting_candles")
    )
    parser.add_argument(
        "--weekly-results",
        type=Path,
        default=Path(
            "results/backtests/fdusd_inventory_exit_parameter_search/weekly_results.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/backtests/live_grid_momentum_model_comparison"),
    )
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reuse-predictions", action="store_true")
    return parser.parse_args()


def _wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _linear_regression_last(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    centered = x - x.mean()
    denominator = np.dot(centered, centered)
    slope = np.dot(centered, values - values.mean()) / denominator
    return float(values.mean() + slope * (x[-1] - x.mean()))


def add_momentum_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build the compact, scale-stable momentum feature set requested."""
    out = bars.copy()
    close, high, low, volume = out.close, out.high, out.low, out.volume
    for length in (1, 5, 20):
        out[f"return_{length}"] = close.pct_change(length)
    out["roc_5"] = close.pct_change(5) * 100
    out["roc_20"] = close.pct_change(20) * 100

    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), 14)
    loss = _wilder(-delta.clip(upper=0), 14)
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)
    out["rsi_slope_3"] = out.rsi_14.diff(3) / 3
    rsi_low = out.rsi_14.rolling(14).min()
    rsi_high = out.rsi_14.rolling(14).max()
    stoch = 100 * (out.rsi_14 - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
    stoch_k = stoch.rolling(3).mean()
    out["stoch_rsi_k_minus_d"] = stoch_k - stoch_k.rolling(3).mean()

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ppo = 100 * (ema12 - ema26) / ema26.replace(0, np.nan)
    out["ppo_hist"] = ppo - ppo.ewm(span=9, adjust=False, min_periods=9).mean()
    out["ppo_hist_slope"] = out.ppo_hist.diff(3) / 3

    tsi_num = delta.ewm(span=25, adjust=False, min_periods=25).mean()
    tsi_num = tsi_num.ewm(span=13, adjust=False, min_periods=13).mean()
    tsi_den = delta.abs().ewm(span=25, adjust=False, min_periods=25).mean()
    tsi_den = tsi_den.ewm(span=13, adjust=False, min_periods=13).mean()
    out["tsi"] = 100 * tsi_num / tsi_den.replace(0, np.nan)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr = _wilder(true_range, 14)
    out["atr_pct"] = atr / close.replace(0, np.nan)
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index
    )
    plus_di = 100 * _wilder(plus_dm, 14) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, 14) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx_14"] = _wilder(dx, 14)
    out["di_spread"] = plus_di - minus_di

    basis = close.rolling(20).mean()
    midpoint = ((high.rolling(20).max() + low.rolling(20).min()) / 2 + basis) / 2
    out["sqzmom_value"] = (close - midpoint).rolling(20).apply(
        _linear_regression_last, raw=True
    )
    out["sqzmom_slope"] = out.sqzmom_value.diff(3) / 3

    volume_mean = volume.rolling(20).mean()
    out["volume_zscore"] = (
        (volume - volume_mean) / volume.rolling(20).std(ddof=0).replace(0, np.nan)
    )
    typical = (high + low + close) / 3
    money = typical * volume
    positive = money.where(typical.diff() > 0, 0.0)
    negative = money.where(typical.diff() < 0, 0.0)
    ratio = positive.rolling(14).sum() / negative.rolling(14).sum().replace(0, np.nan)
    out["mfi_14"] = 100 - 100 / (1 + ratio)
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    out["obv_slope"] = obv.diff(5) / volume.rolling(20).sum().replace(0, np.nan)
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    out["price_to_ema20_atr"] = (close - ema20) / atr.replace(0, np.nan)
    return out


def load_hourly_bars(cache_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    hourly: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        path = cache_dir / f"binance_{pair}_5m.csv"
        raw = read_cache(path).sort_values("timestamp").drop_duplicates("timestamp")
        original_count = len(pd.read_csv(path, usecols=["timestamp"]))
        invalid = (
            (raw.high < raw[["open", "close"]].max(axis=1))
            | (raw.low > raw[["open", "close"]].min(axis=1))
            | (raw.high < raw.low)
            | (raw.volume < 0)
        )
        timestamps = raw.timestamp.astype("int64")
        gaps = timestamps.diff().dropna() // FIVE_MINUTES - 1
        raw["datetime"] = pd.to_datetime(raw.timestamp, unit="s", utc=True)
        frame = raw.set_index("datetime").resample(
            "1h", label="left", closed="left", origin="epoch"
        ).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), source_rows=("close", "size"),
        )
        incomplete = int((frame.source_rows != 12).sum())
        frame = frame[frame.source_rows == 12].drop(columns="source_rows")
        hourly[pair] = frame
        quality_rows.append({
            "pair": pair,
            "source_path": str(path),
            "raw_rows": original_count,
            "duplicate_rows_removed": original_count - len(raw),
            "missing_5m_rows": int(gaps.clip(lower=0).sum()),
            "invalid_ohlcv_rows": int(invalid.sum()),
            "incomplete_hourly_bars_removed": incomplete,
            "complete_hourly_bars": len(frame),
            "start_utc": frame.index.min().isoformat(),
            "end_utc": frame.index.max().isoformat(),
        })
    return hourly, pd.DataFrame(quality_rows)


def build_feature_panel(
    hourly: Mapping[str, pd.DataFrame], horizon_hours: int
) -> pd.DataFrame:
    featured = {pair: add_momentum_features(frame) for pair, frame in hourly.items()}
    btc = featured["BTC-FDUSD"]
    btc_return = btc.return_1.rename("btc_return_1")
    btc_volatility = btc.return_1.rolling(20).std(ddof=0).rename("btc_volatility_20")
    rows: list[pd.DataFrame] = []
    for pair, frame in featured.items():
        item = frame.copy()
        item["btc_return_1"] = btc_return.reindex(item.index)
        item["btc_volatility_20"] = btc_volatility.reindex(item.index)
        item["btc_corr_48"] = item.return_1.rolling(48, min_periods=24).corr(btc_return)
        item["hour_sin"] = np.sin(2 * np.pi * item.index.hour / 24)
        item["hour_cos"] = np.cos(2 * np.pi * item.index.hour / 24)
        item["dow_sin"] = np.sin(2 * np.pi * item.index.dayofweek / 7)
        item["dow_cos"] = np.cos(2 * np.pi * item.index.dayofweek / 7)
        item["pair_is_eth"] = float(pair == "ETH-FDUSD")
        future_low = pd.concat(
            [item.low.shift(-offset) for offset in range(1, horizon_hours + 1)], axis=1
        ).min(axis=1, skipna=False)
        item["future_min_return"] = future_low / item.close - 1
        item["adverse_threshold"] = np.maximum(0.004, item.atr_pct)
        item["target"] = (item.future_min_return <= -item.adverse_threshold).astype(float)
        item.loc[future_low.isna(), "target"] = np.nan
        item["bar_open_ts"] = item.index.astype("int64") // 10**9
        item["signal_ts"] = item.bar_open_ts + BAR_SECONDS
        item["label_ready_ts"] = item.signal_ts + horizon_hours * BAR_SECONDS
        item["pair"] = pair
        rows.append(item.reset_index(names="bar_open_utc"))
    panel = pd.concat(rows, ignore_index=True)
    panel[list(CORE_FEATURES)] = panel[list(CORE_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    )
    panel = panel.dropna(subset=list(CORE_FEATURES)).sort_values(
        ["signal_ts", "pair"]
    ).reset_index(drop=True)
    return panel


def load_online_selections(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "period", "scenario", "fold", "train_end", "test_start", "test_end",
        "half_range", "min_spread", "take_profit", "move_threshold",
        "move_cooldown_seconds",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Weekly results are missing columns: {sorted(missing)}")
    frame = frame[frame.scenario == "online"].copy()
    if frame.empty:
        raise ValueError("No online weekly selections were found")
    for column in ("fold", "train_end", "test_start", "test_end", "move_cooldown_seconds"):
        frame[column] = frame[column].astype("int64")
    order = pd.Categorical(frame.period, categories=["development", "holdout"], ordered=True)
    frame = frame.assign(_period_order=order).sort_values(
        ["_period_order", "test_start", "fold"]
    ).drop(columns="_period_order")
    if frame[["period", "fold"]].duplicated().any():
        raise ValueError("Online weekly selections contain duplicate period/fold rows")
    return frame.reset_index(drop=True)


def candidate_from_row(row: Any) -> Candidate:
    return Candidate(
        float(row.half_range), float(row.min_spread), float(row.take_profit),
        float(row.move_threshold), int(row.move_cooldown_seconds),
    )


def build_models(seed: int) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError("lightgbm, xgboost and catboost are required") from exc
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=240, learning_rate=0.04, num_leaves=31,
            min_child_samples=60, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=240, learning_rate=0.04, max_depth=5,
            min_child_weight=15, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, random_state=seed, n_jobs=4, tree_method="hist",
            eval_metric="logloss",
        ),
        "CatBoost": CatBoostClassifier(
            iterations=240, learning_rate=0.04, depth=6, l2_leaf_reg=3.0,
            loss_function="Logloss", random_seed=seed, thread_count=4,
            verbose=False, allow_writing_files=False,
        ),
        "Gradient Boosting Tree": GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.04, max_depth=3,
            min_samples_leaf=40, subsample=0.85, random_state=seed,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=2, min_samples_leaf=40, random_state=seed
            ),
            n_estimators=180, learning_rate=0.04, random_state=seed,
        ),
    }


def train_weekly_predictions(
    panel: pd.DataFrame, selections: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for selection in selections.itertuples(index=False):
        training = panel[
            panel.target.notna() & (panel.label_ready_ts <= int(selection.train_end))
        ]
        testing = panel[
            (panel.signal_ts >= int(selection.test_start))
            & (panel.signal_ts < int(selection.test_end))
        ].copy()
        if training.empty or testing.empty:
            raise RuntimeError(f"Empty model partition at {selection.period} fold {selection.fold}")
        if int(training.label_ready_ts.max()) > int(selection.train_end):
            raise AssertionError("Label purge failed")
        weights = compute_sample_weight(class_weight="balanced", y=training.target.astype(int))
        output = testing[[
            "pair", "bar_open_utc", "bar_open_ts", "signal_ts", "label_ready_ts",
            "target", "future_min_return", "adverse_threshold",
        ]].copy()
        output["period"] = selection.period
        output["fold"] = int(selection.fold)
        models = build_models(seed + int(selection.fold))
        for model_name, model in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    training[list(CORE_FEATURES)], training.target.astype(int),
                    sample_weight=weights,
                )
            output[model_name] = model.predict_proba(testing[list(CORE_FEATURES)])[:, 1]
            values = getattr(model, "feature_importances_", None)
            if values is not None:
                values = np.asarray(values, dtype=float)
                total = values.sum()
                if total > 0:
                    values = values / total
                importance_rows.extend(
                    {
                        "period": selection.period,
                        "fold": int(selection.fold),
                        "model": model_name,
                        "feature": feature,
                        "importance": float(value),
                    }
                    for feature, value in zip(CORE_FEATURES, values)
                )
        prediction_rows.append(output)
        audit_rows.append({
            "period": selection.period,
            "fold": int(selection.fold),
            "train_cutoff_ts": int(selection.train_end),
            "train_rows": len(training),
            "train_start_signal_ts": int(training.signal_ts.min()),
            "train_last_signal_ts": int(training.signal_ts.max()),
            "train_last_label_ready_ts": int(training.label_ready_ts.max()),
            "test_rows": len(testing),
            "test_start_signal_ts": int(testing.signal_ts.min()),
            "test_end_signal_ts": int(testing.signal_ts.max()),
            "train_target_rate": float(training.target.mean()),
            "test_target_rate": float(testing.target.mean()),
        })
    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(importance_rows),
        pd.DataFrame(audit_rows),
    )


def probability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    holdout = predictions[(predictions.period == "holdout") & predictions.target.notna()]
    y = holdout.target.astype(int)
    for model in MODEL_NAMES:
        probability = holdout[model].clip(1e-8, 1 - 1e-8)
        rows.append({
            "model": model,
            "roc_auc": float(roc_auc_score(y, probability)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, probability)),
            "balanced_accuracy_0_5": float(balanced_accuracy_score(y, probability >= 0.5)),
            "rows": len(holdout),
            "positive_rate": float(y.mean()),
        })
    return pd.DataFrame(rows)


def build_stop_timeline(
    predictions: pd.DataFrame,
    model: str,
    threshold: float,
    start_ts: int,
    end_ts: int,
    horizon_hours: int,
) -> dict[str, dict[int, float]]:
    """Expand closed-hour triggers into a fixed, causal risk-off window."""
    timelines: dict[str, dict[int, float]] = {pair: {} for pair in PAIRS}
    rows = predictions[
        (predictions.signal_ts >= start_ts)
        & (predictions.signal_ts < end_ts)
        & (predictions[model] >= threshold)
    ]
    duration = horizon_hours * BAR_SECONDS
    for _, row in rows.iterrows():
        trigger = int(row.signal_ts)
        score = float(row[model])
        right = min(trigger + duration, end_ts)
        for timestamp in range(trigger, right, FIVE_MINUTES):
            timelines[str(row.pair)][timestamp] = max(
                score, timelines[str(row.pair)].get(timestamp, 0.0)
            )
    return timelines


def slice_candles(
    candles: Mapping[str, pd.DataFrame], start_ts: int, end_ts: int
) -> dict[str, pd.DataFrame]:
    return {
        pair: frame[(frame.timestamp >= start_ts) & (frame.timestamp < end_ts)].reset_index(drop=True)
        for pair, frame in candles.items()
    }


def replay_scenario(
    candles: Mapping[str, pd.DataFrame],
    technical_gate: Mapping[int, bool],
    selections: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
    model: str | None = None,
    threshold: float | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    record_details: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    cumulative = {period: 0.0 for period in selections.period.unique()}
    for selection in selections.itertuples(index=False):
        start, end = int(selection.test_start), int(selection.test_end)
        timeline = None
        if model is not None:
            if predictions is None or threshold is None:
                raise ValueError("Model replay requires predictions and threshold")
            fold_predictions = predictions[
                (predictions.period == selection.period) & (predictions.fold == selection.fold)
            ]
            timeline = build_stop_timeline(
                fold_predictions, model, threshold, start, end, horizon_hours
            )
        trade_log: list[dict[str, Any]] | None = [] if record_details else None
        result, curve, pairs = simulate(
            slice_candles(candles, start, end), candidate_from_row(selection),
            maker_fee=0.0, taker_fee=TAKER_FEE,
            order_refresh_seconds=ORDER_REFRESH_SECONDS,
            technical_buy_gate=dict(technical_gate),
            momentum_stop_timeline=timeline,
            momentum_stop_threshold=float(threshold or 0.5),
            trade_log=trade_log, risk_breakers_enabled=True,
            cost_floor_enabled=True, inventory_exit_policy=None,
            record_curve=record_details,
        )
        weekly_rows.append({
            "scenario": model or "Online Grid", "period": selection.period,
            "fold": int(selection.fold), "test_start": start, "test_end": end,
            "threshold": threshold, **asdict(candidate_from_row(selection)), **result,
        })
        pair_rows.extend({
            "scenario": model or "Online Grid", "period": selection.period,
            "fold": int(selection.fold), "pair": pair, **metrics,
        } for pair, metrics in pairs.items())
        if record_details and not curve.empty:
            item = curve.copy()
            item["scenario"] = model or "Online Grid"
            item["period"] = selection.period
            item["fold"] = int(selection.fold)
            item["cumulative_oos_pnl"] = (
                cumulative[selection.period] + item.equity - INITIAL_EQUITY
            )
            curve_rows.append(item)
        if trade_log is not None:
            event_rows.extend({
                "scenario": model or "Online Grid", "period": selection.period,
                "fold": int(selection.fold), "test_end": end, **event,
            } for event in trade_log)
        cumulative[selection.period] += float(result["net_pnl_quote"])
    return (
        pd.DataFrame(weekly_rows), pd.DataFrame(pair_rows),
        pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame(),
        pd.DataFrame(event_rows),
    )


def summarize_replay(weekly: pd.DataFrame, pairs: pd.DataFrame, period: str) -> dict[str, Any]:
    selected = weekly[weekly.period == period]
    pair_selected = pairs[pairs.period == period]
    returns = selected.net_pnl_quote / INITIAL_EQUITY
    weekly_std = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / weekly_std * math.sqrt(52)) if weekly_std > 0 else 0.0
    return {
        "period": period,
        "folds": len(selected),
        "oos_pnl_fdusd": float(selected.net_pnl_quote.sum()),
        "oos_return_on_weekly_capital_pct": float(selected.net_pnl_quote.sum() / (INITIAL_EQUITY * len(selected)) * 100),
        "positive_folds": int((selected.net_pnl_quote > 0).sum()),
        "worst_fold_pnl_fdusd": float(selected.net_pnl_quote.min()),
        "worst_fold_drawdown_pct": float(selected.max_drawdown_pct.min() * 100),
        "weekly_sharpe": sharpe,
        "portfolio_stop_events": int(selected.liquidated.astype(bool).sum()),
        "pair_stop_events": int(pair_selected.liquidations.sum()),
        "momentum_stop_exits": int(selected.momentum_stop_exits.sum()),
        "momentum_risk_off_pair_hours": float(pair_selected.momentum_risk_off_seconds.sum() / 3600),
        "trades": int(selected.trades.sum()),
        "fees_fdusd": float(selected.fees_quote.sum()),
        "selection_score": float(
            ((selected.net_pnl_pct - 1.5 * selected.max_drawdown_pct.abs()) * 100).sum()
        ),
    }


def select_development_thresholds(
    candles: Mapping[str, pd.DataFrame],
    gate: Mapping[int, bool],
    selections: pd.DataFrame,
    predictions: pd.DataFrame,
    horizon_hours: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    development_selections = selections[selections.period == "development"]
    development_predictions = predictions[predictions.period == "development"]
    rows: list[dict[str, Any]] = []
    chosen: dict[str, float] = {}
    quantiles = (0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99)
    for model in MODEL_NAMES:
        thresholds = sorted({
            float(development_predictions[model].quantile(quantile)) for quantile in quantiles
        })
        model_rows = []
        for threshold in thresholds:
            weekly, pairs, _, _ = replay_scenario(
                candles, gate, development_selections,
                predictions=development_predictions, model=model, threshold=threshold,
                horizon_hours=horizon_hours, record_details=False,
            )
            metrics = summarize_replay(weekly, pairs, "development")
            row = {"model": model, "threshold": threshold, **metrics}
            rows.append(row)
            model_rows.append(row)
        winner = max(
            model_rows,
            key=lambda row: (
                row["selection_score"], row["oos_pnl_fdusd"],
                row["worst_fold_drawdown_pct"], -row["momentum_stop_exits"],
            ),
        )
        chosen[model] = float(winner["threshold"])
    return pd.DataFrame(rows), chosen


def parity_audit(replayed: pd.DataFrame, source: pd.DataFrame) -> dict[str, Any]:
    columns = ["period", "fold", "net_pnl_quote", "max_drawdown_pct", "trades", "fees_quote"]
    expected = source[columns].copy()
    actual = replayed[columns].copy()
    merged = actual.merge(expected, on=["period", "fold"], suffixes=("_actual", "_source"))
    differences = {}
    for column in columns[2:]:
        delta = (merged[f"{column}_actual"] - merged[f"{column}_source"]).abs()
        differences[column] = float(delta.max())
    return {
        "rows_compared": len(merged),
        "source_rows": len(expected),
        "replayed_rows": len(actual),
        "max_absolute_differences": differences,
        "passed": len(merged) == len(expected) == len(actual)
        and differences["net_pnl_quote"] < 1e-8
        and differences["max_drawdown_pct"] < 1e-10,
    }


def create_interactive_chart(
    comparison: pd.DataFrame,
    curves: pd.DataFrame,
    weekly: pd.DataFrame,
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    importance: pd.DataFrame,
    thresholds: Mapping[str, float],
    best_model: str,
    output_path: Path,
) -> None:
    fig = make_subplots(
        rows=6, cols=1,
        specs=[[{}], [{}], [{}], [{}], [{"type": "table"}], [{}]],
        row_heights=[0.26, 0.15, 0.14, 0.16, 0.16, 0.13],
        vertical_spacing=0.047,
        subplot_titles=(
            "锁定样本外：累计周度 OOS 盈亏",
            "逐周盈亏（线上基线与最佳模型）",
            "逐折回撤路径（每周资金重新初始化）",
            f"{best_model}：小时风险概率与实际库存止损",
            "锁定样本外模型比较",
            f"{best_model}：平均特征重要性 Top 15",
        ),
    )
    holdout_curves = curves[curves.period == "holdout"]
    ordered = ["Online Grid"] + comparison.sort_values(
        ["selection_score", "oos_pnl_fdusd"], ascending=False
    ).model.tolist()
    for scenario in ordered:
        frame = holdout_curves[holdout_curves.scenario == scenario]
        if frame.empty:
            continue
        is_best = scenario == best_model
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(frame.timestamp, unit="s", utc=True),
            y=frame.cumulative_oos_pnl,
            mode="lines", name=scenario,
            line={
                "color": MODEL_COLORS.get(scenario, "#94A3B8"),
                "width": 2.8 if is_best else 1.5,
                "dash": "dash" if scenario == "Online Grid" else "solid",
            },
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>%{y:.2f} FDUSD<extra></extra>",
        ), row=1, col=1)

    fold_data = weekly[
        (weekly.period == "holdout") & weekly.scenario.isin(["Online Grid", best_model])
    ]
    for scenario in ("Online Grid", best_model):
        item = fold_data[fold_data.scenario == scenario]
        fig.add_trace(go.Bar(
            x=[f"W{int(value)}" for value in item.fold], y=item.net_pnl_quote,
            name=f"{scenario} 周盈亏", legendgroup=scenario,
            showlegend=False,
            marker_color=MODEL_COLORS[scenario], opacity=0.78,
            hovertemplate="%{x}<br>%{y:.2f} FDUSD<extra></extra>",
        ), row=2, col=1)

    for scenario in ("Online Grid", best_model):
        item = holdout_curves[holdout_curves.scenario == scenario]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(item.timestamp, unit="s", utc=True),
            y=item.drawdown_pct * 100,
            mode="lines", name=f"{scenario} 回撤", legendgroup=scenario,
            showlegend=False,
            line={"color": MODEL_COLORS[scenario], "width": 1.5,
                  "dash": "dash" if scenario == "Online Grid" else "solid"},
        ), row=3, col=1)

    best_predictions = predictions[predictions.period == "holdout"]
    for pair, color in (("BTC-FDUSD", "#0891B2"), ("ETH-FDUSD", "#7C3AED")):
        item = best_predictions[best_predictions.pair == pair]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(item.signal_ts, unit="s", utc=True), y=item[best_model],
            mode="lines", name=f"{pair} 风险概率", line={"color": color, "width": 1.2},
            hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>风险概率 %{y:.3f}<extra></extra>",
        ), row=4, col=1)
    best_events = events[
        (events.period == "holdout") & (events.scenario == best_model)
        & (events.reason == "momentum_stop_exit")
    ] if not events.empty else pd.DataFrame()
    if not best_events.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(best_events.timestamp, unit="s", utc=True),
            y=best_events.momentum_score,
            mode="markers", name="实际超额库存止损",
            marker={"color": "#DC2626", "size": 10, "symbol": "x"},
            customdata=best_events[["pair", "quote_notional"]],
            hovertemplate=("%{x|%Y-%m-%d %H:%M UTC}<br>%{customdata[0]}"
                           "<br>概率 %{y:.3f}<br>名义金额 %{customdata[1]:.2f} FDUSD<extra></extra>"),
        ), row=4, col=1)
    fig.add_hline(
        y=float(thresholds[best_model]), row=4, col=1,
        line={"color": "#DC2626", "dash": "dash", "width": 1.3},
        annotation_text=f"开发集锁定阈值 {thresholds[best_model]:.3f}",
    )

    display = comparison.sort_values(
        ["selection_score", "oos_pnl_fdusd"], ascending=False
    ).copy()
    fig.add_trace(go.Table(
        header={"values": ["模型", "阈值", "OOS盈亏", "最差回撤", "周Sharpe", "组合停止", "库存止损", "AUC"],
                "fill_color": "#E2E8F0", "align": "left"},
        cells={"values": [
            display.model,
            display.threshold.map(lambda value: f"{value:.3f}"),
            display.oos_pnl_fdusd.map(lambda value: f"{value:+.2f}"),
            display.worst_fold_drawdown_pct.map(lambda value: f"{value:.2f}%"),
            display.weekly_sharpe.map(lambda value: f"{value:.2f}"),
            display.portfolio_stop_events,
            display.momentum_stop_exits,
            display.roc_auc.map(lambda value: f"{value:.3f}"),
        ], "fill_color": "#FFFFFF", "align": "left", "height": 25},
    ), row=5, col=1)

    ranked_importance = importance[importance.model == best_model].groupby(
        "feature", as_index=False
    ).importance.mean().sort_values("importance").tail(15)
    fig.add_trace(go.Bar(
        x=ranked_importance.importance, y=ranked_importance.feature,
        orientation="h", name="特征重要性", marker_color=MODEL_COLORS[best_model],
        showlegend=False,
        hovertemplate="%{y}<br>%{x:.3f}<extra></extra>",
    ), row=6, col=1)

    fig.update_layout(
        title={
            "text": (
                f"线上 FDUSD Grid 动量止损模型比较：{best_model} 风险调整后最佳"
                "<br><sup>阈值仅在开发集选择；最终 8 周锁定样本外仅用于比较。"
                "模型只平超额库存，不改变每周网格参数与基准币仓。</sup>"
            ),
            "x": 0.02, "xanchor": "left",
        },
        template="plotly_white", height=1650, hovermode="x unified", barmode="group",
        margin={"l": 88, "r": 270, "t": 135, "b": 65},
        legend={
            "orientation": "v", "x": 1.01, "xanchor": "left",
            "y": 0.99, "yanchor": "top", "font": {"size": 10},
        },
        font={"family": "Arial, Microsoft YaHei, sans-serif", "color": "#1F2937"},
    )
    fig.update_yaxes(title_text="累计盈亏 (FDUSD)", row=1, col=1)
    fig.update_yaxes(title_text="周盈亏", row=2, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=3, col=1)
    fig.update_yaxes(title_text="风险概率", range=[0, 1], row=4, col=1)
    fig.update_yaxes(dtick=1, tickfont={"size": 9}, row=6, col=1)
    fig.update_xaxes(title_text="UTC 时间", row=4, col=1)
    fig.write_html(
        output_path, include_plotlyjs=True, full_html=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )


def write_feature_availability(output_dir: Path) -> None:
    rows = [
        {"feature_group": "价格/趋势/成交量/波动率", "status": "used",
         "details": ", ".join(CORE_FEATURES[:20])},
        {"feature_group": "BTC市场状态", "status": "used",
         "details": "BTC return, volatility, rolling pair correlation"},
        {"feature_group": "时间", "status": "used", "details": "hour and weekday cyclic encodings"},
        {"feature_group": "资金费率", "status": "unavailable",
         "details": "not present in the six-column local FDUSD candle cache"},
        {"feature_group": "持仓量 OI", "status": "unavailable",
         "details": "not present in the six-column local FDUSD candle cache"},
        {"feature_group": "主动买入占比", "status": "unavailable",
         "details": "not present in the six-column local FDUSD candle cache"},
        {"feature_group": "宏观/FOMC门", "status": "excluded",
         "details": "historical approval-state series is unavailable; excluded from all scenarios"},
    ]
    pd.DataFrame(rows).to_csv(output_dir / "feature_availability.csv", index=False)


def run_experiment(
    cache_dir: Path,
    weekly_results_path: Path,
    output_dir: Path,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    seed: int = DEFAULT_SEED,
    reuse_predictions: bool = False,
) -> dict[str, Any]:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = load_online_selections(weekly_results_path)
    candles = {
        pair: read_cache(cache_dir / f"binance_{pair}_5m.csv") for pair in PAIRS
    }
    hourly, quality = load_hourly_bars(cache_dir)
    panel = build_feature_panel(hourly, horizon_hours)
    quality.to_csv(output_dir / "data_quality.csv", index=False)
    panel[[
        "pair", "bar_open_utc", "bar_open_ts", "signal_ts", "label_ready_ts",
        "target", "future_min_return", "adverse_threshold", *CORE_FEATURES,
    ]].to_csv(output_dir / "feature_panel.csv", index=False)
    write_feature_availability(output_dir)

    prediction_path = output_dir / "weekly_predictions.csv"
    importance_path = output_dir / "feature_importance.csv"
    audit_path = output_dir / "training_audit.csv"
    if reuse_predictions and prediction_path.exists() and importance_path.exists() and audit_path.exists():
        predictions = pd.read_csv(prediction_path)
        importance = pd.read_csv(importance_path)
        training_audit = pd.read_csv(audit_path)
    else:
        predictions, importance, training_audit = train_weekly_predictions(
            panel, selections, seed
        )
        predictions.to_csv(prediction_path, index=False)
        importance.to_csv(importance_path, index=False)
        training_audit.to_csv(audit_path, index=False)

    gate = technical_buy_gate_timeline(candles["BTC-FDUSD"])
    baseline_weekly, baseline_pairs, baseline_curves, baseline_events = replay_scenario(
        candles, gate, selections, record_details=True
    )
    parity = parity_audit(baseline_weekly, selections)
    threshold_search, thresholds = select_development_thresholds(
        candles, gate, selections, predictions, horizon_hours
    )
    threshold_search.to_csv(output_dir / "development_threshold_search.csv", index=False)

    all_weekly = [baseline_weekly]
    all_pairs = [baseline_pairs]
    all_curves = [baseline_curves]
    all_events = [baseline_events]
    comparison_rows = []
    probability = probability_metrics(predictions)
    for model in MODEL_NAMES:
        weekly, pairs, curves, events = replay_scenario(
            candles, gate, selections, predictions=predictions, model=model,
            threshold=thresholds[model], horizon_hours=horizon_hours,
            record_details=True,
        )
        all_weekly.append(weekly)
        all_pairs.append(pairs)
        all_curves.append(curves)
        all_events.append(events)
        comparison_rows.append({
            "model": model, "threshold": thresholds[model],
            **summarize_replay(weekly, pairs, "holdout"),
        })
    comparison = pd.DataFrame(comparison_rows).merge(probability, on="model", how="left")
    comparison = comparison.sort_values(
        ["selection_score", "oos_pnl_fdusd"], ascending=False
    ).reset_index(drop=True)
    best_model = str(comparison.iloc[0].model)
    baseline_holdout = summarize_replay(baseline_weekly, baseline_pairs, "holdout")
    baseline_development = summarize_replay(baseline_weekly, baseline_pairs, "development")
    best_holdout = comparison.iloc[0].to_dict()

    weekly_frame = pd.concat(all_weekly, ignore_index=True)
    pair_frame = pd.concat(all_pairs, ignore_index=True)
    curve_frame = pd.concat(all_curves, ignore_index=True)
    event_frame = pd.concat(all_events, ignore_index=True)
    weekly_frame.to_csv(output_dir / "weekly_replay_metrics.csv", index=False)
    pair_frame.to_csv(output_dir / "pair_replay_metrics.csv", index=False)
    curve_frame.to_csv(output_dir / "equity_curves.csv", index=False)
    event_frame.to_csv(output_dir / "trade_and_stop_events.csv", index=False)
    comparison.to_csv(output_dir / "holdout_model_comparison.csv", index=False)

    create_interactive_chart(
        comparison, curve_frame, weekly_frame, predictions, event_frame,
        importance, thresholds, best_model, output_dir / "best_model_interactive.html",
    )

    quality_passed = bool(
        (quality.duplicate_rows_removed == 0).all()
        and (quality.invalid_ohlcv_rows == 0).all()
        and (quality.missing_5m_rows == 0).all()
    )
    purge_passed = bool(
        (training_audit.train_last_label_ready_ts <= training_audit.train_cutoff_ts).all()
    )
    best_outperformed = bool(
        best_holdout["selection_score"] > baseline_holdout["selection_score"]
    )
    summary = {
        "schema_version": "live-grid-momentum-model-comparison-v1",
        "scope": {
            "pairs": list(PAIRS),
            "initial_equity_fdusd": INITIAL_EQUITY,
            "maker_fee": 0.0,
            "taker_fee": TAKER_FEE,
            "order_refresh_seconds": ORDER_REFRESH_SECONDS,
            "weekly_selection_source": str(weekly_results_path),
            "online_grid_controls_preserved": [
                "weekly candidate selection", "ROC/SQZMOM technical BUY gate",
                "moving-average cost floor", "pair loss/drawdown breaker",
                "portfolio loss/drawdown breaker",
            ],
        },
        "method": {
            "bar_interval": "1h complete bars",
            "label": "next-six-hour minimum return <= -max(0.4%, ATR_pct)",
            "signal_effective": "hour close; first eligible 5m replay timestamp",
            "risk_off_duration_hours": horizon_hours,
            "stop_action": "cancel BUY and Taker-sell excess grid inventory only",
            "development_threshold_quantiles": [0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99],
            "ranking_objective": "sum of weekly (return - 1.5 * absolute drawdown)",
            "feature_count": len(CORE_FEATURES),
            "features": list(CORE_FEATURES),
        },
        "periods": {
            period: {
                "start_utc": pd.to_datetime(group.test_start.min(), unit="s", utc=True).isoformat(),
                "end_utc": pd.to_datetime(group.test_end.max(), unit="s", utc=True).isoformat(),
                "folds": len(group),
            }
            for period, group in selections.groupby("period", sort=False)
        },
        "baseline_development": baseline_development,
        "baseline_holdout": baseline_holdout,
        "best_model": best_model,
        "best_model_holdout": best_holdout,
        "best_outperformed_online_baseline_on_locked_objective": best_outperformed,
        "selected_thresholds": thresholds,
        "audits": {
            "data_quality_passed": quality_passed,
            "label_purge_passed": purge_passed,
            "online_replay_parity": parity,
            "all_probabilities_finite": bool(
                np.isfinite(predictions[list(MODEL_NAMES)].to_numpy()).all()
            ),
        },
        "limitations": [
            "5m OHLC limit-touch fills do not model queue position or partial fills.",
            "Funding, OI and taker-buy ratio are unavailable in the local six-column cache.",
            "The macro/FOMC gate is excluded from every scenario because historical approval states are unavailable.",
            "Weekly folds restart from 420 FDUSD, matching the project's online-model validation convention.",
            "The holdout ranks models descriptively; it is not reused to tune thresholds.",
        ],
        "deployment_authorized": False,
        "deployment_note": "Research output only; runtime configuration and OCI state were not changed.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = (
        "# 线上 FDUSD Grid 动量止损模型比较\n\n"
        f"- 最佳模型（锁定样本外风险调整目标）：**{best_model}**\n"
        f"- 最佳模型 OOS 盈亏：**{best_holdout['oos_pnl_fdusd']:+.2f} FDUSD**\n"
        f"- 线上 Grid 基线 OOS 盈亏：**{baseline_holdout['oos_pnl_fdusd']:+.2f} FDUSD**\n"
        f"- 最佳模型最差周内回撤：**{best_holdout['worst_fold_drawdown_pct']:.2f}%**\n"
        f"- 是否优于线上基线锁定目标：**{'是' if best_outperformed else '否'}**\n"
        "- 部署状态：**仅研究，未授权上线**\n\n"
        "阈值只在开发集选择；最终八周为锁定样本外。模型止损仅处理网格新增的超额库存，"
        "不卖出初始基准币仓，也不改动线上每周网格参数。\n"
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = run_experiment(
        args.cache_dir, args.weekly_results, args.output_dir,
        horizon_hours=args.horizon_hours, seed=args.seed,
        reuse_predictions=args.reuse_predictions,
    )
    print(json.dumps({
        "best_model": summary["best_model"],
        "best_model_holdout": summary["best_model_holdout"],
        "baseline_holdout": summary["baseline_holdout"],
        "audits": summary["audits"],
        "interactive_chart": str(args.output_dir / "best_model_interactive.html"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
