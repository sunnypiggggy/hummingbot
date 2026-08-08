#!/usr/bin/env python3
"""Feature-selected pair-independent XGBoost long/short Grid BUY gate v11.

This research entry point keeps the production-like Grid and replaces only the
ordinary-BUY technical gate.  It never emits a market-sell action and never
falls back to Mechanism 1.  The complete 180-day path is previously inspected,
so every result is explicitly targeted revalidation evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing as mp
import os
import urllib.parse
import urllib.request
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import optimize_xgboost_roc_sqz_pair_risk_gate_v8 as engine
import optimize_xgboost_grid_risk_gate_v7 as v7
import tune_xgboost_momentum_stop_v2 as tuner
from backtest_xgboost_long_risk_gate_180d import build_multi_horizon_panel
from compare_independent_gate_ml_stops import HOUR, PAIRS, hourly_bars, load_candles
from tune_xgboost_momentum_stop_v2 import fit_one_group, sha256_file, split_mature_training, write_json


MODEL_VERSION = "xgboost-feature-selected-pair-risk-gate-v11"
OUTPUT_DIR = Path("results/backtests/xgboost_feature_selected_pair_risk_gate_v11")
SOURCE_DIR = Path("results/backtests/xgboost_grid_risk_gate_v7")
OLD_BEST_PNL = 4.08906229455954
OLD_BEST_DRAWDOWN = -9.263364315297606
OLD_BEST_PAIR_STOPS = 7
SEED = 42
CORRELATION_LIMIT = 0.92
MIN_SELECTION_FREQUENCY = 0.60
MIN_POSITIVE_PERMUTATION_FREQUENCY = 0.60
BEAM_WIDTH = 5
MIN_FEATURES = 3
MAX_FEATURES = 8

BASE_LONG = ("adx_14", "di_spread", "atr_pct", "btc_volatility_20")
BASE_SHORT = ("price_to_ema20_atr", "volume_zscore", "di_spread")
COMMON_LONG = (
    *BASE_LONG,
    "drawdown_from_high_72h", "drawdown_from_high_168h",
    "drawdown_duration_168h", "below_ema20_ratio_72h", "lower_low_ratio_72h",
    "downside_semivariance_ratio_24h", "downside_semivariance_ratio_72h",
    "rv_24h_percentile_30d", "vol_of_vol_72h", "trend_efficiency_72h",
    "ema20_slope_atr_12h", "historical_var_72h", "expected_shortfall_72h",
    "negative_skew_72h", "taker_sell_share_24h", "taker_sell_share_72h",
    "trade_count_zscore_72h",
)
COMMON_SHORT = (
    *BASE_SHORT,
    "downside_semivariance_1h", "rv_ratio_1h_24h", "range_zscore_1h",
    "close_location_1h", "signed_volume_imbalance_1h", "amihud_zscore_24h",
    "volume_price_divergence", "max_negative_return_5m_1h", "mad_jump_score_1h",
    "taker_buy_ratio_1h", "taker_sell_imbalance_1h", "trade_count_zscore_1h",
)
PAIR_EXTRA = {
    "BTC-FDUSD": {
        "long": ("eth_sync_down_ratio_72h",),
        "short": ("eth_return_5m", "eth_return_15m", "eth_return_60m",
                  "eth_short_corr_1h", "eth_beta_change_6h"),
    },
    "ETH-FDUSD": {
        "long": ("btc_downside_beta_72h", "eth_btc_relative_drawdown_72h"),
        "short": ("btc_return_5m", "btc_return_15m", "btc_return_60m",
                  "btc_short_corr_1h", "btc_beta_change_6h"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "select", "search", "finalize", "plot", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/backtesting_candles"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--xgb-threads", type=int, default=2)
    parser.add_argument("--screen-top", type=int, default=10)
    return parser.parse_args()


def candidate_features(pair: str, target: str) -> tuple[str, ...]:
    channel = str(v7.TARGETS[target]["channel"])
    common = COMMON_LONG if channel == "long" else COMMON_SHORT
    return tuple(dict.fromkeys((*common, *PAIR_EXTRA[pair][channel])))


def _zscore(series: pd.Series, window: int, minimum: int) -> pd.Series:
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def _rolling_percentile(series: pd.Series, window: int = 720, minimum: int = 168) -> pd.Series:
    return series.rolling(window, min_periods=minimum).apply(
        lambda values: float(np.mean(values <= values[-1])), raw=True
    )


def _rolling_duration_from_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window // 2).apply(
        lambda values: float(len(values) - 1 - int(np.argmax(values))), raw=True
    )


def _downside_beta(asset: pd.Series, market: pd.Series, window: int) -> pd.Series:
    mask = market.lt(0).astype(float)
    count = mask.rolling(window, min_periods=window // 2).sum().replace(0, np.nan)
    mx = (asset * mask).rolling(window, min_periods=window // 2).sum() / count
    my = (market * mask).rolling(window, min_periods=window // 2).sum() / count
    cov = (asset * market * mask).rolling(window, min_periods=window // 2).sum() / count - mx * my
    var = (market.pow(2) * mask).rolling(window, min_periods=window // 2).sum() / count - my.pow(2)
    return cov / var.replace(0, np.nan)


def normalize_micro_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {"quote_volume": "quote_asset_volume", "trades": "n_trades",
              "taker_base": "taker_buy_base_volume", "taker_quote": "taker_buy_quote_volume"}
    frame = frame.rename(columns=rename)
    required = ["timestamp", "open", "high", "low", "close", "volume", "quote_asset_volume",
                "n_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"micro source is missing columns {missing}: {path}")
    item = frame[required].copy()
    item["timestamp"] = pd.to_numeric(item.timestamp, errors="raise")
    if float(item.timestamp.median()) > 1e11:
        item["timestamp"] = item.timestamp // 1000
    item["timestamp"] = item.timestamp.astype("int64")
    for column in required[1:]:
        item[column] = pd.to_numeric(item[column], errors="raise")
    return item.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def fetch_binance_klines(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    rows: list[list[Any]] = []
    cursor = int(start_ts) * 1000
    final = int(end_ts) * 1000
    while cursor < final:
        query = urllib.parse.urlencode({
            "symbol": symbol, "interval": "5m", "startTime": cursor,
            "endTime": final - 1, "limit": 1000,
        })
        with urllib.request.urlopen(
            f"https://api.binance.com/api/v3/klines?{query}", timeout=30
        ) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 300_000
    columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
               "quote_asset_volume", "n_trades", "taker_buy_base_volume",
               "taker_buy_quote_volume", "ignore"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame["timestamp"] = frame.timestamp.astype("int64") // 1000
    return normalize_micro_source_frame(frame)


def normalize_micro_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keep = ["timestamp", "open", "high", "low", "close", "volume", "quote_asset_volume",
            "n_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]
    item = frame[keep].copy()
    item["timestamp"] = pd.to_numeric(item.timestamp, errors="raise").astype("int64")
    for column in keep[1:]:
        item[column] = pd.to_numeric(item[column], errors="raise")
    return item.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def prepare_micro_data(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    quality = []
    cache = args.output_dir / "source_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for pair, symbol in (("BTC-FDUSD", "BTCUSDT"), ("ETH-FDUSD", "ETHUSDT")):
        target = cache / f"binance_{symbol}_5m_micro.csv.gz"
        if args.resume and target.exists():
            item = pd.read_csv(target)
        else:
            source = args.cache_dir / f"{symbol}_5m.csv"
            item = normalize_micro_source(source)
            needed_end = engine.END_TS
            next_ts = int(item.timestamp.max()) + 300
            if next_ts < needed_end:
                fetched = fetch_binance_klines(symbol, next_ts, needed_end)
                item = pd.concat([item, fetched], ignore_index=True).sort_values("timestamp")
                item = item.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
            item = item[(item.timestamp < needed_end)].copy()
            item.to_csv(target, index=False, compression="gzip")
        usable = item[(item.timestamp >= int(pd.Timestamp("2026-01-04T00:00:00Z").timestamp()))
                      & (item.timestamp < engine.END_TS)].copy()
        gaps = usable.timestamp.diff().dropna().astype(int)
        missing_rows = int(((gaps // 300) - 1).clip(lower=0).sum())
        if missing_rows or int(usable.timestamp.max()) < engine.END_TS - 300:
            raise RuntimeError(f"incomplete {symbol} micro data: missing={missing_rows}, last={usable.timestamp.max()}")
        output[pair] = usable
        quality.append({
            "pair": pair, "symbol": symbol, "rows": len(usable), "missing_5m_rows": missing_rows,
            "start_utc": pd.to_datetime(usable.timestamp.min(), unit="s", utc=True),
            "end_utc": pd.to_datetime(usable.timestamp.max(), unit="s", utc=True),
            "sha256": sha256_file(target), "source": str(target),
        })
    return output, pd.DataFrame(quality)


def build_extended_features(
    candles: Mapping[str, pd.DataFrame], micro: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    hourly = hourly_bars(candles)
    parts: dict[str, pd.DataFrame] = {}
    hourly_returns: dict[str, pd.Series] = {}
    five_returns: dict[str, pd.Series] = {}
    for pair in PAIRS:
        raw = candles[pair].copy()
        raw["time"] = pd.to_datetime(raw.timestamp, unit="s", utc=True)
        raw = raw.set_index("time").sort_index()
        r5 = np.log(raw.close).diff()
        five_returns[pair] = r5
        micro_item = micro[pair].copy()
        micro_item["time"] = pd.to_datetime(micro_item.timestamp, unit="s", utc=True)
        micro_item = micro_item.set_index("time").sort_index()
        bars = hourly[pair].copy()
        r1 = np.log(bars.close).diff()
        hourly_returns[pair] = r1
        negative = r1.clip(upper=0).pow(2)
        total = r1.pow(2)
        ema20 = bars.close.ewm(span=20, adjust=False).mean()
        rv24 = total.rolling(24, min_periods=12).sum().pow(0.5)
        typical = (bars.high + bars.low + bars.close) / 3.0
        micro_hour = micro_item.resample("1h", label="left", closed="left", origin="epoch").agg(
            volume=("volume", "sum"), quote=("quote_asset_volume", "sum"),
            trades=("n_trades", "sum"), taker_buy=("taker_buy_base_volume", "sum"),
        )
        taker_ratio = micro_hour.taker_buy / micro_hour.volume.replace(0, np.nan)
        r5_hour = r5.resample("1h", label="left", closed="left", origin="epoch")
        v5 = raw.volume
        signed_volume = (np.sign(r5) * v5).resample("1h", label="left", closed="left", origin="epoch").sum()
        volume_sum = v5.resample("1h", label="left", closed="left", origin="epoch").sum()
        downside_1h = r5.clip(upper=0).pow(2).resample("1h", label="left", closed="left", origin="epoch").sum()
        rv_1h = r5.pow(2).resample("1h", label="left", closed="left", origin="epoch").sum().pow(0.5)
        max_negative = r5_hour.min()
        dollar_volume_5m = raw.close * raw.volume
        amihud_hour = (r5.abs() / dollar_volume_5m.replace(0, np.nan)).resample(
            "1h", label="left", closed="left", origin="epoch"
        ).mean()
        out = pd.DataFrame(index=bars.index)
        out["drawdown_from_high_72h"] = bars.close / bars.close.rolling(72, min_periods=36).max() - 1
        out["drawdown_from_high_168h"] = bars.close / bars.close.rolling(168, min_periods=84).max() - 1
        out["drawdown_duration_168h"] = _rolling_duration_from_high(bars.close, 168)
        out["below_ema20_ratio_72h"] = bars.close.lt(ema20).astype(float).rolling(72, min_periods=36).mean()
        out["lower_low_ratio_72h"] = bars.low.lt(bars.low.shift()).astype(float).rolling(72, min_periods=36).mean()
        out["downside_semivariance_ratio_24h"] = negative.rolling(24, min_periods=12).sum() / total.rolling(24, min_periods=12).sum().replace(0, np.nan)
        out["downside_semivariance_ratio_72h"] = negative.rolling(72, min_periods=36).sum() / total.rolling(72, min_periods=36).sum().replace(0, np.nan)
        out["rv_24h_percentile_30d"] = _rolling_percentile(rv24)
        out["vol_of_vol_72h"] = rv24.rolling(72, min_periods=36).std(ddof=0) / rv24.rolling(72, min_periods=36).mean().replace(0, np.nan)
        out["trend_efficiency_72h"] = (bars.close - bars.close.shift(72)).abs() / bars.close.diff().abs().rolling(72, min_periods=36).sum().replace(0, np.nan)
        atr_price = bars.close * (bars.high - bars.low).rolling(14, min_periods=7).mean() / bars.close.rolling(14, min_periods=7).mean()
        out["ema20_slope_atr_12h"] = (ema20 - ema20.shift(12)) / atr_price.replace(0, np.nan)
        out["historical_var_72h"] = r1.rolling(72, min_periods=36).quantile(0.05)
        out["expected_shortfall_72h"] = r1.rolling(72, min_periods=36).apply(
            lambda x: float(np.mean(np.sort(x)[:max(1, int(math.ceil(.05 * len(x))))])), raw=True
        )
        out["negative_skew_72h"] = -r1.rolling(72, min_periods=36).skew()
        out["taker_sell_share_24h"] = (1 - taker_ratio).rolling(24, min_periods=12).mean().reindex(out.index)
        out["taker_sell_share_72h"] = (1 - taker_ratio).rolling(72, min_periods=36).mean().reindex(out.index)
        out["trade_count_zscore_72h"] = _zscore(micro_hour.trades, 72, 36).reindex(out.index)
        out["downside_semivariance_1h"] = downside_1h.reindex(out.index)
        out["rv_ratio_1h_24h"] = rv_1h.reindex(out.index) / (rv24 / math.sqrt(24)).replace(0, np.nan)
        range_pct = (bars.high - bars.low) / bars.open.replace(0, np.nan)
        out["range_zscore_1h"] = _zscore(range_pct, 168, 72)
        out["close_location_1h"] = (bars.close - bars.low) / (bars.high - bars.low).replace(0, np.nan)
        out["signed_volume_imbalance_1h"] = signed_volume.reindex(out.index) / volume_sum.reindex(out.index).replace(0, np.nan)
        out["amihud_zscore_24h"] = _zscore(amihud_hour.reindex(out.index), 168, 72)
        price_z = _zscore(r1, 72, 36)
        volume_z = _zscore(np.log1p(bars.volume).diff(), 72, 36)
        out["volume_price_divergence"] = -price_z + volume_z
        out["max_negative_return_5m_1h"] = max_negative.reindex(out.index)
        mad = max_negative.rolling(168, min_periods=72).apply(lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True)
        out["mad_jump_score_1h"] = -max_negative.reindex(out.index) / mad.replace(0, np.nan)
        out["taker_buy_ratio_1h"] = taker_ratio.reindex(out.index)
        out["taker_sell_imbalance_1h"] = (1 - 2 * taker_ratio).reindex(out.index)
        out["trade_count_zscore_1h"] = _zscore(micro_hour.trades, 168, 72).reindex(out.index)
        out["bar_open_ts"] = out.index.astype("int64") // 10**9
        out["pair"] = pair
        parts[pair] = out

    btc1, eth1 = hourly_returns["BTC-FDUSD"], hourly_returns["ETH-FDUSD"]
    btc_dd = hourly["BTC-FDUSD"].close / hourly["BTC-FDUSD"].close.rolling(72, min_periods=36).max() - 1
    eth_dd = hourly["ETH-FDUSD"].close / hourly["ETH-FDUSD"].close.rolling(72, min_periods=36).max() - 1
    parts["ETH-FDUSD"]["btc_downside_beta_72h"] = _downside_beta(eth1, btc1, 72)
    parts["ETH-FDUSD"]["eth_btc_relative_drawdown_72h"] = eth_dd - btc_dd
    parts["BTC-FDUSD"]["eth_sync_down_ratio_72h"] = (btc1.lt(0) & eth1.lt(0)).astype(float).rolling(72, min_periods=36).mean()
    for pair, leader in (("ETH-FDUSD", "BTC-FDUSD"), ("BTC-FDUSD", "ETH-FDUSD")):
        prefix = "btc" if leader.startswith("BTC") else "eth"
        lead5 = five_returns[leader]
        own5 = five_returns[pair]
        parts[pair][f"{prefix}_return_5m"] = lead5.resample("1h", label="left", closed="left").last()
        parts[pair][f"{prefix}_return_15m"] = lead5.rolling(3).sum().resample("1h", label="left", closed="left").last()
        parts[pair][f"{prefix}_return_60m"] = lead5.rolling(12).sum().resample("1h", label="left", closed="left").last()
        corr = own5.rolling(12, min_periods=6).corr(lead5).resample("1h", label="left", closed="left").last()
        beta = own5.rolling(12, min_periods=6).cov(lead5) / lead5.rolling(12, min_periods=6).var().replace(0, np.nan)
        beta = beta.resample("1h", label="left", closed="left").last()
        parts[pair][f"{prefix}_short_corr_1h"] = corr
        parts[pair][f"{prefix}_beta_change_6h"] = beta - beta.shift(6)
    return pd.concat([parts[pair].reset_index(drop=True) for pair in PAIRS], ignore_index=True)


def prepare_panel(args: argparse.Namespace, candles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    panel_path = args.output_dir / "feature_panel.csv.gz"
    if args.resume and panel_path.exists():
        cached = pd.read_csv(panel_path)
        if not cached.empty:
            return cached
    micro, quality = prepare_micro_data(args)
    quality.to_csv(args.output_dir / "micro_data_quality.csv", index=False)
    source = pd.read_csv(args.source_dir / "dual_target_feature_panel.csv.gz")
    extended = build_extended_features(candles, micro)
    added = sorted({feature for pair, target in product(PAIRS, v7.TARGETS)
                    for feature in candidate_features(pair, target)} - set(source.columns))
    panel = source.merge(extended[["pair", "bar_open_ts", *added]], on=["pair", "bar_open_ts"], how="left", validate="one_to_one")
    all_features = sorted({feature for pair, target in product(PAIRS, v7.TARGETS)
                           for feature in candidate_features(pair, target)})
    panel[all_features] = panel[all_features].replace([np.inf, -np.inf], np.nan)
    pair_parts = []
    for pair in PAIRS:
        required = sorted(set(candidate_features(pair, "long_72h") + candidate_features(pair, "short_1h_6h")))
        pair_parts.append(panel[panel.pair.eq(pair)].dropna(subset=required))
    panel = pd.concat(pair_parts, ignore_index=True).sort_values(["signal_ts", "pair"]).reset_index(drop=True)
    panel.to_csv(panel_path, index=False, compression="gzip")
    definitions = [{"pair": pair, "target": target, "channel": v7.TARGETS[target]["channel"], "feature": feature}
                   for pair, target in product(PAIRS, v7.TARGETS) for feature in candidate_features(pair, target)]
    pd.DataFrame(definitions).to_csv(args.output_dir / "feature_definitions.csv", index=False)
    return panel


def build_pressure_panel(
    args: argparse.Namespace, candles: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Build the exact v11 feature contract on a synthetic pressure path."""
    micro, _ = prepare_micro_data(args)
    source = v7.relabel_panel(build_multi_horizon_panel(candles), candles)
    extended = build_extended_features(candles, micro)
    added = sorted({
        feature for pair, target in product(PAIRS, v7.TARGETS)
        for feature in candidate_features(pair, target)
    } - set(source.columns))
    panel = source.merge(
        extended[["pair", "bar_open_ts", *added]],
        on=["pair", "bar_open_ts"], how="left", validate="one_to_one",
    )
    pair_parts = []
    for pair in PAIRS:
        required = sorted(set(
            candidate_features(pair, "long_72h")
            + candidate_features(pair, "short_1h_6h")
        ))
        item = panel[panel.pair.eq(pair)].copy()
        item[required] = item[required].replace([np.inf, -np.inf], np.nan)
        pair_parts.append(item.dropna(subset=required))
    return pd.concat(pair_parts, ignore_index=True).sort_values(
        ["signal_ts", "pair"]
    ).reset_index(drop=True)


def correlation_filter(frame: pd.DataFrame, features: Sequence[str]) -> tuple[list[str], list[dict[str, Any]]]:
    usable = [feature for feature in features if frame[feature].notna().mean() >= .99 and frame[feature].nunique() > 1]
    relevance = {feature: abs(float(frame[[feature, "target"]].corr(method="spearman").iloc[0, 1])) for feature in usable}
    ordered = sorted(usable, key=lambda value: (-np.nan_to_num(relevance[value]), value))
    correlation = frame[ordered].corr(method="spearman").abs()
    kept: list[str] = []
    rows: list[dict[str, Any]] = []
    for feature in ordered:
        conflicts = [representative for representative in kept if float(correlation.loc[feature, representative]) > CORRELATION_LIMIT]
        if conflicts:
            rows.append({"feature": feature, "representative": conflicts[0], "correlation": float(correlation.loc[feature, conflicts[0]]), "kept": False})
        else:
            kept.append(feature)
            rows.append({"feature": feature, "representative": feature, "correlation": 1.0, "kept": True})
    return kept, rows


def rotate_24h_blocks(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(copy=True)
    blocks = [array[index:index + 24] for index in range(0, len(array), 24)]
    if len(blocks) > 1:
        blocks = blocks[1:] + blocks[:1]
    return np.concatenate(blocks) if blocks else array


def selection_audit(panel: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = tuner.xgb_configurations()[1]
    rows, clusters = [], []
    for pair, target in product(PAIRS, v7.TARGETS):
        working = v7.working_target(panel, target)
        working = working[working.pair.eq(pair)].copy()
        candidates = candidate_features(pair, target)
        for block in selections.itertuples(index=False):
            mature, core, validation = split_mature_training(working, int(block.train_end))
            kept, cluster_rows = correlation_filter(mature, candidates)
            clusters.extend({"pair": pair, "target": target, "fold": int(block.fold), **item} for item in cluster_rows)
            if len(kept) < MIN_FEATURES or mature.target.nunique() < 2 or validation.target.nunique() < 2:
                continue
            model, fit = fit_one_group(config, kept, mature, core, validation)
            base_probability = model.predict_proba(validation[kept])[:, 1]
            base_loss = float(log_loss(validation.target.astype(int), base_probability, labels=[0, 1]))
            gains = np.asarray(model.feature_importances_, dtype=float)
            if gains.sum() > 0:
                gains = gains / gains.sum()
            permutation: dict[str, float] = {}
            for feature in kept:
                altered = validation[kept].copy()
                altered[feature] = rotate_24h_blocks(altered[feature])
                loss = float(log_loss(validation.target.astype(int), model.predict_proba(altered)[:, 1], labels=[0, 1]))
                permutation[feature] = loss - base_loss
            ranked = sorted(kept, key=lambda f: (-(permutation[f] > 0), -permutation[f], -gains[kept.index(f)], f))
            chosen = ranked[:MAX_FEATURES]
            if len(chosen) < MIN_FEATURES:
                chosen = ranked[:MIN_FEATURES]
            for feature, gain in zip(kept, gains):
                rows.append({
                    "pair": pair, "target": target, "channel": v7.TARGETS[target]["channel"],
                    "fold": int(block.fold), "train_cutoff_ts": int(block.train_end),
                    "last_label_ready_ts": int(mature.label_ready_ts.max()),
                    "feature": feature, "selected": feature in chosen,
                    "gain": float(gain), "permutation_logloss_increase": permutation[feature],
                    "positive_permutation": permutation[feature] > 0,
                    "best_tree_count": fit["best_tree_count"],
                })
    return pd.DataFrame(rows), pd.DataFrame(clusters)


def unique_subsets(ranking: list[str], stable: list[str], baseline: Sequence[str]) -> list[tuple[str, ...]]:
    seeds: list[Sequence[str]] = [stable[:MAX_FEATURES], stable[:6], stable[:5], stable[:4], stable[:3]]
    base = [feature for feature in baseline if feature in ranking]
    seeds.append([*base, *[feature for feature in ranking if feature not in base]][:MAX_FEATURES])
    output: list[tuple[str, ...]] = []
    for seed in seeds + [ranking[:size] for size in range(MAX_FEATURES, MIN_FEATURES - 1, -1)]:
        subset = tuple(dict.fromkeys(seed))
        if len(subset) < MIN_FEATURES:
            subset = tuple(ranking[:MIN_FEATURES])
        subset = subset[:MAX_FEATURES]
        if subset not in output:
            output.append(subset)
        if len(output) == BEAM_WIDTH:
            break
    if len(output) != BEAM_WIDTH:
        raise RuntimeError(f"could not create {BEAM_WIDTH} unique feature subsets")
    return output


def select_subsets(args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame) -> list[dict[str, Any]]:
    audit_path = args.output_dir / "feature_selection_fold_audit.csv"
    cluster_path = args.output_dir / "feature_correlation_clusters.csv"
    subset_path = args.output_dir / "selected_feature_subsets.json"
    if args.resume and audit_path.exists() and cluster_path.exists() and subset_path.exists():
        return json.loads(subset_path.read_text(encoding="utf-8"))["subsets"]
    audit, clusters = selection_audit(panel, selections)
    audit.to_csv(audit_path, index=False)
    clusters.to_csv(cluster_path, index=False)
    stability = audit.groupby(["pair", "target", "channel", "feature"], as_index=False).agg(
        folds=("fold", "nunique"), selection_frequency=("selected", "mean"),
        positive_permutation_frequency=("positive_permutation", "mean"),
        median_permutation=("permutation_logloss_increase", "median"), median_gain=("gain", "median"),
    )
    stability.to_csv(args.output_dir / "feature_stability.csv", index=False)
    subsets: list[dict[str, Any]] = []
    for pair, target in product(PAIRS, v7.TARGETS):
        group = stability[(stability.pair == pair) & (stability.target == target)].copy()
        group = group.sort_values(["selection_frequency", "positive_permutation_frequency", "median_permutation", "median_gain", "feature"], ascending=[False, False, False, False, True])
        ranking = group.feature.tolist()
        stable = group[(group.selection_frequency >= MIN_SELECTION_FREQUENCY) & (group.positive_permutation_frequency >= MIN_POSITIVE_PERMUTATION_FREQUENCY)].feature.tolist()
        if len(stable) < MIN_FEATURES:
            stable = ranking[:MIN_FEATURES]
        baseline = BASE_LONG if v7.TARGETS[target]["channel"] == "long" else BASE_SHORT
        for index, features in enumerate(unique_subsets(ranking, stable, baseline)):
            subsets.append({
                "pair": pair, "target": target, "channel": v7.TARGETS[target]["channel"],
                "subset_id": f"s{index}", "features": list(features),
                "feature_count": len(features),
            })
    write_json(subset_path, {
        "schema": "xgboost-v11-feature-subsets-v1", "model_version": MODEL_VERSION,
        "selection_frequency_minimum": MIN_SELECTION_FREQUENCY,
        "positive_permutation_frequency_minimum": MIN_POSITIVE_PERMUTATION_FREQUENCY,
        "beam_width": BEAM_WIDTH, "subsets": subsets,
    })
    return subsets


def configuration_provider(subsets: Sequence[Mapping[str, Any]]):
    base = tuner.xgb_configurations()
    lookup = {(str(row["pair"]), str(row["target"])): [] for row in subsets}
    for row in subsets:
        lookup[(str(row["pair"]), str(row["target"]))].append(row)

    def provider(target: str, pair: str) -> list[dict[str, Any]]:
        prefix = ("b" if pair.startswith("BTC") else "e") + target.replace("short_1h_6h", "s").replace("long_72h", "l72").replace("long_120h", "l120")
        output = []
        for subset in lookup[(pair, target)]:
            for config in base:
                item = dict(config)
                item["base_config_id"] = config["config_id"]
                item["subset_id"] = subset["subset_id"]
                item["features"] = tuple(subset["features"])
                item["config_id"] = f"{prefix}{subset['subset_id']}{config['config_id'].replace('xgb_', 'x')}"
                output.append(item)
        return output
    return provider


def configure_engine(args: argparse.Namespace, subsets: Sequence[Mapping[str, Any]]) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    tuner.XGB_N_JOBS = int(args.xgb_threads)
    engine.MODEL_VERSION = MODEL_VERSION
    engine.OUTPUT_DIR = args.output_dir
    engine.SOURCE_DIR = args.source_dir
    engine.MODEL_ARTIFACT_FILENAME = "xgboost_feature_selected_pair_risk_gate_v11.joblib"
    engine.MODEL_SCHEMA = "xgboost-feature-selected-pair-risk-gate-v11-model-v1"
    engine.LOCK_SCHEMA = "xgboost-feature-selected-pair-risk-gate-v11-lock-v1"
    engine.SUMMARY_SCHEMA = "xgboost-feature-selected-pair-risk-gate-v11-summary-v1"
    engine.PREDICTION_CACHE_SCHEMA = "xgboost-feature-selected-pair-v11-prediction-cache-v1"
    engine.STRATEGY_LABEL = "XGBoost v11 feature-selected independent long/short BUY gate"
    engine.PLOT_FILENAME = "xgboost_v11_feature_selected_riskoff_plotly.html"
    engine.PLOT_TITLE = "XGBoost v11：BTC/ETH独立特征筛选Risk-off驱动Grid"
    engine.FEATURE_NOTE = "长期/短期特征由嵌套时序稳定性、24h块置换及Grid目标筛选"
    engine.FEATURE_LIMITATION = "Order-book, OI, funding, basis and liquidation history are excluded for incomplete 180-day coverage."
    engine.LONG_CHANNEL_LABEL = "长期持续下跌风险"
    engine.SHORT_CHANNEL_LABEL = "1h快速插针风险"
    engine.PARAMETERS_FILENAME = "xgboost_v11_feature_subset_parameters.csv"
    engine.IMPORTANCE_FILENAME = "xgboost_v11_gain_feature_importance.csv"
    engine.SCREEN_FILENAME = "model_screen_5subsets_x40_x2pairs_x3targets_x8.csv"
    engine.SINGLE_FILENAME = "single_pair_channel_refined_search.csv"
    engine.PAIR_FILENAME = "pair_independent_long_short_search.csv"
    engine.PORTFOLIO_FILENAME = "btc_eth_independent_portfolio_search.csv"
    engine.FEATURES_BY_PAIR_TARGET = {
        (pair, target): candidate_features(pair, target) for pair, target in product(PAIRS, v7.TARGETS)
    }
    engine.CONFIGURATION_PROVIDER = configuration_provider(subsets)
    engine.CRASH_PANEL_BUILDER = lambda candles: build_pressure_panel(args, candles)


def write_subset_grid_scores(args: argparse.Namespace, screen: pd.DataFrame) -> None:
    configurations = pd.DataFrame(engine.all_configurations())[["config_id", "base_config_id", "subset_id", "features"]]
    joined = screen.merge(configurations, on="config_id", how="left", validate="many_to_one")
    grouped = joined.sort_values("rank").groupby(["pair", "target", "channel", "subset_id"], as_index=False).first()
    grouped.to_csv(args.output_dir / "feature_subset_grid_scores.csv", index=False)
    best = grouped.sort_values("rank").groupby(["pair", "channel", "subset_id"], as_index=False).first()
    rows = []
    for item in best.itertuples(index=False):
        features = list(item.features) if isinstance(item.features, (list, tuple)) else list(ast.literal_eval(str(item.features)))
        full_score = float(item.objective_score)
        for feature in features:
            rows.append({"pair": item.pair, "channel": item.channel, "subset_id": item.subset_id,
                         "feature": feature, "full_grid_objective": full_score,
                         "drop_column_status": "scheduled_in_finalists_only"})
    pd.DataFrame(rows).to_csv(args.output_dir / "drop_column_grid_ablation.csv", index=False)


_ABLATION_PANEL: pd.DataFrame | None = None
_ABLATION_SELECTIONS: pd.DataFrame | None = None


def _init_ablation_worker(panel: pd.DataFrame, selections: pd.DataFrame) -> None:
    global _ABLATION_PANEL, _ABLATION_SELECTIONS
    _ABLATION_PANEL, _ABLATION_SELECTIONS = panel, selections


def _ablation_prediction_worker(job: tuple[str, str, str, str, dict[str, Any]]):
    if _ABLATION_PANEL is None or _ABLATION_SELECTIONS is None:
        raise RuntimeError("ablation worker is not initialized")
    pair, channel, target, feature, config = job
    prediction, _ = engine.weekly_prediction(
        _ABLATION_PANEL, _ABLATION_SELECTIONS, target, pair, config
    )
    return pair, channel, target, feature, prediction


def run_drop_column_ablation(
    args: argparse.Namespace, panel: pd.DataFrame, selections: pd.DataFrame,
    candles: Mapping[str, pd.DataFrame], pair_ranked: pd.DataFrame,
    portfolio: pd.DataFrame, predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Measure every locked feature by replaying the complete weekly Grid."""
    path = args.output_dir / "drop_column_grid_ablation.csv"
    if args.resume and path.exists():
        cached = pd.read_csv(path)
        if "evaluation_scope" in cached and cached.evaluation_scope.eq(
            "weekly_walk_forward_full_grid"
        ).all():
            return cached
    winner = portfolio.iloc[0].to_dict()
    pair_rows = engine.selected_pair_rows(winner, pair_ranked)
    configs = {item["config_id"]: item for item in engine.all_configurations()}
    jobs = []
    for pair in PAIRS:
        row = pair_rows[pair]
        for channel in ("long", "short"):
            key = str(row[f"{channel}_model_key"])
            target, _, config_id = key.split("|")
            original = configs[config_id]
            for feature in original["features"]:
                reduced = dict(original)
                reduced["features"] = tuple(
                    value for value in original["features"] if value != feature
                )
                reduced["config_id"] = f"{config_id}-drop-{feature}"
                jobs.append((pair, channel, target, str(feature), reduced))
    workers = max(1, int(args.workers))
    if workers == 1:
        _init_ablation_worker(panel, selections)
        iterator = map(_ablation_prediction_worker, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            workers, initializer=_init_ablation_worker,
            initargs=(panel, selections), maxtasksperchild=4,
        )
        iterator = pool.imap_unordered(_ablation_prediction_worker, jobs, chunksize=1)
    reduced_predictions = {}
    try:
        for pair, channel, target, feature, prediction in iterator:
            reduced_predictions[(pair, channel, target, feature)] = prediction
            print(f"ABLATION {pair} {channel} -{feature}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    full = engine.detailed_replay(
        candles, selections,
        engine.locked_specifications(winner, pair_ranked, predictions),
        "XGBoost v11 locked full features",
    )["summary"]
    rows = []
    for pair, channel, target, feature, _ in jobs:
        key = str(pair_rows[pair][f"{channel}_model_key"])
        replaced = dict(predictions)
        replaced[key] = reduced_predictions[(pair, channel, target, feature)]
        drop = engine.detailed_replay(
            candles, selections,
            engine.locked_specifications(winner, pair_ranked, replaced),
            f"drop {pair} {channel} {feature}",
        )["summary"]
        contribution = (
            (float(full["oos_pnl_fdusd"]) - float(drop["oos_pnl_fdusd"])) / 420.0
            + (float(full["stitched_max_drawdown_pct"])
               - float(drop["stitched_max_drawdown_pct"])) / 100.0
        )
        rows.append({
            "pair": pair, "channel": channel, "target": target,
            "feature": feature, "evaluation_scope": "weekly_walk_forward_full_grid",
            "full_pnl_fdusd": full["oos_pnl_fdusd"],
            "drop_pnl_fdusd": drop["oos_pnl_fdusd"],
            "pnl_contribution_fdusd": float(full["oos_pnl_fdusd"]) - float(drop["oos_pnl_fdusd"]),
            "full_stitched_drawdown_pct": full["stitched_max_drawdown_pct"],
            "drop_stitched_drawdown_pct": drop["stitched_max_drawdown_pct"],
            "drawdown_contribution_pct": float(full["stitched_max_drawdown_pct"]) - float(drop["stitched_max_drawdown_pct"]),
            "full_pair_stops": full["pair_stop_events"], "drop_pair_stops": drop["pair_stop_events"],
            "full_portfolio_stops": full["portfolio_stop_events"], "drop_portfolio_stops": drop["portfolio_stop_events"],
            "grid_composite_contribution": contribution,
            "positive_grid_contribution": contribution > 0,
        })
    result = pd.DataFrame(rows).sort_values(
        ["pair", "channel", "grid_composite_contribution"],
        ascending=[True, True, False],
    )
    result.to_csv(path, index=False)
    return result


def strict_postprocess(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    lock_path = args.output_dir / "locked_configuration.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pressure = pd.read_csv(args.output_dir / "pressure_tests.csv")
    metrics = summary["winner_metrics"]
    pair_rows = summary["pair_winners"]
    strict = {
        "positive_and_beats_v9": bool(metrics["oos_pnl_fdusd"] > OLD_BEST_PNL),
        "drawdown_not_worse_than_v9": bool(metrics["stitched_max_drawdown_pct"] >= OLD_BEST_DRAWDOWN),
        "btc_non_negative": bool(metrics["btc_pnl_fdusd"] >= 0),
        "eth_non_negative": bool(metrics["eth_pnl_fdusd"] >= 0),
        "zero_portfolio_stops": bool(int(metrics["portfolio_stop_events"]) == 0),
        "fewer_than_7_pair_stops": bool(int(metrics["pair_stop_events"]) < OLD_BEST_PAIR_STOPS),
        "BTC_anchor_pass": bool(pair_rows["BTC-FDUSD"]["anchor_pass"]),
        "ETH_anchor_pass": bool(pair_rows["ETH-FDUSD"]["anchor_pass"]),
        "BTC_overlap_pass": bool(pair_rows["BTC-FDUSD"]["active_jaccard"] <= .15),
        "ETH_overlap_pass": bool(pair_rows["ETH-FDUSD"]["active_jaccard"] <= .15),
        "all_pressure_scenarios_no_stops": bool(pressure.no_stops.all()),
    }
    research_passed = bool(all(strict.values()))
    summary["acceptance"] = strict
    summary["research_gate_passed"] = research_passed
    summary["deployment_allowed"] = False
    summary["verdict"] = "NEXT_STAGE_JOINT_VALIDATION" if research_passed else "NO-GO"
    summary["evidence_status"] = "full_180d_in_sample_targeted_revalidation"
    lock["acceptance"] = strict
    lock["research_gate_passed"] = research_passed
    lock["deployment_allowed"] = False
    lock["evidence_status"] = summary["evidence_status"]
    write_json(args.output_dir / "summary.json", summary)
    write_json(lock_path, lock)
    comparison = []
    baseline = summary["baseline"]
    comparison.append({"version": "Mechanism 1", **baseline})
    for folder, label in (("xgboost_roc_sqz_pair_risk_gate_v8", "XGBoost v8"),
                          ("xgboost_regime_spike_pair_risk_gate_v9", "XGBoost v9"),
                          ("lightgbm_regime_spike_pair_risk_gate_v10", "LightGBM v10")):
        previous = json.loads((Path("results/backtests") / folder / "summary.json").read_text(encoding="utf-8"))
        comparison.append({"version": label, **previous["winner_metrics"]})
    comparison.append({"version": "XGBoost v11", **metrics})
    pd.DataFrame(comparison).to_csv(args.output_dir / "previous_version_comparison.csv", index=False)
    return summary


def run_pipeline(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candles, quality = load_candles(args.cache_dir)
    quality.to_csv(args.output_dir / "data_quality.csv", index=False)
    selections = pd.read_csv(args.source_dir / "grid_selections.csv")
    selections.to_csv(args.output_dir / "grid_selections.csv", index=False)
    panel = prepare_panel(args, candles)
    v7.target_quality(panel).to_csv(args.output_dir / "target_quality.csv", index=False)
    if args.stage == "prepare":
        return 0
    subsets = select_subsets(args, panel, selections)
    if args.stage == "select":
        return 0
    configure_engine(args, subsets)
    pd.DataFrame(engine.all_configurations()).to_csv(args.output_dir / engine.PARAMETERS_FILENAME, index=False)
    baseline_path = args.output_dir / "mechanism1_baseline.json"
    if args.resume and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = engine.baseline_metrics(candles, selections)
        write_json(baseline_path, baseline)
    screen_path = args.output_dir / engine.SCREEN_FILENAME
    if args.stage in {"search", "all"}:
        screen = engine.screen_models(args, panel, candles, selections, baseline)
        write_subset_grid_scores(args, screen)
    elif screen_path.exists():
        screen = pd.read_csv(screen_path)
    else:
        raise FileNotFoundError("run --stage search first")
    selected = engine.finalists(screen, args.screen_top)
    selected.to_csv(args.output_dir / "model_finalists.csv", index=False)
    predictions, audit = engine.load_weekly_predictions(args, panel, selections, selected)
    single_path = args.output_dir / engine.SINGLE_FILENAME
    pair_path = args.output_dir / engine.PAIR_FILENAME
    portfolio_path = args.output_dir / engine.PORTFOLIO_FILENAME
    if args.stage in {"search", "all"}:
        single = engine.refine_single(args, candles, selections, selected, predictions, baseline)
        pair_ranked = engine.pair_dual_search(args, candles, selections, single, predictions, baseline)
        portfolio = engine.portfolio_search(args, candles, selections, pair_ranked, predictions, baseline)
    elif all(path.exists() for path in (single_path, pair_path, portfolio_path)):
        pair_ranked, portfolio = pd.read_csv(pair_path), pd.read_csv(portfolio_path)
    else:
        raise FileNotFoundError("run --stage search first")
    if args.stage == "search":
        return 0
    if args.stage in {"finalize", "all"}:
        summary = engine.finalize(args, panel, candles, selections, baseline, pair_ranked, portfolio, predictions, audit)
        summary = strict_postprocess(args, summary)
        run_drop_column_ablation(
            args, panel, selections, candles, pair_ranked, portfolio, predictions
        )
        print(json.dumps({"verdict": summary["verdict"], "metrics": summary["winner_metrics"]}, ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"plot", "all"}:
        plot = engine.build_plot(args)
        print(json.dumps({"plotly": str(plot)}, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
