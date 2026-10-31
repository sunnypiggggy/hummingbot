"""Process workers for the frozen second-round cash-only study.

The parent validates and hashes the cache before dispatch. Each process loads
its own read-only-by-contract copy once; simulator helpers copy candle inputs.
Workers write disjoint run directories and return only small reviewed outputs,
never full minute-equity frames. No strategy or execution rule is changed here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from scripts import spot_strategy_suite_reporting as reporting
from scripts.spot_strategy_round2 import IDS, LABELS, simulate_strategy
from scripts.spot_strategy_suite_engine import simulate_buy_hold


_FRAMES: dict[str, pd.DataFrame] = {}
_SIGNALS: dict[Path, pd.DataFrame] = {}
_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def initialize_worker(cache_paths: Mapping[str, str | Path]) -> None:
    """Load the parent's already-validated normalized pickles once per process.

    ``cache_paths`` maps BTCUSDT and ETHUSDT to normalized ``.pkl.gz`` files.
    This initializer performs no network request and does not modify the cache.
    The complete evaluation window remains checked by the unchanged simulator.
    """
    if set(cache_paths) != set(_SYMBOLS):
        raise ValueError("Workers require the two BTCUSDT/ETHUSDT cache paths")
    frames = {}
    for symbol in _SYMBOLS:
        path = Path(cache_paths[symbol]).resolve()
        frame = pd.read_pickle(path)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"Normalized cache is not a DataFrame: {symbol}")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"Normalized cache needs UTC timestamps: {symbol}")
        if str(frame.index.tz) != "UTC" or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Normalized cache timestamps are invalid: {symbol}")
        if not {"open", "high", "low", "close", "volume"}.issubset(frame.columns):
            raise ValueError(f"Normalized cache lacks OHLCV columns: {symbol}")
        frames[symbol] = frame
    _FRAMES.clear()
    _FRAMES.update(frames)
    _SIGNALS.clear()
    reporting.LABELS.update(LABELS)  # process-local display names, no source edit


def _signals(task: Mapping[str, Any], symbol: str, strategy: str) -> pd.DataFrame:
    overrides = task.get("signals_paths", {})
    path = Path(overrides.get(symbol, Path(task["out"]) / f"signals_{strategy}_{symbol}.csv")).resolve()
    if path not in _SIGNALS:
        frame = pd.read_csv(path, dtype={"enter": "boolean", "exit": "boolean"})
        if not {"timestamp", "enter", "exit"}.issubset(frame.columns):
            raise ValueError(f"Signal file lacks required fields: {path}")
        if frame[["timestamp", "enter", "exit"]].isna().any().any():
            raise ValueError(f"Signal file contains missing values: {path}")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True), name="timestamp")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Signal timestamps must be unique and increasing: {path}")
        _SIGNALS[path] = frame[["enter", "exit"]].astype(bool)
    return _SIGNALS[path]


def run_family(task: Mapping[str, Any]) -> dict[str, Any]:
    """Run one strategy/cost/exit family, or one cost's four benchmarks.

    Required common keys are ``out``, ``symbols``, ``start``, ``end``,
    ``capital`` (total group budget), ``costs`` (fee/slippage), ``filters``,
    ``scenario`` and ``repeat_check``. Strategy tasks also provide ``strategy``,
    ``variant`` and ``stop`` (None or the frozen .05). Optional ``signals_paths``
    maps symbols to signal CSVs; otherwise files under ``out`` are read.

    ``mode='benchmark'`` runs buy-and-hold for each coin, their sum, and cash.
    Returned keys are summary_rows, histories, months, runs and
    repeat_verifications. Histories and months retain the existing daily/monthly
    DataFrame schemas; only persisted run files contain minute equity.
    """
    if set(_FRAMES) != set(_SYMBOLS):
        raise RuntimeError("Call initialize_worker before dispatching a family")
    symbols = tuple(task["symbols"])
    if len(symbols) != 2 or set(symbols) != set(_SYMBOLS):
        raise ValueError("Family requires BTCUSDT/ETHUSDT independent sleeves")
    if float(task["capital"]) <= 0:
        raise ValueError("Family capital must be positive")
    mode = task.get("mode", "strategy")
    if mode not in ("strategy", "benchmark"):
        raise ValueError(f"Unknown worker mode: {mode}")
    strategy = task.get("strategy")
    if mode == "strategy" and strategy not in IDS:
        raise ValueError(f"Unknown frozen strategy: {strategy}")
    output = Path(task["out"]).resolve()
    start, end = pd.Timestamp(task["start"]), pd.Timestamp(task["end"])
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    scenario = str(task["scenario"])
    variant = "benchmark" if mode == "benchmark" else str(task["variant"])
    if mode == "strategy" and variant not in ("original", "stop5"):
        raise ValueError(f"Unknown frozen exit variant: {variant}")
    stop = task.get("stop")
    if mode == "strategy" and stop != (None if variant == "original" else .05):
        raise ValueError("The original and stop5 families must retain their frozen stop settings")
    total_capital = float(task["capital"])
    costs = dict(task["costs"])
    results: dict[str, Any] = {
        "summary_rows": [], "histories": [], "months": [],
        "runs": {}, "repeat_verifications": [],
    }

    def record(result: dict, label: str, pair: str, repeat=None) -> None:
        run_id = f"{scenario}__{variant}__{label}__{pair}"
        fingerprints = reporting.save_result(output / "runs" / run_id, result)
        if repeat is not None and task["repeat_check"]:
            if reporting.result_fingerprints(repeat()) != fingerprints:
                raise AssertionError(f"Non-deterministic run: {run_id}")
            results["repeat_verifications"].append({"run_id": run_id, "passed": True, "fingerprints": fingerprints})
        row, history, monthly = reporting.summarize(
            result, run_id=run_id, strategy=label, pair=pair,
            scenario=scenario, variant=variant, start=start, end=end,
        )
        results["summary_rows"].append(row)
        results["histories"].append(history)
        results["months"].append(monthly)
        results["runs"][run_id] = {"fingerprints": fingerprints, "metadata": result["metadata"]}

    sleeves = []
    for symbol in symbols:
        common = dict(symbol=symbol, start=start, end=end, capital=total_capital / 2,
                      **costs, filters=task["filters"][symbol])
        if mode == "benchmark":
            run = lambda s=symbol, kw=common: simulate_buy_hold(_FRAMES[s], **kw)
            label = "buy_hold"
        else:
            signals = _signals(task, symbol, strategy)
            run = lambda s=symbol, sig=signals, kw=common: simulate_strategy(
                _FRAMES[s], sig, strategy=strategy, protective_stop=stop, **kw)
            label = strategy
        result = run()
        record(result, label, symbol, run)
        sleeves.append(result)
    combined_label = "buy_hold_equal" if mode == "benchmark" else strategy
    record(reporting.combine_sleeves(sleeves), combined_label, "PORTFOLIO")
    if mode == "benchmark":
        empty = sleeves[0]
        cash = {
            "equity": pd.DataFrame({"equity": total_capital, "cash": total_capital,
                                    "base_value": 0., "exposure": 0.}, index=empty["equity"].index),
            "fills": empty["fills"].iloc[:0].copy(), "trades": empty["trades"].iloc[:0].copy(),
            "metadata": {"initial_capital": total_capital, "final_cash": total_capital,
                         "final_positions": {}, "fees": 0, "total_notional": 0,
                         "slippage_cost": 0, "final_liquidation_equity": total_capital,
                         "unrealized_pnl": 0},
        }
        record(cash, "cash", "PORTFOLIO")
    return results
