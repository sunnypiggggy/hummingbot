"""Independent arithmetic audit of each single-pair RL cash/coin account.

Does not import an environment, ledger, runner, trainer, policy, or financial
summary helper. Targets and actual fills are checked against original hourly
market inputs. No exchange calls, model inference, or output repair occur.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
ALGORITHMS = ("sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq")
SEEDS = tuple(range(5))
BASELINES = ("cash", "buy_hold", "sma200_trend", "momentum180")
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
SPLITS = {"validation": ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"),
          "test": ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")}
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv",
                "metadata.json", "fingerprints.json", "actions.json", "summary.json",
                "daily_equity.csv", "monthly_returns.csv")


def utc(value):
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root, relative):
    root = Path(root).resolve()
    relative = str(relative).replace("\\", "/")
    if ":" in relative or relative.startswith("/") or ".." in relative.split("/"):
        raise ValueError("Only safe root-relative audit inputs are allowed")
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


class Checks:
    def __init__(self, label=""):
        self.label, self.count, self.failed, self.failures = label, 0, 0, []
        self.maximum_errors = {}

    def check(self, condition, label, details=None):
        self.count += 1
        okay = bool(condition)
        if not okay:
            self.failed += 1
            self.failures.append({"case": self.label, "check": label, **({"details": details} if details is not None else {})})
        return okay

    def close(self, actual, expected, label, atol=1e-6, rtol=1e-10):
        try:
            actual, expected = float(actual), float(expected)
            error = abs(actual - expected)
            self.maximum_errors[label] = max(error, self.maximum_errors.get(label, 0.))
            okay = math.isfinite(actual) and math.isfinite(expected) and error <= atol + rtol * abs(expected)
        except (ValueError, TypeError, OverflowError):
            okay = False
        return self.check(okay, label, {"actual": _json(actual), "expected": _json(expected)})

    def vector(self, actual, expected, label, atol=1e-6, rtol=1e-10):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if not self.check(actual.shape == expected.shape, label + ":shape", {"actual": list(actual.shape), "expected": list(expected.shape)}):
            return False
        left, right = actual.astype(float), expected.astype(float)
        errors = np.abs(left - right)
        bad = ~np.isfinite(left) | ~np.isfinite(right) | (errors > atol + rtol * np.abs(right))
        self.maximum_errors[label] = float(errors.max()) if errors.size else 0.
        self.count += int(errors.size)
        if bad.any():
            self.failed += int(bad.sum())
            # Preserve every failing index/value, rather than silently dropping
            # a long tail of failures after a corrupt cash or equity state.
            self.failures.append({"case": self.label, "check": label, "mismatches": [
                {"index": list(index), "actual": _json(left[index]), "expected": _json(right[index])}
                for index in zip(*np.where(bad))]})
            return False
        return True

    def result(self):
        return {"passed": self.failed == 0, "checks_total": self.count, "failure_count": self.failed,
                "failures": self.failures, "maximum_errors": self.maximum_errors}


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def expected_cases(algorithms):
    return [(symbol, algorithm, seed, cost) for symbol in SYMBOLS for algorithm in algorithms for seed in SEEDS for cost in COSTS] + [
        (symbol, baseline, None, cost) for symbol in SYMBOLS for baseline in BASELINES for cost in COSTS]


def case_relative(split, symbol, algorithm, seed, scenario, repeat=False):
    return Path("repeat" if repeat else "evaluation") / split / symbol / algorithm / ("baseline" if seed is None else f"seed{seed}") / scenario


def _read_frame(path, *, index=False):
    frame = pd.read_csv(path, float_precision="round_trip")
    if index:
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
        frame = frame.set_index("timestamp")
    return frame


def _market(data, symbol, start, end):
    """Only selected-coin columns are inspected; no cross-coin dependencies."""
    frames, reference = {}, None
    for field in ("open", "high", "low", "close", "volume"):
        full = data["hourly_" + field]
        if not isinstance(full, pd.DataFrame) or list(full.columns).count(symbol) != 1:
            raise ValueError("Selected OHLCV column missing/duplicated")
        frame = full.loc[full.index < end, [symbol]].copy()
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError("Market timestamps must be timezone aware")
        frame.index = frame.index.tz_convert("UTC")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing or not frame.index.equals(frame.index.floor("h")):
            raise ValueError("Market index must be unique ordered hour opens")
        if reference is not None and not frame.index.equals(reference):
            raise ValueError("Selected OHLCV market frames are not aligned")
        frames[field], reference = frame[symbol].astype(float), frame.index
    wanted = pd.date_range(start, end, freq="h", inclusive="left")
    period = {field: series.reindex(wanted).to_numpy() for field, series in frames.items()}
    for field, values in period.items():
        if not (np.isfinite(values) & (values >= 0 if field == "volume" else values > 0)).all():
            raise ValueError("Selected-coin evaluation OHLCV is incomplete or invalid: " + field)
    previous = frames["close"].get(start - pd.Timedelta(hours=1), np.nan)
    if not np.isfinite(previous) or previous <= 0:
        raise ValueError("Previous completed close is missing")
    closes = frames["close"].loc[reference.hour == 23]
    closes.index = closes.index.normalize()
    dates = pd.date_range(reference[0].normalize(), (end - pd.Timedelta(hours=1)).normalize(), freq="D")
    closes = closes.reindex(dates).where(lambda x: np.isfinite(x) & x.gt(0))
    sma = closes / closes.rolling(200, min_periods=200).mean() - 1
    momentum = closes / closes.shift(180) - 1
    days = pd.date_range(start, end, freq="D", inclusive="left")
    signal_dates = days - pd.Timedelta(days=1)
    signals = {"sma200_trend": sma.reindex(signal_dates).to_numpy(), "momentum180": momentum.reindex(signal_dates).to_numpy()}
    if any(not np.isfinite(values).all() for values in signals.values()):
        raise ValueError("Causal selected-coin baseline dependencies are incomplete")
    return {**period, "index": wanted, "days": days, "previous_close": float(previous), "signals": signals}


def _targets(actions, market, algorithm, seed, check):
    days = market["days"]
    if seed is None:
        check.check(actions == [], "baseline has no learned-policy actions")
        return [None if algorithm == "buy_hold" and i else .95 if algorithm == "buy_hold" or algorithm in market["signals"] and market["signals"][algorithm][i] > 0 else 0.
                for i in range(len(days))]
    if not check.check(isinstance(actions, list) and len(actions) == len(days), "exactly one learned action per evaluation day"):
        raise ValueError("Action history does not cover every day")
    targets = []
    for day, record in zip(days, actions):
        check.check(utc(record["date"]) == day, "causal daily action date")
        action = np.asarray(record["action"], dtype=float)
        if not check.check(action.shape == (1,) and np.isfinite(action).all(), "one finite action, no ensemble/second asset"):
            raise ValueError("Malformed single-pair action")
        weight = .95 * max(float(np.clip(action[0], -1, 1)), 0.)
        check.vector(record["target"], [weight, 1 - weight], "action maps to exactly [coin,cash]", atol=1e-12, rtol=1e-12)
        targets.append(weight)
    return targets


def arithmetic_path(market, targets, costs):
    """Closed-form one-coin target execution, independently of saved fills.

    Reference-notional reduction d=(held-w*N)/(1-w*cSell), then increase
    d=(w*N-held)/(1+w*cBuy), constrained by the cash reserve. These equations
    replace the environment's iterative NAV solver and inventory ledger.
    """
    fee, slip = costs["fee"], costs["slippage"]
    sell_cost, buy_cost = 1 - (1 - slip) * (1 - fee), (1 + slip) * (1 + fee) - 1
    cash, qty, basis, realized, high = 10000., 0., 0., 0., 10000.
    last_nav, last_close = 10000., market["previous_close"]
    values, fills, weights, boundary = [], [], [], [(market["days"][0], 10000., 0.)]
    for day_index, (day, target) in enumerate(zip(market["days"], targets)):
        before, initial_dd = last_nav, max(0., 1 - last_nav / high)
        deepest = initial_dd
        actual_nav = cash + qty * last_close
        actual = np.array([qty * last_close / actual_nav, cash / actual_nav])
        requested = None if target is None else np.array([target, 1 - target])
        execute = requested is not None and (target <= 1e-12 and actual[0] > 0 or np.max(np.abs(requested - actual)) >= .025)
        if execute:
            reference = market["open"][day_index * 24]
            nav, held = cash + qty * reference, qty * reference
            reduction = max(0., (held - target * nav) / (1 - target * sell_cost))
            if reduction > 1e-9:
                quantity = min(reduction / reference, qty)
                price = reference * (1 - slip)
                notional, commission = quantity * price, quantity * price * fee
                removed_basis = basis * quantity / qty
                gain = notional - commission - removed_basis
                cash, qty, basis, realized = cash + notional - commission, qty - quantity, basis - removed_basis, realized + gain
                if qty <= 1e-14:
                    qty, basis = 0., 0.
                fills.append(dict(timestamp=day, side="SELL", qty=quantity, reference_price=reference, price=price,
                    notional=notional, fee=commission, slippage_cost=quantity * abs(price - reference),
                    cash_after=cash, quantity_after=qty, realized_pnl=gain, reason="target_reduction"))
        for hour in range(24):
            at = day_index * 24 + hour
            if execute and hour == 1:
                reference = market["open"][at]
                nav, held = cash + qty * reference, qty * reference
                increase = max(0., (target * nav - held) / (1 + target * buy_cost))
                reserve = max(.05, 1 - target)
                affordable = max(0., (cash - reserve * nav) / (1 + buy_cost - reserve * buy_cost))
                increase = min(increase, affordable)
                if increase > 1e-9:
                    quantity, price = increase / reference, reference * (1 + slip)
                    notional, commission = quantity * price, quantity * price * fee
                    cash, qty, basis = cash - notional - commission, qty + quantity, basis + notional + commission
                    fills.append(dict(timestamp=day + pd.Timedelta(hours=1), side="BUY", qty=quantity, reference_price=reference, price=price,
                        notional=notional, fee=commission, slippage_cost=quantity * abs(price - reference),
                        cash_after=cash, quantity_after=qty, realized_pnl=0., reason="target_increase"))
            last_close = market["close"][at]
            last_nav = cash + qty * last_close
            high = max(high, last_nav)
            deepest = max(deepest, 1 - last_nav / high)
            if cash < -1e-7 or qty < 0 or last_nav <= 0:
                raise ValueError("Closed-form path violates funded spot-account constraints")
            values.append((last_nav, cash, qty * last_close, qty * last_close / last_nav))
        reward = 100 * (math.log(last_nav / before) - .25 * max(deepest - initial_dd, 0.))
        boundary.append((day + pd.Timedelta(days=1), last_nav, reward))
        weights.append((day, target, qty * last_close / last_nav, bool(execute)))
    return {"values": np.asarray(values), "fills": fills, "weights": weights, "boundary": boundary,
            "qty": qty, "basis": basis, "realized": realized, "high_water": high}


def _identity(row, *, split, symbol, algorithm, seed, scenario, start, end, check):
    expected = {"run_id": f"{split}__{scenario}__{symbol}__{algorithm}__{'baseline' if seed is None else seed}",
                "strategy": algorithm, "symbol": symbol, "algorithm": algorithm, "seed": seed,
                "scenario": scenario, "split": split, "ensemble": False, "independent_capital": True,
                "primary_seed": seed == 0, "drawdown_grain": "1h_close"}
    for key, value in expected.items():
        actual = row.get(key)
        if key == "seed" and seed is None and (actual is None or pd.isna(actual)):
            actual = None
        check.check(actual == value, "summary identity:" + key, {"actual": _json(actual), "expected": value})
    check.check(utc(row["start"]) == start and utc(row["end_exclusive"]) == end, "summary exact window")
    check.close(row.get("initial_capital"), 10000., "summary independent capital")


def audit_case(output, market, split, symbol, algorithm, seed, scenario, start, end):
    """Read-only audit of one account; callable on synthetic fixtures in tests."""
    relative = case_relative(split, symbol, algorithm, seed, scenario)
    check = Checks(relative.as_posix())
    directory = _path(output, relative)
    inputs = {}
    try:
        for name in RESULT_FILES:
            original, repeat = _path(directory, name), _path(output, case_relative(split, symbol, algorithm, seed, scenario, True) / name)
            inputs[(relative / name).as_posix()] = sha(original)
            inputs[(case_relative(split, symbol, algorithm, seed, scenario, True) / name).as_posix()] = sha(repeat)
            check.check(inputs[(relative / name).as_posix()] == inputs[(case_relative(split, symbol, algorithm, seed, scenario, True) / name).as_posix()], "exact deterministic repeated file:" + name)
        fingerprints = read(directory / "fingerprints.json")
        check.check(set(fingerprints) == {"equity", "fills", "weights", "boundary_equity"}, "all four canonical result fingerprints")
        for key, filename in (("equity", "equity_1h.csv.gz"), ("fills", "fills.csv"), ("weights", "weights.csv"), ("boundary_equity", "boundary_equity.csv")):
            payload = gzip.decompress((directory / filename).read_bytes()) if filename.endswith(".gz") else (directory / filename).read_bytes()
            check.check(hashlib.sha256(payload).hexdigest() == fingerprints.get(key), "canonical fingerprint matches actual CSV:" + key)
        meta, row, actions = read(directory / "metadata.json"), read(directory / "summary.json"), read(directory / "actions.json")
        _identity(row, split=split, symbol=symbol, algorithm=algorithm, seed=seed, scenario=scenario, start=start, end=end, check=check)
        expected_meta = {"pair": symbol, "no_ensemble": True, "tradable_symbols": [symbol], "cash_symbol": "USDT",
                         "observation_dim": 13, "action_dim": 1, "bnb_fee_discount": False, "forced_final_liquidation": False,
                         "valuation_frequency": "1h", "training": False, "record_history": True}
        for key, value in expected_meta.items():
            check.check(meta.get(key) == value, "metadata identity:" + key)
        for key, value in (("independent_capital", 10000.), ("initial_capital", 10000.), ("cash_floor_target", .05),
                           ("max_asset_weight_target", .95), ("rebalance_band", .025), ("drawdown_penalty", .25),
                           ("fee_rate", COSTS[scenario]["fee"]), ("slippage", COSTS[scenario]["slippage"])):
            check.close(meta.get(key), value, "metadata:" + key, atol=1e-12)
        check.check(utc(meta["episode_start"]) == start and utc(meta["episode_end"]) == end and utc(meta["completed_until"]) == end, "metadata completed full window")
        check.check(len(meta.get("observation_features", [])) == 13 and all(not any(other in feature for other in SYMBOLS if other != symbol) for feature in meta.get("observation_features", [])), "no cross-coin feature identity")
        targets = _targets(actions, market, algorithm, seed, check)
        oracle = arithmetic_path(market, targets, COSTS[scenario])
        equity, fills = _read_frame(directory / "equity_1h.csv.gz", index=True), _read_frame(directory / "fills.csv")
        weights, boundary = _read_frame(directory / "weights.csv"), _read_frame(directory / "boundary_equity.csv", index=True)
        check.check(equity.index.equals(market["index"]), "every evaluation hour exactly once")
        check.vector(equity[["equity", "cash", "base_value", "exposure"]].to_numpy(), oracle["values"], "independent hourly cash inventory NAV exposure")
        check.check(len(fills) == len(oracle["fills"]), "actual fill count equals independently requested transactions")
        if len(fills):
            check.check(set(fills.asset_key) == {symbol} and set(fills.symbol) == {symbol[:-4]}, "fills contain only owned coin")
            times = pd.to_datetime(fills.timestamp, utc=True)
            check.check(times.is_monotonic_increasing and ((times >= start) & (times < end)).all(), "fills ordered inside window")
            check.check(((fills.side.eq("SELL") & times.dt.hour.eq(0)) | (fills.side.eq("BUY") & times.dt.hour.eq(1))).all() and times.dt.minute.eq(0).all() and times.dt.second.eq(0).all(), "sell00 buy01 execution only")
        for actual, expected in zip(fills.to_dict("records"), oracle["fills"]):
            check.check(utc(actual["timestamp"]) == expected["timestamp"] and actual["side"] == expected["side"] and actual["reason"] == expected["reason"], "fill identity/timing/reason")
            for field in ("qty", "reference_price", "price", "notional", "fee", "slippage_cost", "cash_after", "quantity_after", "realized_pnl"):
                check.close(actual[field], expected[field], "independent fill:" + field, atol=1e-6)
        check.check(len(weights) == len(market["days"]), "one actual weight record per day")
        if len(weights):
            check.check(set(weights.asset_key) == {symbol} and set(weights.symbol) == {symbol[:-4]} and set(weights.slot) == {0}, "one real coin weight slot only")
        for actual, expected in zip(weights.to_dict("records"), oracle["weights"]):
            day, target, actual_weight, executed = expected
            check.check(utc(actual["date"]) == day and actual["rebalanced"] == executed, "weight day and no-trade band")
            if target is None:
                check.check(pd.isna(actual["target_weight"]), "hold baseline has no rebalance target")
            else:
                check.close(actual["target_weight"], target, "weight requested target", atol=1e-12)
            check.close(actual["actual_weight"], actual_weight, "weight actual ownership", atol=1e-10)
        check.check(boundary.index.equals(pd.DatetimeIndex([item[0] for item in oracle["boundary"]], name="timestamp")), "all daily boundaries including initial capital")
        check.vector(boundary[["equity", "reward"]].to_numpy(), [[nav, reward] for _, nav, reward in oracle["boundary"]], "independent daily equity and hourly drawdown reward")
        # Reconstruct every hourly cash/quantity state from ACTUAL fill records
        # as a second check distinct from the target-derived closed-form path.
        cash, qty, actual_replay, total_fees, total_notional, total_slip = 10000., 0., [], 0., 0., 0.
        grouped = {}
        for item in fills.to_dict("records"):
            grouped.setdefault(utc(item["timestamp"]), []).append(item)
        for i, timestamp in enumerate(market["index"]):
            for item in grouped.get(timestamp, []):
                quantity, price = float(item["qty"]), float(item["price"])
                commission = quantity * price * COSTS[scenario]["fee"]
                reference = market["open"][i]
                direction = 1 if item["side"] == "BUY" else -1
                check.close(price, reference * (1 + direction * COSTS[scenario]["slippage"]), "actual replay adverse open price")
                check.close(item["fee"], commission, "actual replay quote fee")
                check.check(quantity > 0 and item["side"] in ("BUY", "SELL"), "actual replay finite positive quantity and valid side")
                if item["side"] == "BUY":
                    cash, qty = cash - quantity * price - commission, qty + quantity
                else:
                    check.check(quantity <= qty + 1e-12, "cannot sell another account's inventory")
                    cash, qty = cash + quantity * price - commission, qty - quantity
                    if qty <= 1e-14:
                        qty = 0.
                check.check(cash >= -1e-7 and qty >= 0, "cash and inventory never borrowed or shared")
                total_fees += commission
                total_notional += quantity * price
                total_slip += quantity * abs(price - reference)
            actual_replay.append((cash + qty * market["close"][i], cash, qty * market["close"][i]))
        check.vector(equity[["equity", "cash", "base_value"]].to_numpy(), actual_replay, "actual fill arithmetic reconstructs every hour")
        nav = oracle["values"][:, 0]
        peaks = np.maximum.accumulate(np.r_[10000., nav])[1:]
        drawdown = nav / peaks - 1
        final, final_cash, final_base = oracle["values"][-1, :3]
        liquidation = final_cash + final_base * (1 - COSTS[scenario]["slippage"]) * (1 - COSTS[scenario]["fee"])
        daily_nav, daily_dd = nav.reshape(-1, 24)[:, -1], drawdown.reshape(-1, 24)[:, -1]
        daily_return = daily_nav / np.r_[10000., daily_nav[:-1]] - 1
        deviation = float(daily_return.std(ddof=1)) if len(daily_return) > 1 else 0.
        sharpe = float(daily_return.mean() / deviation * np.sqrt(365)) if deviation > 1e-15 else None
        cagr = math.expm1(math.log(final / 10000.) * 365 / len(market["days"]))
        maximum_dd = float(drawdown.min())
        metrics = {"final_equity": final, "net_pnl": final - 10000., "net_return": final / 10000. - 1,
                   "max_drawdown": maximum_dd, "cagr": cagr, "sharpe_daily": sharpe,
                   "calmar": cagr / abs(maximum_dd) if maximum_dd < -1e-15 else None,
                   "fills": len(fills), "buy_fills": int(fills.side.eq("BUY").sum()) if len(fills) else 0,
                   "sell_fills": int(fills.side.eq("SELL").sum()) if len(fills) else 0,
                   "fees": total_fees, "total_notional": total_notional, "turnover": total_notional / 10000.,
                   "average_exposure": float(oracle["values"][:, 3].mean()), "exposure": float(oracle["values"][:, 3].mean()),
                   "final_cash": final_cash, "final_base_value": final_base, "net_liquidation_equity": liquidation,
                   "net_liquidation_return": liquidation / 10000. - 1, "slippage_cost": total_slip, "observed_hours": len(nav)}
        for key, expected in metrics.items():
            if expected is None:
                check.check(row.get(key) is None, "summary undefined metric:" + key)
            else:
                check.close(row.get(key), expected, "recomputed summary:" + key)
        for key, expected in (("final_equity", final), ("final_cash", final_cash), ("fees", total_fees),
                              ("total_notional", total_notional), ("slippage_cost", total_slip), ("fill_count", len(fills)),
                              ("final_liquidation_equity", liquidation), ("hourly_high_water", float(peaks.max())),
                              ("current_drawdown", max(0., 1 - final / peaks[-1])), ("realized_pnl", oracle["realized"]),
                              ("unrealized_pnl", final_base - oracle["basis"])):
            check.close(meta.get(key), expected, "recomputed metadata:" + key)
        positions = meta.get("final_positions", {})
        check.check(set(positions).issubset({symbol}) and (oracle["qty"] > 0) == (symbol in positions), "final ownership only own coin")
        check.close(positions.get(symbol, 0.), oracle["qty"], "final owned quantity", atol=1e-10)
        holdings = meta.get("final_holdings", [])
        check.check(len(holdings) == int(oracle["qty"] > 0), "final holdings count")
        for holding in holdings:
            check.check(holding["asset_key"] == symbol and holding["symbol"] == symbol[:-4], "final holding identity")
            for key, expected in (("qty", oracle["qty"]), ("mark_price", market["close"][-1]), ("market_value", final_base), ("weight", final_base / final)):
                check.close(holding.get(key), expected, "final holding:" + key)
        daily = _read_frame(directory / "daily_equity.csv")
        expected_times = pd.DatetimeIndex([start, *[day + pd.Timedelta(days=1) for day in market["days"]]])
        check.check(pd.DatetimeIndex(pd.to_datetime(daily.timestamp, utc=True)).equals(expected_times), "daily chart exact boundaries")
        check.vector(daily[["equity", "net_return", "drawdown"]].to_numpy(), np.column_stack([np.r_[10000., daily_nav], np.r_[0., daily_nav / 10000. - 1], np.r_[0., daily_dd]]), "daily chart reconciles hourly marks and peaks")
        monthly = _read_frame(directory / "monthly_returns.csv")
        month_close = pd.Series(daily_nav, index=market["days"]).groupby(market["days"].strftime("%Y-%m")).last()
        month_return = month_close.to_numpy() / np.r_[10000., month_close.to_numpy()[:-1]] - 1
        check.check(monthly.month.tolist() == month_close.index.tolist(), "real partial-month labels")
        check.vector(monthly["return"].to_numpy(), month_return, "monthly returns compound from actual boundaries", atol=1e-10)
        check.close(float(np.prod(1 + monthly["return"].to_numpy()) - 1), final / 10000. - 1, "monthly compounding equals total net return", atol=1e-10)
        for name, frame in (("daily", daily), ("monthly", monthly)):
            for key in ("run_id", "strategy", "scenario", "split"):
                check.check(frame[key].eq(row[key]).all(), name + " chart identity:" + key)
            if seed is None:
                check.check(frame.seed.isna().all(), name + " baseline is not a seed")
            else:
                check.check(frame.seed.eq(seed).all(), name + " seed identity")
        for relative_name, expected_sha in inputs.items():
            check.check(sha(_path(output, relative_name)) == expected_sha, "case input unchanged during audit:" + relative_name)
    except Exception as exc:
        check.check(False, "case exception", {"type": type(exc).__name__, "error": str(exc)})
        row = None
    return {**check.result(), "relative_path": relative.as_posix(), "inputs_sha256": inputs, "summary": row}


def _protocol(spec, dataset, check):
    for key, expected in (("symbols", list(SYMBOLS)), ("seeds", list(SEEDS)), ("primary_seed", 0),
                          ("primary_seed_fixed_before_training", True), ("capital_per_account", 10000.),
                          ("capital_shared_between_pairs", False), ("ensemble", False), ("costs", COSTS),
                          ("baselines", list(BASELINES)), ("test_year_already_observed", True), ("fresh_holdout", False)):
        check.check(spec.get(key) == expected, "frozen protocol:" + key)
    algorithms = spec["algorithms"]
    check.check(bool(algorithms) and len(set(algorithms)) == len(algorithms) and set(algorithms) <= set(ALGORITHMS), "unique declared algorithms only")
    for split, bounds in SPLITS.items():
        check.check(spec.get(split) == list(bounds), "fixed frozen window:" + split)
    check.check(spec.get("train") == ["2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "fixed training cutoff")
    for key, expected in (("observation_dim", 13), ("action_dim", 1), ("cash_floor", .05), ("max_coin_target", .95),
                          ("rebalance_band", .025), ("drawdown_penalty", .25), ("episode_days", 32)):
        check.check(spec.get("environment", {}).get(key) == expected, "frozen environment:" + key)
    check.check(spec.get("validation_selection", {}).get("seed_selection") is False, "no selected/cherry-picked seed")
    check.check(sha(dataset) == spec.get("dataset_sha256"), "original dataset hash")
    check.check(sha(Path(dataset).parent / "manifest.json") == spec.get("dataset_manifest_sha256"), "original dataset manifest hash")


_WORKER = None


def _initialize_worker(output, markets, split, start, end):
    global _WORKER
    _WORKER = (output, markets, split, start, end)


def _worker(case):
    output, markets, split, start, end = _WORKER
    symbol, algorithm, seed, scenario = case
    return audit_case(output, markets[symbol], split, symbol, algorithm, seed, scenario, start, end)


def _cpu_workers(count):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        limit, period = quota.read_text().split()
        if limit != "max":
            available = min(available, max(1, int(limit) // int(period)))
    return min(count, max(1, int(available * .85)))


def audit_phase(output, dataset, split, *, workers=None):
    """Audit the complete frozen phase and retain a timestamped failure trail."""
    if split not in SPLITS:
        raise ValueError("Only validation or test can be audited")
    output, dataset = Path(output).resolve(), Path(dataset).resolve()
    check = Checks(split)
    results, inputs, expected, actual_workers = [], {}, [], 0
    started = datetime.now(timezone.utc).isoformat()
    try:
        spec = read(output / "protocol.json")
        _protocol(spec, dataset, check)
        inputs["protocol.json"] = sha(output / "protocol.json")
        for relative, expected_sha in spec["sources"].items():
            check.check(sha(_path(ROOT, relative)) == expected_sha, "executed source hash:" + relative)
            check.check(sha(_path(output, Path("frozen_source") / relative)) == expected_sha, "frozen source hash:" + relative)
        if check.failed:
            raise ValueError("Frozen protocol/source/data verification failed")
        with gzip.open(dataset, "rb") as stream:
            data = pickle.load(stream)
        start, end = map(utc, SPLITS[split])
        markets = {symbol: _market(data, symbol, start, end) for symbol in SYMBOLS}
        expected = expected_cases(spec["algorithms"])
        wanted = {case_relative(split, *case).relative_to(Path("evaluation") / split).as_posix() for case in expected}
        for kind in ("evaluation", "repeat"):
            phase_root = output / kind / split
            observed = {path.parent.relative_to(phase_root).as_posix() for path in phase_root.rglob("summary.json")}
            check.check(observed == wanted, "complete unique case population:" + kind, {"missing": sorted(wanted - observed), "unexpected": sorted(observed - wanted)})
        completion = read(output / f"{split}_complete.json")
        inputs[f"{split}_complete.json"] = sha(output / f"{split}_complete.json")
        check.check(completion.get("accounts") == len(expected) and completion.get("repeats") == len(expected) and completion.get("repeat_equal") is True, "phase completion counts and repeated execution")
        actual_workers = _cpu_workers(len(expected)) if workers is None else max(1, min(len(expected), int(workers)))
        if actual_workers == 1:
            results = [audit_case(output, markets[case[0]], split, *case, start, end) for case in expected]
        else:
            with ProcessPoolExecutor(max_workers=actual_workers, initializer=_initialize_worker, initargs=(output, markets, split, start, end)) as pool:
                results = list(pool.map(_worker, expected))
        aggregate = _read_frame(output / f"{split}_summary.csv")
        inputs[f"{split}_summary.csv"] = sha(output / f"{split}_summary.csv")
        check.check(len(aggregate) == len(expected) and not aggregate.run_id.duplicated().any(), "aggregate has exactly one row per independent account")
        for result in results:
            row = result.get("summary")
            if row is None:
                continue
            matched = aggregate.loc[aggregate.run_id.eq(row["run_id"])]
            if check.check(len(matched) == 1, "aggregate includes case:" + row["run_id"]):
                actual = matched.iloc[0].to_dict()
                for key, value in row.items():
                    if value is None:
                        check.check(actual.get(key) is None or pd.isna(actual.get(key)), "aggregate nullable identity/metric:" + key)
                    elif isinstance(value, bool) or isinstance(value, str):
                        check.check(actual.get(key) == value, "aggregate identity:" + key)
                    else:
                        check.close(actual.get(key), value, "aggregate value:" + key)
        completion_ids = [item.get("summary", {}).get("run_id") for item in completion.get("results", [])]
        check.check(len(completion_ids) == len(expected) and set(completion_ids) == set(aggregate.run_id), "completion evidence has every actual account exactly once")
        if split == "test":
            selection = read(output / "validation_selection.json")
            check.check(selection.get("uses_test") is False and selection.get("seed_selection") is False, "annual selection was validation-only")
            for symbol in SYMBOLS:
                choice = selection["selection"][symbol]
                check.check(choice.get("algorithm") in spec["algorithms"] and choice.get("seed") == 0, "annual selected account uses fixed primary seed:" + symbol)
        for relative_name, expected_sha in inputs.items():
            check.check(sha(_path(output, relative_name)) == expected_sha, "phase input unchanged during audit:" + relative_name)
        check.check(sha(dataset) == spec["dataset_sha256"], "dataset unchanged through audit completion")
    except Exception as exc:
        check.check(False, "phase exception", {"type": type(exc).__name__, "error": str(exc)})
    output.mkdir(parents=True, exist_ok=True)
    failed = check.failed + sum(item["failure_count"] for item in results)
    result = {"phase": split, "started_at": started, "audited_at": datetime.now(timezone.utc).isoformat(),
              "passed": failed == 0, "checks_total": check.count + sum(item["checks_total"] for item in results),
              "failure_count": failed, "failures": check.failures + [failure for item in results for failure in item["failures"]],
              "expected_accounts": len(expected), "accounts_checked": len(results), "repeats_checked": len(results),
              "accounts_count": len(results), "summary_sha256": inputs.get(f"{split}_summary.csv"),
              "protocol_sha256": inputs.get("protocol.json"),
              "worker_count": actual_workers, "capital_per_account": 10000., "capital_shared_between_pairs": False,
              "ensemble": False, "primary_seed": 0, "independent_oracle": "closed-form selected-coin targets plus actual-fill cash/inventory replay; no environment/ledger/policy imports",
              "inputs_sha256": inputs, "cases": results}
    text = json.dumps(_json(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # A new attempt remains inspectable even if a previous failed audit exists.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    history = output / "audit_history"
    history.mkdir(exist_ok=True)
    (history / f"{split}_audit_{stamp}.json").write_text(text, encoding="utf-8")
    temporary = output / f"{split}_audit.json.pending"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output / f"{split}_audit.json")
    return _json(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("split", choices=tuple(SPLITS))
    parser.add_argument("--workers", type=int)
    arguments = parser.parse_args()
    audit = audit_phase(arguments.output, arguments.dataset, arguments.split, workers=arguments.workers)
    print(json.dumps({key: audit[key] for key in ("phase", "passed", "checks_total", "failure_count", "accounts_checked", "worker_count")}))
    raise SystemExit(0 if audit["passed"] else 1)
