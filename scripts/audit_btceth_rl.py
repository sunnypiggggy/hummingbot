"""Independent BTC/ETH RL evidence and cash/inventory audit.

No trainer, environment, PortfolioLedger, policy loading or model inference is
used here. Explicit fills are replayed with arithmetic against original market
prices. Returned evidence uses relative paths instead of remote absolute paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT")
ALGORITHMS = ("recurrent_ppo", "tqc", "crossq", "sac")
PROFILES = ("return", "balanced")
SEEDS = tuple(range(5))
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
COSTS = {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}}
SPLITS = {"train": ("2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
          "validation": ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"),
          "test": ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")}


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def relative_file(root, value):
    """Never trust a saved remote absolute path or parent-directory escape."""
    root = Path(root).resolve()
    text = str(value).replace("\\", "/")
    if text.startswith("/") or ":" in text or ".." in text.split("/"):
        raise ValueError("Evidence path must be safe and root-relative")
    path = (root / text).resolve()
    path.relative_to(root)
    return path


class Checks:
    def __init__(self):
        self.count, self.failed, self.failures = 0, 0, []
        self.maximum_errors = {}

    def check(self, condition, label):
        self.count += 1
        if not bool(condition):
            self.failed += 1
            if len(self.failures) < 250:
                self.failures.append(str(label))
        return bool(condition)

    def close(self, actual, expected, label, *, atol=1e-6, rtol=1e-10):
        try:
            actual, expected = float(actual), float(expected)
            valid = math.isfinite(actual) and math.isfinite(expected)
            valid = valid and abs(actual - expected) <= atol + rtol * abs(expected)
        except (ValueError, TypeError, OverflowError):
            valid = False
        return self.check(valid, label)

    def vector_close(self, actual, expected, label, *, atol=1e-6, rtol=1e-10):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        valid = actual.shape == expected.shape
        if valid:
            difference = np.abs(actual - expected)
            maximum = float(difference.max()) if difference.size else 0.
            key = label.split(":")[-1]
            self.maximum_errors[key] = max(self.maximum_errors.get(key, 0.), maximum)
            valid = (np.isfinite(actual).all() and np.isfinite(expected).all()
                     and np.all(difference <= atol + rtol * np.abs(expected)))
        return self.check(valid, label)


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def read_equity(path):
    frame = read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    return frame.set_index("timestamp")


def verify_saved_fingerprints(directory, checks, label):
    expected = read_json(directory / "fingerprints.json")
    actual = {}
    for name, filename in (("equity", "equity_1h.csv.gz"), ("fills", "fills.csv"),
                           ("weights", "weights.csv"), ("boundary_equity", "boundary_equity.csv")):
        path = directory / filename
        payload = gzip.decompress(path.read_bytes()) if name == "equity" else path.read_bytes()
        actual[name] = hashlib.sha256(payload).hexdigest()
    checks.check(actual == expected, label + ": exact saved CSV fingerprints")
    return actual


def reconstruct_fills(dataset, fills, equity, metadata, *, start, end, fee, slippage, checks=None, label="run"):
    """Replay quantities independently; no environment sizing code is shared."""
    checks = Checks() if checks is None else checks
    start, end = utc(start), utc(end)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    if not checks.check(equity.index.equals(hours), label + ": complete unique ordered hourly equity"):
        return None
    opening = dataset["hourly_open"].reindex(hours).loc[:, list(SYMBOLS)]
    closing = dataset["hourly_close"].reindex(hours).loc[:, list(SYMBOLS)]
    checks.check(np.isfinite(opening).all().all() and opening.gt(0).all().all(), label + ": real opening prices")
    checks.check(np.isfinite(closing).all().all() and closing.gt(0).all().all(), label + ": real closing prices")
    cash, quantity, basis = 10000., {symbol: 0. for symbol in SYMBOLS}, {symbol: 0. for symbol in SYMBOLS}
    realized = fees = notional = slip_cost = 0.
    trades = fills.copy()
    if len(trades):
        trades["timestamp"] = pd.to_datetime(trades.timestamp, utc=True)
        checks.check(trades.timestamp.is_monotonic_increasing, label + ": fills chronological")
        checks.check(((trades.timestamp >= start) & (trades.timestamp < end)).all(), label + ": fills within split")
        checks.check((trades.timestamp == trades.timestamp.dt.floor("h")).all(), label + ": exact hour fills")
    grouped = {stamp: frame for stamp, frame in trades.groupby("timestamp", sort=False)} if len(trades) else {}
    computed, quantities = [], []
    empty = pd.DataFrame()
    for stamp in hours:
        for row in grouped.get(stamp, empty).itertuples(index=False):
            symbol, side = str(row.asset_key), str(row.side)
            if not checks.check(symbol in SYMBOLS and side in ("BUY", "SELL"), label + ": owned coin identity and side"):
                continue
            amount = float(row.qty)
            if not checks.check(math.isfinite(amount) and amount > 0, label + ": positive finite quantity"):
                continue
            checks.check(stamp.hour == (1 if side == "BUY" else 0), label + ": 00UTC sell / 01UTC buy")
            reference = float(opening.loc[stamp, symbol])
            price = reference * (1 + slippage if side == "BUY" else 1 - slippage)
            traded, charge = amount * price, amount * price * fee
            checks.close(row.reference_price, reference, label + ": reference price", atol=1e-10, rtol=1e-12)
            checks.close(row.price, price, label + ": adverse fill price", atol=1e-10, rtol=1e-12)
            checks.close(row.notional, traded, label + ": fill notional")
            checks.close(row.fee, charge, label + ": fill fee")
            checks.close(row.slippage_cost, amount * abs(price - reference), label + ": fill slippage")
            realized_fill = 0.
            if side == "BUY":
                checks.check(cash + 1e-7 >= traded + charge, label + ": buy funded including fee")
                cash -= traded + charge
                quantity[symbol] += amount
                basis[symbol] += traded + charge
            else:
                owned = quantity[symbol]
                if not checks.check(owned > 0 and amount <= owned + 1e-10, label + ": sell bounded by strategy inventory"):
                    continue
                removed = basis[symbol] * min(amount, owned) / owned
                quantity[symbol] -= min(amount, owned)
                basis[symbol] -= removed
                cash += traded - charge
                realized_fill = traded - charge - removed
                realized += realized_fill
            fees += charge
            notional += traded
            slip_cost += amount * abs(price - reference)
            checks.close(row.cash_after, cash, label + ": per-fill cash")
            checks.close(row.quantity_after, quantity[symbol], label + ": per-fill inventory", atol=1e-10)
            checks.close(row.realized_pnl, realized_fill, label + ": realized fill PNL")
        checks.check(cash >= -1e-7 and all(value >= -1e-10 for value in quantity.values()), label + ": nonnegative cash/inventory")
        base = sum(quantity[symbol] * float(closing.loc[stamp, symbol]) for symbol in SYMBOLS)
        value = cash + base
        computed.append((value, cash, base, base / value))
        quantities.append([quantity[symbol] for symbol in SYMBOLS])
    computed = pd.DataFrame(computed, index=hours, columns=("equity", "cash", "base_value", "exposure"))
    for column in computed:
        checks.vector_close(equity[column], computed[column], label + ": reconstructed " + column,
                            atol=1e-6 if column != "exposure" else 1e-10)
    final, final_cash, final_base = computed.iloc[-1][["equity", "cash", "base_value"]]
    peaks = np.maximum.accumulate(np.r_[10000., computed.equity.to_numpy()])[1:]
    drawdown = computed.equity / peaks - 1
    unrealized = sum(quantity[symbol] * float(closing.iloc[-1][symbol]) - basis[symbol] for symbol in SYMBOLS)
    metrics = {"initial_capital": 10000., "final_equity": final, "final_cash": final_cash,
        "final_base_value": final_base, "net_pnl": final - 10000., "net_return": final / 10000. - 1,
        "max_drawdown": float(drawdown.min()), "fees": fees, "total_notional": notional,
        "turnover": notional / 10000., "slippage_cost": slip_cost,
        "average_exposure": float(computed.exposure.mean()), "exposure": float(computed.exposure.mean()),
        "fills": len(fills), "buy_fills": int(fills.side.eq("BUY").sum()) if len(fills) else 0,
        "sell_fills": int(fills.side.eq("SELL").sum()) if len(fills) else 0,
        "observed_hours": len(hours), "realized_pnl": realized, "unrealized_pnl": unrealized,
        "net_liquidation_equity": final_cash + final_base * (1 - slippage) * (1 - fee)}
    metrics["net_liquidation_return"] = metrics["net_liquidation_equity"] / 10000. - 1
    daily = computed.equity.resample("D").last()
    daily_return = daily.pct_change(fill_method=None)
    daily_return.iloc[0] = daily.iloc[0] / 10000. - 1
    standard_deviation = float(daily_return.std(ddof=1))
    metrics["sharpe_daily"] = float(daily_return.mean() / standard_deviation * np.sqrt(365)) if standard_deviation > 1e-15 else None
    metrics["cagr"] = math.expm1(math.log(final / 10000.) * 365 / ((end - start).total_seconds() / 86400))
    metrics["calmar"] = metrics["cagr"] / abs(metrics["max_drawdown"]) if metrics["max_drawdown"] < -1e-15 else None
    for key in ("initial_capital", "final_equity", "final_cash", "fees", "total_notional", "slippage_cost", "realized_pnl", "unrealized_pnl"):
        checks.close(metadata.get(key), metrics[key], label + ": metadata " + key)
    checks.close(metadata.get("final_liquidation_equity"), metrics["net_liquidation_equity"], label + ": metadata net liquidation")
    checks.close(final - 10000., realized + unrealized, label + ": realized/open PNL bridge")
    recorded_positions = metadata.get("final_positions", {})
    checks.check(set(recorded_positions).issubset(SYMBOLS), label + ": final position identities")
    for symbol in SYMBOLS:
        checks.close(recorded_positions.get(symbol, 0.), quantity[symbol], label + ": final inventory " + symbol, atol=1e-10)
    checks.check(metadata.get("bnb_fee_discount") is False and metadata.get("fee_asset") == "USDT", label + ": no BNB discount, quote-paid fees")
    checks.close(metadata.get("fee_rate"), fee, label + ": metadata fee rate", atol=1e-15)
    checks.close(metadata.get("slippage"), slippage, label + ": metadata slippage", atol=1e-15)
    checks.check(metadata.get("record_history") is True and metadata.get("training") is False, label + ": complete evaluation history")
    checks.check(metadata.get("forced_final_liquidation") is False, label + ": true terminal holding valuation")
    checks.check(metadata.get("fill_count") == len(fills), label + ": metadata fill count")
    checks.close(metadata.get("hourly_high_water"), max(10000., float(computed.equity.max())), label + ": hourly high-water")
    checks.close(metadata.get("current_drawdown"), -float(drawdown.iloc[-1]), label + ": terminal current drawdown", atol=1e-10)
    holdings = metadata.get("final_holdings", [])
    holding_keys = [item.get("asset_key") for item in holdings]
    checks.check(len(holding_keys) == len(set(holding_keys)) and set(holding_keys).issubset(SYMBOLS),
                 label + ": distinct final holding identities")
    for item in holdings:
        symbol = item.get("asset_key")
        if symbol not in SYMBOLS:
            continue
        checks.close(item.get("qty"), quantity[symbol], label + ": final holding quantity", atol=1e-10)
        checks.close(item.get("mark_price"), float(closing.iloc[-1][symbol]), label + ": final holding price")
        value = quantity[symbol] * float(closing.iloc[-1][symbol])
        checks.close(item.get("market_value"), value, label + ": final holding value")
        checks.close(item.get("weight"), value / final, label + ": final holding weight", atol=1e-10)
    for symbol in SYMBOLS:
        checks.check(quantity[symbol] * float(closing.iloc[-1][symbol]) <= 1e-6 or symbol in holding_keys,
                     label + ": nonzero final inventory represented in holdings")
    monthly = daily.resample("ME").last()
    monthly_returns = monthly.pct_change(fill_method=None)
    monthly_returns.iloc[0] = monthly.iloc[0] / 10000. - 1
    return {"metrics": metrics, "equity": computed, "quantity": pd.DataFrame(quantities, index=hours, columns=SYMBOLS),
            "drawdown": drawdown, "monthly": dict(zip(monthly.index.strftime("%Y-%m"), monthly_returns.to_numpy())),
            "checks": checks}


def independent_training_calendar(dataset, start, end, episode_days):
    """Dependency-derived readiness; no environment feature code is imported."""
    fields = {name: dataset["hourly_" + name].loc[:, list(SYMBOLS)] for name in ("open", "high", "low", "close", "volume")}
    index = fields["close"].index
    daily_index = pd.date_range(index[0].normalize(), index[-1].normalize(), freq="D")
    close = fields["close"].loc[index.hour == 23].copy()
    close.index = close.index.normalize()
    close = close.reindex(daily_index)
    price_ready = (np.isfinite(close) & close.gt(0)).rolling(200, min_periods=200).sum().eq(200).all(axis=1)
    good_volume = np.isfinite(fields["volume"]) & fields["volume"].ge(0)
    volumes = fields["volume"].where(good_volume).resample("D").sum(min_count=24)
    volume_days = good_volume.resample("D").sum().eq(24)
    volume_ready = (volume_days.rolling(30, min_periods=30).sum().eq(30)
                    & volumes.rolling(30, min_periods=30).mean().gt(0)).all(axis=1)
    ready = price_ready & volume_ready
    ready.index += pd.Timedelta(days=1)
    complete = pd.DataFrame(True, index=index, columns=SYMBOLS)
    for name, frame in fields.items():
        complete &= np.isfinite(frame) & (frame.ge(0) if name == "volume" else frame.gt(0))
    market = complete.resample("D").sum().eq(24).all(axis=1)
    dates = pd.date_range(utc(start), utc(end), freq="D", inclusive="left")
    good = np.array([bool(ready.get(day, False) and market.get(day, False)) for day in dates])
    eligible = [dates[i] for i in range(len(dates) - episode_days + 1) if good[i:i + episode_days].all()]
    covered = set(day for first in eligible for day in pd.date_range(first, periods=episode_days, freq="D"))
    return {"eligible": set(eligible), "covered_days": len(covered), "market": market, "features": ready,
            "episode_days": int(episode_days),
            "requested_train_days": len(dates), "candidate_windows": len(dates) - episode_days + 1,
            "eligible_windows": len(eligible), "excluded_windows": len(dates) - episode_days + 1 - len(eligible)}


def score_candidate(base, stress):
    def one(row):
        drawdown = abs(float(row["max_drawdown"]))
        return math.log1p(float(row["net_return"])) - .5 * drawdown - 2 * max(drawdown - .20, 0.)
    return (one(base) + one(stress)) / 2


def recompute_selection(summary):
    summary = summary.loc[summary.split.eq("validation")]
    candidates, choices = [], {}
    for algorithm in ALGORITHMS:
        options = []
        for profile in PROFILES:
            rows = summary.loc[(summary.algorithm == algorithm) & (summary.profile == profile) & summary.primary_policy.eq(True)]
            if len(rows) != 2 or set(rows.scenario) != set(COSTS):
                raise ValueError("Need each profile's two validation ensemble rows")
            base = rows.loc[rows.scenario == "base"].iloc[0].to_dict()
            stress = rows.loc[rows.scenario == "stress"].iloc[0].to_dict()
            candidate = {"algorithm": algorithm, "profile": profile, "score": score_candidate(base, stress), "base": base, "stress": stress}
            options.append(candidate)
            candidates.append(candidate)
        choices[algorithm] = sorted(options, key=lambda item: (-item["score"], item["profile"]))[0]["profile"]
    overall = sorted(candidates, key=lambda item: (-item["score"], item["algorithm"], item["profile"]))[0]
    return {"selected_profiles": choices, "primary_validation_leader": {key: overall[key] for key in ("algorithm", "profile")},
            "candidates": candidates}


def _check_summary(row, expected, checks, label):
    # Realized/unrealized PNL are metadata fields, not public summary columns.
    for key, value in expected.items():
        if key in ("realized_pnl", "unrealized_pnl"):
            continue
        if value is None:
            checks.check(key in row and (row[key] is None or pd.isna(row[key])), label + ": metric " + key)
        else:
            checks.close(row.get(key), value, label + ": metric " + key,
                         atol=1e-6 if key in ("final_equity", "net_pnl", "fees", "total_notional", "final_cash", "final_base_value",
                                             "slippage_cost", "net_liquidation_equity") else 1e-10)


def _audit_training(output, spec, dataset, checks):
    records, hashes = {}, {}
    completion = read_json(output / "training_completion.json")
    hashes["training_completion.json"] = sha(output / "training_completion.json")
    checks.check(completion.get("model_count") == 40 and len(completion.get("records", [])) == 40, "40 completed model records")
    spec_hash = sha(output / "frozen_spec.json")
    checks.check(completion.get("frozen_spec_sha256") == spec_hash, "training completion frozen protocol")
    calendar = independent_training_calendar(dataset, *spec["train"], int(spec["train_episode_days"]))
    identities = []
    for algorithm in ALGORITHMS:
        for profile in PROFILES:
            for seed in SEEDS:
                identity = (algorithm, profile, seed)
                directory = output / "training" / f"{algorithm}__{profile}__seed{seed}"
                record_path = directory / "record.json"
                record = read_json(record_path)
                identities.append((record.get("algorithm"), record.get("profile"), record.get("seed")))
                checks.check(identities[-1] == identity, f"{identity}: training identity")
                checks.check(record.get("frozen_spec_sha256") == spec_hash, f"{identity}: training protocol SHA")
                checkpoint = relative_file(output, record["model_relative_path"])
                checks.check(checkpoint == (directory / "model.zip").resolve(), f"{identity}: unique checkpoint path")
                checks.check(sha(checkpoint) == record["model_sha256"], f"{identity}: frozen model SHA256")
                hashes[record_path.relative_to(output).as_posix()] = sha(record_path)
                hashes[checkpoint.relative_to(output).as_posix()] = sha(checkpoint)
                checks.check(record.get("actual_timesteps") == spec["timesteps_per_model"], f"{identity}: training budget")
                device = record.get("device", {})
                checks.check(device.get("device") == "cuda:0" and "V100" in device.get("name", "")
                             and device.get("capability") == [7, 0] and bool(device.get("visible_gpu_uuid")),
                             f"{identity}: isolated V100 execution")
                quality = record.get("training_window_quality", {})
                for name in ("requested_train_days", "candidate_windows", "eligible_windows", "excluded_windows"):
                    checks.check(quality.get(name) == calendar[name], f"{identity}: independent training readiness {name}")
                checks.check(quality.get("covered_training_days") == calendar["covered_days"], f"{identity}: covered training days")
                completed_steps = 0
                history = record.get("training_episode_history", [])
                checks.check(bool(history), f"{identity}: actual episode history exists")
                for episode in history:
                    first, last, until = (utc(episode[name]) for name in ("start", "end_exclusive", "completed_until"))
                    steps = int(episode["completed_steps"])
                    checks.check(first in calendar["eligible"], f"{identity}: complete episode market/feature window")
                    checks.check(last - first == pd.Timedelta(days=spec["train_episode_days"]), f"{identity}: fragment length")
                    checks.check(utc(spec["train"][0]) <= first <= until <= last <= utc(spec["train"][1]),
                                 f"{identity}: episode strictly within training")
                    checks.check(until - first == pd.Timedelta(days=steps), f"{identity}: completed step dates")
                    checks.check(episode.get("terminated") is False and
                                 episode.get("truncated") == (steps == spec["train_episode_days"]),
                                 f"{identity}: random fragments time-truncated")
                    completed_steps += steps
                checks.check(completed_steps == record["actual_timesteps"], f"{identity}: actual transition count")
                checks.check(len(record.get("features", [])) == 27 and len(set(record.get("features", []))) == 27,
                             f"{identity}: unique causal state features")
                records[identity] = record
    checks.check(len(set(identities)) == 40, "40 unique algorithm/profile/seed identities")
    checks.check(len(hashes) == 81, "81 immutable training evidence files")
    completed_records = {(r.get("algorithm"), r.get("profile"), r.get("seed")): r for r in completion["records"]}
    checks.check(completed_records == records, "completion preserves all original model records exactly")
    runtime = read_json(output / "runtime.json")
    gpus = runtime.get("physical_gpus", [])
    gpu_ids = {item.get("uuid") for item in gpus}
    checks.check(len(gpus) == len(gpu_ids) == 2 and all("V100" in item.get("name", "") for item in gpus),
                 "actual runtime declares two distinct V100 GPUs")
    checks.check({record["device"]["visible_gpu_uuid"] for record in records.values()} == gpu_ids,
                 "formal model grid used both UUID-isolated V100 devices")
    packages = runtime.get("packages", {})
    checks.check(packages.get("stable-baselines3") == packages.get("sb3-contrib") == "2.9.0",
                 "matched pinned SB3 and SB3-Contrib runtime")
    return records, hashes, {name: value for name, value in calendar.items() if name not in ("eligible", "market", "features")}


def _audit_run(directory, dataset, spec, records, row, monthly_rows, daily_rows, *, split, scenario,
               algorithm, profile, key, checks):
    label = str(row["run_id"])
    metadata = read_json(directory / "metadata.json")
    equity, fills = read_equity(directory / "equity_1h.csv.gz"), read_csv(directory / "fills.csv")
    fingerprints = verify_saved_fingerprints(directory, checks, label)
    result = reconstruct_fills(dataset, fills, equity, metadata, start=spec[split][0], end=spec[split][1],
                               **spec["costs"][scenario], checks=checks, label=label)
    if result is None:
        return {"fingerprints": fingerprints, "reconstructed": False}
    _check_summary(row, result["metrics"], checks, label)
    checks.check(row.get("split") == split and row.get("scenario") == scenario, label + ": summary split/scenario")
    checks.check(utc(row["start"]) == utc(spec[split][0]) and utc(row["end_exclusive"]) == utc(spec[split][1]),
                 label + ": summary dates")
    for name, value in (("cash_floor_target", .05), ("max_asset_weight_target", .95), ("rebalance_band", .025)):
        checks.close(metadata.get(name), value, label + ": target policy " + name, atol=1e-14)
    checks.close(metadata.get("drawdown_penalty"), 0. if profile in ("return", "baseline") else .25,
                 label + ": declared risk reward", atol=1e-14)
    if algorithm != "baseline":
        selected = ([records[(algorithm, profile, seed)]["parameter_sha256"] for seed in SEEDS] if key == "ensemble"
                    else [records[(algorithm, profile, int(key))]["parameter_sha256"]])
        checks.check(metadata.get("parameter_sha256_before") == selected and metadata.get("parameter_sha256_after") == selected,
                     label + ": unchanged policy parameters")
        checks.check(metadata.get("model_count") == len(selected), label + ": actual model count")
    else:
        checks.check(metadata.get("model_count") == 0 and metadata.get("baseline") == key, label + ": model-free baseline")
    months = monthly_rows.loc[monthly_rows.run_id == label]
    checks.check(len(months) == len(result["monthly"]) and not months.month.duplicated().any(), label + ": monthly coverage")
    for month, expected in result["monthly"].items():
        selected = months.loc[months.month == month]
        checks.check(len(selected) == 1, label + ": unique month " + month)
        if len(selected) == 1:
            checks.close(selected.iloc[0]["return"], expected, label + ": month return " + month, atol=1e-11)
    days = result["equity"].equity.resample("D").last()
    expected_times = pd.DatetimeIndex([utc(spec[split][0]), *[day + pd.Timedelta(days=1) for day in days.index]])
    histories = daily_rows.loc[daily_rows.run_id == label]
    checks.check(pd.DatetimeIndex(pd.to_datetime(histories.timestamp, utc=True)).equals(expected_times),
                 label + ": daily display timestamps")
    if len(histories) == len(expected_times):
        checks.vector_close(histories.equity, np.r_[10000., days.to_numpy()], label + ": daily display equity")
        checks.vector_close(histories.net_return, np.r_[0., days.to_numpy() / 10000. - 1], label + ": daily display returns", atol=1e-11)
        checks.vector_close(histories.drawdown, np.r_[0., result["drawdown"].resample("D").last().to_numpy()],
                            label + ": daily display drawdown", atol=1e-11)
    weights = read_csv(directory / "weights.csv")
    weights["date"] = pd.to_datetime(weights.date, utc=True)
    expected_days = pd.date_range(*[utc(value) for value in spec[split]], freq="D", inclusive="left")
    checks.check(len(weights) == 2 * len(expected_days), label + ": two coin weight rows per day")
    for day, frame in weights.groupby("date", sort=False):
        checks.check(day in expected_days and frame.asset_key.tolist() == list(SYMBOLS), label + ": daily weight identities")
        stamp = day + pd.Timedelta(hours=23)
        if stamp not in result["quantity"].index:
            continue
        expected = (result["quantity"].loc[stamp] * dataset["hourly_close"].loc[stamp, list(SYMBOLS)]
                    / result["equity"].loc[stamp, "equity"])
        checks.vector_close(frame.actual_weight, expected.to_numpy(), label + ": actual coin weights", atol=1e-10)
        if frame.target_weight.notna().any():
            target = frame.target_weight.to_numpy(dtype=float)
            checks.check(np.isfinite(target).all() and (target >= -1e-12).all() and (target <= .95 + 1e-12).all()
                         and target.sum() <= .95 + 1e-10, label + ": funded targets and minimum cash")
    boundary = read_equity(directory / "boundary_equity.csv")
    checks.check(boundary.index.equals(expected_times), label + ": reward boundary dates")
    if len(boundary) == len(expected_times):
        checks.vector_close(boundary.equity, np.r_[10000., days.to_numpy()], label + ": boundary equity")
        rewards, previous_nav, previous_dd = [0.], 10000., 0.
        penalty = float(metadata["drawdown_penalty"])
        for _, day_frame in result["equity"].groupby(result["equity"].index.normalize()):
            deepest = -float(result["drawdown"].loc[day_frame.index].min())
            final = float(day_frame.equity.iloc[-1])
            rewards.append(100 * (math.log(final / previous_nav) - penalty * max(deepest - previous_dd, 0.)))
            previous_nav, previous_dd = final, -float(result["drawdown"].loc[day_frame.index[-1]])
        checks.vector_close(boundary.reward, rewards, label + ": independent hourly risk-shaped reward", atol=1e-9)
    return {"fingerprints": fingerprints, "reconstructed": True, "hours": len(equity),
            "net_return": result["metrics"]["net_return"], "max_drawdown": result["metrics"]["max_drawdown"]}


def _compare_tables(actual, expected, checks, label):
    checks.check(set(actual.columns) == set(expected.columns) and len(actual) == len(expected), label + ": table schema and row count")
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        return
    for column in expected:
        left, right = actual[column].reset_index(drop=True), expected[column].reset_index(drop=True)
        missing = left.isna() & right.isna()
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(right.dtype):
            checks.vector_close(left.loc[~missing], right.loc[~missing], label + ": " + column, atol=1e-10)
        else:
            checks.check((missing | left.eq(right)).all(), label + ": " + column)


def _verify_receipt(output, checks):
    path = output / "transfer_receipt.json"
    if not path.is_file():
        return {"present": False, "note": "No transfer receipt supplied; source/model/run hashes are checked directly."}
    receipt = read_json(path)
    files = receipt["exactfiles"]
    if isinstance(files, dict):
        entries = [{"path": name, "sha256": value if isinstance(value, str) else value["sha256"]} for name, value in files.items()]
    elif isinstance(files, list):
        entries = [{"path": item.get("relative_path", item.get("path")), "sha256": item["sha256"]} for item in files]
    else:
        raise ValueError("transfer_receipt.exactfiles must map relative files to SHA256")
    seen = set()
    for item in entries:
        target = relative_file(output, item["path"])
        checks.check(target not in seen, "transfer receipt unique file path")
        seen.add(target)
        checks.check(sha(target) == item["sha256"], "transfer receipt exact bytes: " + str(item["path"]))
    return {"present": True, "sha256": sha(path), "files_verified": len(entries)}


def _audit_group(output, dataset, spec, records, checks, *, split, algorithm, profile, runs, source_tables):
    baseline = algorithm == "baseline"
    directory = output / "evaluation" / split / ("baselines" if baseline else algorithm)
    if not baseline:
        directory /= profile
    summary = read_csv(directory / "summary.csv")
    monthly, daily = read_csv(directory / "monthly_returns.csv"), read_csv(directory / "daily_equity.csv")
    completion = read_json(directory / "completion.json")
    source_tables["summary.csv"].append(summary)
    source_tables["monthly_returns.csv"].append(monthly)
    source_tables["daily_equity.csv"].append(daily)
    keys = BASELINES if baseline else (*[str(seed) for seed in SEEDS], "ensemble")
    expected_ids = {f"{split}__{scenario}__{key}" if baseline else f"{split}__{scenario}__{algorithm}__{profile}__{key}"
                    for scenario in COSTS for key in keys}
    checks.check(set(summary.run_id) == expected_ids and not summary.run_id.duplicated().any(),
                 str(directory.relative_to(output)) + ": exact financial run set")
    checks.check(len(completion.get("rows", [])) == len(expected_ids), str(directory.relative_to(output)) + ": completed row count")
    if not baseline:
        checks.check(completion.get("frozen_spec_sha256") == sha(output / "frozen_spec.json"), str(directory.relative_to(output)) + ": completion protocol")
    repeated = {row["run_id"]: row for row in completion.get("repeats", [])}
    if not baseline:
        checks.check(set(repeated) == expected_ids, str(directory.relative_to(output)) + ": complete independent replay receipts")
    completion_rows = {row["run_id"]: row for row in completion.get("rows", [])}
    for scenario in COSTS:
        for key in keys:
            run_id = f"{split}__{scenario}__{key}" if baseline else f"{split}__{scenario}__{algorithm}__{profile}__{key}"
            selected = summary.loc[summary.run_id == run_id]
            if not checks.check(len(selected) == 1, run_id + ": unique summary row"):
                continue
            row = selected.iloc[0].to_dict()
            checks.check(row.get("algorithm") == algorithm and row.get("profile") == profile, run_id + ": summary model identity")
            checks.check(bool(row.get("primary_policy")) == (key == "ensemble"), run_id + ": actual primary ensemble marker")
            expected_seed = None if key == "ensemble" or baseline else int(key)
            checks.check(pd.isna(row.get("seed")) if expected_seed is None else row.get("seed") == expected_seed, run_id + ": seed identity")
            result = _audit_run(directory / scenario / key, dataset, spec, records, row, monthly, daily,
                                split=split, scenario=scenario, algorithm=algorithm, profile=profile, key=key, checks=checks)
            result["relative_directory"] = (directory / scenario / key).relative_to(output).as_posix()
            result["split"], result["algorithm"], result["profile"] = split, algorithm, profile
            runs[run_id] = result
            if not baseline:
                checks.check(repeated.get(run_id, {}).get("passed") is True
                             and repeated.get(run_id, {}).get("fingerprints") == result["fingerprints"],
                             run_id + ": repeated inference financial fingerprints unchanged")
            if run_id in completion_rows:
                for name in ("net_return", "max_drawdown", "fees", "final_equity", "initial_capital"):
                    checks.close(completion_rows[run_id].get(name), row.get(name), run_id + ": completion summary " + name)
    return summary


def _validate_selection(output, checks, summary, training_hashes):
    selection = read_json(output / "validation_selection.json")
    derived = recompute_selection(summary)
    checks.check(selection.get("frozen_spec_sha256") == sha(output / "frozen_spec.json"), "selection frozen protocol SHA")
    checks.check(selection.get("selected_profiles") == derived["selected_profiles"], "profiles independently selected from validation only")
    checks.check(selection.get("primary_validation_leader") == derived["primary_validation_leader"], "overall validation leader independently reproduced")
    checks.check(selection.get("test_results_available_to_selector") is False and selection.get("selection_uses_test") is False,
                 "selection declares no test-based tuning")
    checks.check(selection.get("training_source_sha256") == training_hashes, "selection fixed all 81 training files")
    expected_source = {f"evaluation/validation/{algorithm}/{profile}/summary.csv":
                       sha(output / "evaluation" / "validation" / algorithm / profile / "summary.csv")
                       for algorithm in ALGORITHMS for profile in PROFILES}
    checks.check(selection.get("validation_source_sha256") == expected_source, "selection fixed all eight validation summaries")
    recorded = {(item["algorithm"], item["profile"]): item for item in selection.get("candidates", [])}
    checks.check(len(recorded) == 8 and len(selection.get("candidates", [])) == 8, "exact eight validation candidates")
    for item in derived["candidates"]:
        previous = recorded.get((item["algorithm"], item["profile"]), {})
        checks.close(previous.get("score"), item["score"], "independent validation score", atol=1e-12)
        for scenario in ("base", "stress"):
            for name in ("run_id", "algorithm", "profile", "split", "scenario"):
                checks.check(previous.get(scenario, {}).get(name) == item[scenario].get(name), "selection candidate identity " + name)
            for name in ("net_return", "max_drawdown", "final_equity", "fees"):
                checks.close(previous.get(scenario, {}).get(name), item[scenario].get(name), "selection candidate metric " + name)
    return selection


def _research_target_result(summary, selection):
    chosen = selection["primary_validation_leader"]
    profile, algorithm = chosen["profile"], chosen["algorithm"]
    rows = summary.loc[(summary.split == "test") & (summary.algorithm == algorithm)
                       & (summary.profile == profile)]
    base = rows.loc[(rows.scenario == "base") & rows.primary_policy.eq(True)].iloc[0]
    stress = rows.loc[(rows.scenario == "stress") & rows.primary_policy.eq(True)].iloc[0]
    benchmark = summary.loc[(summary.run_id == "test__base__equal_hold")].iloc[0]
    seeds = rows.loc[(rows.scenario == "base") & rows.seed.notna()]
    gates = {"base_return_at_least_5pct": float(base.net_return) >= .05,
             "base_max_drawdown_at_most_20pct": abs(float(base.max_drawdown)) <= .20,
             "stress_return_nonnegative": float(stress.net_return) >= 0,
             "beats_equal_hold_return_and_drawdown": (float(base.net_return) > float(benchmark.net_return)
                                                     and float(base.max_drawdown) > float(benchmark.max_drawdown)),
             "at_least_three_positive_training_seeds": int((seeds.net_return > 0).sum()) >= 3}
    trend = summary.loc[summary.run_id.isin(("test__base__sma200_trend", "test__base__momentum180"))]
    controls = {row.strategy: bool(base.net_return > row.net_return and base.max_drawdown > row.max_drawdown)
                for row in trend.itertuples(index=False)}
    return {"primary_validation_leader": chosen, "base_net_return": float(base.net_return),
            "base_max_drawdown": float(base.max_drawdown), "stress_net_return": float(stress.net_return),
            "gates": gates, "all_predeclared_primary_targets_met": all(gates.values()),
            "dominates_simple_trend_controls": controls,
            "stronger_claim_supported": all(gates.values()) and len(controls) == 2 and all(controls.values()),
            "note": "Accounting-audit passage is separate from whether the strategy meets the user's performance objective."}


def _output_input_snapshot(output, complete_test):
    names = ("frozen_spec.json", "validation_selection.json", "training_completion.json", "runtime.json",
             "validation_completion.json", "summary.csv", "daily_equity.csv", "monthly_returns.csv", "transfer_receipt.json")
    if complete_test:
        names += ("test_started.json", "test_completion.json")
    files = {output / name for name in names if (output / name).is_file()}
    filenames = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json",
                 "metadata.json", "fingerprints.json", "equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv")
    for split in (("validation", "test") if complete_test else ("validation",)):
        root = output / "evaluation" / split
        for name in filenames:
            files.update(root.rglob(name))
    return {path.resolve(): sha(path) for path in files}


def audit(output, dataset_path, *, allow_validation_only=False, source_root=ROOT):
    output, dataset_path, source_root = Path(output).resolve(), Path(dataset_path).resolve(), Path(source_root).resolve()
    checks, runs, complete_test = Checks(), {}, False
    report = {"audit_started_at": datetime.now(timezone.utc).isoformat(), "output_directory": str(output),
              "dataset_path": str(dataset_path), "scope": "independent saved-fill accounting; no policy re-inference or training",
              "policy_causality_scope": "Frozen source hashes plus independently reconstructed pre-test market/feature readiness; no claim of a fresh blind holdout."}
    try:
        spec = read_json(output / "frozen_spec.json")
        spec_hash = sha(output / "frozen_spec.json")
        report["source_sha256"] = dict(spec["code_sha256"])
        report["dataset_sha256"] = spec["dataset_sha256"]
        report["frozen_spec_sha256"] = spec_hash
        data_manifest = read_json(dataset_path.parent / "manifest.json")
        checks.check(sha(dataset_path) == spec["dataset_sha256"] == data_manifest["dataset_sha256"], "immutable prepared dataset SHA256")
        checks.check(sha(dataset_path.parent / "manifest.json") == spec["dataset_manifest_sha256"], "immutable dataset manifest SHA256")
        frozen_files = {output / "frozen_spec.json": spec_hash, dataset_path: spec["dataset_sha256"],
                        dataset_path.parent / "manifest.json": spec["dataset_manifest_sha256"]}
        frozen_files.update(_output_input_snapshot(output, (output / "test_completion.json").is_file()))
        for name, expected in spec["code_sha256"].items():
            path = relative_file(source_root, name)
            checks.check(sha(path) == expected, "frozen source SHA256: " + name)
            frozen_files[path] = expected
        for split, boundaries in SPLITS.items():
            checks.check([utc(value) for value in spec[split]] == [utc(value) for value in boundaries], "fixed chronological " + split + " dates")
        checks.check(spec["algorithms"] == list(ALGORITHMS) and list(spec["profiles"]) == list(PROFILES)
                     and spec["seeds"] == list(SEEDS), "fixed four algorithms/two profiles/five seeds")
        checks.check(spec["costs"] == COSTS and spec["capital"] == 10000., "uniform fee/slippage and 10000 USDT")
        checks.check(spec["train_episode_days"] == 32 and spec["timesteps_per_model"] == 32768, "fixed 32-day fragments and 32768-step training budget")
        checks.check(spec["baseline_names"] == list(BASELINES), "seven declared controls")
        with gzip.open(dataset_path, "rb") as stream:
            dataset = pickle.load(stream)
        index = dataset["hourly_close"].index
        checks.check(isinstance(index, pd.DatetimeIndex) and index.tz is not None
                     and index.is_monotonic_increasing and not index.has_duplicates, "unique timezone-aware ordered market hours")
        checks.check(index.equals(pd.date_range(index[0], index[-1], freq="h")), "market grid has no omitted timestamp rows")
        for name in ("open", "high", "low", "close", "volume"):
            frame = dataset["hourly_" + name]
            checks.check(frame.index.equals(index) and list(frame.columns) == list(SYMBOLS), "two coins aligned in hourly " + name)
        records, training_hashes, calendar = _audit_training(output, spec, dataset, checks)
        report["training_calendar"] = calendar
        report["models_checked"] = len(records)
        for name, expected in training_hashes.items():
            frozen_files[relative_file(output, name)] = expected
        source_tables = {name: [] for name in ("summary.csv", "daily_equity.csv", "monthly_returns.csv")}
        validation_frames = []
        for algorithm in ALGORITHMS:
            for profile in PROFILES:
                validation_frames.append(_audit_group(output, dataset, spec, records, checks, split="validation",
                                                       algorithm=algorithm, profile=profile, runs=runs, source_tables=source_tables))
        _audit_group(output, dataset, spec, records, checks, split="validation", algorithm="baseline", profile="baseline",
                     runs=runs, source_tables=source_tables)
        validation = pd.concat(validation_frames, ignore_index=True)
        selection = _validate_selection(output, checks, validation, training_hashes)
        frozen_files[output / "validation_selection.json"] = sha(output / "validation_selection.json")
        for name, expected in selection["validation_source_sha256"].items():
            frozen_files[relative_file(output, name)] = expected
        checks.check(sum(value["split"] == "validation" for value in runs.values()) == 110, "96 validation RL plus 14 baseline runs")
        validation_completion = read_json(output / "validation_completion.json")
        checks.check(validation_completion.get("model_groups") == 8 and validation_completion.get("baselines") == 7
                     and validation_completion.get("frozen_spec_sha256") == spec_hash, "formal validation completed")
        complete_test = (output / "test_completion.json").is_file()
        if not complete_test and not allow_validation_only:
            checks.check(False, "Final audit requires all 62 test runs and formal test completion")
        if complete_test:
            marker = read_json(output / "test_started.json")
            checks.check(marker.get("validation_selection_sha256") == sha(output / "validation_selection.json")
                         and marker.get("frozen_spec_sha256") == spec_hash, "test dispatch used the frozen validation selection")
            checks.check(utc(selection["frozen_at"]) <= utc(marker["at"]), "selection frozen before test dispatch")
            for algorithm in ALGORITHMS:
                _audit_group(output, dataset, spec, records, checks, split="test", algorithm=algorithm,
                             profile=selection["selected_profiles"][algorithm], runs=runs, source_tables=source_tables)
            _audit_group(output, dataset, spec, records, checks, split="test", algorithm="baseline", profile="baseline",
                         runs=runs, source_tables=source_tables)
            completion = read_json(output / "test_completion.json")
            checks.check(completion.get("model_groups") == 4 and completion.get("baselines") == 7
                         and completion.get("frozen_spec_sha256") == spec_hash, "formal test completed")
            checks.check(sum(value["split"] == "test" for value in runs.values()) == 62, "48 test RL plus 14 baseline runs")
            checks.check(len(runs) == 172, "exact 172 final accounting runs")
        expected_directories = {relative_file(output, value["relative_directory"]) for value in runs.values()}
        scope_splits = ("validation", "test") if complete_test else ("validation",)
        actual_directories = {path.parent.resolve() for split in scope_splits
                              for path in (output / "evaluation" / split).rglob("fingerprints.json")}
        checks.check(actual_directories == expected_directories, "no missing or unannounced run evidence directories")
        combined = {}
        for name, frames in source_tables.items():
            expected = pd.concat(frames, ignore_index=True)
            # The runner assembles in sorted path order, so compare sorted by a
            # full stable row key rather than depending on traversal order.
            actual = read_csv(output / name)
            keys = ["run_id"] + (["timestamp"] if name == "daily_equity.csv" else ["month"] if name == "monthly_returns.csv" else [])
            _compare_tables(actual.sort_values(keys, kind="stable").reset_index(drop=True),
                            expected.sort_values(keys, kind="stable").reset_index(drop=True), checks, "assembled " + name)
            combined[name] = actual
        if complete_test:
            report["research_targets"] = _research_target_result(combined["summary.csv"], selection)
        report["transfer_receipt"] = _verify_receipt(output, checks)
        for path, expected in frozen_files.items():
            checks.check(sha(path) == expected, "input unchanged during independent audit: " + path.name)
        report["input_sha256"] = {path.relative_to(output).as_posix(): value for path, value in frozen_files.items()
                                   if path.is_relative_to(output)}
        report["status"] = "passed" if complete_test else "partial_test_pending"
        report["frozen_spec_sha256"] = spec_hash
        report["policy_replay_receipts_checked"] = sum(value["algorithm"] != "baseline" for value in runs.values())
        report["baseline_replay_limitation"] = "Runner repeats baseline evaluations before completion but does not persist a second replay fingerprint; saved baseline fingerprints and all economics are independently checked."
    except Exception as error:
        checks.check(False, f"{type(error).__name__}: {error}")
        report["exception"] = {"type": type(error).__name__, "message": str(error)}
    if checks.failed:
        report["status"] = "failed"
    coverage = {f"{split}_{kind}": sum(value["split"] == split and (value["algorithm"] == "baseline") == (kind == "baseline")
                                         for value in runs.values())
                for split in ("validation", "test") for kind in ("rl", "baseline")}
    test_complete = complete_test and len(runs) == 172 and checks.failed == 0
    report.update(passed=checks.failed == 0, test_complete=test_complete,
                  checks=checks.count, checks_total=checks.count, failure_count=checks.failed, failures=checks.failures,
                  runs_checked=len(runs), run_count=len(runs), coverage=coverage,
                  expected_final_coverage={"validation_rl": 96, "validation_baseline": 14, "test_rl": 48, "test_baseline": 14},
                  runs=runs, maximum_reconstruction_errors=checks.maximum_errors,
                  audit_completed_at=datetime.now(timezone.utc).isoformat())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_btceth_20261029/dataset.pkl.gz")
    parser.add_argument("--allow-validation-only", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--audit-file", type=Path)
    args = parser.parse_args()
    report = audit(args.output_dir, args.dataset, allow_validation_only=args.allow_validation_only, source_root=args.source_root)
    target = args.audit_file or args.output_dir / ("validation_audit.json"
              if args.allow_validation_only and not (args.output_dir / "test_completion.json").is_file()
              else "independent_audit.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: report[name] for name in ("status", "checks", "failure_count", "runs_checked")}, ensure_ascii=False))
    raise SystemExit(1 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
