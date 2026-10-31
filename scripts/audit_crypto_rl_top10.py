#!/usr/bin/env python3
"""Independent serialized RL-portfolio audit; imports no env/training/metrics."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import gzip
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_digest_from_zip(path):
    """Hash only safely loaded policy tensors; no model/env construction."""
    import torch
    with ZipFile(path) as archive:
        tensors = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        array = tensor.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str((array.shape, array.dtype)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def stamp(value):
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def announced_exit(assets, key, date):
    asset = assets.get(key, {})
    notice, until = asset.get("delisting_notice_at"), asset.get("effective_until")
    return notice is not None and until is not None and stamp(notice) <= date < stamp(until) <= date + pd.Timedelta(days=1)


def reconstruct(dataset, start, end, fills, initial, fee, slippage):
    index = pd.date_range(start, end, freq="h", inclusive="left")
    keys = [str(k) for k in dataset["hourly_close"].columns]
    cash_change = np.zeros(len(index))
    quantity_change = {key: np.zeros(len(index)) for key in keys}
    positions, basis, actual_trades = {}, {}, []
    errors = []
    fees = notional_total = realized = slippage_total = 0.
    fills = fills.copy()
    if len(fills):
        fills["timestamp"] = pd.to_datetime(fills.timestamp, utc=True)
        fills["asset_key"] = fills.asset_key.astype(str)
        offsets = index.get_indexer(fills.timestamp)
        if (offsets < 0).any():
            raise AssertionError("Fill outside fixed evaluation interval")
    else:
        offsets = []
    for offset, row in zip(offsets, fills.itertuples(index=False)):
        key, qty, price, fee_paid = row.asset_key, float(row.qty), float(row.price), float(row.fee)
        reference = float(dataset["hourly_open"].loc[row.timestamp, key])
        if not np.isfinite(reference) or reference <= 0 or qty <= 0:
            errors.append("invalid real execution quote/quantity")
        expected_price = reference * (1 + slippage if row.side == "BUY" else 1 - slippage)
        if abs(price - expected_price) > max(1e-7, abs(expected_price) * 1e-10):
            errors.append("execution/slippage")
        notional = qty * price
        if abs(fee_paid - notional * fee) > 1e-6:
            errors.append("per-leg fee")
        if row.side == "BUY":
            if row.timestamp.hour != 1:
                errors.append("buy not at 01UTC")
            cash_change[offset] -= notional + fee_paid
            quantity_change[key][offset] += qty
            positions[key] = positions.get(key, 0.) + qty
            basis[key] = basis.get(key, 0.) + notional + fee_paid
        elif row.side == "SELL":
            if row.timestamp.hour != 0:
                errors.append("sell not at 00UTC")
            owned = positions.get(key, 0.)
            if owned <= 0 or qty > owned + 1e-9:
                errors.append("selling unowned inventory")
                continue
            removed = basis[key] * min(qty / owned, 1.)
            pnl = notional - fee_paid - removed
            realized += pnl
            actual_trades.append(pnl)
            cash_change[offset] += notional - fee_paid
            quantity_change[key][offset] -= qty
            remaining = max(owned - qty, 0.)
            if remaining < 1e-12:
                positions.pop(key, None)
                basis.pop(key, None)
            else:
                positions[key] = remaining
                basis[key] -= removed
            if row.reason == "known_delisting_protection" and not announced_exit(dataset["assets"], key, row.timestamp):
                errors.append("delisting protection without previously known announcement")
        else:
            errors.append("unknown side")
        fees += fee_paid
        notional_total += notional
        slippage_total += qty * abs(price - reference)
    cash = initial + cash_change.cumsum()
    market_value = np.zeros(len(index))
    inventories = {}
    for key in keys:
        quantity = quantity_change[key].cumsum()
        inventories[key] = quantity
        if quantity.min() < -1e-8:
            errors.append("negative owned inventory")
        close = dataset["hourly_close"].loc[index, key].to_numpy()
        owned = np.abs(quantity) > 1e-10
        if (owned & (~np.isfinite(close) | (close <= 0))).any():
            errors.append("held asset missing hourly mark")
        values = np.zeros(len(index))
        np.multiply(quantity, close, out=values, where=owned)
        market_value += values
    if cash.min() < -1e-6:
        errors.append("negative cash")
    equity = cash + market_value
    result = pd.DataFrame({"cash": cash, "base_value": market_value, "equity": equity}, index=index)
    open_pnl = sum(q * float(dataset["hourly_close"].loc[index[-1], key]) - basis[key] for key, q in positions.items())
    return result, inventories, {
        "fees": fees, "total_notional": notional_total, "slippage_cost": slippage_total,
        "realized_pnl": realized, "unrealized_pnl": open_pnl, "final_positions": positions,
        "errors": sorted(set(errors)),
    }


def independent_training_candidates(dataset, start, end, episode_days=128):
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    universe = {date: frame.sort_values("slot") for date, frame in dataset["universe"].groupby("date")}
    assets = dataset["assets"]
    good_open = (np.isfinite(dataset["hourly_open"]) & dataset["hourly_open"].gt(0)).resample("D").sum().eq(24)
    good_close = (np.isfinite(dataset["hourly_close"]) & dataset["hourly_close"].gt(0)).resample("D").sum().eq(24)
    bad, active_by_day = set(), {}
    for day in dates:
        if day not in universe:
            bad.add(day)
            active_by_day[day] = []
            continue
        active = [str(k) for k in universe[day].asset_key if pd.notna(k) and not announced_exit(assets, str(k), day)]
        active_by_day[day] = active
        if any(not (good_open.loc[day, key] and good_close.loc[day, key]) for key in active):
            bad.add(day)
        previous = universe.get(day - pd.Timedelta(days=1))
        before = [] if previous is None else [
            str(k) for k in previous.asset_key
            if pd.notna(k) and not announced_exit(assets, str(k), day - pd.Timedelta(days=1))]
        for key in set(before) - set(active):
            for h in (0, 1):
                value = dataset["hourly_open"].loc[day + pd.Timedelta(hours=h), key]
                if not np.isfinite(value) or value <= 0:
                    bad.add(day)
    eligible, covered = set(), set()
    for offset in range(len(dates) - episode_days + 1):
        selected = dates[offset:offset + episode_days]
        finish = selected[-1] + pd.Timedelta(days=1)
        if any(day in bad for day in selected):
            continue
        if finish < end and any(
            not np.isfinite(dataset["hourly_open"].loc[finish, key]) or dataset["hourly_open"].loc[finish, key] <= 0
            for key in active_by_day[selected[-1]]
        ):
            continue
        eligible.add(selected[0])
        covered.update(selected)
    return eligible, covered, bad


def audit(output, dataset_path):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    data_manifest = json.loads((dataset_path.parent / "manifest.json").read_text(encoding="utf-8"))
    checks, failures, runs = 0, [], {}

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not bool(condition):
            failures.append(label)

    check(sha(dataset_path) == data_manifest["dataset_sha256"], "prepared data SHA256")
    check(sha(dataset_path) == manifest["frozen_inputs"]["dataset_sha256"], "training used this immutable prepared data")
    for path, expected in manifest["source_sha256"].items():
        if Path(path).name in ("backtest_crypto_rl_top10.py", "backtest_rl_a2c_comparison.py", "crypto_rl_portfolio_env.py",
                                "crypto_rl_data.py", "crypto_rl_reporting.py"):
            check(sha(ROOT / path) == expected, "unchanged frozen financial source: " + path)
    dataset = pd.read_pickle(dataset_path)
    dataset["assets"] = {str(k): v for k, v in dataset["assets"].items()}
    universe = dataset["universe"].copy()
    for field in ("date", "snapshot_date", "available_after", "anchor"):
        universe[field] = pd.to_datetime(universe[field], utc=True)
    universe["asset_key"] = universe.asset_key.astype(str)
    dataset["universe"] = universe
    complete = None
    for field in ("open", "high", "low", "close", "volume"):
        quotes = dataset["hourly_" + field]
        available = np.isfinite(quotes) & (quotes.ge(0) if field == "volume" else quotes.gt(0))
        complete = available if complete is None else complete & available
    complete_days = complete.resample("D").sum().eq(24)
    daily_volume = dataset["hourly_volume"].resample("D").sum(min_count=24).where(complete_days)
    ready = complete_days.rolling(31, min_periods=31).sum().eq(31) & daily_volume.rolling(30, min_periods=30).mean().gt(0)
    ready.index = ready.index + pd.Timedelta(days=1)
    check((universe.available_after == universe.snapshot_date + pd.Timedelta(days=2)).all(), "declared two-day historical snapshot lag")
    check((universe.available_after <= universe.anchor).all() and (universe.anchor <= universe.date).all(), "universe information available before decision")
    check(universe.groupby("date").slot.apply(lambda x: sorted(x) == list(range(10))).all(), "ten unique historical slots each day")
    check(universe.groupby("date").asset_key.nunique().eq(10).all(), "no duplicate asset identity")
    eligible, covered, bad = independent_training_candidates(dataset, stamp(config["train_start"]), stamp(config["train_end"]))
    for seed in manifest["models"]:
        check(sha(seed["model_path"]) == seed["model_sha256"], f"seed{seed['seed']}: frozen model file")
        check(policy_digest_from_zip(seed["model_path"]) == seed["parameter_sha256"],
              f"seed{seed['seed']}: independent policy tensor digest")
        after = [item for item in manifest.get("models_unchanged_after_evaluation", []) if item["seed"] == seed["seed"]]
        check(len(after) == 1 and after[0]["parameters_and_checkpoint_unchanged_after_evaluation"],
              f"seed{seed['seed']}: producer before/after evaluation memory+file guard passed")
        quality = seed["training_window_metadata"]
        check(quality["eligible_windows"] == len(eligible), f"seed{seed['seed']}: eligible train starts")
        check(quality["covered_training_days"] == len(covered), f"seed{seed['seed']}: effective train days")
        check({stamp(row["date"]) for row in quality["incomplete_days"]} == bad, f"seed{seed['seed']}: excluded train gap dates")
        history = seed.get("training_episode_history", [])
        check(bool(history), f"seed{seed['seed']}: actual sampled episode history exists")
        total_steps = 0
        for i, episode in enumerate(history):
            start, end = stamp(episode["start"]), stamp(episode["end_exclusive"])
            completed = stamp(episode["completed_until"])
            check(start in eligible and end == start + pd.Timedelta(days=128), f"seed{seed['seed']}: episode{i} only eligible TRAIN data")
            check(start <= completed <= end and (completed - start).days == episode["completed_steps"], f"seed{seed['seed']}: episode{i} completion")
            total_steps += episode["completed_steps"]
        check(total_steps == seed["timesteps"] == 32768, f"seed{seed['seed']}: bounded actual training budget")
    summary = pd.read_csv(output / "summary.csv", float_precision="round_trip")
    daily_table = pd.read_csv(output / "daily_equity.csv", float_precision="round_trip")
    monthly_table = pd.read_csv(output / "monthly_returns.csv", float_precision="round_trip")
    maximum_errors = {"cash": 0., "base_value": 0., "equity": 0.}
    univ = {date: rows.sort_values("slot") for date, rows in universe.groupby("date")}
    repeats = manifest.get("repeat_verifications", [])
    check(len(repeats) == len(summary) == 40 and all(item["passed"] for item in repeats),
          "all forty frozen evaluation repeat checks recorded")
    for row in summary.to_dict("records"):
        run_id = row["run_id"]
        folder = output / "runs" / run_id
        fingerprints = json.loads((folder / "fingerprints.json").read_text(encoding="utf-8"))
        for name in ("equity", "fills", "weights", "boundary_equity"):
            if name == "equity":
                with gzip.open(folder / "equity_1h.csv.gz", "rb") as stream:
                    actual_hash = hashlib.sha256(stream.read()).hexdigest()
            else:
                actual_hash = sha(folder / (name + ".csv"))
            check(actual_hash == fingerprints[name] == manifest["run_results"][run_id]["fingerprints"][name],
                  run_id + ": serialized " + name + " fingerprint")
        start, end = stamp(row["start"]), stamp(row["end_exclusive"])
        costs = config["costs"][row["scenario"]]
        fills = pd.read_csv(folder / "fills.csv", float_precision="round_trip")
        equity = pd.read_csv(folder / "equity_1h.csv.gz", index_col="timestamp", float_precision="round_trip")
        equity.index = pd.to_datetime(equity.index, utc=True)
        initial = float(row["initial_capital"])
        rebuilt, inventories, economics = reconstruct(dataset, start, end, fills, initial, **costs)
        check(initial == 10000, run_id + ": equal initial capital")
        check(equity.index.equals(rebuilt.index), run_id + ": all evaluation hours once")
        check(not economics["errors"], run_id + ": fill/ownership constraints " + str(economics["errors"]))
        entry_information_ok = True
        for entry in fills.loc[fills.side.eq("BUY")].itertuples(index=False):
            date, key = stamp(entry.timestamp).normalize(), str(entry.asset_key)
            entry_information_ok &= key in set(univ[date].asset_key)
            entry_information_ok &= date in ready.index and bool(ready.loc[date, key])
            entry_information_ok &= not announced_exit(dataset["assets"], key, date)
        check(entry_information_ok, run_id + ": every purchase has eligible identity and 31 prior complete days")
        for field in maximum_errors:
            error = float(np.abs(rebuilt[field].to_numpy() - equity[field].to_numpy()).max())
            maximum_errors[field] = max(maximum_errors[field], error)
            check(error < .001, run_id + ": hourly " + field)
        values = rebuilt.equity.to_numpy()
        peaks = np.maximum.accumulate(np.r_[initial, values])[1:]
        drawdown = values / peaks - 1
        day_close = rebuilt.equity.resample("D").last()
        day_return = day_close.pct_change()
        day_return.iloc[0] = day_close.iloc[0] / initial - 1
        sd = day_return.std(ddof=1)
        sharpe = day_return.mean() / sd * np.sqrt(365) if sd > 1e-15 else np.nan
        result_metrics = {
            "final_equity": values[-1], "net_pnl": values[-1] - initial,
            "net_return": values[-1] / initial - 1, "max_drawdown": drawdown.min(),
            "sharpe_daily": sharpe, "fees": economics["fees"], "total_notional": economics["total_notional"],
            "turnover": economics["total_notional"] / initial, "slippage_cost": economics["slippage_cost"],
            "final_cash": rebuilt.cash.iloc[-1], "final_base_value": rebuilt.base_value.iloc[-1],
            "net_liquidation_equity": rebuilt.cash.iloc[-1] + rebuilt.base_value.iloc[-1] * (1 - costs["fee"]) * (1 - costs["slippage"]),
            "average_exposure": float((rebuilt.base_value / rebuilt.equity).mean()),
        }
        for metric, value in result_metrics.items():
            tolerance = .001 if metric in ("final_equity", "net_pnl", "fees", "total_notional", "slippage_cost", "final_cash", "final_base_value", "net_liquidation_equity") else 1e-8
            check((pd.isna(value) and pd.isna(row[metric])) or abs(value - row[metric]) < tolerance, run_id + ": " + metric)
        check(abs(values[-1] - initial - economics["realized_pnl"] - economics["unrealized_pnl"]) < .001, run_id + ": realized/open PNL bridge")
        history = daily_table.loc[daily_table.run_id == run_id]
        check(len(history) == len(day_close) + 1 and np.abs(history.equity.to_numpy() - np.r_[initial, day_close.to_numpy()]).max() < .001, run_id + ": daily plot")
        daily_dd = pd.Series(drawdown, index=rebuilt.index).resample("D").last().to_numpy()
        check(np.abs(history.drawdown.to_numpy() - np.r_[0., daily_dd]).max() < 1e-8,
              run_id + ": daily plotted drawdown retains hourly peaks")
        month = monthly_table.loc[monthly_table.run_id == run_id]
        last_month = day_close.resample("ME").last().to_numpy()
        expected_month = last_month / np.r_[initial, last_month[:-1]] - 1
        check(len(month) == len(expected_month) and np.abs(month["return"].to_numpy() - expected_month).max() < 1e-8, run_id + ": monthly returns")
        weights = pd.read_csv(folder / "weights.csv", float_precision="round_trip")
        weights["date"] = pd.to_datetime(weights.date, utc=True)
        weights["asset_key"] = weights.asset_key.astype(str)
        check(weights.date.nunique() == len(day_close) and len(weights) == len(day_close) * 10,
              run_id + ": complete daily target/actual slot records")
        identity_ok = True
        actual_ok = True
        target_ok = True
        for day, records in weights.groupby("date", sort=True):
            current = univ[day].set_index("slot")
            at = rebuilt.index.get_loc(day + pd.Timedelta(hours=23))
            for w in records.itertuples(index=False):
                key = str(current.loc[w.slot, "asset_key"])
                identity_ok &= str(w.asset_key) == key
                q = inventories[key][at]
                price = float(dataset["hourly_close"].loc[rebuilt.index[at], key])
                expected_weight = q * price / values[at] if abs(q) > 1e-10 else 0.
                actual_ok &= abs(expected_weight - w.actual_weight) < 1e-8
                if pd.notna(w.target_weight):
                    cap = 1. if row["strategy"] == "btc_eth_hold" else .25
                    target_ok &= -1e-10 <= w.target_weight <= cap + 1e-10
                    if w.target_weight > 1e-10:
                        target_ok &= bool(ready.loc[day, key]) and not announced_exit(dataset["assets"], key, day)
            target_sum = records.target_weight.sum()
            target_ok &= target_sum <= .95 + 1e-10
        check(identity_ok, run_id + ": holdings attached to stable identity, not rank")
        check(actual_ok, run_id + ": actual weights reconstruct from holdings")
        check(target_ok, run_id + ": declared target concentration and cash floor")
        boundary_path = folder / "boundary_equity.csv"
        check(boundary_path.is_file(), run_id + ": reward boundary evidence exists")
        if boundary_path.is_file():
            boundary = pd.read_csv(boundary_path, index_col="timestamp", float_precision="round_trip")
            boundary.index = pd.to_datetime(boundary.index, utc=True)
            check(boundary.index.equals(pd.date_range(start, end, freq="D")),
                  run_id + ": complete unique reward decision boundaries")
            expected_values = [initial]
            for date in boundary.index[1:]:
                if date == end:
                    expected_values.append(float(values[-1]))
                else:
                    prior = rebuilt.index.get_loc(date) - 1
                    amount = float(rebuilt.cash.iloc[prior])
                    for key, q in inventories.items():
                        if abs(q[prior]) > 1e-10:
                            amount += q[prior] * float(dataset["hourly_open"].loc[date, key])
                    expected_values.append(amount)
            check(len(boundary) == len(day_close) + 1 and np.abs(boundary.equity.to_numpy() - expected_values).max() < .001, run_id + ": next-open/last-close boundary NAV")
            expected_reward = np.r_[0., 100 * np.log(np.array(expected_values[1:]) / np.array(expected_values[:-1]))]
            check(np.abs(boundary.reward.to_numpy() - expected_reward).max() < 1e-7, run_id + ": actual net-log rewards")
            check(abs(boundary.reward.sum() - 100 * np.log(values[-1] / initial)) < 1e-7, run_id + ": reward telescopes to actual terminal net return")
        runs[run_id] = {"hours": len(rebuilt), "fills": len(fills), "net_return": result_metrics["net_return"],
                        "max_drawdown": result_metrics["max_drawdown"], "fees": economics["fees"]}
        print(f"AUDIT {run_id}: {len(rebuilt)} hours, {len(fills)} fills", flush=True)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "passed" if not failures else "failed",
        "independent_of_env_training_and_metric_helpers": True,
        "checks_total": checks, "checks_failed": len(failures), "failures": failures,
        "runs_checked": len(runs), "max_absolute_reconstruction_error_usdt": maximum_errors,
        "training_quality": {"eligible_windows": len(eligible), "covered_days": len(covered),
                             "bad_days": [str(t) for t in sorted(bad)]},
        "snapshot_scope": "retrospective vendor snapshots with declared lag, not proof of original publication vintage",
        "runs": runs,
    }


def self_check():
    start, end = stamp("2025-01-01"), stamp("2025-01-03")
    index = pd.date_range(start, end, freq="h", inclusive="left")
    prices = pd.DataFrame({"a": 100., "b": np.nan}, index=index)
    qty = 900 / (100 * 1.02 * 1.01)
    sale_time = start + pd.Timedelta(days=1)
    fills = pd.DataFrame([
        {"timestamp": start + pd.Timedelta(hours=1), "asset_key": "a", "side": "BUY", "qty": qty,
         "price": 102., "fee": qty * 102 * .01, "reason": "target_increase"},
        {"timestamp": sale_time, "asset_key": "a", "side": "SELL", "qty": qty,
         "price": 98., "fee": qty * 98 * .01, "reason": "target_reduction"},
    ])
    rebuilt, _, info = reconstruct({"hourly_open": prices, "hourly_close": prices, "assets": {}},
                                   start, end, fills, 1000, .01, .02)
    assert not info["errors"]
    assert abs(rebuilt.cash.iloc[0] - 1000) < 1e-9
    assert abs(rebuilt.cash.iloc[1] - 100) < 1e-9
    assert abs(rebuilt.equity.iloc[-1] - (100 + qty * 98 * .99)) < 1e-9
    assert np.isfinite(rebuilt.to_numpy()).all(), "unheld prelisting NaN must not create phantom positions"
    print("AUDIT SELF-CHECK passed: independent buy/sell fee, slippage, cash and unheld missing-asset mark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/crypto_rl_top10_20261024")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_top10/dataset.pkl.gz")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    result = audit(args.output_dir.resolve(), args.dataset.resolve())
    (args.output_dir / "independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "runs_checked", "checks_total", "checks_failed", "max_absolute_reconstruction_error_usdt")}))
    return int(result["checks_failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
