"""Independent input/target/economic checks, with no model or env imports."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_crypto_rl_top10 import reconstruct, stamp, announced_exit, sha


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def daily_inputs(dataset):
    """Reject partial UTC days jointly across five real hourly fields."""
    frames = {name: dataset["hourly_" + name] for name in ("open", "high", "low", "close", "volume")}
    first = frames["open"]
    complete = pd.DataFrame(True, index=first.index, columns=first.columns)
    if not first.index.equals(pd.date_range(first.index[0], first.index[-1], freq="h")):
        raise AssertionError("Omitted or duplicate hourly grid rows")
    for name, frame in frames.items():
        if not frame.index.equals(first.index) or not frame.columns.equals(first.columns):
            raise AssertionError("Hourly matrices do not share identity/time axes")
        complete &= np.isfinite(frame) & (frame.ge(0) if name == "volume" else frame.gt(0))
    valid = complete.resample("D").sum().eq(24)
    out = {"open": frames["open"].resample("D").first().where(valid),
           "high": frames["high"].resample("D").max().where(valid),
           "low": frames["low"].resample("D").min().where(valid),
           "close": frames["close"].resample("D").last().where(valid),
           "volume": frames["volume"].resample("D").sum(min_count=24).where(valid)}
    volume_ready = out["volume"].rolling(30, min_periods=30).mean().gt(0)
    out["ready31"] = valid.rolling(31, min_periods=31).sum().eq(31) & volume_ready
    out["ready128"] = valid.rolling(128, min_periods=128).sum().eq(128) & volume_ready
    returns = out["close"].pct_change(fill_method=None)
    out["vol30"] = returns.rolling(30, min_periods=30).std(ddof=1)
    return out


def target_weights(returns, volatility, *, valid=None, hurdle=.0024, floor=.01, cash=.05, cap=.25):
    """Independent fixed-score portfolio, cap excess remains cash."""
    returns, volatility = np.asarray(returns, float), np.asarray(volatility, float)
    if returns.shape != volatility.shape:
        raise ValueError("Return and volatility arrays must share the same slots")
    okay = np.isfinite(returns) & np.isfinite(volatility) & (volatility >= 0)
    if valid is not None:
        okay &= np.asarray(valid, bool)
    scores = np.zeros(len(returns))
    scores[okay] = np.maximum(returns[okay] - hurdle, 0) / np.maximum(volatility[okay], floor)
    weights = np.zeros(len(returns) + 1)
    total = scores.sum()
    if total > 0:
        for i, score in enumerate(scores):
            weights[i] = min((1 - cash) * score / total, cap)
    weights[-1] = 1 - sum(weights[:-1])
    return weights


class Checks:
    def __init__(self):
        self.count, self.failures = 0, []

    def check(self, condition, label):
        self.count += 1
        if not bool(condition):
            self.failures.append(str(label))


def audit_case_economics(folder, dataset_path, *, expected_runs=None, target_provider=None):
    """Rebuild every saved hour from saved fills, not a simulated environment."""
    folder, dataset_path = Path(folder), Path(dataset_path)
    review = Checks()
    check = review.check
    manifest = read(folder / "manifest.json")
    config = manifest["config"]
    dataset = pd.read_pickle(dataset_path)
    dataset["assets"] = {str(k): v for k, v in dataset["assets"].items()}
    universe = dataset["universe"].copy()
    universe["date"] = pd.to_datetime(universe.date, utc=True)
    universe["asset_key"] = universe.asset_key.astype(str)
    univ = {date: rows.sort_values("slot").set_index("slot") for date, rows in universe.groupby("date")}
    daily = daily_inputs(dataset)
    ready31 = daily["ready31"].copy()
    ready31.index = ready31.index + pd.Timedelta(days=1)
    check(sha(dataset_path) == manifest["frozen_inputs"]["dataset_sha256"], "same immutable hourly dataset")
    summary = pd.read_csv(folder / "summary.csv", float_precision="round_trip")
    histories = pd.read_csv(folder / "daily_equity.csv", float_precision="round_trip")
    months = pd.read_csv(folder / "monthly_returns.csv", float_precision="round_trip")
    check(summary.run_id.is_unique, "one financial summary per run identity")
    if expected_runs is not None:
        check(set(summary.run_id) == set(expected_runs), "complete intended run family")
    check(set(summary.run_id) == set(manifest["run_results"]), "manifest agrees with summary coverage")
    repeats = manifest.get("repeat_verifications", [])
    repeat_by_id = {entry["run_id"]: entry for entry in repeats}
    check(len(repeats) == len(summary) and set(repeat_by_id) == set(summary.run_id), "all repeat verifications recorded exactly once")
    maxima = {name: 0. for name in ("cash", "base_value", "equity")}
    results = {}
    for row in summary.to_dict("records"):
        run_id = row["run_id"]
        run = folder / "runs" / run_id
        fingerprints = read(run / "fingerprints.json")
        for field, filename in (("equity", "equity_1h.csv.gz"), ("fills", "fills.csv"),
                                ("weights", "weights.csv"), ("boundary_equity", "boundary_equity.csv")):
            if field == "equity":
                with gzip.open(run / filename, "rb") as stream:
                    digest = hashlib.sha256(stream.read()).hexdigest()
            else:
                digest = sha(run / filename)
            expected = manifest["run_results"][run_id]["fingerprints"][field]
            repeated = repeat_by_id[run_id]
            check(digest == fingerprints[field] == expected == repeated["fingerprints"][field]
                  and repeated.get("passed") is True, run_id + ": actual and repeat " + field + " fingerprints")
        start, end = stamp(row["start"]), stamp(row["end_exclusive"])
        costs = config["costs"][row["scenario"]]
        fills = pd.read_csv(run / "fills.csv", float_precision="round_trip")
        equity = pd.read_csv(run / "equity_1h.csv.gz", index_col="timestamp", float_precision="round_trip")
        equity.index = pd.to_datetime(equity.index, utc=True)
        initial = float(row["initial_capital"])
        rebuilt, inventories, economics = reconstruct(dataset, start, end, fills, initial, **costs)
        check(initial == 10000, run_id + ": equal capital")
        check(equity.index.equals(rebuilt.index), run_id + ": every evaluation hour once")
        check(not economics["errors"], run_id + ": ownership, cost, cash constraints " + str(economics["errors"]))
        for field in maxima:
            error = float(np.abs(rebuilt[field].to_numpy() - equity[field].to_numpy()).max())
            maxima[field] = max(maxima[field], error)
            check(error < 1e-6, run_id + ": independent hourly " + field)
        values = rebuilt.equity.to_numpy()
        peaks = np.maximum.accumulate(np.r_[initial, values])[1:]
        drawdown = values / peaks - 1
        day_close = rebuilt.equity.resample("D").last()
        daily_return = day_close.pct_change()
        daily_return.iloc[0] = day_close.iloc[0] / initial - 1
        deviation = daily_return.std(ddof=1)
        sharpe = daily_return.mean() / deviation * np.sqrt(365) if deviation > 1e-15 else np.nan
        annual = np.expm1(np.log(values[-1] / initial) * 365 / ((end - start).total_seconds() / 86400))
        metrics = {"final_equity": values[-1], "net_pnl": values[-1] - initial,
                   "net_return": values[-1] / initial - 1, "max_drawdown": drawdown.min(),
                   "sharpe_daily": sharpe, "cagr": annual,
                   "calmar": annual / abs(drawdown.min()) if drawdown.min() < -1e-15 else np.nan,
                   "fees": economics["fees"], "total_notional": economics["total_notional"],
                   "turnover": economics["total_notional"] / initial,
                   "slippage_cost": economics["slippage_cost"], "final_cash": rebuilt.cash.iloc[-1],
                   "final_base_value": rebuilt.base_value.iloc[-1],
                   "net_liquidation_equity": rebuilt.cash.iloc[-1] + rebuilt.base_value.iloc[-1] * (1 - costs["fee"]) * (1 - costs["slippage"]),
                   "average_exposure": float((rebuilt.base_value / rebuilt.equity).mean()),
                   "fills": len(fills), "buy_fills": int(fills.side.eq("BUY").sum()),
                   "sell_fills": int(fills.side.eq("SELL").sum())}
        metrics["net_liquidation_return"] = metrics["net_liquidation_equity"] / initial - 1
        check(row.get("drawdown_grain") == "1h_close", run_id + ": honest drawdown grain")
        for name, value in metrics.items():
            if name in row:
                tolerance = 1e-6 if name in ("final_equity", "net_pnl", "fees", "total_notional", "slippage_cost", "final_cash", "final_base_value", "net_liquidation_equity") else 1e-8
                check((pd.isna(value) and pd.isna(row[name])) or abs(value - row[name]) < tolerance, run_id + ": " + name)
        check(abs(values[-1] - initial - economics["realized_pnl"] - economics["unrealized_pnl"]) < 1e-6,
              run_id + ": realized plus open PNL bridge")
        plotted = histories.loc[histories.run_id == run_id]
        check(len(plotted) == len(day_close) + 1 and np.max(np.abs(plotted.equity.to_numpy() - np.r_[initial, day_close.to_numpy()])) < 1e-6,
              run_id + ": daily equity plot")
        expected_drawdown = np.r_[0., pd.Series(drawdown, index=rebuilt.index).resample("D").last().to_numpy()]
        check(len(plotted) == len(expected_drawdown) and np.max(np.abs(plotted.drawdown.to_numpy() - expected_drawdown)) < 1e-8,
              run_id + ": daily plot retains hourly peaks")
        month = months.loc[months.run_id == run_id]
        month_end = day_close.resample("ME").last().to_numpy()
        expected_monthly = month_end / np.r_[initial, month_end[:-1]] - 1
        check(len(month) == len(expected_monthly) and np.max(np.abs(month["return"].to_numpy() - expected_monthly)) < 1e-8,
              run_id + ": monthly returns")
        weights = pd.read_csv(run / "weights.csv", float_precision="round_trip", dtype={"asset_key": str})
        weights["date"] = pd.to_datetime(weights.date, utc=True)
        check(len(weights) == len(day_close) * 10 and weights.date.nunique() == len(day_close), run_id + ": all daily slot records")
        actual_ok = identity_ok = target_ok = formula_ok = True
        for day, records in weights.groupby("date", sort=True):
            current = univ[day]
            at = rebuilt.index.get_loc(day + pd.Timedelta(hours=23))
            if target_provider is not None and row["strategy"] not in ("cash", "top10_equal", "top10_cap", "btc_eth_hold"):
                expected_target = target_provider(row, day, current)
                ordered = records.sort_values("slot")
                formula_ok &= np.max(np.abs(ordered.target_weight.to_numpy() - expected_target[:10])) < 1e-8
            for item in records.itertuples(index=False):
                key = str(current.loc[item.slot, "asset_key"])
                identity_ok &= str(item.asset_key) == key
                quantity = inventories[key][at]
                price = float(dataset["hourly_close"].loc[rebuilt.index[at], key])
                expected_actual = quantity * price / values[at] if abs(quantity) > 1e-10 else 0.
                actual_ok &= abs(expected_actual - item.actual_weight) < 1e-8
                if pd.notna(item.target_weight):
                    target_ok &= -1e-10 <= item.target_weight <= (1. if row["strategy"] == "btc_eth_hold" else .25) + 1e-10
                    if item.target_weight > 1e-10:
                        target_ok &= bool(ready31.loc[day, key]) and not announced_exit(dataset["assets"], key, day)
            target_ok &= records.target_weight.sum() <= .95 + 1e-10
        check(identity_ok, run_id + ": stable economic identities")
        check(actual_ok, run_id + ": actual weights reconstructed")
        check(target_ok, run_id + ": target cap/cash/eligibility")
        if target_provider is not None:
            check(formula_ok, run_id + ": independently reproduced forecast-to-target rule")
        boundary = pd.read_csv(run / "boundary_equity.csv", index_col="timestamp", float_precision="round_trip")
        boundary.index = pd.to_datetime(boundary.index, utc=True)
        check(boundary.index.equals(pd.date_range(start, end, freq="D")), run_id + ": all net-reward boundary times")
        expected_boundary = [initial]
        for date in boundary.index[1:]:
            if date == end:
                expected_boundary.append(float(values[-1]))
            else:
                prior = rebuilt.index.get_loc(date) - 1
                nav = float(rebuilt.cash.iloc[prior])
                for key, quantities in inventories.items():
                    if abs(quantities[prior]) > 1e-10:
                        nav += quantities[prior] * float(dataset["hourly_open"].loc[date, key])
                expected_boundary.append(nav)
        check(np.max(np.abs(boundary.equity.to_numpy() - expected_boundary)) < 1e-6, run_id + ": actual boundary NAV")
        expected_rewards = np.r_[0., 100 * np.log(np.array(expected_boundary[1:]) / np.array(expected_boundary[:-1]))]
        check(np.max(np.abs(boundary.reward.to_numpy() - expected_rewards)) < 1e-7, run_id + ": log NAV rewards")
        check(abs(boundary.reward.sum() - 100 * np.log(values[-1] / initial)) < 1e-7, run_id + ": rewards telescope")
        results[run_id] = {"hours": len(rebuilt), "fills": len(fills), "net_return": metrics["net_return"],
                           "max_drawdown": metrics["max_drawdown"], "fees": metrics["fees"]}
        print(f"AUDIT {folder.name}/{run_id}: {len(rebuilt)} hours, {len(fills)} fills", flush=True)
    return {"status": "passed" if not review.failures else "failed", "checks_total": review.count,
            "checks_failed": len(review.failures), "failures": review.failures,
            "runs_checked": len(results), "runs": results,
            "max_absolute_reconstruction_error_usdt": maxima,
            "independent_of_model_env_and_reporting_implementations": True}
