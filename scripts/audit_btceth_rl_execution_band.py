"""Independent, validation-only audit of one post-annual execution-band trial.

This file does not import any environment, execution ledger or trained policy.
Only frozen pure-audit arithmetic is reused. Passing accounts never constitutes
fresh holdout evidence: the first annual test had already been observed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_btceth_rl import (
    Checks, _check_summary, independent_training_calendar, read_csv, read_equity,
    reconstruct_fills, verify_saved_fingerprints,
)
from scripts.audit_btceth_rl_exposure_cap import (
    CASE_FILES, COSTS, FINANCIAL, GROUP_FILES, SEEDS, SYMBOLS,
    attach_inventory_replay, candidate_score, compare_group_tables, load_market,
    mapped_target, read_json, safe_file, sha, utc,
)

VALIDATION = ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z")
FAMILY, STRATEGY = "cap075_band05", "tqc_cap075_band05"
BAND, REFERENCE_BAND, MULTIPLIER = .05, .025, .75
FLAGS = ("designed_after_first_annual", "test_year_already_seen", "not_fresh_holdout")
SOURCE_FILES = (*FINANCIAL, "scripts/btceth_rl_goal_candidate.py",
    "scripts/btceth_rl_exposure_cap.py", "scripts/audit_btceth_rl.py",
    "scripts/audit_btceth_rl_exposure_cap.py", "scripts/btceth_rl_execution_band.py",
    "scripts/audit_btceth_rl_execution_band.py")
TRACE_FIELDS = {
    "day", "seed", "raw_action0", "raw_action1", "raw_btc", "raw_eth", "raw_cash",
    "capped_btc", "capped_eth", "capped_cash", "target_btc", "target_eth", "target_cash",
    "observation_sha256", "observation_btc_weight", "observation_eth_weight",
    "observation_cash_weight", "observation_drawdown",
}


def now():
    return datetime.now(timezone.utc).isoformat()


class Inputs:
    """Collect immutable hashes without trusting saved remote absolute paths."""
    def __init__(self, output, main, reference, source, checks):
        self.roots = {key: Path(value).resolve() for key, value in
                      (("extension", output), ("main", main), ("reference", reference), ("source", source))}
        self.hashes = {key: {} for key in self.roots}
        self.checks = checks

    def capture(self, root, name):
        path = safe_file(self.roots[root], name)
        digest, key = sha(path), str(name).replace("\\", "/")
        previous = self.hashes[root].get(key)
        if previous is not None:
            self.checks.check(previous == digest, "Input changed between reads: " + root + ":" + key)
        self.hashes[root][key] = digest
        return path

    def json(self, root, name):
        return read_json(self.capture(root, name))

    def csv(self, root, name):
        return read_csv(self.capture(root, name))

    def pin_map(self, root, values, required, label):
        if not self.checks.check(isinstance(values, dict) and set(required).issubset(values),
                                 label + ": complete immutable pin set"):
            raise ValueError(label + " is missing required immutable evidence")
        for name, expected in values.items():
            self.checks.check(sha(self.capture(root, name)) == expected, label + ": " + name)

    def unchanged(self):
        for root, values in self.hashes.items():
            for name, expected in values.items():
                self.checks.check(sha(safe_file(self.roots[root], name)) == expected,
                                  "Input unchanged: " + root + ":" + name)


def validation_files():
    return {"validation_completion.json", *GROUP_FILES[:3],
        *(f"validation/balanced/{name}" for name in GROUP_FILES),
        *(f"validation/balanced/{cost}/{key}/{name}" for cost in COSTS
          for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def reference_files():
    """Root aggregates are deliberately excluded: they contain annual rows."""
    return {*(f"validation/balanced/{name}" for name in GROUP_FILES),
        *(f"validation/balanced/{cost}/{key}/{name}" for cost in COSTS
          for key in (*map(str, SEEDS), "ensemble") for name in CASE_FILES)}


def rebalance_decision(target, prior, band=BAND):
    target, prior = np.asarray(target, dtype=float), np.asarray(prior, dtype=float)
    if (target.shape != (3,) or prior.shape not in ((3,), (4,))
            or not np.isfinite(target).all() or not np.isfinite(prior).all()
            or band not in (BAND, REFERENCE_BAND)):
        raise ValueError("Three finite target/prior weights and a declared band required")
    cash_force = target[:2].sum() <= 1e-12 and prior[:2].sum() > 0
    return bool(cash_force or np.max(np.abs(target - prior[:3])) >= band)


def independent_market_features(dataset, *, cutoff=VALIDATION[1]):
    """27-state market prefix, constructed from actual input dependencies only.

    Output has 23 market features indexed by the next decision day. A strict
    cutoff is applied before every operation; the current day's close cannot
    influence that day's observation. Missing close23 or volume is not filled.
    """
    frames, index = {}, None
    for field in ("open", "high", "low", "close", "volume"):
        frame = dataset["hourly_" + field]
        if (not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex)
                or frame.index.tz is None or list(frame.columns) != list(SYMBOLS)):
            raise ValueError("Two aligned timezone-aware BTCUSDT/ETHUSDT input frames required")
        frame = frame.loc[frame.index < utc(cutoff)].copy()
        if (frame.index.has_duplicates or not frame.index.is_monotonic_increasing
                or (frame.index != frame.index.floor("h")).any()
                or (index is not None and not index.equals(frame.index))):
            raise ValueError("Hourly two-coin data must be unique, ordered and exactly aligned")
        index = frame.index
        frames[field] = frame.astype(float)
    if not len(index) or not index.equals(pd.date_range(index[0], index[-1], freq="h")):
        raise ValueError("No omitted hourly rows or fabricated time alignment")
    days = pd.date_range(index[0].normalize(), index[-1].normalize(), freq="D")
    close = frames["close"].loc[index.hour == 23].copy()
    close.index = close.index.normalize()
    close = close.reindex(days).where(lambda x: np.isfinite(x) & x.gt(0))
    volume = frames["volume"]
    valid_volume = np.isfinite(volume) & volume.ge(0)
    volume = volume.where(valid_volume).resample("D").sum(min_count=24).where(
        valid_volume.resample("D").sum().eq(24)).reindex(days)
    returns = close.pct_change(fill_method=None)
    components = [returns / .1, (close / close.shift(7) - 1) / .3,
        (close / close.shift(30) - 1) / .5, close / close.shift(90) - 1,
        (close / close.shift(180) - 1) / 2,
        (close / close.rolling(20, min_periods=20).mean() - 1) / .2,
        (close / close.rolling(60, min_periods=60).mean() - 1) / .35,
        (close / close.rolling(200, min_periods=200).mean() - 1) / .5,
        returns.rolling(30, min_periods=30).std(ddof=1) / .1,
        (volume / volume.rolling(30, min_periods=30).mean() - 1) / 2]
    coin = np.stack([frame.to_numpy() for frame in components], axis=2).reshape(len(close), 20)
    ratio = close.ETHUSDT / close.BTCUSDT
    relative = np.column_stack([(ratio / ratio.shift(n) - 1).to_numpy() / scale
                               for n, scale in ((7, .3), (30, .5), (90, 1.))])
    return pd.DataFrame(np.clip(np.column_stack([coin, relative]), -10., 10.),
                        index=days + pd.Timedelta(days=1))


def audit_member_targets(trace, weights, rebuilt, fills, *, start, end, seeds,
                         checks, label, band=BAND, market_features=None,
                         require_observation_values=False):
    required = set(TRACE_FIELDS)
    if require_observation_values:
        required.update(("observation_values_json", "observation_asof", "feature_input_cutoff"))
    if not checks.check(required.issubset(trace), label + ": complete original-action/member/common-state schema"):
        return
    trace, weights = trace.copy(), weights.copy()
    trace["day"] = pd.to_datetime(trace.day, utc=True)
    weights["date"] = pd.to_datetime(weights.date, utc=True)
    days = pd.date_range(utc(start), utc(end), freq="D", inclusive="left")
    checks.check(len(trace) == len(days) * len(seeds) and not trace[["day", "seed"]].duplicated().any()
                 and set(trace.day) == set(days) and set(trace.seed) == set(seeds),
                 label + ": exact member/day Cartesian product")
    checks.check(len(weights) == len(days) * 2 and set(weights.date) == set(days)
                 and not weights[["date", "asset_key"]].duplicated().any(),
                 label + ": every actual two-coin daily weight")
    if not checks.check({"inventory_quantity", "inventory_presence", "coin_values"}.issubset(rebuilt),
                         label + ": independent exact SELL-only dust inventory replay"):
        return
    eq = rebuilt["equity"]
    peak = eq.equity.cummax().clip(lower=10000.)
    fill_days = set(pd.to_datetime(fills.timestamp, utc=True).dt.normalize()) if len(fills) else set()
    for day in days:
        member = trace.loc[trace.day.eq(day)].sort_values("seed")
        if not checks.check(member.seed.tolist() == sorted(seeds),
                             label + ": fixed seed population once per day"):
            continue
        hashes = member.observation_sha256.astype(str)
        checks.check(hashes.nunique() == 1 and all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in hashes),
                     label + ": all policies share one actual-account observation")
        previous = day - pd.Timedelta(hours=1)
        if day == utc(start):
            prior = np.array([0., 0., 1., 0.])
        else:
            state = eq.loc[previous]
            prior = np.r_[rebuilt["coin_values"].loc[previous].to_numpy() / state.equity,
                          state.cash / state.equity, 1 - state.equity / peak.loc[previous]]
        observed_columns = ["observation_btc_weight", "observation_eth_weight",
                            "observation_cash_weight", "observation_drawdown"]
        checks.vector_close(member[observed_columns], np.tile(prior.astype(np.float32), (len(member), 1)),
                            label + ": current weights/drawdown use this actual ledger", atol=1e-7, rtol=0)
        if "observation_values_json" in member:
            for row in member.itertuples(index=False):
                values = np.asarray(json.loads(row.observation_values_json), dtype=float)
                if not checks.check(values.shape == (27,) and np.isfinite(values).all(),
                                    label + ": full finite 27-state observation"):
                    continue
                encoded = np.asarray(values, dtype="<f4").tobytes()
                checks.check(hashlib.sha256(encoded).hexdigest() == row.observation_sha256,
                             label + ": observation SHA binds the explicit float32 input")
                checks.vector_close(values[-4:], prior, label + ": full input actual account state", atol=1e-7, rtol=0)
                if market_features is not None:
                    checks.vector_close(values[:23], market_features.loc[day].to_numpy().astype(np.float32),
                                        label + ": full market input independently uses only prior closes", atol=1e-7, rtol=0)
                checks.check(utc(row.observation_asof) == previous and utc(row.feature_input_cutoff) == previous,
                             label + ": feature/input cutoff is strictly before execution day")
        elif require_observation_values:
            checks.check(False, label + ": complete causal observation evidence required")
        targets = []
        for row in member.itertuples(index=False):
            raw, mapped = mapped_target(row.raw_action0, row.raw_action1)
            checks.vector_close([row.raw_btc, row.raw_eth, row.raw_cash], raw,
                                label + ": original clipped action cash/gross/share mapping", atol=1e-12)
            checks.vector_close([row.capped_btc, row.capped_eth, row.capped_cash], mapped,
                                label + ": .75 coin targets and residual cash", atol=1e-12)
            targets.append(mapped)
        target = np.asarray(targets).mean(axis=0)
        checks.check(target[:2].sum() <= .7125 + 1e-12 and target[2] >= .2875 - 1e-12
                     and (target >= 0).all(), label + ": funded .7125 requested gross cap")
        checks.vector_close(member[["target_btc", "target_eth", "target_cash"]],
                            np.tile(target, (len(member), 1)),
                            label + ": true arithmetic mean of all member targets", atol=1e-12)
        daily = weights.loc[weights.date.eq(day)]
        if not checks.check(len(daily) == 2 and set(daily.asset_key) == set(SYMBOLS),
                             label + ": original coin identities"):
            continue
        daily = daily.set_index("asset_key").loc[list(SYMBOLS)]
        checks.check(daily.slot.tolist() == [0, 1] and daily.symbol.tolist() == ["BTC", "ETH"],
                     label + ": fixed BTC/ETH slots")
        checks.vector_close(daily.target_weight, target[:2], label + ": funded mean target in execution history", atol=1e-12)
        closing = day + pd.Timedelta(hours=23)
        checks.vector_close(daily.actual_weight,
            rebuilt["coin_values"].loc[closing].to_numpy() / eq.loc[closing, "equity"],
            label + ": close weights independently reconstruct inventory", atol=1e-10)
        decision = rebalance_decision(target, prior, band)
        checks.check(daily.rebalanced.map(lambda value: isinstance(value, (bool, np.bool_))).all()
                     and daily.rebalanced.eq(decision).all(), label + ": declared band and complete cash exit")
        if not decision:
            checks.check(day not in fill_days, label + ": skipped band cannot contain trades")
        elif day == utc(start) and target[:2].sum() >= band:
            initial = fills.loc[pd.to_datetime(fills.timestamp, utc=True).dt.normalize().eq(day)] if len(fills) else fills
            checks.check(len(initial) > 0 and initial.side.eq("BUY").any(),
                         label + ": funded material initial target actually buys")


def validate_group_rows(frame, *, reference=False):
    required = {"run_id", "algorithm", "profile", "family", "strategy", "scenario", "split", "seed",
                "primary_policy", "initial_capital", "start", "end_exclusive", "net_return",
                "max_drawdown", "final_equity", "fees", "exposure_multiplier"}
    if not required.issubset(frame) or len(frame) != 12 or frame.run_id.duplicated().any():
        raise ValueError("Validation group must contain exactly five seeds and one ensemble at both costs")
    family, strategy = ("cap075", "tqc_cap075") if reference else (FAMILY, STRATEGY)
    fixed = {"algorithm": "tqc", "profile": "balanced", "family": family, "strategy": strategy,
             "split": "validation", "initial_capital": 10000., "exposure_multiplier": MULTIPLIER}
    if any(not frame[key].eq(value).all() for key, value in fixed.items()):
        raise ValueError("Only the one frozen TQC balanced capped validation variant is permitted")
    if not reference and ("rebalance_band" not in frame or not frame.rebalance_band.eq(BAND).all()):
        raise ValueError("New summary must explicitly retain the one5pp execution band")
    if (not pd.to_datetime(frame.start, utc=True).eq(utc(VALIDATION[0])).all()
            or not pd.to_datetime(frame.end_exclusive, utc=True).eq(utc(VALIDATION[1])).all()):
        raise ValueError("Validation must remain exactly295 days and may not include annual rows")
    ids = {f"validation__{cost}__{family}__tqc__balanced__{key}" for cost in COSTS
           for key in (*map(str, SEEDS), "ensemble")}
    if set(frame.run_id) != ids or not frame.primary_policy.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("Case IDs/primary flags differ from the declared twelve-case population")
    for cost in COSTS:
        group = frame.loc[frame.scenario.eq(cost)]
        primary, members = group.loc[group.primary_policy], group.loc[~group.primary_policy]
        if (len(primary) != 1 or primary.seed.notna().any() or len(members) != 5
                or members.seed.isna().any() or set(members.seed) != set(SEEDS)):
            raise ValueError("Cannot select individual seeds or omit stress costs")
    numeric = frame[["net_return", "max_drawdown", "final_equity", "fees"]].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (frame.net_return <= -1).any()
            or (frame.max_drawdown > 0).any() or (frame.max_drawdown <= -1).any()
            or (frame.fees < 0).any() or (frame.final_equity <= 0).any()
            or (frame.final_equity / 10000 - 1 - frame.net_return).abs().max() > 1e-10):
        raise ValueError("Validation financial summary is inconsistent")


def compare_validation_candidates(new, reference):
    """Same two-cost score; an exact score tie always keeps the old execution."""
    validate_group_rows(new)
    validate_group_rows(reference, reference=True)
    candidates = []
    for family, frame, band in (("cap075", reference, REFERENCE_BAND), (FAMILY, new, BAND)):
        costs = {cost: frame.loc[frame.scenario.eq(cost) & frame.primary_policy].iloc[0] for cost in COSTS}
        candidates.append({"family": family, "algorithm": "tqc", "profile": "balanced",
            "rebalance_band": band, "score": candidate_score(costs["base"], costs["stress"]),
            **{cost: {"run_id": str(values.run_id), **{key: float(values[key]) for key in
                ("net_return", "max_drawdown", "fees", "final_equity")}} for cost, values in costs.items()}})
    return sorted(candidates, key=lambda row: (-row["score"], 0 if row["family"] == "cap075" else 1))


def funding_nav(initial, cost, values, targets, *, buying):
    """Independent monotone root for after-cost value, without execution code."""
    values, targets = np.asarray(values, dtype=float), np.asarray(targets, dtype=float)
    lo, hi = 0., float(initial)
    for _ in range(100):
        nav = (lo + hi) / 2
        volume = np.maximum(targets * nav - values if buying else values - targets * nav, 0).sum()
        if nav + cost * volume > initial:
            hi = nav
        else:
            lo = nav
    return (lo + hi) / 2


def audit_execution_quantities(dataset, trace, rebuilt, fills, *, start, end, band, checks, label):
    """Reconstruct both sizing legs from prices and the original funded cash.

    This complements the independent fill/PnL replay: economically consistent
    fills still fail if the strategy filled a quantity unrelated to its target.
    """
    eq = rebuilt["equity"]
    quantity = rebuilt["inventory_quantity"]
    trace, trades = trace.copy(), fills.copy()
    trace["day"] = pd.to_datetime(trace.day, utc=True)
    if len(trades):
        trades["timestamp"] = pd.to_datetime(trades.timestamp, utc=True)
    fee, slip = rebuilt["costs"]["fee"], rebuilt["costs"]["slippage"]
    for day in pd.date_range(utc(start), utc(end), freq="D", inclusive="left"):
        rows = trace.loc[trace.day.eq(day)]
        if len(rows) == 0:
            continue
        target = rows.iloc[0][["target_btc", "target_eth", "target_cash"]].to_numpy(dtype=float)
        previous = day - pd.Timedelta(hours=1)
        cash = 10000. if day == utc(start) else float(eq.at[previous, "cash"])
        owned = np.zeros(2) if day == utc(start) else quantity.loc[previous].to_numpy(dtype=float).copy()
        mark = dataset["hourly_close"].loc[previous, list(SYMBOLS)].to_numpy(dtype=float)
        previous_nav = cash + float((owned * mark).sum())
        prior = np.r_[owned * mark / previous_nav, cash / previous_nav]
        current = trades.loc[trades.timestamp.dt.normalize().eq(day)] if len(trades) else trades
        if not rebalance_decision(target, prior, band):
            checks.check(current.empty, label + ": no cash/inventory mutations inside skipped band")
            continue
        expected = {}
        opening = dataset["hourly_open"].loc[day, list(SYMBOLS)].to_numpy(dtype=float)
        values = owned * opening
        initial = cash + values.sum()
        sell_cost = 1 - (1 - slip) * (1 - fee)
        after_sell = funding_nav(initial, sell_cost, values, target[:2], buying=False)
        for i, symbol in enumerate(SYMBOLS):
            reduction = max(values[i] - target[i] * after_sell, 0.)
            if reduction > 1e-9:
                sold = min(reduction / opening[i], owned[i])
                expected[("SELL", symbol)] = sold
                cash += sold * opening[i] * (1 - slip) * (1 - fee)
                owned[i] -= sold
                if owned[i] <= 1e-14:
                    owned[i] = 0.
        opening = dataset["hourly_open"].loc[day + pd.Timedelta(hours=1), list(SYMBOLS)].to_numpy(dtype=float)
        values = owned * opening
        initial = cash + values.sum()
        buy_cost = (1 + slip) * (1 + fee) - 1
        after_buy = funding_nav(initial, buy_cost, values, target[:2], buying=True)
        desired = np.maximum(target[:2] * after_buy - values, 0.)
        reserve = max(.05, target[2])
        affordable = max(0., (cash - reserve * initial) / (1 + buy_cost - reserve * buy_cost))
        scale = min(1., affordable / desired.sum()) if desired.sum() > 0 else 0.
        for i, symbol in enumerate(SYMBOLS):
            if desired[i] * scale > 1e-9:
                expected[("BUY", symbol)] = desired[i] * scale / opening[i]
        actual = {(str(row.side), str(row.asset_key)): float(row.qty) for row in current.itertuples(index=False)}
        checks.check(len(actual) == len(current) and set(actual) == set(expected),
                     label + ": exactly the funded target sell/buy legs")
        for key, amount in expected.items():
            checks.close(actual.get(key), amount, label + ": independently solved target quantity " + str(key),
                         atol=1e-11, rtol=1e-10)


def audit_case(directory, dataset, row, records, protocol, checks, *, reference=False, features=None):
    label, cost = row["run_id"], row["scenario"]
    active = list(SEEDS) if row["primary_policy"] else [int(row["seed"])]
    equity, fills = read_equity(directory / "equity_1h.csv.gz"), read_csv(directory / "fills.csv")
    metadata, weights = read_json(directory / "metadata.json"), read_csv(directory / "weights.csv")
    fingerprints = verify_saved_fingerprints(directory, checks, label)
    rebuilt = reconstruct_fills(dataset, fills, equity, metadata, start=VALIDATION[0], end=VALIDATION[1],
                               **COSTS[cost], checks=checks, label=label)
    if rebuilt is None:
        raise ValueError("No complete independent hourly account: " + label)
    inventory = attach_inventory_replay(rebuilt, fills, dataset["hourly_close"],
                                        checks=checks, label=label, metadata=metadata)
    _check_summary(row, rebuilt["metrics"], checks, label)
    family, band = ("cap075", REFERENCE_BAND) if reference else (FAMILY, BAND)
    parameters = [records[seed]["parameter_sha256"] for seed in active]
    checks.check(metadata.get("algorithm") == "tqc" and metadata.get("profile") == "balanced"
                 and metadata.get("family") == family and metadata.get("model_count") == len(active),
                 label + ": fixed untrained model population")
    checks.check(metadata.get("parameter_sha256_before") == parameters
                 and metadata.get("parameter_sha256_after") == parameters,
                 label + ": inference leaves trained parameters immutable")
    for key, value in (("rebalance_band", band), ("cash_floor_target", .05),
                       ("max_asset_weight_target", .95), ("drawdown_penalty", .25),
                       ("exposure_multiplier", MULTIPLIER)):
        checks.close(metadata.get(key), value, label + ": financial parameter " + key, atol=1e-15, rtol=0)
    checks.check(metadata.get("old_actions_or_equity_scaled") is False,
                 label + ": actual funded account rather than old NAV scaling")
    if not reference:
        checks.check(metadata.get("band_change_only") is True
                     and all(metadata.get(flag) is True for flag in FLAGS),
                     label + ": only band changed, observed annual history disclosed")
    trace = read_csv(directory / "member_targets.csv")
    audit_member_targets(trace, weights, rebuilt, fills, start=VALIDATION[0], end=VALIDATION[1],
        seeds=active, checks=checks, label=label, band=band, market_features=features,
        require_observation_values=not reference)
    rebuilt["costs"] = COSTS[cost]
    audit_execution_quantities(dataset, trace, rebuilt, fills, start=VALIDATION[0], end=VALIDATION[1],
                               band=band, checks=checks, label=label)
    replay = read_json(directory / "replay.json")
    checks.check(replay.get("run_id") == label and replay.get("passed") is True
                 and replay.get("repeat_exact") is True
                 and replay.get("fingerprints") == fingerprints == replay.get("repeated_fingerprints"),
                 label + ": complete repeated financial fingerprints")
    checks.check(replay.get("member_targets_sha256") == sha(directory / "member_targets.csv")
                 == replay.get("repeated_member_targets_sha256"), label + ": exact repeated member/state trace")
    checks.check(replay.get("parameter_sha256_before") == parameters
                 and replay.get("parameter_sha256_after") == parameters,
                 label + ": repeat uses immutable original policy parameters")
    model_set = [item for item in protocol["models"] if item["seed"] in active]
    checks.check(replay.get("model_records") == model_set, label + ": original model and record bytes")
    device = replay.get("device", {})
    checks.check(device.get("device") == "cuda:0" and device.get("capability") == [7, 0]
                 and "V100" in device.get("name", "")
                 and device.get("visible_gpu_uuid") in {g["uuid"] for g in protocol["physical_gpus"]},
                 label + ": original isolated physical V100 inference")
    daily = rebuilt["equity"].equity.resample("D").last()
    daily_dd = rebuilt["drawdown"].resample("D").last()
    history = pd.DataFrame({"timestamp": [utc(VALIDATION[0]), *list(daily.index + pd.Timedelta(days=1))],
        "equity": [10000., *daily.tolist()], "net_return": [0., *(daily / 10000 - 1).tolist()],
        "drawdown": [0., *daily_dd.tolist()]})
    boundary = read_equity(directory / "boundary_equity.csv")
    checks.check(boundary.index.equals(pd.DatetimeIndex(history.timestamp)), label + ": complete daily boundary timeline")
    checks.vector_close(boundary.equity, history.equity, label + ": independent daily boundary NAV", atol=1e-8)
    rewards, previous_nav, previous_dd = [0.], 10000., 0.
    for _, hours in rebuilt["equity"].groupby(rebuilt["equity"].index.normalize()):
        deepest = -float(rebuilt["drawdown"].loc[hours.index].min())
        final = float(hours.equity.iloc[-1])
        rewards.append(100 * (math.log(final / previous_nav) - .25 * max(deepest - previous_dd, 0.)))
        previous_nav, previous_dd = final, -float(rebuilt["drawdown"].loc[hours.index[-1]])
    checks.vector_close(boundary.reward, rewards, label + ": unchanged balanced daily reward", atol=1e-9)
    result = {"run_id": label, "relative_directory": str(directory), "hours": len(equity),
        "reconstructed": True, "repeat_exact": True, "fingerprints": fingerprints,
        "member_targets_sha256": sha(directory / "member_targets.csv"), "inventory_replay": inventory,
        **{key: rebuilt["metrics"][key] for key in ("net_return", "max_drawdown", "fees", "final_equity")}}
    return result, history, rebuilt["monthly"]


def verify_protocol(inputs, dataset_path):
    checks = inputs.checks
    p, receipt = inputs.json("extension", "protocol.json"), inputs.json("extension", "protocol_receipt.json")
    spec = inputs.json("main", "frozen_spec.json")
    checks.check(receipt.get("protocol_file") == "protocol.json"
                 and receipt.get("protocol_sha256") == sha(inputs.roots["extension"] / "protocol.json")
                 and utc(p["prepared_at"]) <= utc(receipt["sealed_at"]),
                 "Declared finite validation protocol sealed before execution")
    fixed = {"algorithm": "tqc", "profile": "balanced", "capital": 10000., "costs": COSTS,
             "seeds": list(SEEDS), "cash_floor": .05, "max_asset_weight": .95, "drawdown_penalty": .25,
             "new_rebalance_band": BAND, "reference_rebalance_band": REFERENCE_BAND,
             "exposure_multiplier": MULTIPLIER, "new_training": 0, "seed_selection": False,
             "selection_uses_test": False, "annual_evaluation_enabled": False,
             "validation_start": VALIDATION[0], "end_exclusive": VALIDATION[1],
             "selection_score": spec["validation_selection"]["score"], "tie_break": "prefer_reference"}
    for key, value in fixed.items():
        checks.check(p.get(key) == value, "One fixed post-annual variant: " + key)
    for key in FLAGS:
        checks.check(p.get(key) is True, "Disclosure of previously observed annual: " + key)
    checks.check(tuple(spec["train"]) == ("2020-01-01T00:00:00Z", VALIDATION[0])
                 and tuple(spec["validation"]) == VALIDATION,
                 "Original training cutoff and validation window remain unchanged")
    sources = p.get("source_sha256", {})
    inputs.pin_map("source", sources, SOURCE_FILES, "Frozen original and new source bytes")
    checks.check(set(sources) == set(SOURCE_FILES), "Exact ten source files, no execution-source replacement")
    checks.check(all(sources.get(name) == spec["code_sha256"].get(name) for name in FINANCIAL),
                 "Original four financial modules retain their frozen bytes")
    for name, expected in sources.items():
        copied = inputs.roots["extension"] / "sources" / name
        if copied.exists():
            checks.check(sha(inputs.capture("extension", "sources/" + name)) == expected,
                         "Archived exact source bytes: " + name)
    required_main = {"frozen_spec.json", "training_completion.json",
        *(f"training/tqc__balanced__seed{seed}/{name}" for seed in SEEDS for name in ("record.json", "model.zip"))}
    originals = p.get("original_inputs", {})
    inputs.pin_map("main", originals.get("main"), required_main, "Original five model/record identities")
    inputs.pin_map("reference", originals.get("reference"),
        reference_files() | {"independent_audit.json", "goal_candidate_evidence.json"},
        "Original audited comparator identities")
    reference_audit = inputs.json("reference", "independent_audit.json")
    checks.check(reference_audit.get("passed") is True and reference_audit.get("status") == "passed"
                 and reference_audit.get("test_complete") is True and reference_audit.get("failures") == []
                 and reference_audit.get("run_count", reference_audit.get("runs_checked")) == 36,
                 "Original reference account audit contains all36 original cases")
    goal = inputs.json("reference", "goal_candidate_evidence.json")
    checks.check(goal.get("audit_sha256") == sha(inputs.roots["reference"] / "independent_audit.json")
                 and goal.get("candidate") == {"family": "cap075", "algorithm": "tqc", "profile": "balanced"},
                 "Original capped candidate and its actual annual goal evidence remain preserved")
    checks.check(utc(reference_audit["audited_at"]) <= utc(p["prepared_at"]),
                 "Reference annual account had already been audited before this new design")
    runtime = inputs.json("main", "runtime.json")
    gpus = p.get("physical_gpus", [])
    checks.check(gpus == runtime.get("physical_gpus") and len(gpus) == 2
                 and len({g.get("uuid") for g in gpus}) == 2
                 and all("V100" in g.get("name", "") for g in gpus),
                 "Original two physical V100 inference identities")
    checks.check(p.get("runtime_packages") == spec.get("runtime_packages")
                 and p.get("gpu_cost_assignments") == {cost: gpu["uuid"] for cost, gpu in zip(COSTS, gpus)},
                 "Frozen dependency versions and one distinct original GPU per cost scenario")
    dataset_path = Path(dataset_path)
    checks.check(sha(dataset_path) == p.get("dataset_sha256") == spec["dataset_sha256"]
                 and sha(dataset_path.parent / "manifest.json") == p.get("dataset_manifest_sha256")
                 == spec["dataset_manifest_sha256"], "Original real dataset and manifest byte identities")
    records, models = {}, []
    completion = inputs.json("main", "training_completion.json")
    complete_records = {(r["algorithm"], r["profile"], r["seed"]): r for r in completion.get("records", [])}
    checks.check(completion.get("model_count") == 40 and len(complete_records) == 40
                 and completion.get("frozen_spec_sha256") == sha(inputs.roots["main"] / "frozen_spec.json"),
                 "Original40 training records remain frozen")
    for seed in SEEDS:
        folder = f"training/tqc__balanced__seed{seed}"
        record = inputs.json("main", folder + "/record.json")
        record_file, model_file = folder + "/record.json", folder + "/model.zip"
        checks.check((record.get("algorithm"), record.get("profile"), record.get("seed")) == ("tqc", "balanced", seed)
                     and record.get("model_relative_path") == model_file
                     and record.get("model_sha256") == sha(inputs.capture("main", model_file))
                     and record.get("frozen_spec_sha256") == sha(inputs.roots["main"] / "frozen_spec.json")
                     and complete_records.get(("tqc", "balanced", seed)) == record
                     and record.get("actual_timesteps") == 32768,
                     "Unchanged trained record/checkpoint/transition budget seed" + str(seed))
        records[seed] = record
        models.append(dict(profile="balanced", seed=seed, record_file=record_file, model_file=model_file,
            record_sha256=sha(inputs.roots["main"] / record_file), model_sha256=sha(inputs.roots["main"] / model_file),
            parameter_sha256=record["parameter_sha256"]))
    checks.check(p.get("models") == models, "Exactly all five original balanced model/parameter SHA identities")
    if checks.failed:
        raise ValueError("Declaration/source/model/reference checks failed before reconstruction")
    return p, spec, records, reference_audit


def verify_training_cutoff(records, spec, dataset, checks):
    calendar = independent_training_calendar(dataset, *spec["train"], 32)
    for seed, record in records.items():
        steps = 0
        history = record.get("training_episode_history", [])
        checks.check(bool(history), f"seed{seed}: original actual training fragment records")
        for item in history:
            first, end, completed = (utc(item[key]) for key in ("start", "end_exclusive", "completed_until"))
            count = int(item["completed_steps"])
            checks.check(first in calendar["eligible"] and end - first == pd.Timedelta(days=32),
                         f"seed{seed}: only complete causal32-day training fragments")
            checks.check(utc(spec["train"][0]) <= first <= completed <= end <= utc(VALIDATION[0])
                         and completed - first == pd.Timedelta(days=count),
                         f"seed{seed}: every transition strictly before validation")
            checks.check(item.get("terminated") is False and item.get("truncated") == (count == 32),
                         f"seed{seed}: random fragment is time truncated")
            steps += count
        checks.check(steps == 32768 == record.get("actual_timesteps"), f"seed{seed}: original fixed training samples")
    return {key: value for key, value in calendar.items() if key not in ("eligible", "market", "features")}


def audit_group(inputs, dataset, records, protocol, *, reference=False, features=None):
    root, folder = ("reference" if reference else "extension"), "validation/balanced"
    checks = inputs.checks
    for name in GROUP_FILES:
        inputs.capture(root, folder + "/" + name)
    summary = inputs.csv(root, folder + "/summary.csv")
    daily, monthly = inputs.csv(root, folder + "/daily_equity.csv"), inputs.csv(root, folder + "/monthly_returns.csv")
    validate_group_rows(summary, reference=reference)
    completion = inputs.json(root, folder + "/completion.json")
    checks.check(completion.get("case_count") == 12, root + ": complete original twelve-case group")
    if not reference:
        checks.check(completion.get("protocol_sha256") == sha(inputs.roots["extension"] / "protocol.json"),
                     "New group carries the one sealed protocol SHA")
    repeats = completion.get("repeats", [])
    proofs = {row.get("run_id"): row for row in repeats}
    checks.check(len(proofs) == len(repeats) == 12 and set(proofs) == set(summary.run_id),
                 root + ": exactly twelve independent repeat receipts")
    expected = {}
    for row in summary.to_dict("records"):
        key = "ensemble" if row["primary_policy"] else str(int(row["seed"]))
        relative = f"{folder}/{row['scenario']}/{key}"
        for name in CASE_FILES:
            inputs.capture(root, relative + "/" + name)
        run, history, months = audit_case(safe_file(inputs.roots[root], relative), dataset, row,
            records, protocol, checks, reference=reference, features=features)
        run["relative_directory"], run["source_root"] = relative, root
        proof = proofs.get(row["run_id"], {})
        checks.check(proof.get("passed") is True and proof.get("fingerprints") == run["fingerprints"]
                     and proof.get("member_targets_sha256") == run["member_targets_sha256"],
                     row["run_id"] + ": group receipt binds actual trace and financial bytes")
        expected[row["run_id"]] = (run, history, months)
    compare_group_tables(summary, daily, monthly, expected, checks, root)
    return summary, daily, monthly, expected


def compare_root_tables(inputs, group):
    for index, name, keys in ((0, "summary.csv", ["run_id"]),
                              (1, "daily_equity.csv", ["run_id", "timestamp"]),
                              (2, "monthly_returns.csv", ["run_id", "month"])):
        actual, expected = inputs.csv("extension", name), group[index]
        inputs.checks.check(set(actual.columns) == set(expected.columns) and len(actual) == len(expected),
                            "No hidden/missing root aggregate rows: " + name)
        if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
            continue
        actual, expected = actual.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
        inputs.checks.check(not actual[keys].duplicated().any(), "Unique root table identity: " + name)
        for column in expected:
            missing = actual[column].isna() & expected[column].isna()
            if pd.api.types.is_numeric_dtype(expected[column]):
                inputs.checks.vector_close(actual.loc[~missing, column], expected.loc[~missing, column],
                                            "Root/source table " + name + ":" + column, atol=1e-8)
            else:
                inputs.checks.check((missing | actual[column].eq(expected[column])).all(),
                                    "Root/source table " + name + ":" + column)


def verify_terminal(output, protocol_sha256, checks):
    output = Path(output)
    checks.check(not any((output / name).exists() for name in
        ("test", "test_started.json", "test_completion.json", "test_process.json")),
        "New study has no annual/test dispatch or results")
    process = read_json(output / "runner_process.json")
    checks.check(process.get("status") == "completed" and process.get("exit_code") == 0,
                 "Real validation supervisor reached terminal completed0")
    completed = read_json(output / "validation_completion.json")
    checks.check(completed.get("case_count") == 12 and completed.get("repeat_count") == 12
                 and completed.get("protocol_sha256") == protocol_sha256,
                 "Exactly twelve validation cases plus twelve exact repeats")
    return process, completed


def verify_process_chain(inputs, protocol, process, completion):
    checks, root = inputs.checks, inputs.roots["extension"]
    receipt = inputs.json("extension", "protocol_receipt.json")
    checks.check(process.get("phase") == "validate" and process.get("protocol_sha256") == sha(root / "protocol.json")
                 and process.get("source_sha256") == protocol["source_sha256"]["scripts/btceth_rl_execution_band.py"],
                 "Actual supervisor loaded this frozen validation-only source and protocol")
    command = process.get("command", [])
    checks.check(isinstance(command, list) and len(command) >= 5
                 and any(str(item).replace("\\", "/").endswith("/btceth_rl_execution_band.py") for item in command)
                 and "--phase" in command and command[command.index("--phase") + 1] == "_validate",
                 "Terminal process evidence records this private validation child command")
    checks.check(isinstance(process.get("supervisor_pid"), int) and process["supervisor_pid"] > 0
                 and isinstance(process.get("child_pid"), int) and process["child_pid"] > 0
                 and process["supervisor_pid"] != process["child_pid"],
                 "Real supervisor and child identities recorded separately")
    checks.check(process.get("log_relative_path") == "validation.log"
                 and process.get("log_sha256") == sha(inputs.capture("extension", "validation.log")),
                 "Terminal log bytes bound by the supervisor")
    checks.check(utc(receipt["sealed_at"]) <= utc(process["started_at"]) <= utc(completion["at"])
                 <= utc(process["completed_at"]), "Sealed protocol precedes execution and true completion")


def verify_worker_groups(inputs, protocol, group, completion):
    checks = inputs.checks
    workers = completion.get("workers", [])
    checks.check(len(workers) == 2 and {row.get("scenario") for row in workers} == set(COSTS)
                 and len({row.get("worker_pid") for row in workers}) == 2,
                 "Exactly two independent cost workers with distinct actual handles")
    for cost in COSTS:
        folder = "validation/balanced/" + cost
        frames = [inputs.csv("extension", folder + "/" + name) for name in GROUP_FILES[:3]]
        for actual, all_rows, keys in zip(frames, group[:3], (["run_id"], ["run_id", "timestamp"], ["run_id", "month"])):
            expected = all_rows.loc[all_rows.run_id.str.startswith("validation__" + cost + "__")]
            checks.check(set(actual.columns) == set(expected.columns) and len(actual) == len(expected),
                         cost + ": worker tables form exact portions of parent aggregate")
            if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
                continue
            a, e = actual.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
            for column in e:
                missing = a[column].isna() & e[column].isna()
                if pd.api.types.is_numeric_dtype(e[column]):
                    checks.vector_close(a.loc[~missing, column], e.loc[~missing, column],
                                        cost + ": worker table:" + column, atol=1e-8)
                else:
                    checks.check((missing | a[column].eq(e[column])).all(), cost + ": worker table:" + column)
        record = inputs.json("extension", folder + "/completion.json")
        worker = next((item for item in workers if item.get("scenario") == cost), {})
        checks.check(record.get("case_count") == worker.get("case_count") == 6
                     and record.get("worker_pid") == worker.get("worker_pid")
                     and record.get("protocol_sha256") == sha(inputs.roots["extension"] / "protocol.json"),
                     cost + ": six cases completed under the frozen worker")
        device = record.get("device", {})
        checks.check(device.get("visible_gpu_uuid") == protocol.get("gpu_cost_assignments", {}).get(cost),
                     cost + ": declared UUID-isolated physical inference device")
        proofs = record.get("repeats", [])
        checks.check(len(proofs) == 6 and {row.get("run_id") for row in proofs}
                     == set(frames[0].run_id), cost + ": all six worker repeat receipts")
        for key in (*map(str, SEEDS), "ensemble"):
            proof = inputs.json("extension", f"{folder}/{key}/replay.json")
            checks.check(proof in proofs and proof.get("worker_pid") == worker.get("worker_pid")
                         and proof.get("device") == device, cost + ": per-case actual worker and full receipt")


def verify_selection_if_present(inputs, new, reference):
    """Optional post-accounting selection, explicitly never fresh test evidence."""
    if not (inputs.roots["extension"] / "selection.json").exists():
        return False
    selected = inputs.json("extension", "selection.json")
    receipt = inputs.json("extension", "selection_receipt.json")
    ranked = compare_validation_candidates(new, reference)
    checks = inputs.checks
    checks.check(receipt.get("selection_file") == "selection.json"
                 and receipt.get("selection_sha256") == sha(inputs.roots["extension"] / "selection.json")
                 and receipt.get("protocol_sha256") == selected.get("protocol_sha256")
                 == sha(inputs.roots["extension"] / "protocol.json"), "Whole-file post-accounting selection receipt")
    checks.check(selected.get("selection_uses_test") is False
                 and selected.get("seed_selection") is False
                 and all(selected.get(flag) is True for flag in FLAGS),
                 "Finite validation comparison retains previously seen annual disclosure")
    actual = selected.get("candidates", [])
    checks.check(len(actual) == 2 and {row.get("family") for row in actual} == {"cap075", FAMILY},
                 "Exactly old and new execution candidates compete")
    for item in actual:
        expected = next((row for row in ranked if row["family"] == item.get("family")), None)
        if expected is None:
            continue
        checks.close(item.get("score"), expected["score"], "Independent unchanged two-cost score", atol=1e-12)
        checks.check(item.get("algorithm") == "tqc" and item.get("profile") == "balanced"
                     and item.get("rebalance_band") == expected["rebalance_band"]
                     and item.get("name") == ("reference_band025" if expected["family"] == "cap075" else "band05"),
                     "Selection keeps exactly the original models and declared execution band")
        for cost in COSTS:
            checks.check(item.get(cost, {}).get("run_id") == expected[cost]["run_id"],
                         "Candidate score comes from actual full ensemble validation")
            for key in ("net_return", "max_drawdown", "fees", "final_equity"):
                checks.close(item.get(cost, {}).get(key), expected[cost][key],
                             "Independent candidate source metric:" + key, atol=1e-10)
    # Frozen score bytes are kept after independently validating numerical
    # agreement; this avoids changing a rank for a libm platform ULP.
    order = sorted(actual, key=lambda row: (-row["score"], 0 if row["family"] == "cap075" else 1))
    checks.check(actual == order and selected.get("candidate") == (order[0] if order else None),
                 "Same validated score ranking and exact ties prefer original execution")
    checks.check(utc(selected["selected_at"]) <= utc(receipt["sealed_at"]),
                 "Selection precedes its immutable whole-file seal")
    previous_audit = inputs.json("extension", "independent_audit.validation.json")
    checks.check(selected.get("audit_sha256") == sha(inputs.roots["extension"] / "independent_audit.validation.json")
                 and previous_audit.get("passed") is True and previous_audit.get("status") == "passed"
                 and previous_audit.get("run_count") == previous_audit.get("comparator_run_count") == 12
                 and previous_audit.get("test_complete") is False
                 and utc(previous_audit["audited_at"]) <= utc(selected["selected_at"]),
                 "Actual independently audited validation evidence preceded selection")
    hashes = selected.get("validation_input_sha256", {})
    checks.check(hashes == previous_audit.get("input_sha256"),
                 "Selection preserves every original validation audit input SHA")
    for root, values in hashes.items():
        for name, expected in values.items():
            checks.check(sha(inputs.capture(root, name)) == expected,
                         "Selected immutable audited input: " + root + ":" + name)
    comparison = inputs.json("extension", "comparison.json")
    checks.check(comparison.get("candidates") == actual
                 and comparison.get("chosen") == selected.get("candidate", {}).get("name")
                 and comparison.get("tie_break") == "prefer_reference",
                 "Reviewable comparison table matches the sealed two-candidate choice")
    return True


def audit(output, main, reference, dataset_path, *, source_root=ROOT):
    """Read only. Never starts, loads or infers a model, or writes source/results."""
    checks, started = Checks(), now()
    inputs = Inputs(output, main, reference, source_root, checks)
    runs, comparator_runs, training_calendar, selection_complete = {}, {}, None, False
    try:
        protocol, spec, records, original_audit = verify_protocol(inputs, dataset_path)
        for name in ("runner_process.json", "validation_completion.json", "validation.log"):
            inputs.capture("extension", name)
        process, completion = verify_terminal(output, sha(Path(output) / "protocol.json"), checks)
        verify_process_chain(inputs, protocol, process, completion)
        if checks.failed:
            raise ValueError("Terminal/protocol evidence failed before financial reconstruction")
        dataset = load_market(dataset_path, VALIDATION[1])
        features = independent_market_features(dataset)
        validation_hours = pd.date_range(utc(VALIDATION[0]), utc(VALIDATION[1]), freq="h", inclusive="left")
        for field in ("open", "high", "low", "close", "volume"):
            data = dataset["hourly_" + field].reindex(validation_hours)
            values = data.to_numpy(dtype=float)
            checks.check(len(data) == 7080 and np.isfinite(values).all()
                         and (values >= 0 if field == "volume" else values > 0).all(),
                         "Every real validation OHLCV hour complete: " + field)
        training_calendar = verify_training_cutoff(records, spec, dataset, checks)
        new = audit_group(inputs, dataset, records, protocol, features=features)
        verify_worker_groups(inputs, protocol, new, completion)
        old = audit_group(inputs, dataset, records, protocol, reference=True, features=features)
        runs = {key: value[0] for key, value in new[3].items()}
        comparator_runs = {key: value[0] for key, value in old[3].items()}
        for run_id, run in comparator_runs.items():
            evidence = original_audit.get("runs", {}).get(run_id, {})
            checks.check(evidence.get("reconstructed") is True and evidence.get("hours") == 7080
                         and evidence.get("fingerprints") == run["fingerprints"],
                         "Original audited twelve-case comparator identity: " + run_id)
        compare_root_tables(inputs, new)
        selection_complete = verify_selection_if_present(inputs, new[0], old[0])
        checks.check(len(runs) == len(comparator_runs) == 12, "All twelve new and twelve reference validation accounts audited")
        inputs.unchanged()
    except Exception as exc:
        checks.check(False, type(exc).__name__ + ": " + str(exc))
    passed = checks.failed == 0 and len(runs) == len(comparator_runs) == 12
    return {"schema_version": 1, "started_at": started, "audited_at": now(),
        "passed": passed, "status": "passed" if passed else "failed",
        "validation_complete": passed, "test_complete": False, "annual_evaluation_enabled": False,
        "selection_complete": selection_complete and passed,
        "run_count": len(runs), "repeat_count": len(runs) if passed else 0,
        "comparator_run_count": len(comparator_runs), "checks_total": checks.count,
        "failure_count": checks.failed, "failures": checks.failures, "maximum_errors": checks.maximum_errors,
        "input_sha256": inputs.hashes, "source_sha256": inputs.hashes["source"],
        "dataset_sha256": sha(dataset_path), "training_calendar": training_calendar,
        "runs": runs, "comparator_runs": comparator_runs,
        **{flag: True for flag in FLAGS},
        "accounting_passage_is_not_goal_success": True,
        "scope": "Only validation295 days; old annual was seen before design; no fresh annual evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_file = args.report or args.output_dir / "independent_audit.validation.json"
    if report_file.exists():
        raise FileExistsError("Preserve existing audit evidence; choose a new --report")
    result = audit(args.output_dir, args.main_output, args.reference_output, args.dataset)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in
        ("passed", "status", "run_count", "repeat_count", "comparator_run_count", "checks_total", "failure_count")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
