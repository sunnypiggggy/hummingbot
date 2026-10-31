"""Independent accounting and pre-test selection audit for the TQC .75 overlay.

No environment, portfolio ledger, policy inference or training is used here.
The original 40 models, 172-run study and goal contract remain separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_btceth_rl import (Checks, _check_summary, read_csv, read_equity,
                                    reconstruct_fills, verify_saved_fingerprints)

ALGORITHMS = ("recurrent_ppo", "tqc", "crossq")
PROFILES = ("return", "balanced")
SEEDS = tuple(range(5))
COSTS = {"base": {"fee": .001, "slippage": .0002},
         "stress": {"fee": .002, "slippage": .0005}}
SPLITS = {"validation": ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"),
          "test": ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")}
MULTIPLIER = .75
MAX_GROSS = .95 * MULTIPLIER
SELL_DUST_QUANTITY = 1e-14
SYMBOLS = ("BTCUSDT", "ETHUSDT")
FINANCIAL = ("scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py",
             "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Audit evidence timestamps require an explicit UTC offset")
    return stamp.tz_convert("UTC")


def sha(path):
    h = hashlib.sha256()
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Do not trust linked evidence files")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_file(root, name):
    text = str(name).replace("\\", "/")
    parts = PurePosixPath(text)
    if not text or parts.is_absolute() or ":" in text or ".." in parts.parts:
        raise ValueError("Audit pins must stay relative to their declared root")
    root = Path(root).resolve()
    result = root
    for part in parts.parts:
        result = result / part
        if result.is_symlink() or (hasattr(result, "is_junction") and result.is_junction()):
            raise ValueError("Audit paths cannot cross links or junctions")
    result.resolve().relative_to(root)
    return result


def candidate_score(base, stress):
    def one(row):
        ret, dd = float(row["net_return"]), abs(float(row["max_drawdown"]))
        if not math.isfinite(ret) or ret <= -1 or not math.isfinite(dd) or dd >= 1:
            raise ValueError("Invalid candidate return or drawdown")
        return math.log1p(ret) - .5 * dd - 2 * max(dd - .20, 0.)
    return (one(base) + one(stress)) / 2


def validate_group_rows(frame, *, split, algorithm, profile, multiplier):
    columns = {"run_id", "algorithm", "profile", "scenario", "split", "seed", "primary_policy",
               "initial_capital", "start", "end_exclusive", "net_return", "max_drawdown", "final_equity", "fees"}
    if not columns.issubset(frame) or len(frame) != 12 or frame.run_id.duplicated().any():
        raise ValueError("A complete group needs exactly five seeds and one true ensemble at both costs")
    start, end = map(utc, SPLITS[split])
    if (not frame.algorithm.eq(algorithm).all() or not frame.profile.eq(profile).all()
            or not frame.split.eq(split).all() or not frame.initial_capital.eq(10000.).all()
            or not pd.to_datetime(frame.start, utc=True).eq(start).all()
            or not pd.to_datetime(frame.end_exclusive, utc=True).eq(end).all()):
        raise ValueError("Group financial identity or chronological window changed")
    if multiplier != 1.:
        if (algorithm != "tqc" or not {"family", "strategy", "exposure_multiplier"}.issubset(frame)
                or not frame.family.eq("cap075").all() or not frame.strategy.eq("tqc_cap075").all()
                or not frame.exposure_multiplier.eq(multiplier).all()):
            raise ValueError("Overlay multiplier differs from its single predeclared value")
    ids = {f"{split}__{scenario}__{'cap075__' if multiplier != 1. else ''}{algorithm}__{profile}__{key}"
           for scenario in COSTS for key in (*map(str, SEEDS), "ensemble")}
    if set(frame.run_id) != ids:
        raise ValueError("Financial run IDs differ from the immutable case identities")
    for scenario in COSTS:
        rows = frame.loc[frame.scenario.eq(scenario)]
        if len(rows) != 6 or not rows.primary_policy.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError("Invalid primary markers or cost coverage")
        primary, seeds = rows.loc[rows.primary_policy], rows.loc[~rows.primary_policy]
        if (len(primary) != 1 or not primary.seed.isna().all() or len(seeds) != 5
                or seeds.seed.isna().any() or seeds.seed.duplicated().any()
                or set(seeds.seed.astype(int)) != set(SEEDS)
                or not np.equal(seeds.seed, seeds.seed.astype(int)).all()):
            raise ValueError("All five original seeds must be retained; no seed selection")
    values = frame[["net_return", "max_drawdown", "final_equity", "fees"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (frame.net_return <= -1).any()
            or (frame.max_drawdown > 0).any() or (frame.max_drawdown <= -1).any()
            or (frame.final_equity <= 0).any() or (frame.fees < 0).any()
            or (frame.final_equity / 10000 - 1 - frame.net_return).abs().max() > 1e-10):
        raise ValueError("Invalid or inconsistent financial summary")


def candidate_row(frame, *, family, algorithm, profile):
    multiplier = 1. if family == "raw" else MULTIPLIER
    validate_group_rows(frame, split="validation", algorithm=algorithm, profile=profile, multiplier=multiplier)
    values = {}
    for scenario in COSTS:
        row = frame.loc[frame.scenario.eq(scenario) & frame.primary_policy].iloc[0]
        values[scenario] = {"run_id": str(row.run_id), **{name: float(row[name]) for name in (
            "net_return", "max_drawdown", "final_equity", "fees")}}
    return {"family": family, "algorithm": algorithm, "profile": profile,
            "score": candidate_score(values["base"], values["stress"]), **values}


def candidate_pool(raw_frames, capped_frames):
    """Only six raw new-model profiles plus the two capped TQC profiles compete."""
    expected = {(a, p) for a in ALGORITHMS for p in PROFILES}
    if not expected.issubset(raw_frames) or set(capped_frames) != set(PROFILES):
        raise ValueError("The fixed candidate population is six raw new profiles and two capped TQC profiles")
    values = [candidate_row(raw_frames[(a, p)], family="raw", algorithm=a, profile=p)
              for a in ALGORITHMS for p in PROFILES]
    values += [candidate_row(capped_frames[p], family="cap075", algorithm="tqc", profile=p) for p in PROFILES]
    return sorted(values, key=lambda row: (-row["score"], 0 if row["family"] == "raw" else 1,
                                          row["algorithm"], row["profile"]))


def mapped_target(action0, action1, multiplier=MULTIPLIER):
    action = np.asarray([action0, action1], dtype=float)
    if not np.isfinite(action).all() or multiplier != MULTIPLIER:
        raise ValueError("Finite original actions and the one fixed .75 multiplier are required")
    a0, a1 = np.clip(action, -1., 1.)
    gross = .95 * max(float(a0), 0.)
    eth_share = (float(a1) + 1.) / 2.
    raw = np.array([gross * (1 - eth_share), gross * eth_share, 1 - gross])
    scaled = raw.copy()
    scaled[:2] *= multiplier
    scaled[2] = 1 - scaled[:2].sum()
    return raw, scaled


def objective_evidence(summary, selected, equal_hold):
    """Accounting passage is deliberately absent from these investment gates."""
    if selected["algorithm"] not in ALGORITHMS or selected.get("family") not in ("raw", "cap075"):
        raise ValueError("SAC and cash cannot satisfy the new-model objective")
    if not summary.split.eq("test").all():
        raise ValueError("Validation is not a final performance result")
    multiplier = 1. if selected["family"] == "raw" else MULTIPLIER
    validate_group_rows(summary, split="test", algorithm=selected["algorithm"],
                        profile=selected["profile"], multiplier=multiplier)
    base = summary.loc[summary.scenario.eq("base") & summary.primary_policy].iloc[0]
    stress = summary.loc[summary.scenario.eq("stress") & summary.primary_policy].iloc[0]
    seeds = summary.loc[summary.scenario.eq("base") & ~summary.primary_policy].sort_values("seed")
    positive = int(seeds.net_return.gt(0).sum())
    gates = {"base_return_at_least_5pct": bool(base.net_return >= .05),
             "base_max_drawdown_at_most_20pct": bool(abs(base.max_drawdown) <= .20),
             "stress_return_nonnegative": bool(stress.net_return >= 0),
             "at_least_three_positive_training_seeds": positive >= 3,
             "beats_equal_hold_return_and_drawdown": bool(base.net_return > equal_hold["net_return"]
                                                           and base.max_drawdown > equal_hold["max_drawdown"])}
    metrics = {"base_net_return": float(base.net_return), "base_max_drawdown": float(base.max_drawdown),
               "stress_net_return": float(stress.net_return), "stress_max_drawdown": float(stress.max_drawdown),
               "base_fees_usdt": float(base.fees), "stress_fees_usdt": float(stress.fees),
               "base_run_id": str(base.run_id), "stress_run_id": str(stress.run_id),
               "positive_base_training_seeds": positive,
               "base_seed_returns": [{"seed": int(r.seed), "net_return": float(r.net_return)} for r in seeds.itertuples(index=False)],
               "equal_hold_base_net_return": float(equal_hold["net_return"]),
               "equal_hold_base_max_drawdown": float(equal_hold["max_drawdown"])}
    return {"evaluated": True, "selected": {key: selected[key] for key in ("family", "algorithm", "profile")},
            "base_net_return": float(base.net_return), "base_max_drawdown": float(base.max_drawdown),
            "stress_net_return": float(stress.net_return), "stress_max_drawdown": float(stress.max_drawdown),
            "positive_base_training_seeds": positive, "gates": gates, "all_predeclared_primary_targets_met": all(gates.values()),
            "metrics": metrics,
            "goal_performance_achieved": bool(all(gates.values())), "accounting_passage_is_not_goal_success": True}


def reconstruct_inventory(fills, hours, *, checks, label):
    """Replay inventory existence, including the frozen SELL-only dust rule.

    This independent pass never trusts quantity_after to set its state and
    never clips BUY inventory. The original cash/NAV/fee reconstruction stays
    unchanged; zero/nonzero state additionally matters to protective exits.
    """
    hours = pd.DatetimeIndex(hours)
    positions, quantities, presence, dust_events = {}, [], [], []
    tiny_buys = 0
    trades = fills.copy()
    required = {"timestamp", "asset_key", "side", "qty", "quantity_after"}
    if len(trades) and not checks.check(required.issubset(trades), label + ": inventory replay fill schema"):
        return None
    if len(trades):
        trades["timestamp"] = pd.to_datetime(trades.timestamp, utc=True)
        checks.check(trades.timestamp.is_monotonic_increasing and trades.timestamp.isin(hours).all(),
                     label + ": inventory replay uses original ordered hourly fills")
    grouped = {day: frame for day, frame in trades.groupby("timestamp", sort=False)} if len(trades) else {}
    empty = pd.DataFrame()
    for hour in hours:
        for row in grouped.get(hour, empty).itertuples(index=False):
            symbol, side, amount = str(row.asset_key), str(row.side), float(row.qty)
            if not checks.check(symbol in SYMBOLS and side in ("BUY", "SELL")
                                and math.isfinite(amount) and amount > 0,
                                label + ": inventory replay valid owned identity/positive fill"):
                continue
            owned = positions.get(symbol, 0.)
            if side == "BUY":
                positions[symbol] = owned + amount
                tiny_buys += int(positions[symbol] <= SELL_DUST_QUANTITY)
            else:
                if not checks.check(owned > 0 and amount <= owned + 1e-12,
                                    label + ": inventory replay sell bounded by present owned quantity"):
                    continue
                remainder = owned - min(amount, owned)
                if remainder <= SELL_DUST_QUANTITY:
                    del positions[symbol]
                    dust_events.append({"timestamp": str(hour), "asset_key": symbol,
                        "owned_before": owned, "sold_quantity": amount, "remainder_before_delete": remainder})
                else:
                    positions[symbol] = remainder
            after = positions.get(symbol, 0.)
            # A tiny but present BUY is economically different from deleted
            # SELL dust even when both are close under a numeric tolerance.
            recorded_after = float(row.quantity_after)
            checks.check(math.isfinite(recorded_after) and recorded_after >= 0
                         and (recorded_after > 0) == (symbol in positions),
                         label + ": per-fill inventory zero/present identity")
            checks.close(recorded_after, after, label + ": precise per-fill inventory quantity", atol=1e-15, rtol=1e-12)
        quantities.append([positions.get(symbol, 0.) for symbol in SYMBOLS])
        presence.append([symbol in positions for symbol in SYMBOLS])
    quantity = pd.DataFrame(quantities, index=hours, columns=SYMBOLS, dtype=float)
    present = pd.DataFrame(presence, index=hours, columns=SYMBOLS, dtype=bool)
    return {"quantity": quantity, "presence": present, "sell_deletions": dust_events,
            "retained_tiny_buy_fills": tiny_buys}


def attach_inventory_replay(rebuilt, fills, closing, *, checks, label, metadata=None):
    exact = reconstruct_inventory(fills, rebuilt["equity"].index, checks=checks, label=label)
    if exact is None:
        raise ValueError("Cannot establish independent inventory state: " + label)
    quantity = exact["quantity"]
    prices = closing.reindex(quantity.index).loc[:, list(SYMBOLS)]
    checks.vector_close(quantity, rebuilt["quantity"], label + ": precise inventory versus original accounting quantities", atol=1e-10)
    values = quantity * prices
    checks.vector_close(values.sum(axis=1), rebuilt["equity"].base_value,
                        label + ": precise inventory versus original accounting coin value", atol=1e-6)
    if metadata is not None:
        final = {symbol for symbol in SYMBOLS if bool(exact["presence"].iloc[-1][symbol])}
        checks.check(set(metadata.get("final_positions", {})) == final,
                     label + ": final inventory zero/present identities")
    rebuilt["inventory_quantity"], rebuilt["inventory_presence"] = quantity, exact["presence"]
    rebuilt["coin_values"] = values
    fingerprint = lambda frame: hashlib.sha256(frame.to_csv(index=True, float_format="%.17g", lineterminator="\n").encode()).hexdigest()
    return {"contract": "BUY retains every positive quantity; only SELL deletes remainder<=1e-14; no blanket quantity clipping",
            "hours": len(quantity), "quantity_sha256": fingerprint(quantity), "presence_sha256": fingerprint(exact["presence"]),
            "sell_deletions": exact["sell_deletions"], "retained_tiny_buy_fills": exact["retained_tiny_buy_fills"],
            "maximum_quantity_difference_from_original_accounting": float(np.abs(quantity.to_numpy()-rebuilt["quantity"].to_numpy()).max()),
            "maximum_coin_value_difference_from_original_accounting": float(np.abs(values.sum(axis=1).to_numpy()-rebuilt["equity"].base_value.to_numpy()).max())}


def rebalance_decision(target, prior):
    """Unchanged frozen band/exit formula, evaluated on exact inventory marks."""
    forced = target[:2].sum() <= 1e-12 and prior[:2].sum() > 0
    return bool(forced or np.max(np.abs(target - prior[:3])) >= .025)


def audit_member_targets(trace, weights, rebuilt, fills, *, start, end, seeds, checks, label):
    required = {"day", "seed", "raw_action0", "raw_action1", "raw_btc", "raw_eth", "raw_cash",
                "capped_btc", "capped_eth", "capped_cash", "target_btc", "target_eth", "target_cash",
                "observation_sha256", "observation_btc_weight", "observation_eth_weight",
                "observation_cash_weight", "observation_drawdown"}
    if not checks.check(required.issubset(trace), label + ": complete member target schema"):
        return
    trace = trace.copy()
    trace["day"] = pd.to_datetime(trace.day, utc=True)
    weights = weights.copy()
    weights["date"] = pd.to_datetime(weights.date, utc=True)
    days = pd.date_range(utc(start), utc(end), freq="D", inclusive="left")
    checks.check(len(trace) == len(days) * len(seeds) and not trace[["day", "seed"]].duplicated().any(),
                 label + ": exact member/day Cartesian product")
    checks.check(set(trace.day) == set(days) and set(trace.seed) == set(seeds), label + ": complete fixed dates and seeds")
    checks.check(len(weights) == len(days) * 2 and set(weights.date) == set(days), label + ": complete daily two-coin weights")
    checks.check(not weights[["date", "asset_key"]].duplicated().any(), label + ": unique coin/day weights")
    fill_days = set(pd.to_datetime(fills.timestamp, utc=True).dt.normalize()) if len(fills) else set()
    eq = rebuilt["equity"]
    if not checks.check("inventory_quantity" in rebuilt and "inventory_presence" in rebuilt,
                        label + ": precise SELL-dust inventory replay required before target decisions"):
        return
    peaks = eq.equity.cummax().clip(lower=10000.)
    for day in days:
        member = trace.loc[trace.day.eq(day)].sort_values("seed")
        if not checks.check(len(member) == len(seeds) and member.seed.tolist() == sorted(seeds), label + ": each member exactly once"):
            continue
        hashes = member.observation_sha256.astype(str)
        checks.check(hashes.nunique() == 1 and all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in hashes),
                     label + ": members receive identical common-account observation")
        if day == utc(start):
            prior = np.array([0., 0., 1., 0.])
        else:
            previous = day - pd.Timedelta(hours=1)
            value = eq.loc[previous]
            # Actual coin values are obtained from independently replayed
            # quantities and original closes, saved by the caller below.
            prior = np.r_[rebuilt["coin_values"].loc[previous].to_numpy() / value.equity,
                          value.cash / value.equity, 1 - value.equity / peaks.loc[previous]]
        obscols = ["observation_btc_weight", "observation_eth_weight", "observation_cash_weight", "observation_drawdown"]
        expected_obs = np.tile(prior.astype(np.float32).astype(float), (len(member), 1))
        checks.vector_close(member[obscols].to_numpy(dtype=float), expected_obs,
                            label + ": observed portfolio matches this ledger prior close", atol=1e-7, rtol=0.)
        capped = []
        for row in member.itertuples(index=False):
            raw, target = mapped_target(row.raw_action0, row.raw_action1)
            checks.vector_close([row.raw_btc, row.raw_eth, row.raw_cash], raw,
                                label + ": original action mapping", atol=1e-12)
            checks.vector_close([row.capped_btc, row.capped_eth, row.capped_cash], target,
                                label + ": .75 acts on coin targets, cash residual", atol=1e-12)
            capped.append(target)
        target = np.asarray(capped).mean(axis=0)
        checks.check(np.isfinite(target).all() and (target >= -1e-12).all()
                     and target[:2].sum() <= MAX_GROSS + 1e-12 and target[2] >= 1 - MAX_GROSS - 1e-12,
                     label + ": strict .7125 requested gross budget")
        checks.vector_close(member[["target_btc", "target_eth", "target_cash"]].to_numpy(dtype=float),
                            np.tile(target, (len(member), 1)), label + ": true mean of all member targets", atol=1e-12)
        daily = weights.loc[weights.date.eq(day)]
        if not checks.check(set(daily.asset_key) == set(SYMBOLS) and len(daily) == 2, label + ": two original spot identities"):
            continue
        daily = daily.set_index("asset_key").loc[list(SYMBOLS)]
        checks.check(daily.slot.tolist() == [0, 1] and daily.symbol.tolist() == ["BTC", "ETH"], label + ": slots cannot swap coin identities")
        checks.vector_close(daily.target_weight.to_numpy(dtype=float), target[:2], label + ": executed mean target in weights", atol=1e-12)
        close = day + pd.Timedelta(hours=23)
        actual = rebuilt["coin_values"].loc[close].to_numpy() / eq.loc[close, "equity"]
        checks.vector_close(daily.actual_weight.to_numpy(dtype=float), actual, label + ": daily actual inventory weights", atol=1e-10)
        # The target is capped. Subsequent price drift is intentionally allowed.
        expected_rebalance = rebalance_decision(target, prior)
        checks.check(daily.rebalanced.map(lambda x: isinstance(x, (bool, np.bool_))).all()
                     and daily.rebalanced.eq(expected_rebalance).all(), label + ": original band and protective cash exit")
        if not expected_rebalance:
            checks.check(day not in fill_days, label + ": no fills when the original band skips execution")
        elif day == utc(start) and target[:2].sum() >= .025:
            first = fills.loc[pd.to_datetime(fills.timestamp, utc=True).dt.normalize().eq(day)] if len(fills) else fills
            checks.check(len(first) > 0 and first.side.eq("BUY").any(), label + ": material initial target must actually invest funded cash")


def audit_case(directory, dataset, row, *, dates, records, profile, key, checks, capped, protocol=None):
    directory = Path(directory)
    label, scenario = row["run_id"], row["scenario"]
    equity, fills = read_equity(directory / "equity_1h.csv.gz"), read_csv(directory / "fills.csv")
    metadata = read_json(directory / "metadata.json")
    fingerprints = verify_saved_fingerprints(directory, checks, label)
    rebuilt = reconstruct_fills(dataset, fills, equity, metadata, start=dates[0], end=dates[1],
                               **COSTS[scenario], checks=checks, label=label)
    if rebuilt is None:
        raise ValueError("Incomplete hourly account: " + label)
    inventory_evidence = attach_inventory_replay(rebuilt, fills, dataset["hourly_close"], checks=checks,
                                                label=label, metadata=metadata)
    _check_summary(row, rebuilt["metrics"], checks, label)
    active = list(SEEDS) if key == "ensemble" else [int(key)]
    expected_parameters = [records[(row["algorithm"], profile, seed)]["parameter_sha256"] for seed in active]
    checks.check(metadata.get("algorithm") == row["algorithm"] and metadata.get("profile") == profile
                 and metadata.get("model_count") == len(active), label + ": original model identity/count")
    checks.check(metadata.get("parameter_sha256_before") == expected_parameters
                 and metadata.get("parameter_sha256_after") == expected_parameters, label + ": inference parameters immutable")
    checks.close(metadata.get("rebalance_band"), .025, label + ": unchanged execution band", atol=1e-15)
    checks.close(metadata.get("cash_floor_target"), .05, label + ": unchanged cash-floor environment", atol=1e-15)
    checks.close(metadata.get("max_asset_weight_target"), .95, label + ": original single-coin environment cap", atol=1e-15)
    checks.close(metadata.get("drawdown_penalty"), .25 if profile == "balanced" else 0., label + ": original risk reward", atol=1e-15)
    weights = read_csv(directory / "weights.csv")
    if capped:
        checks.check(metadata.get("family") == "cap075" and metadata.get("exposure_multiplier") == MULTIPLIER
                     and metadata.get("old_actions_or_equity_scaled") is False, label + ": actual capped account, not scaled old NAV")
        trace = read_csv(directory / "member_targets.csv")
        audit_member_targets(trace, weights, rebuilt, fills, start=dates[0], end=dates[1],
                             seeds=active, checks=checks, label=label)
        replay = read_json(directory / "replay.json")
        checks.check(replay.get("run_id") == label and replay.get("passed") is True and replay.get("repeat_exact") is True
                     and replay.get("fingerprints") == fingerprints == replay.get("repeated_fingerprints"), label + ": exact financial replay")
        checks.check(replay.get("member_targets_sha256") == sha(directory / "member_targets.csv")
                     == replay.get("repeated_member_targets_sha256"), label + ": exact member replay")
        checks.check(replay.get("parameter_sha256_before") == expected_parameters
                     and replay.get("parameter_sha256_after") == expected_parameters, label + ": repeat parameter hashes")
        if protocol is not None:
            expected_models = [item for item in protocol["models"] if item["profile"] == profile and item["seed"] in active]
            checks.check(replay.get("model_records") == expected_models, label + ": original checkpoint byte identities")
            device = replay.get("device", {})
            uuids = {item["uuid"] for item in protocol["physical_gpus"]}
            checks.check(device.get("device") == "cuda:0" and device.get("visible_gpu_uuid") in uuids
                         and "V100" in device.get("name", "") and device.get("capability") == [7, 0], label + ": isolated original V100 inference")
    else:
        weights["date"] = pd.to_datetime(weights.date, utc=True)
        dates_expected = pd.date_range(utc(dates[0]), utc(dates[1]), freq="D", inclusive="left")
        checks.check(len(weights) == len(dates_expected) * 2 and set(weights.date) == set(dates_expected)
                     and not weights[["date", "asset_key"]].duplicated().any(), label + ": raw daily two-coin coverage")
        for day, frame in weights.groupby("date"):
            if not checks.check(set(frame.asset_key) == set(SYMBOLS), label + ": raw coin identity"):
                continue
            frame = frame.set_index("asset_key").loc[list(SYMBOLS)]
            close = day + pd.Timedelta(hours=23)
            actual = rebuilt["coin_values"].loc[close].to_numpy() / equity.loc[close, "equity"]
            checks.vector_close(frame.actual_weight, actual, label + ": original actual inventory weights", atol=1e-10)
            checks.check(frame.slot.tolist() == [0, 1], label + ": original slot identity")
            target = frame.target_weight.to_numpy(dtype=float)
            checks.check(np.isfinite(target).all() and (target >= -1e-12).all() and target.sum() <= .95 + 1e-10,
                         label + ": original funded raw targets")
    # Derive daily/monthly records independently from the reconstructed ledger.
    daily = rebuilt["equity"].equity.resample("D").last()
    daily_drawdown = rebuilt["drawdown"].resample("D").last()
    history = pd.DataFrame({"timestamp": [utc(dates[0]), *list(daily.index + pd.Timedelta(days=1))],
        "equity": [10000., *daily.to_list()], "net_return": [0., *(daily / 10000 - 1).to_list()],
        "drawdown": [0., *daily_drawdown.to_list()]})
    monthly = rebuilt["monthly"]
    boundary = read_equity(directory / "boundary_equity.csv")
    checks.check(boundary.index.equals(pd.DatetimeIndex(history.timestamp)), label + ": daily reward boundary coverage")
    checks.vector_close(boundary.equity.to_numpy(), history.equity.to_numpy(), label + ": daily NAV boundary", atol=1e-8)
    rewards, previous_nav, previous_dd = [0.], 10000., 0.
    penalty = .25 if profile == "balanced" else 0.
    for _, hours in rebuilt["equity"].groupby(rebuilt["equity"].index.normalize()):
        deepest = -float(rebuilt["drawdown"].loc[hours.index].min())
        final = float(hours.equity.iloc[-1])
        rewards.append(100 * (math.log(final / previous_nav) - penalty * max(deepest - previous_dd, 0.)))
        previous_nav, previous_dd = final, -float(rebuilt["drawdown"].loc[hours.index[-1]])
    checks.vector_close(boundary.reward.to_numpy(), rewards, label + ": unchanged independent daily reward", atol=1e-9)
    result = {"run_id": label, "split": row["split"], "scenario": scenario, "profile": profile,
              "hours": len(equity), "reconstructed": True, "fingerprints": fingerprints,
              "net_return": rebuilt["metrics"]["net_return"], "max_drawdown": rebuilt["metrics"]["max_drawdown"],
              "fees": rebuilt["metrics"]["fees"], "inventory_replay": inventory_evidence}
    return result, history, monthly


CASE_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json", "fingerprints.json", "member_targets.csv", "replay.json")
GROUP_FILES = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "completion.json")


def extension_validation_files():
    return {"validation_completion.json",
            *(f"validation/{p}/{f}" for p in PROFILES for f in GROUP_FILES),
            *(f"validation/{p}/{c}/{k}/{f}" for p in PROFILES for c in COSTS
              for k in (*map(str, SEEDS), "ensemble") for f in CASE_FILES)}


class Inputs:
    def __init__(self, output, main, source_root, checks):
        self.roots = {"extension": Path(output).resolve(), "main": Path(main).resolve(), "source": Path(source_root).resolve()}
        self.hashes = {key: {} for key in self.roots}
        self.checks = checks

    def capture(self, root, name):
        path = safe_file(self.roots[root], name)
        digest = sha(path)
        canonical = str(name).replace("\\", "/")
        previous = self.hashes[root].get(canonical)
        if previous is not None:
            self.checks.check(previous == digest, "Input changed between reads: " + root + ":" + canonical)
        else:
            self.hashes[root][canonical] = digest
        return path

    def json(self, root, name):
        return read_json(self.capture(root, name))

    def csv(self, root, name):
        return read_csv(self.capture(root, name))

    def pin_map(self, root, values, names, label):
        if not self.checks.check(isinstance(values, dict) and set(values) == set(names), label + ": exact immutable file set"):
            raise ValueError(label + " has a partial or extra pin set")
        for name, expected in values.items():
            self.checks.check(sha(self.capture(root, name)) == expected, label + ": " + name)

    def unchanged(self):
        for root, files in self.hashes.items():
            for name, expected in files.items():
                self.checks.check(sha(safe_file(self.roots[root], name)) == expected, "Input unchanged: " + root + ":" + name)


def verify_protocol(inputs, dataset_path):
    p = inputs.json("extension", "protocol.json")
    receipt = inputs.json("extension", "protocol_receipt.json")
    spec = inputs.json("main", "frozen_spec.json")
    checks = inputs.checks
    checks.check(receipt.get("protocol_file") == "protocol.json"
                 and receipt.get("protocol_sha256") == sha(inputs.roots["extension"] / "protocol.json")
                 and utc(p["prepared_at"]) <= utc(receipt["sealed_at"]), "Independent extension declaration receipt")
    checks.check(p.get("multiplier") == MULTIPLIER and p.get("capital") == 10000.
                 and p.get("costs") == COSTS and p.get("profiles") == list(PROFILES) and p.get("seeds") == list(SEEDS)
                 and p.get("effective_max_coin_target_total") == .7125 and p.get("rebalance_band") == .025
                 and p.get("cash_floor") == .05 and p.get("max_asset_weight") == .95
                 and p.get("new_training") == 0 and p.get("multiplier_search") is False
                 and p.get("seed_selection") is False and p.get("selection_uses_test") is False,
                 "Exactly one capped inference variant, no new training or test tuning")
    checks.check(p.get("original_frozen_spec_sha256") == sha(inputs.roots["main"] / "frozen_spec.json")
                 and p.get("dates_utc") == {key: spec[key] for key in ("train", "validation", "test")}
                 and all(tuple(spec[key]) == dates for key, dates in SPLITS.items())
                 and p.get("research_targets") == spec["research_targets"], "Original study and chronological windows preserved")
    goal_file = inputs.capture("main", "goal_candidate_protocol.json")
    checks.check(p.get("original_goal_protocol_sha256") == sha(goal_file), "Original goal declaration preserved")
    for name, expected in p.get("source_sha256", {}).items():
        checks.check(sha(inputs.capture("source", name)) == expected, "Declared source SHA: " + name)
        copy = inputs.capture("extension", "sources/" + name)
        checks.check(sha(copy) == expected, "Exact independent byte snapshot: " + name)
    required_source = {*FINANCIAL, "scripts/btceth_rl_exposure_cap.py", "scripts/btceth_rl_goal_candidate.py"}
    checks.check(set(p.get("source_sha256", {})) == required_source, "Exact financial/goal/overlay source population")
    checks.check(all(p["source_sha256"].get(name) == spec["code_sha256"].get(name) for name in FINANCIAL), "Original four financial modules unchanged")
    runtime = inputs.json("main", "runtime.json")
    checks.check(p.get("physical_gpus") == runtime.get("physical_gpus") and len(p["physical_gpus"]) == 2
                 and len({g["uuid"] for g in p["physical_gpus"]}) == 2
                 and all("V100" in g["name"] for g in p["physical_gpus"]), "Original two physical V100 identities")
    checks.check(p.get("runtime_packages") == spec.get("runtime_packages"), "Unchanged formal inference dependencies")
    dataset_path = Path(dataset_path)
    data_hash, manifest_hash = sha(dataset_path), sha(dataset_path.parent / "manifest.json")
    checks.check(data_hash == p.get("dataset_sha256") == spec["dataset_sha256"]
                 and manifest_hash == p.get("dataset_manifest_sha256") == spec["dataset_manifest_sha256"], "Frozen source candle file and manifest bytes")
    model_names, records, expected_models = set(), {}, []
    for profile in PROFILES:
        for seed in SEEDS:
            folder = f"training/tqc__{profile}__seed{seed}"
            record_file, model_file = folder + "/record.json", folder + "/model.zip"
            record = inputs.json("main", record_file)
            model = inputs.capture("main", model_file)
            checks.check((record.get("algorithm"), record.get("profile"), record.get("seed")) == ("tqc", profile, seed)
                         and record.get("model_relative_path") == model_file
                         and record.get("model_sha256") == sha(model)
                         and record.get("frozen_spec_sha256") == p["original_frozen_spec_sha256"], "Original trained TQC identity: " + folder)
            expected_models.append(dict(profile=profile, seed=seed, record_file=record_file, model_file=model_file,
                record_sha256=sha(inputs.roots["main"] / record_file), model_sha256=sha(model), parameter_sha256=record["parameter_sha256"]))
            records[("tqc", profile, seed)] = record
            model_names.update((record_file, model_file))
    inputs.pin_map("main", p["training_source_sha256"], model_names, "Exactly twenty original TQC files")
    checks.check(p.get("models") == expected_models, "All ten original checkpoint/parameter identities")
    if checks.failed:
        raise ValueError("Protocol/source/model checks failed before financial reconstruction")
    return p, spec, records, (data_hash, manifest_hash)


def compare_group_tables(summary, daily, monthly, expected, checks, label):
    for run_id, (run, history, month_values) in expected.items():
        d = daily.loc[daily.run_id.eq(run_id)].copy()
        m = monthly.loc[monthly.run_id.eq(run_id)].copy()
        checks.check(len(d) == len(history) and not d.timestamp.duplicated().any(), label + ": daily coverage " + run_id)
        if len(d) == len(history):
            checks.check(pd.to_datetime(d.timestamp, utc=True).tolist() == pd.to_datetime(history.timestamp, utc=True).tolist(), label + ": daily dates " + run_id)
            for key in ("equity", "net_return", "drawdown"):
                checks.vector_close(d[key], history[key], label + ": daily " + key, atol=1e-8 if key == "equity" else 1e-10)
        checks.check(len(m) == len(month_values) and not m.month.duplicated().any() and set(m.month) == set(month_values), label + ": monthly coverage " + run_id)
        for item in m.itertuples(index=False, name=None):
            values = dict(zip(m.columns, item))
            if values["month"] in month_values:
                checks.close(values["return"], month_values[values["month"]], label + ": independent monthly " + run_id, atol=1e-10)
    ids = set(summary.run_id)
    checks.check(set(daily.run_id) == ids == set(monthly.run_id), label + ": no hidden/missing table cases")


def audit_group(inputs, dataset, records, protocol, *, split, algorithm, profile, capped):
    root = "extension" if capped else "main"
    folder = f"{split}/{profile}" if capped else f"evaluation/{split}/{algorithm}/{profile}"
    for name in GROUP_FILES:
        inputs.capture(root, folder + "/" + name)
    summary = inputs.csv(root, folder + "/summary.csv")
    daily, monthly = inputs.csv(root, folder + "/daily_equity.csv"), inputs.csv(root, folder + "/monthly_returns.csv")
    validate_group_rows(summary, split=split, algorithm=algorithm, profile=profile, multiplier=MULTIPLIER if capped else 1.)
    completion = inputs.json(root, folder + "/completion.json")
    if capped:
        inputs.checks.check(completion.get("case_count") == 12 and completion.get("protocol_sha256") == sha(inputs.roots["extension"] / "protocol.json"), folder + ": completed full twelve-case group")
    repeated = {r["run_id"]: r for r in completion.get("repeats", [])}
    inputs.checks.check(set(repeated) == set(summary.run_id), folder + ": all twelve independent replay receipts")
    expected = {}
    for row in summary.to_dict("records"):
        key = "ensemble" if row["primary_policy"] else str(int(row["seed"]))
        case = f"{folder}/{row['scenario']}/{key}"
        for name in CASE_FILES if capped else CASE_FILES[:6]:
            inputs.capture(root, case + "/" + name)
        run, history, month_values = audit_case(safe_file(inputs.roots[root], case), dataset, row, dates=SPLITS[split],
            records=records, profile=profile, key=key, checks=inputs.checks, capped=capped, protocol=protocol if capped else None)
        run.update(source_root=root, relative_directory=case)
        proof = repeated.get(row["run_id"], {})
        inputs.checks.check(proof.get("passed") is True and proof.get("fingerprints") == run["fingerprints"], row["run_id"] + ": group replay matches actual saved account")
        if capped:
            inputs.checks.check(proof.get("member_targets_sha256") == sha(safe_file(inputs.roots[root], case + "/member_targets.csv")), row["run_id"] + ": group member replay hash")
        expected[row["run_id"]] = (run, history, month_values)
    compare_group_tables(summary, daily, monthly, expected, inputs.checks, folder)
    return summary, daily, monthly, expected


def compare_candidate(actual, expected, checks, label):
    checks.check((actual.get("algorithm"), actual.get("profile")) == (expected["algorithm"], expected["profile"]), label + ": identity")
    checks.close(actual.get("score"), expected["score"], label + ": independent score", atol=1e-12, rtol=1e-12)
    for scenario in COSTS:
        source, truth = actual.get(scenario, {}), expected[scenario]
        checks.check(source.get("run_id") == truth["run_id"], label + ": actual ensemble run")
        for metric in ("net_return", "max_drawdown", "final_equity", "fees"):
            checks.close(source.get(metric), truth[metric], label + ": " + metric, atol=1e-10)


def independent_original_context(inputs, *, require_test):
    """Validate old immutable choices without running its strict-libm selector.

    Financial values/scores are independently recomputed with explicit numeric
    tolerance. Validated frozen score bytes retain the original exact ranking,
    including genuine ties, rather than changing a choice for a platform ULP.
    No root/test financial summary is read by this function.
    """
    checks = inputs.checks
    spec = inputs.json("main", "frozen_spec.json")
    original = inputs.json("main", "validation_selection.json")
    protocol = inputs.json("main", "goal_candidate_protocol.json")
    selected = inputs.json("main", "goal_candidate_selection.json")
    receipt = inputs.json("main", "goal_candidate_selection_receipt.json")
    spec_hash = sha(inputs.roots["main"] / "frozen_spec.json")
    original_hash = sha(inputs.roots["main"] / "validation_selection.json")
    goal_hash = sha(inputs.roots["main"] / "goal_candidate_protocol.json")
    checks.check(original.get("frozen_spec_sha256") == spec_hash
                 and original.get("selection_uses_test") is False
                 and original.get("test_results_available_to_selector") is False, "Original pre-test validation choice identity")
    checks.check(protocol.get("original_frozen_spec_sha256") == spec_hash
                 and protocol.get("original_financial_code_sha256") == spec["code_sha256"]
                 and protocol.get("new_algorithms") == list(ALGORITHMS) and protocol.get("control_algorithm") == "sac"
                 and protocol.get("seeds") == list(SEEDS) and protocol.get("selection_score") == spec["validation_selection"]["score"]
                 and protocol.get("research_targets") == spec["research_targets"], "Original new-only goal contract preserved")
    expected_goal_source = {"scripts/btceth_rl_goal_candidate.py": sha(inputs.capture("source", "scripts/btceth_rl_goal_candidate.py"))}
    checks.check(protocol.get("source_sha256") == selected.get("source_sha256") == expected_goal_source, "Original goal-helper immutable source bytes")
    checks.check(receipt.get("selection_file") == "goal_candidate_selection.json"
                 and receipt.get("selection_sha256") == sha(inputs.roots["main"] / "goal_candidate_selection.json")
                 and receipt.get("declared_protocol_sha256") == selected.get("declared_protocol_sha256") == goal_hash
                 and receipt.get("original_validation_selection_sha256") == selected.get("original_validation_selection_sha256") == original_hash
                 and selected.get("original_frozen_spec_sha256") == spec_hash
                 and selected.get("seed_selection") is False and selected.get("selection_uses_test") is False,
                 "Original new-only candidate full-byte selection receipt")
    checks.check(utc(spec["frozen_at"]) <= utc(protocol["declared_at"]) <= utc(selected["selected_at"])
                 <= utc(receipt["sealed_at"]) and utc(original["frozen_at"]) <= utc(selected["selected_at"]), "Original declaration/selection/seal chronology")
    all_algorithms = (*ALGORITHMS, "sac")
    names = {"training_completion.json", *(f"training/{a}__{p}__seed{s}/{f}"
             for a in all_algorithms for p in PROFILES for s in SEEDS for f in ("record.json", "model.zip"))}
    inputs.pin_map("main", original["training_source_sha256"], names, "Exactly81 original trained-model files")
    training = inputs.json("main", "training_completion.json")
    entries = training.get("records", [])
    expected_identities = {(a, p, s) for a in all_algorithms for p in PROFILES for s in SEEDS}
    checks.check(training.get("model_count") == len(entries) == 40
                 and {(r["algorithm"], r["profile"], r["seed"]) for r in entries} == expected_identities
                 and training.get("frozen_spec_sha256") == spec_hash, "All40 original training records completed")
    for record in entries:
        folder = f"training/{record['algorithm']}__{record['profile']}__seed{record['seed']}"
        disk = inputs.json("main", folder + "/record.json")
        checks.check(disk == record and record.get("model_relative_path") == folder + "/model.zip"
                     and record.get("frozen_spec_sha256") == spec_hash
                     and record.get("actual_timesteps") == spec["timesteps_per_model"]
                     and record.get("model_sha256") == sha(inputs.capture("main", folder + "/model.zip")), "Original model completion matches original checkpoint: " + folder)
    val_names = {f"evaluation/validation/{a}/{p}/summary.csv" for a in all_algorithms for p in PROFILES}
    inputs.pin_map("main", original["validation_source_sha256"], val_names, "Exactly8 original immutable validation groups")
    complete = inputs.json("main", "validation_completion.json")
    checks.check(complete.get("model_groups") == 8 and complete.get("baselines") == 7
                 and complete.get("frozen_spec_sha256") == spec_hash, "Original full110 validation phase completed")
    calculated = []
    for a in all_algorithms:
        for p in PROFILES:
            frame = inputs.csv("main", f"evaluation/validation/{a}/{p}/summary.csv")
            calculated.append(candidate_row(frame, family="raw", algorithm=a, profile=p))
    prior = original.get("candidates", [])
    checks.check(len(prior) == 8 and {(r.get("algorithm"),r.get("profile")) for r in prior} == {(r["algorithm"],r["profile"]) for r in calculated}, "All8 original validation candidates retained")
    by_key = {(r["algorithm"],r["profile"]):r for r in prior}
    for expected in calculated:
        compare_candidate(by_key.get((expected["algorithm"], expected["profile"]), {}), expected, checks, "Original candidate")
    # Ranking the validated original frozen numbers avoids a spurious ranking
    # change from a Linux/Windows last-bit difference in log1p.
    ranked = sorted(prior, key=lambda r:(-r["score"],r["algorithm"],r["profile"]))
    choices = {a:sorted([r for r in prior if r["algorithm"] == a],key=lambda r:(-r["score"],r["profile"]))[0]["profile"] for a in all_algorithms}
    global_leader = {k:ranked[0][k] for k in ("algorithm","profile")}
    checks.check(original.get("selected_profiles") == choices and original.get("primary_validation_leader") == global_leader, "Exact original stored-score ranking preserved after independent numeric checks")
    new_calculated = {(r["algorithm"],r["profile"]):r for r in calculated if r["algorithm"] in ALGORITHMS}
    new_recorded = selected.get("new_validation_candidates", [])
    checks.check(len(new_recorded) == 6 and {(r.get("algorithm"),r.get("profile")) for r in new_recorded}
                 == set(new_calculated), "Original six new-only candidates retained, no SAC")
    for actual in new_recorded:
        key=(actual.get("algorithm"),actual.get("profile"))
        if key in new_calculated:
            compare_candidate(actual,new_calculated[key],checks,"Original new-only candidate")
    ranked_new=sorted(new_recorded,key=lambda r:(-r["score"],r["algorithm"],r["profile"]))
    checks.check(new_recorded==ranked_new,"Original new-only exact frozen-score order")
    leader = {k:ranked_new[0][k] for k in ("algorithm","profile")}
    checks.check(selected.get("new_model_validation_leader") == leader
                 and selected.get("original_global_validation_leader") == global_leader
                 and selected.get("training_source_sha256") == original["training_source_sha256"]
                 and selected.get("validation_source_sha256") == original["validation_source_sha256"], "Original new-only leader and immutable sources preserved")
    compare_candidate(selected.get("candidate",{}),new_calculated[(leader["algorithm"],leader["profile"])],checks,"Original sealed new-only winner")
    checks.check(choices[leader["algorithm"]] == leader["profile"]
                 and (global_leader["algorithm"] not in ALGORITHMS or leader==global_leader), "Original global/new-only choices remain consistent")
    marker = None
    if (inputs.roots["main"] / "test_started.json").is_file():
        marker = inputs.json("main", "test_started.json")
        checks.check(marker.get("validation_selection_sha256") == original_hash and marker.get("frozen_spec_sha256") == spec_hash
                     and utc(receipt["sealed_at"]) < utc(marker["at"]) and utc(original["frozen_at"]) <= utc(marker["at"]), "Original test marker postdates both fixed choices")
    elif require_test or (inputs.roots["main"] / "test_completion.json").exists() or (inputs.roots["main"] / "evaluation/test").exists():
        raise ValueError("Original test requires a pre-test selection-bound dispatch marker")
    return {"spec":spec,"protocol":protocol,"selection":selected,"receipt":receipt,
            "original_selection":original,"new_model_validation_leader":leader,"test_started":marker,"test_pnl_read":False}


def verify_selection(inputs, protocol, cap_frames, *, require_test):
    context = independent_original_context(inputs, require_test=require_test)
    checks = inputs.checks
    for name in ("validation_selection.json", "goal_candidate_protocol.json", "goal_candidate_selection.json", "goal_candidate_selection_receipt.json", "training_completion.json", "validation_completion.json"):
        inputs.capture("main", name)
    original = context["original_selection"]
    for name, digest in original["training_source_sha256"].items():
        checks.check(sha(inputs.capture("main", name)) == digest, "Main sealed training file: " + name)
    frames = {}
    for algorithm in (*ALGORITHMS, "sac"):
        for profile in PROFILES:
            name = f"evaluation/validation/{algorithm}/{profile}/summary.csv"
            frame = inputs.csv("main", name)
            checks.check(inputs.hashes["main"][name] == original["validation_source_sha256"][name], "Original validation group remains selection-pinned: " + name)
            validate_group_rows(frame, split="validation", algorithm=algorithm, profile=profile, multiplier=1.)
            frames[(algorithm, profile)] = frame
    derived = candidate_pool(frames, cap_frames)
    s, r = inputs.json("extension", "selection.json"), inputs.json("extension", "selection_receipt.json")
    checks.check(r.get("selection_file") == "selection.json" and r.get("selection_sha256") == sha(inputs.roots["extension"] / "selection.json")
                 and r.get("protocol_sha256") == s.get("protocol_sha256") == sha(inputs.roots["extension"] / "protocol.json"), "Independent whole-file extension selection receipt")
    protocol_receipt = inputs.json("extension", "protocol_receipt.json")
    checks.check(utc(protocol["prepared_at"]) <= utc(protocol_receipt["sealed_at"])
                 <= utc(s["selected_at"]) <= utc(r["sealed_at"]), "Predeclared and fully sealed extension selection timeline")
    checks.check(s.get("selection_uses_test") is False and s.get("seed_selection") is False, "No test or seed selection")
    checks.check(s.get("original_validation_selection_sha256") == sha(inputs.roots["main"] / "validation_selection.json")
                 and s.get("original_goal_selection_sha256") == sha(inputs.roots["main"] / "goal_candidate_selection.json")
                 and s.get("original_goal_receipt_sha256") == sha(inputs.roots["main"] / "goal_candidate_selection_receipt.json"), "Original global and new-only goal choices preserved")
    checks.check(s.get("training_source_sha256") == original["training_source_sha256"]
                 and s.get("validation_source_sha256") == original["validation_source_sha256"], "Long-term selection pins immutable group evidence, not root aggregates")
    inputs.pin_map("extension", s["extension_source_sha256"], extension_validation_files(), "Complete immutable twenty-four-case extension validation")
    checks.check(s.get("original_global_validation_leader") == original["primary_validation_leader"]
                 and s.get("original_new_model_validation_leader") == context["new_model_validation_leader"], "Original validation leaders unchanged")
    recorded = s.get("candidates", [])
    independent = {(r["family"],r["algorithm"],r["profile"]):r for r in derived}
    checks.check(len(recorded) == 8 and {(r.get("family"),r.get("algorithm"),r.get("profile")) for r in recorded}
                 == set(independent), "Independent eight-candidate population, SAC control excluded")
    for actual in recorded:
        identity=(actual.get("family"),actual.get("algorithm"),actual.get("profile"))
        if identity not in independent:
            continue
        expected=independent[identity]
        checks.close(actual.get("score"), expected["score"], "Independent fixed validation score", atol=1e-12)
        for scenario in COSTS:
            checks.check(actual.get(scenario, {}).get("run_id") == expected[scenario]["run_id"], "Candidate actual ensemble identity")
            for metric in ("net_return", "max_drawdown", "fees", "final_equity"):
                checks.close(actual.get(scenario, {}).get(metric), expected[scenario][metric], "Candidate reconstructed source " + metric, atol=1e-10)
    frozen_order=sorted(recorded,key=lambda r:(-r["score"],0 if r["family"]=="raw" else 1,r["algorithm"],r["profile"]))
    checks.check(recorded==frozen_order,"Validated frozen scores preserve exact raw-first ties across platforms")
    checks.check(s.get("candidate") == frozen_order[0] if frozen_order else False, "Only the previously ranked candidate may reach test")
    checks.check({k: s.get("candidate", {}).get(k) for k in ("family", "algorithm", "profile")}
                 == {k: frozen_order[0][k] for k in ("family", "algorithm", "profile")}, "Winner independently reproduced without test ranking")
    if (inputs.roots["main"] / "test_started.json").is_file():
        marker = inputs.json("main", "test_started.json")
        checks.check(utc(r["sealed_at"]) < utc(marker["at"]), "Extension selection sealed before original test dispatch")
    return frozen_order[0], context


def compare_aggregate(inputs, groups):
    """Current root tables may legitimately grow from24 to36; never seal them as selection inputs."""
    for index, name, keys in ((0, "summary.csv", ["run_id"]),
                              (1, "daily_equity.csv", ["run_id", "timestamp"]),
                              (2, "monthly_returns.csv", ["run_id", "month"])):
        actual = inputs.csv("extension", name)
        expected = pd.concat([group[index] for group in groups], ignore_index=True)
        inputs.checks.check(set(actual.columns) == set(expected.columns) and len(actual) == len(expected), "Current root table dimensions: " + name)
        if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
            continue
        actual, expected = actual.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
        inputs.checks.check(not actual[keys].duplicated().any(), "Unique root table identities: " + name)
        for column in expected:
            missing = actual[column].isna() & expected[column].isna()
            if pd.api.types.is_numeric_dtype(expected[column]):
                inputs.checks.vector_close(actual.loc[~missing, column], expected.loc[~missing, column], "Root table " + name + ": " + column, atol=1e-8 if column == "equity" else 1e-10)
            else:
                inputs.checks.check((missing | actual[column].eq(expected[column])).all(), "Root table " + name + ": " + column)


def verify_main_final_audit(inputs, selected_runs):
    name = "independent_audit.json"
    if not (inputs.roots["main"] / name).is_file() and (inputs.roots["main"] / "independent_audit.local.json").is_file():
        name = "independent_audit.local.json"
    report = inputs.json("main", name)
    checks = inputs.checks
    checks.check(report.get("status") == "passed" and report.get("passed") is True
                 and report.get("test_complete") is True and report.get("failure_count") == 0
                 and report.get("failures") == [] and report.get("run_count", report.get("runs_checked")) == 172,
                 "Original accounting audit really contains all172 runs")
    for path, digest in report.get("input_sha256", {}).items():
        checks.check(sha(inputs.capture("main", path)) == digest, "Original final audit input: " + path)
    for run_id, run in selected_runs.items():
        if run["source_root"] != "main":
            continue
        original = report.get("runs", {}).get(run_id, {})
        checks.check(original.get("reconstructed") is True and original.get("hours") == 8760
                     and original.get("fingerprints") == run["fingerprints"], "Original audited raw test reference: " + run_id)
    return name, report


def audit_equal_hold(inputs, dataset, main_audit):
    folder = "evaluation/test/baselines"
    summary = inputs.csv("main", folder + "/summary.csv")
    row = summary.loc[summary.run_id.eq("test__base__equal_hold")]
    if len(row) != 1:
        raise ValueError("Need the original funded BTC/ETH equal-hold reference")
    row = row.iloc[0].to_dict()
    if (row["split"] != "test" or row["scenario"] != "base" or row["strategy"] != "equal_hold"
            or row["initial_capital"] != 10000. or utc(row["start"]) != utc(SPLITS["test"][0])
            or utc(row["end_exclusive"]) != utc(SPLITS["test"][1])):
        raise ValueError("Equal-hold reference has another asset/cost/window/capital identity")
    directory = folder + "/base/equal_hold"
    for name in CASE_FILES[:6]:
        inputs.capture("main", directory + "/" + name)
    path = safe_file(inputs.roots["main"], directory)
    fingerprints = verify_saved_fingerprints(path, inputs.checks, row["run_id"])
    rebuilt = reconstruct_fills(dataset, read_csv(path / "fills.csv"), read_equity(path / "equity_1h.csv.gz"),
        read_json(path / "metadata.json"), start=SPLITS["test"][0], end=SPLITS["test"][1],
        **COSTS["base"], checks=inputs.checks, label="Independent equal-hold reference")
    if rebuilt is None:
        raise ValueError("Equal-hold reference lacks full real hourly accounting")
    _check_summary(row, rebuilt["metrics"], inputs.checks, row["run_id"])
    original = main_audit.get("runs", {}).get(row["run_id"], {})
    inputs.checks.check(original.get("reconstructed") is True and original.get("hours") == 8760
                        and original.get("fingerprints") == fingerprints, "Equal hold belongs to the passed172 main audit")
    return rebuilt["metrics"]


def load_market(path, cutoff):
    with gzip.open(path, "rb") as stream:
        original = pickle.load(stream)
    result = {key: value for key, value in original.items() if not key.startswith("hourly_")}
    for field in ("open", "high", "low", "close", "volume"):
        frame = original["hourly_" + field]
        if list(frame.columns) != list(SYMBOLS):
            raise ValueError("Prepared market columns must be fixed BTCUSDT,ETHUSDT")
        result["hourly_" + field] = frame.loc[frame.index < utc(cutoff)].copy()
    return result


def audit(output, main, dataset_path, *, allow_validation_only=False, source_root=ROOT):
    """Read-only audit API. Only final mode can access true test financial rows."""
    output, main, dataset_path = map(Path, (output, main, dataset_path))
    checks, runs, groups = Checks(), {}, []
    inputs = Inputs(output, main, source_root, checks)
    validation_complete = selection_complete = test_complete = False
    target = {"evaluated": False, "reason": "No complete, preselected test was audited"}
    main_audit_path, selected, data_hashes = None, None, None
    audit_sources = {"scripts/audit_btceth_rl_exposure_cap.py": sha(Path(__file__).resolve()),
                     "scripts/audit_btceth_rl.py": sha(ROOT / "scripts/audit_btceth_rl.py")}
    try:
        if output.resolve() == main.resolve() or output.resolve().is_relative_to(main.resolve()) or main.resolve().is_relative_to(output.resolve()):
            raise ValueError("Independent extension must not overlap the original study")
        if not allow_validation_only:
            for root, name in (("extension", "selection.json"), ("extension", "selection_receipt.json"),
                               ("extension", "test_started.json"), ("extension", "test_completion.json"),
                               ("main", "test_started.json"), ("main", "test_completion.json")):
                if not (inputs.roots[root] / name).is_file():
                    raise ValueError("Final audit is closed until the complete preselected test exists: " + name)
        p, spec, records, data_hashes = verify_protocol(inputs, dataset_path)
        process = inputs.json("main", "fit_process.json")
        checks.check(process.get("phase") == "fit" and process.get("status") == "completed"
                     and process.get("exit_code") == 0 and process.get("frozen_spec_sha256") == p["original_frozen_spec_sha256"]
                     and bool(process.get("completed_at")), "Original40 fit terminal0 before extension inference")
        if process.get("log_relative_path"):
            checks.check(sha(inputs.capture("main", process["log_relative_path"])) == process.get("log_sha256"), "Immutable original fit terminal log")
        else:
            checks.check(False, "Missing original fit terminal log proof")
        data = load_market(dataset_path, SPLITS["validation"][1])
        cap_frames = {}
        for profile in PROFILES:
            group = audit_group(inputs, data, records, p, split="validation", algorithm="tqc", profile=profile, capped=True)
            groups.append(group[:3])
            cap_frames[profile] = group[0]
            runs.update({name: values[0] for name, values in group[3].items()})
        done = inputs.json("extension", "validation_completion.json")
        checks.check(done.get("case_count") == 24 and done.get("repeat_count") == 24
                     and done.get("profiles") == list(PROFILES) and done.get("split") == "validation"
                     and done.get("protocol_sha256") == sha(output / "protocol.json"), "All24 capped validation cases and repeats completed")
        validation_complete = len(runs) == 24 and checks.failed == 0
        if (output / "selection.json").exists() or (output / "selection_receipt.json").exists() or not allow_validation_only:
            selected, context = verify_selection(inputs, p, cap_frames, require_test=not allow_validation_only)
            selection_complete = checks.failed == 0
        if not allow_validation_only:
            if not validation_complete or not selection_complete:
                raise ValueError("No test financial data may be audited after failed pre-test validation/selection checks")
            marker = inputs.json("extension", "test_started.json")
            main_marker = inputs.json("main", "test_started.json")
            receipt = read_json(output / "selection_receipt.json")
            checks.check(marker.get("selection_sha256") == sha(output / "selection.json")
                         and marker.get("protocol_sha256") == sha(output / "protocol.json")
                         and marker.get("main_test_started_sha256") == sha(main / "test_started.json")
                         and utc(receipt["sealed_at"]) < utc(main_marker["at"]) <= utc(marker["at"]), "Immutable preselected extension test dispatch")
            done = inputs.json("extension", "test_completion.json")
            original_done = inputs.json("main", "test_completion.json")
            checks.check(original_done.get("model_groups") == 4 and original_done.get("baselines") == 7
                         and original_done.get("frozen_spec_sha256") == p["original_frozen_spec_sha256"], "Original172-case test completed first")
            checks.check(done.get("case_count") == done.get("repeat_count") == 12
                         and done.get("protocol_sha256") == sha(output / "protocol.json")
                         and done.get("selection_sha256") == sha(output / "selection.json")
                         and done.get("main_test_completion_sha256") == sha(main / "test_completion.json"), "Only12 tests of the frozen selected candidate")
            if checks.failed:
                raise ValueError("Test-dispatch/completion contract mismatch before test financial reads")
            data = load_market(dataset_path, SPLITS["test"][1])
            if selected["family"] == "cap075":
                checks.check(done.get("mode") == "cap075_evaluation" and done.get("raw_source_sha256") == {}, "Cap selection cannot be replaced with raw returns")
                checks.check({d.name for d in (output / "test").iterdir() if d.is_dir()} == {selected["profile"]}, "Only the selected capped profile was tested")
                group = audit_group(inputs, data, records, p, split="test", algorithm="tqc", profile=selected["profile"], capped=True)
            else:
                checks.check(done.get("mode") == "raw_reference" and not (output / "test").exists(), "Raw winner reuses existing tests without new inference")
                algorithm, profile = selected["algorithm"], selected["profile"]
                checks.check(context["original_selection"]["selected_profiles"][algorithm] == profile, "Original raw profile cannot be replaced after test")
                folder = f"evaluation/test/{algorithm}/{profile}"
                names = {*(f"{folder}/{f}" for f in GROUP_FILES),
                         *(f"{folder}/{c}/{k}/{f}" for c in COSTS for k in (*map(str, SEEDS), "ensemble") for f in CASE_FILES[:6])}
                inputs.pin_map("main", done["raw_source_sha256"], names, "Exactly76 immutable reused raw test files")
                for seed in SEEDS:
                    record_file = f"training/{algorithm}__{profile}__seed{seed}/record.json"
                    records[(algorithm, profile, seed)] = inputs.json("main", record_file)
                group = audit_group(inputs, data, records, p, split="test", algorithm=algorithm, profile=profile, capped=False)
                group[0]["family"], group[0]["exposure_multiplier"] = "raw", 1.
            groups.append(group[:3])
            runs.update({name: values[0] for name, values in group[3].items()})
            main_audit_path, original_audit = verify_main_final_audit(inputs, runs)
            equal_hold = audit_equal_hold(inputs, data, original_audit)
            target = objective_evidence(group[0], selected, equal_hold)
            test_complete = len(runs) == 36 and checks.failed == 0
        compare_aggregate(inputs, groups)
        inputs.unchanged()
        if data_hashes is not None:
            checks.check(sha(dataset_path) == data_hashes[0]
                         and sha(dataset_path.parent / "manifest.json") == data_hashes[1], "Prepared public market files remain unchanged")
    except Exception as error:
        checks.check(False, type(error).__name__ + ": " + str(error))
    for name, expected in audit_sources.items():
        try:
            checks.check(sha(ROOT / name) == expected, "Actual audit implementation unchanged during run: " + name)
        except OSError as error:
            checks.check(False, "Audit source changed: " + str(error))
    complete = validation_complete and (allow_validation_only or (selection_complete and test_complete))
    passed = checks.failed == 0 and complete
    return {"schema_version": 1, "audited_at": now(),
            "status": ("passed_validation_only" if allow_validation_only else "passed") if passed else "failed",
            "passed": bool(passed), "validation_complete": bool(validation_complete),
            "selection_complete": bool(selection_complete), "test_complete": bool(test_complete and not allow_validation_only),
            "checks": checks.count, "failure_count": checks.failed, "failures": checks.failures,
            "maximum_errors": checks.maximum_errors, "run_count": len(runs), "runs": runs,
            "main_final_audit_path": main_audit_path, "research_targets": target,
            "input_sha256": inputs.hashes["extension"], "main_input_sha256": inputs.hashes["main"],
            "source_sha256": {**inputs.hashes["source"], **audit_sources},
            "dataset_sha256": data_hashes[0] if data_hashes else None,
            "test_financial_reads_enabled": not allow_validation_only,
            "accounting_audit_is_not_performance_success": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--allow-validation-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.output_dir, args.main_output, args.dataset,
                   allow_validation_only=args.allow_validation_only, source_root=args.source_root)
    path = args.report or args.output_dir / ("independent_audit.validation.json" if args.allow_validation_only else "independent_audit.json")
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False, default=scalar)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "run_count", "checks", "failure_count", "test_complete")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
