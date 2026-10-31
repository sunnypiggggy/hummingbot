"""Independent audit acceptance and economically meaningful corruption cases.

Environment output is the test subject. The auditor never imports or calls
that environment, its ledger, or the report's metric calculator.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import audit_single_pair_rl as audit
from scripts.crypto_rl_reporting import canonical_csv, save_result, summarize
from scripts.single_pair_rl_env import SinglePairEnv


START = pd.Timestamp("2025-10-23T00:00:00Z")
END = START + pd.Timedelta(days=3)


@pytest.fixture(scope="module")
def market_data():
    index = pd.date_range("2025-01-01T00:00:00Z", END + pd.Timedelta(days=2), freq="h", inclusive="left")
    rng = np.random.default_rng(71)
    base = pd.DataFrame({"BTCUSDT": 40000. * np.exp(np.cumsum(rng.normal(.00001, .001, len(index)))),
                         "ETHUSDT": 2000. * np.exp(np.cumsum(rng.normal(.00001, .001, len(index))))}, index=index)
    return {"hourly_open": base.copy(), "hourly_high": base * 1.002,
            "hourly_low": base * .998, "hourly_close": base * 1.00005,
            "hourly_volume": base * 0 + 100.}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def make_case(tmp_path, data, *, symbol="BTCUSDT", scenario="base", algorithm="sac", seed=0):
    env = SinglePairEnv(data, symbol, START, END, costs=audit.COSTS[scenario])
    env.reset(seed=0)
    actions = []
    for index, action in enumerate((1., .2, 0.)):
        day = env.current_date
        if seed is None:
            if algorithm == "buy_hold" and index:
                env.step_hold()
            else:
                on = algorithm == "buy_hold" or algorithm == "sma200_trend" and env._features[day][7] > 0 or algorithm == "momentum180" and env._features[day][4] > 0
                env.step_target([.95 if on else 0., .05 if on else 1.])
        else:
            *_, info = env.step([action])
            actions.append({"date": str(day), "action": [action], "target": info["target_weights"].tolist()})
    result = env.get_result()
    run_id = f"test__{scenario}__{symbol}__{algorithm}__{'baseline' if seed is None else seed}"
    row, daily, monthly = summarize(result, run_id=run_id, strategy=algorithm, seed=seed, scenario=scenario,
                                     split="test", start=START, end=END, **audit.COSTS[scenario])
    row.update(symbol=symbol, algorithm=algorithm, primary_seed=seed == 0, ensemble=False, independent_capital=True)
    directory = tmp_path / audit.case_relative("test", symbol, algorithm, seed, scenario)
    save_result(directory, result)
    write_json(directory / "actions.json", actions)
    write_json(directory / "summary.json", row)
    (directory / "daily_equity.csv").write_bytes(canonical_csv(daily))
    (directory / "monthly_returns.csv").write_bytes(canonical_csv(monthly))
    repeat = tmp_path / audit.case_relative("test", symbol, algorithm, seed, scenario, True)
    shutil.copytree(directory, repeat)
    return directory, repeat


def case_audit(tmp_path, data, *, symbol="BTCUSDT", scenario="base", algorithm="sac", seed=0):
    return audit.audit_case(tmp_path, audit._market(data, symbol, START, END), "test", symbol, algorithm, seed, scenario, START, END)


def recopy_and_rehash(directory, repeat):
    fingerprints = {}
    for key, name in (("equity", "equity_1h.csv.gz"), ("fills", "fills.csv"), ("weights", "weights.csv"), ("boundary_equity", "boundary_equity.csv")):
        payload = gzip.decompress((directory / name).read_bytes()) if name.endswith(".gz") else (directory / name).read_bytes()
        fingerprints[key] = hashlib.sha256(payload).hexdigest()
    write_json(directory / "fingerprints.json", fingerprints)
    for name in audit.RESULT_FILES:
        shutil.copyfile(directory / name, repeat / name)


def assert_failed(result, label):
    assert result["passed"] is False
    assert result["failure_count"] > 0
    assert any(label in failure["check"] for failure in result["failures"]), result["failures"]


@pytest.mark.parametrize("symbol", audit.SYMBOLS)
@pytest.mark.parametrize("scenario", audit.COSTS)
@pytest.mark.parametrize("algorithm,seed", [("sac", 0), ("cash", None), ("buy_hold", None), ("sma200_trend", None), ("momentum180", None)])
def test_real_result_passes_independent_closed_form_and_actual_fill_replay(tmp_path, market_data, symbol, scenario, algorithm, seed):
    make_case(tmp_path, market_data, symbol=symbol, scenario=scenario, algorithm=algorithm, seed=seed)
    result = case_audit(tmp_path, market_data, symbol=symbol, scenario=scenario, algorithm=algorithm, seed=seed)
    assert result["passed"], result["failures"]
    assert result["checks_total"] > 500
    assert len(result["inputs_sha256"]) == len(audit.RESULT_FILES) * 2


def test_foreign_coin_fills_fail_even_when_all_saved_hashes_and_repeat_are_forged_consistently(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    frame = pd.read_csv(directory / "fills.csv", float_precision="round_trip")
    frame.loc[0, ["asset_key", "symbol"]] = ["ETHUSDT", "ETH"]
    (directory / "fills.csv").write_bytes(canonical_csv(frame))
    recopy_and_rehash(directory, repeat)
    assert_failed(case_audit(tmp_path, market_data), "fills contain only owned coin")


def test_forged_fill_price_and_fee_are_checked_against_original_open(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    frame = pd.read_csv(directory / "fills.csv", float_precision="round_trip")
    frame.loc[0, "price"] *= 1.01
    frame.loc[0, "fee"] = frame.loc[0, "qty"] * frame.loc[0, "price"] * .001
    (directory / "fills.csv").write_bytes(canonical_csv(frame))
    recopy_and_rehash(directory, repeat)
    assert_failed(case_audit(tmp_path, market_data), "actual replay adverse open price")


def test_early_buy_fails_even_with_consistent_repeated_fingerprints(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    frame = pd.read_csv(directory / "fills.csv", float_precision="round_trip")
    frame.loc[0, "timestamp"] = str(START)
    (directory / "fills.csv").write_bytes(canonical_csv(frame))
    recopy_and_rehash(directory, repeat)
    assert_failed(case_audit(tmp_path, market_data), "sell00 buy01 execution only")


def test_hourly_capital_injection_cannot_be_hidden_by_balancing_reported_equity(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    frame = pd.read_csv(directory / "equity_1h.csv.gz", float_precision="round_trip")
    frame.loc[12:, "cash"] += 1000
    frame.loc[12:, "equity"] += 1000
    frame.loc[12:, "exposure"] = frame.loc[12:, "base_value"] / frame.loc[12:, "equity"]
    (directory / "equity_1h.csv.gz").write_bytes(gzip.compress(canonical_csv(frame), mtime=0))
    recopy_and_rehash(directory, repeat)
    assert_failed(case_audit(tmp_path, market_data), "actual fill arithmetic reconstructs every hour")


def test_missing_hour_is_not_interpolated_or_accepted(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    frame = pd.read_csv(directory / "equity_1h.csv.gz", float_precision="round_trip").drop(index=7)
    (directory / "equity_1h.csv.gz").write_bytes(gzip.compress(canonical_csv(frame), mtime=0))
    recopy_and_rehash(directory, repeat)
    assert_failed(case_audit(tmp_path, market_data), "every evaluation hour exactly once")


@pytest.mark.parametrize("field,value", [("net_return", .99), ("max_drawdown", 0.), ("primary_seed", False), ("ensemble", True)])
def test_summary_performance_and_primary_account_are_not_trusted(tmp_path, market_data, field, value):
    directory, repeat = make_case(tmp_path, market_data)
    row = audit.read(directory / "summary.json")
    row[field] = value
    write_json(directory / "summary.json", row)
    recopy_and_rehash(directory, repeat)
    result = case_audit(tmp_path, market_data)
    assert result["passed"] is False
    assert any(field in item["check"] for item in result["failures"])


def test_one_dimensional_action_and_exact_target_mapping_are_mandatory(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    actions = audit.read(directory / "actions.json")
    actions[0]["action"] = [1., 0.]
    write_json(directory / "actions.json", actions)
    recopy_and_rehash(directory, repeat)
    assert_failed(case_audit(tmp_path, market_data), "one finite action")


def test_target_record_is_not_accepted_as_policy_output_without_mapping_check(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    actions = audit.read(directory / "actions.json")
    actions[0]["target"] = [.5, .5]
    write_json(directory / "actions.json", actions)
    recopy_and_rehash(directory, repeat)
    assert_failed(case_audit(tmp_path, market_data), "action maps to exactly [coin,cash]")


def test_monthly_and_daily_chart_values_must_reconcile(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    monthly = pd.read_csv(directory / "monthly_returns.csv", float_precision="round_trip")
    monthly.loc[0, "return"] = .8
    (directory / "monthly_returns.csv").write_bytes(canonical_csv(monthly))
    recopy_and_rehash(directory, repeat)
    assert_failed(case_audit(tmp_path, market_data), "monthly returns compound")


def test_repeat_is_verified_from_actual_bytes_not_a_success_flag(tmp_path, market_data):
    directory, repeat = make_case(tmp_path, market_data)
    row = audit.read(repeat / "summary.json")
    row["net_return"] += .2
    write_json(repeat / "summary.json", row)
    assert_failed(case_audit(tmp_path, market_data), "exact deterministic repeated file:summary.json")


def test_market_ignores_other_coin_and_future_inputs(market_data):
    base = audit._market(market_data, "BTCUSDT", START, END)
    changed = deepcopy(market_data)
    for frame in changed.values():
        frame.loc[:, "ETHUSDT"] = np.nan
        frame.loc[frame.index >= END, "BTCUSDT"] = 1e30
    actual = audit._market(changed, "BTCUSDT", START, END)
    for key in ("open", "high", "low", "close", "volume", "previous_close"):
        np.testing.assert_array_equal(base[key], actual[key])
    for name in base["signals"]:
        np.testing.assert_array_equal(base["signals"][name], actual["signals"][name])


def test_expected_population_is_all_five_seeds_two_pairs_two_costs_plus_independent_baselines():
    cases = audit.expected_cases(audit.ALGORITHMS)
    assert len(cases) == 136 and len(set(cases)) == 136
    assert sum(seed is not None for _, _, seed, _ in cases) == 120
    assert sum(seed is None for _, _, seed, _ in cases) == 16
    assert sum(seed == 0 for _, _, seed, _ in cases) == 24
    assert len(audit.expected_cases(["sac"])) == 36


def test_failed_phase_is_retained_and_never_changes_source_evidence(tmp_path):
    (tmp_path / "protocol.json").write_text('{}\n', encoding="utf-8")
    original = (tmp_path / "protocol.json").read_bytes()
    result = audit.audit_phase(tmp_path, tmp_path / "absent.pkl.gz", "validation", workers=1)
    assert result["passed"] is False and result["failure_count"] > 0
    assert (tmp_path / "protocol.json").read_bytes() == original
    assert audit.read(tmp_path / "validation_audit.json") == result
    assert len(list((tmp_path / "audit_history").glob("validation_audit_*.json"))) == 1
    again = audit.audit_phase(tmp_path, tmp_path / "absent.pkl.gz", "validation", workers=1)
    assert again["passed"] is False
    assert len(list((tmp_path / "audit_history").glob("validation_audit_*.json"))) == 2


def test_auditor_does_not_import_environment_ledger_runner_or_report_calculator():
    import ast
    source = Path(audit.__file__).read_text(encoding="utf-8")
    modules = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert not any(module and module.startswith("scripts.") for module in modules)


def test_complete_fixed_365day_phase_checks_all_accounts_and_aggregate_hash(tmp_path):
    """Real full-year serialization + parallel independent financial audit.

    This is a labelled synthetic fixture, not trained research or performance
    evidence. Zero policy actions deliberately exercise cash-degenerate models.
    """
    start, end = map(audit.utc, audit.SPLITS["test"])
    index = pd.date_range("2024-01-01T00:00:00Z", end, freq="h", inclusive="left")
    t = np.arange(len(index), dtype=float)
    prices = pd.DataFrame({"BTCUSDT": 100 * np.exp(.00001 * t + .04 * np.sin(t / 300)),
                           "ETHUSDT": 200 * np.exp(-.000004 * t + .06 * np.cos(t / 400))}, index=index)
    data = {"hourly_" + field: prices.copy() for field in ("open", "high", "low", "close")}
    data["hourly_volume"] = prices * 0 + 100
    dataset = tmp_path / "synthetic.pkl.gz"
    with gzip.open(dataset, "wb") as stream:
        pickle.dump(data, stream)
    write_json(tmp_path / "manifest.json", {"synthetic_test_fixture_only": True})
    output = tmp_path / "study"
    output.mkdir()
    spec = {"symbols": list(audit.SYMBOLS), "seeds": list(audit.SEEDS), "algorithms": ["sac"],
            "primary_seed": 0, "primary_seed_fixed_before_training": True, "capital_per_account": 10000.,
            "capital_shared_between_pairs": False, "ensemble": False, "costs": audit.COSTS,
            "baselines": list(audit.BASELINES), "test_year_already_observed": True, "fresh_holdout": False,
            "train": ["2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            **{name: list(bounds) for name, bounds in audit.SPLITS.items()},
            "environment": {"observation_dim": 13, "action_dim": 1, "cash_floor": .05, "max_coin_target": .95,
                            "rebalance_band": .025, "drawdown_penalty": .25, "episode_days": 32},
            "validation_selection": {"seed_selection": False}, "sources": {},
            "dataset_sha256": audit.sha(dataset), "dataset_manifest_sha256": audit.sha(tmp_path / "manifest.json")}
    write_json(output / "protocol.json", spec)
    rows = []
    completions = []
    for symbol in audit.SYMBOLS:
        for scenario in audit.COSTS:
            cached_cash = None
            for algorithm, seed in [(kind, None) for kind in audit.BASELINES] + [("sac", seed) for seed in audit.SEEDS]:
                actions = []
                if algorithm == "sac":
                    result = cached_cash
                    actions = [{"date": str(day), "action": [0.], "target": [0., 1.]} for day in pd.date_range(start, end, freq="D", inclusive="left")]
                else:
                    env = SinglePairEnv(data, symbol, start, end, costs=audit.COSTS[scenario])
                    env.reset(seed=0)
                    first, done = True, False
                    while not done:
                        day = env.current_date
                        if algorithm == "buy_hold" and not first:
                            _, _, terminated, truncated, _ = env.step_hold()
                        else:
                            on = algorithm == "buy_hold" or algorithm == "sma200_trend" and env._features[day][7] > 0 or algorithm == "momentum180" and env._features[day][4] > 0
                            _, _, terminated, truncated, _ = env.step_target([.95 if on else 0., .05 if on else 1.])
                        first, done = False, terminated or truncated
                    result = env.get_result()
                    if algorithm == "cash":
                        cached_cash = result
                run_id = f"test__{scenario}__{symbol}__{algorithm}__{'baseline' if seed is None else seed}"
                row, daily, monthly = summarize(result, run_id=run_id, strategy=algorithm, seed=seed, scenario=scenario,
                                                 split="test", start=start, end=end, **audit.COSTS[scenario])
                row.update(symbol=symbol, algorithm=algorithm, primary_seed=seed == 0, ensemble=False, independent_capital=True)
                directory = output / audit.case_relative("test", symbol, algorithm, seed, scenario)
                save_result(directory, result)
                write_json(directory / "summary.json", row)
                write_json(directory / "actions.json", actions)
                (directory / "daily_equity.csv").write_bytes(canonical_csv(daily))
                (directory / "monthly_returns.csv").write_bytes(canonical_csv(monthly))
                repeat = output / audit.case_relative("test", symbol, algorithm, seed, scenario, True)
                shutil.copytree(directory, repeat)
                rows.append(row)
                completions.append({"summary": row})
    (output / "test_summary.csv").write_bytes(canonical_csv(pd.DataFrame(rows)))
    write_json(output / "test_complete.json", {"accounts": 36, "repeats": 36, "repeat_equal": True, "results": completions})
    write_json(output / "validation_selection.json", {"uses_test": False, "seed_selection": False,
               "selection": {symbol: {"algorithm": "sac", "seed": 0} for symbol in audit.SYMBOLS}})
    original_hash = audit.sha(output / "test_summary.csv")
    result = audit.audit_phase(output, dataset, "test", workers=2)
    assert result["passed"], result["failures"][:5]
    assert result["accounts_count"] == result["accounts_checked"] == result["repeats_checked"] == 36
    assert result["summary_sha256"] == original_hash
    assert result["protocol_sha256"] == audit.sha(output / "protocol.json")
    assert result["checks_total"] > 2_000_000
    assert result["worker_count"] == 2
    assert audit.sha(output / "test_summary.csv") == original_hash
