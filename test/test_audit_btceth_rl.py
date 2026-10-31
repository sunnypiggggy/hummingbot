"""Hand-built accounting fixtures; neither environment nor ledger is used."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.audit_btceth_rl import (
    ALGORITHMS, Checks, PROFILES, _audit_run, independent_training_calendar, recompute_selection,
    reconstruct_fills, relative_file, score_candidate, verify_saved_fingerprints,
)


START = pd.Timestamp("2025-01-01T00:00Z")
END = START + pd.Timedelta(days=2)


def hand_case():
    hours = pd.date_range(START, END, freq="h", inclusive="left")
    prices = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=hours)
    prices.loc[START + pd.Timedelta(days=1):, "BTCUSDT"] = 110.
    data = {"hourly_" + name: prices.copy() for name in ("open", "high", "low", "close")}
    data["hourly_volume"] = prices * 0 + 100.
    # Buy 10 BTC at 100.02; sell 4 the following midnight at 109.978.
    buy_fee, buy_notional = 1.0002, 1000.2
    first_cash = 10000 - buy_notional - buy_fee
    sale_notional, sale_fee = 439.912, .439912
    final_cash = first_cash + sale_notional - sale_fee
    removed_basis = (buy_notional + buy_fee) * .4
    realized = sale_notional - sale_fee - removed_basis
    fills = pd.DataFrame([
        {"timestamp": START + pd.Timedelta(hours=1), "asset_key": "BTCUSDT", "symbol": "BTC",
         "side": "BUY", "qty": 10., "reference_price": 100., "price": 100.02, "notional": buy_notional,
         "fee": buy_fee, "slippage_cost": .2, "cash_after": first_cash, "quantity_after": 10., "realized_pnl": 0.},
        {"timestamp": START + pd.Timedelta(days=1), "asset_key": "BTCUSDT", "symbol": "BTC",
         "side": "SELL", "qty": 4., "reference_price": 110., "price": 109.978, "notional": sale_notional,
         "fee": sale_fee, "slippage_cost": .088, "cash_after": final_cash, "quantity_after": 6., "realized_pnl": realized},
    ])
    equity = pd.DataFrame(index=hours, columns=("equity", "cash", "base_value", "exposure"), dtype=float)
    equity.iloc[0] = [10000., 10000., 0., 0.]
    equity.iloc[1:24, 0] = first_cash + 1000.
    equity.iloc[1:24, 1] = first_cash
    equity.iloc[1:24, 2] = 1000.
    equity.iloc[24:, 0] = final_cash + 660.
    equity.iloc[24:, 1] = final_cash
    equity.iloc[24:, 2] = 660.
    equity["exposure"] = equity.base_value / equity.equity
    final = final_cash + 660.
    metadata = {"initial_capital": 10000., "final_equity": final, "final_cash": final_cash,
                "fees": buy_fee + sale_fee, "total_notional": buy_notional + sale_notional,
                "slippage_cost": .288, "realized_pnl": realized, "unrealized_pnl": 660. - (buy_notional + buy_fee) * .6,
                "final_liquidation_equity": final_cash + 660. * .9998 * .999, "final_positions": {"BTCUSDT": 6.},
                "bnb_fee_discount": False, "fee_asset": "USDT", "fee_rate": .001, "slippage": .0002,
                "record_history": True, "training": False, "forced_final_liquidation": False,
                "hourly_high_water": final, "current_drawdown": 0., "fill_count": 2,
                "final_holdings": [{"asset_key": "BTCUSDT", "qty": 6., "mark_price": 110.,
                                    "market_value": 660., "weight": 660. / final}]}
    return data, fills, equity, metadata


def rebuild(case, checks=None):
    data, fills, equity, metadata = case
    return reconstruct_fills(data, fills, equity, metadata, start=START, end=END,
                             fee=.001, slippage=.0002, checks=checks)


def test_independent_round_trip_reconciles_hand_cash_inventory_fees_pnl_and_drawdown():
    result = rebuild(hand_case())
    assert result["checks"].failed == 0, result["checks"].failures
    assert result["metrics"]["final_cash"] == pytest.approx(9438.271888)
    assert result["metrics"]["final_equity"] == pytest.approx(10098.271888)
    assert result["metrics"]["fees"] == pytest.approx(1.440112)
    assert result["metrics"]["max_drawdown"] == pytest.approx(-.00012002)
    assert result["quantity"].iloc[-1].tolist() == [6., 0.]
    assert result["metrics"]["realized_pnl"] + result["metrics"]["unrealized_pnl"] == pytest.approx(98.271888)


@pytest.mark.parametrize("column", ["cash", "base_value", "equity", "exposure"])
def test_one_hour_cash_or_marked_inventory_tampering_is_detected(column):
    case = hand_case()
    case[2].loc[START + pd.Timedelta(hours=3), column] += 1
    result = rebuild(case)
    assert result["checks"].failed > 0
    assert any("reconstructed " + column in failure for failure in result["checks"].failures)


@pytest.mark.parametrize("field,label", [("quantity_after", "per-fill inventory"),
                                       ("cash_after", "per-fill cash"),
                                       ("realized_pnl", "realized fill PNL"),
                                       ("fee", "fill fee"), ("price", "adverse fill price"),
                                       ("reference_price", "reference price"),
                                       ("slippage_cost", "fill slippage")])
def test_per_fill_cash_inventory_and_cost_tampering_are_detected(field, label):
    case = hand_case()
    case[1].loc[0, field] += 1
    result = rebuild(case)
    assert any(label in failure for failure in result["checks"].failures)


def test_missing_hour_is_a_failure_even_when_remaining_values_are_correct():
    data, fills, equity, metadata = hand_case()
    checks = Checks()
    result = rebuild((data, fills, equity.drop(equity.index[12]), metadata), checks)
    assert result is None and checks.failed == 1
    assert "complete unique ordered hourly equity" in checks.failures[0]


def test_duplicate_hour_cannot_hide_inside_correct_row_count():
    data, fills, equity, metadata = hand_case()
    altered = equity.copy()
    index = list(altered.index)
    index[12] = index[11]
    altered.index = pd.DatetimeIndex(index)
    checks = Checks()
    assert rebuild((data, fills, altered, metadata), checks) is None
    assert checks.failed


def test_selling_without_inventory_is_rejected_independently():
    case = hand_case()
    case[1].loc[0, "side"] = "SELL"
    case[1].loc[0, "timestamp"] = START
    result = rebuild(case)
    assert any("sell bounded by strategy inventory" in failure for failure in result["checks"].failures)


def test_buys_cannot_borrow_cash_to_cover_notional_and_fee():
    case = hand_case()
    case[1].loc[0, "qty"] = 1000.
    result = rebuild(case)
    assert any("buy funded including fee" in failure for failure in result["checks"].failures)
    assert any("nonnegative cash/inventory" in failure for failure in result["checks"].failures)


def test_wrong_execution_hour_and_bnb_discount_are_detected():
    case = hand_case()
    case[1].loc[0, "timestamp"] = START
    case[3]["bnb_fee_discount"] = True
    result = rebuild(case)
    assert any("00UTC sell / 01UTC buy" in failure for failure in result["checks"].failures)
    assert any("no BNB discount" in failure for failure in result["checks"].failures)


def test_serialized_metadata_does_not_override_reconstructed_economics():
    case = hand_case()
    case[3]["final_equity"] += 100
    case[3]["final_positions"]["BTCUSDT"] = 7.
    result = rebuild(case)
    assert result["metrics"]["final_equity"] == pytest.approx(10098.271888)
    assert any("metadata final_equity" in failure for failure in result["checks"].failures)
    assert any("final inventory BTCUSDT" in failure for failure in result["checks"].failures)


def test_saved_csv_fingerprints_verify_uncompressed_original_bytes(tmp_path):
    payloads = {"equity": b"timestamp,equity\n2025-01-01,10000\n",
                "fills": b"timestamp,qty\n", "weights": b"date,weight\n", "boundary_equity": b"timestamp,equity\n"}
    names = {"equity": "equity_1h.csv.gz", "fills": "fills.csv", "weights": "weights.csv",
             "boundary_equity": "boundary_equity.csv"}
    for name, content in payloads.items():
        (tmp_path / names[name]).write_bytes(gzip.compress(content, mtime=0) if name == "equity" else content)
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    (tmp_path / "fingerprints.json").write_text(json.dumps(hashes), encoding="utf-8")
    checks = Checks()
    assert verify_saved_fingerprints(tmp_path, checks, "fixture") == hashes
    assert checks.failed == 0
    (tmp_path / "weights.csv").write_bytes(b"date,weight\n2025-01-01,0.95\n")
    verify_saved_fingerprints(tmp_path, checks, "tampered")
    assert checks.failed == 1


@pytest.mark.parametrize("path", ["../escape", "/home/sunny/model.zip", "C:/key", "training/../../escape"])
def test_remote_paths_and_parent_escapes_are_not_trusted(tmp_path, path):
    with pytest.raises(ValueError):
        relative_file(tmp_path, path)
    assert relative_file(tmp_path, "training/model.zip") == tmp_path / "training/model.zip"


def selection_rows():
    return pd.DataFrame([{"algorithm": algorithm, "profile": profile, "scenario": scenario,
                          "split": "validation", "primary_policy": True,
                          "net_return": .1 if profile == "balanced" else .05,
                          "max_drawdown": -.08 if profile == "balanced" else -.1}
                         for algorithm in ALGORITHMS for profile in PROFILES for scenario in ("base", "stress")])


def test_selection_recomputes_risk_penalty_and_never_uses_test_returns():
    rows = selection_rows()
    first = recompute_selection(rows)
    assert set(first["selected_profiles"].values()) == {"balanced"}
    assert first["primary_validation_leader"] == {"algorithm": "crossq", "profile": "balanced"}
    test = rows.copy()
    test["split"] = "test"
    test.loc[test.profile == "return", "net_return"] = 1000
    assert recompute_selection(pd.concat([rows, test], ignore_index=True)) == first
    manual = np.log1p(.1) - .5 * .3 - 2 * (.3 - .2)
    assert score_candidate({"net_return": .1, "max_drawdown": -.3},
                           {"net_return": .1, "max_drawdown": -.3}) == pytest.approx(manual)


def test_validation_ties_have_fixed_lexicographic_resolution():
    rows = selection_rows()
    rows["net_return"], rows["max_drawdown"] = .05, -.1
    selected = recompute_selection(rows.sample(frac=1, random_state=17))
    assert selected["primary_validation_leader"] == {"algorithm": "crossq", "profile": "balanced"}
    assert set(selected["selected_profiles"].values()) == {"balanced"}


def test_training_calendar_uses_actual_close_volume_dependencies_and_cannot_cross_market_gaps():
    index = pd.date_range("2019-01-01T00:00Z", periods=700 * 24, freq="h")
    frame = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=index)
    data = {"hourly_" + field: frame.copy() for field in ("open", "high", "low", "close", "volume")}
    gap = index[300 * 24 + 13]
    for item in data.values():
        item.loc[gap, "BTCUSDT"] = np.nan
    start, end = index[250 * 24], index[650 * 24]
    calendar = independent_training_calendar(data, start, end, 32)
    assert not calendar["market"][gap.normalize()]
    assert not calendar["features"][gap.normalize() + pd.Timedelta(days=30)]
    assert calendar["features"][gap.normalize() + pd.Timedelta(days=31)]
    assert calendar["features"][gap.normalize() + pd.Timedelta(days=90)]
    assert all(not first <= gap.normalize() < first + pd.Timedelta(days=32) for first in calendar["eligible"])
    shortened = {name: value.loc[:end - pd.Timedelta(hours=1)].copy() for name, value in data.items()}
    second = independent_training_calendar(shortened, start, end, 32)
    assert calendar["eligible"] == second["eligible"]
    bad_close = deepcopy(data)
    bad_close["hourly_close"].loc[gap.normalize() + pd.Timedelta(hours=23), "BTCUSDT"] = np.nan
    third = independent_training_calendar(bad_close, start, end, 32)
    assert not third["features"][gap.normalize() + pd.Timedelta(days=200)]
    assert third["features"][gap.normalize() + pd.Timedelta(days=201)]


def test_complete_saved_result_summary_monthly_weights_and_reward_are_independently_auditable(tmp_path):
    # Integration only: production research output is fed to the independent
    # auditor. The separate hand fixture above supplies the economic oracle.
    from scripts.btceth_rl_env import BTCETHPortfolioEnv
    from scripts.crypto_rl_reporting import save_result, summarize
    index = pd.date_range("2024-01-01T00:00Z", periods=500 * 24, freq="h")
    frame = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=index)
    data = {"hourly_" + field: frame.copy() for field in ("open", "high", "low", "close", "volume")}
    start, end = pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-04T00:00Z")
    e = BTCETHPortfolioEnv(data, start, end, drawdown_penalty=0)
    e.reset()
    e.step_target([.475, .475, .05])
    e.step_hold()
    e.step_hold()
    result = e.get_result()
    result["metadata"].update(model_count=0, baseline="equal_hold")
    save_result(tmp_path, result)
    row, daily, monthly = summarize(result, run_id="validation__base__equal_hold", strategy="equal_hold",
                                     scenario="base", split="validation", start=start, end=end)
    checks = Checks()
    audited = _audit_run(tmp_path, data, {"validation": [start, end], "costs": {"base": {"fee": .001, "slippage": .0002}}},
                         {}, row, monthly, daily, split="validation", scenario="base", algorithm="baseline",
                         profile="baseline", key="equal_hold", checks=checks)
    assert audited["reconstructed"]
    assert checks.failed == 0, checks.failures
    corrupted = monthly.copy()
    corrupted["return"] += .01
    second = Checks()
    _audit_run(tmp_path, data, {"validation": [start, end], "costs": {"base": {"fee": .001, "slippage": .0002}}},
               {}, row, corrupted, daily, split="validation", scenario="base", algorithm="baseline",
               profile="baseline", key="equal_hold", checks=second)
    assert any("month return" in failure for failure in second.failures)


def test_incomplete_directory_cannot_receive_a_false_success(tmp_path):
    from scripts.audit_btceth_rl import audit
    result = audit(tmp_path, tmp_path / "absent-dataset.pkl.gz", allow_validation_only=True)
    assert result["status"] == "failed"
    assert result["passed"] is False and result["test_complete"] is False
    assert result["checks_total"] > 0 and result["failure_count"] > 0
    assert result["run_count"] == 0


def test_cli_keeps_partial_validation_proof_separate_from_final_audit(tmp_path, monkeypatch):
    import sys
    from scripts import audit_btceth_rl as module
    partial = {"status": "partial_test_pending", "passed": True, "test_complete": False,
               "checks": 17, "checks_total": 17, "failure_count": 0, "runs_checked": 110,
               "run_count": 110, "failures": []}
    monkeypatch.setattr(module, "audit", lambda *args, **kwargs: partial)
    monkeypatch.setattr(sys, "argv", ["audit_btceth_rl.py", "--output-dir", str(tmp_path), "--allow-validation-only"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 0
    assert (tmp_path / "validation_audit.json").is_file()
    assert not (tmp_path / "independent_audit.json").exists()
    assert json.loads((tmp_path / "validation_audit.json").read_text(encoding="utf-8"))["test_complete"] is False


def test_cli_final_failure_stays_failed_and_uses_final_artifact_even_with_allow_flag(tmp_path, monkeypatch):
    import sys
    from scripts import audit_btceth_rl as module
    (tmp_path / "test_completion.json").write_text("{}", encoding="utf-8")
    failed = {"status": "failed", "passed": False, "test_complete": False,
              "checks": 200, "checks_total": 200, "failure_count": 1, "runs_checked": 171,
              "run_count": 171, "failures": ["missing one run"]}
    monkeypatch.setattr(module, "audit", lambda *args, **kwargs: failed)
    monkeypatch.setattr(sys, "argv", ["audit_btceth_rl.py", "--output-dir", str(tmp_path), "--allow-validation-only"])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 1
    assert (tmp_path / "independent_audit.json").is_file()
    assert not (tmp_path / "validation_audit.json").exists()
