"""Synthetic data-only risk reader guards; no original annual prices opened."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import btceth_rl_report_risk_controls as risk


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("change", [{"cases": 5}, {"full_repeats": 0}, {"hours_reconstructed": 6 * 7080}, {"formal_rankings_modified": True}])
def test_incomplete_wrong_window_or_mixed_ranking_rejected_before_pickle(tmp_path, monkeypatch, change):
    main, reference = tmp_path / "main", tmp_path / "risk"
    main.mkdir()
    completion = dict(passed=True, cases=6, full_repeats=6, hours_reconstructed=6 * 8760, failures=0, independent_checks=1, formal_rankings_modified=False)
    completion.update(change)
    write(reference / "test/completion.json", completion)
    monkeypatch.setattr(risk.pickle, "load", lambda p: pytest.fail("No price decoding before complete contracts"))
    with pytest.raises(ValueError, match="six separate"):
        risk.load_risk_controls(reference, dataset_path=tmp_path / "not_real_data.pkl.gz", main_output=main)


def test_manifest_exactness_preserves_original_and_allows_separate_local_audit(tmp_path):
    target = tmp_path / "test"; target.mkdir()
    (target / "summary.csv").write_bytes(b"SYNTHETIC")
    write(target / "file_manifest.json", risk.inventory(target))
    expected = risk.risk_manifest(target)
    (target / "independent_audit.local.json").write_bytes(b"LOCAL_REVIEW_ONLY")
    assert risk.risk_manifest(target) == expected
    (target / "summary.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        risk.risk_manifest(target)


def test_manifest_path_escape_is_not_read(tmp_path):
    target = tmp_path / "test"; target.mkdir()
    write(target / "file_manifest.json", [{"path": "../outside", "size": 0, "sha256": "0" * 64}])
    with pytest.raises(ValueError):
        risk.risk_manifest(target)


def covariance_sample():
    index = pd.date_range("2025-01-01T00:00Z", periods=40 * 24, freq="h")
    daily_returns = np.tile([.01, .02, .03], 13)[:39]
    btc = np.r_[100., 100 * np.cumprod(1 + daily_returns)]
    eth = np.r_[50., 50 * np.cumprod(1 + 2 * daily_returns)]
    frame = pd.DataFrame({"BTCUSDT": np.repeat(btc, 24), "ETHUSDT": np.repeat(eth, 24)}, index=index)
    data = {"hourly_" + name: frame.copy() for name in ("open", "high", "low", "close")}
    data["hourly_volume"] = frame * 0 + 100
    return data, daily_returns


def test_independent_covariance_matches_hand_formula_and_is_causal():
    data, returns = covariance_sample()
    day = pd.Timestamp("2025-02-01T00:00Z")
    target = risk.independent_vol_targets(data, [day])[0]
    variance = sum((returns[:30] - returns[:30].mean()) ** 2) * 365 / 29
    gross = min(.95, .20 / (4 / 3 * np.sqrt(variance)))
    np.testing.assert_allclose(target, [gross * 2 / 3, gross / 3, 1 - gross], atol=1e-12)
    future = {key: frame.copy() for key, frame in data.items()}
    for frame in future.values(): frame.loc[frame.index >= day] *= 1000
    truncated = {key: frame.loc[frame.index < day].copy() for key, frame in data.items()}
    np.testing.assert_array_equal(risk.independent_vol_targets(future, [day]), risk.independent_vol_targets(truncated, [day]))


def test_covariance_missing_hour_is_rejected_not_filled():
    data, _ = covariance_sample()
    data["hourly_close"].iloc[100, 0] = np.nan
    with pytest.raises(ValueError, match="complete days"):
        risk.independent_vol_targets(data, [pd.Timestamp("2025-02-01T00:00Z")])


def display_sample():
    days = pd.date_range(*risk.TEST, freq="D")
    values = np.linspace(10000., 11000., len(days))
    hours = pd.date_range(*risk.TEST, freq="h", inclusive="left")
    nav = pd.Series(np.repeat(values[1:], 24), index=hours)
    dd = pd.Series(np.zeros(len(hours)), index=hours)
    month_values = nav.resample("D").last().resample("ME").last()
    months = month_values.pct_change(); months.iloc[0] = month_values.iloc[0] / 10000 - 1
    run_id = "test__base__btc25_cash75_hold"
    history = pd.DataFrame(dict(run_id=run_id, timestamp=days, equity=values, net_return=values / 10000 - 1, drawdown=0.))
    monthly = pd.DataFrame(dict(run_id=run_id, month=month_values.index.strftime("%Y-%m"), **{"return": months.to_numpy()}))
    rebuilt = {"equity": pd.DataFrame({"equity": nav}), "drawdown": dd, "monthly": dict(zip(monthly.month, monthly["return"]))}
    return history, monthly, SimpleNamespace(run_id=run_id, net_return=.10), rebuilt


def test_daily_curve_and_monthly_values_are_checked_against_same_ledger():
    history, monthly, row, rebuilt = display_sample()
    risk.verify_curve_months(history, monthly, row, rebuilt)
    history.loc[len(history) - 1, "equity"] += 100
    with pytest.raises(ValueError, match="display curve"):
        risk.verify_curve_months(history, monthly, row, rebuilt)


def test_monthly_product_and_month_identity_cannot_disagree_with_return():
    history, monthly, row, rebuilt = display_sample()
    row.net_return = .30
    with pytest.raises(ValueError, match="product"):
        risk.verify_curve_months(history, monthly, row, rebuilt)
    monthly.loc[0, "month"] = monthly.loc[1, "month"]
    with pytest.raises(ValueError, match="coverage"):
        risk.verify_curve_months(history, monthly, row, rebuilt)


def test_reader_never_imports_goal_selector_environment_or_model():
    tree = ast.parse(Path(risk.__file__).read_text(encoding="utf-8"))
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(any(token in name for token in ("btceth_rl_goal_candidate", "btceth_rl_env", "backtest_btceth_rl", "stable_baselines")) for name in imports)


@pytest.mark.parametrize("change", [None, "financial_rules_changed", "original_validator_restored", "main_marker_sha256"])
def test_shape_adapter_provenance_is_pinned_and_never_changes_financial_rules(tmp_path, change):
    ref, main, source = tmp_path / "reference", tmp_path / "main", tmp_path / "repo"
    directory = ref / "adapter_attempts/compat_20261030"
    write(main / "test_started.json", {"synthetic": True})
    write(ref / "test_protocol.json", {"synthetic": True})
    write(ref / "test_protocol_receipt.json", {"synthetic": True})
    source_map = {}
    for name in ("scripts/btceth_rl_risk_controls.py", "scripts/btceth_rl_goal_candidate.py", "scripts/btceth_rl_risk_controls_adapter.py"):
        for root in (source, directory / "sources"):
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(("SYNTHETIC " + name).encode())
        source_map[name] = risk.sha(source / name)
    write(directory / "original_failure/process.json", {"synthetic": True})
    (directory / "original_failure/run.log").write_bytes(b"SYNTHETIC ORIGINAL FAILURE")
    p = dict(financial_rules_changed=False, annual_data_decoded_during_preparation=False,
        target_absent_at_preparation=True, source_sha256=source_map,
        original_risk_sha256=source_map["scripts/btceth_rl_risk_controls.py"], original_goal_sha256=source_map["scripts/btceth_rl_goal_candidate.py"],
        main_marker_sha256=risk.sha(main / "test_started.json"), risk_protocol_sha256=risk.sha(ref / "test_protocol.json"),
        risk_receipt_sha256=risk.sha(ref / "test_protocol_receipt.json"),
        failed_log_sha256=risk.sha(directory / "original_failure/run.log"), failed_process_sha256=risk.sha(directory / "original_failure/process.json"),
        original_reference_files=[])
    if change == "financial_rules_changed": p[change] = True
    if change == "main_marker_sha256": p[change] = "0" * 64
    write(directory / "protocol.json", p)
    write(directory / "preparation.json", dict(passed=True, test_data_loaded=False, protocol_sha256=risk.sha(directory / "protocol.json")))
    execution = dict(passed=True, exit_code=0, protocol_sha256=risk.sha(directory / "protocol.json"), main_marker_sha256=risk.sha(main / "test_started.json"),
        original_validator_restored=True, original_validation_unchanged=True, cuda_initialized=False, source_sha256=source_map,
        result={"passed": True, "formal_rankings_modified": False, "cases": 6, "full_repeats": 6})
    if change == "original_validator_restored": execution[change] = False
    write(directory / "execution.json", execution)
    if change:
        with pytest.raises(ValueError, match="adapter provenance"):
            risk.verify_adapter_provenance(ref, main, source)
    else:
        values = risk.verify_adapter_provenance(ref, main, source)
        assert values[(directory / "protocol.json").resolve().as_posix()] == risk.sha(directory / "protocol.json")
        assert (source / "scripts/btceth_rl_risk_controls_adapter.py").resolve().as_posix() in values


@pytest.mark.parametrize("fails", [False, True])
def test_audit_source_import_does_not_write_cache_and_restores_setting(tmp_path, monkeypatch, fails):
    source = tmp_path / "auditor.py"
    source.write_text("raise RuntimeError('synthetic')\n" if fails else "answer=42\n", encoding="utf-8")
    monkeypatch.setattr(risk.sys, "dont_write_bytecode", False)
    if fails:
        with pytest.raises(RuntimeError, match="synthetic"):
            risk.load_auditor(source)
    else:
        assert risk.load_auditor(source).answer == 42
    assert risk.sys.dont_write_bytecode is False
    assert not (tmp_path / "__pycache__").exists()
