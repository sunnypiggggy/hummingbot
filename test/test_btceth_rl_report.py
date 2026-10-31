"""Reject misleading or incomplete BTC/ETH reports; no HTML, inference or network.

Synthetic rows exist only in pytest fixtures. A passed bookkeeping audit does
not itself imply that the validation-selected strategy meets performance gates.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import btceth_rl_report_app as report
from scripts.audit_btceth_rl import _research_target_result, recompute_selection


ALGOS = ("recurrent_ppo", "tqc", "crossq", "sac")
BASELINES = ("cash", "btc_hold", "eth_hold", "equal_hold", "equal_rebalance", "sma200_trend", "momentum180")
PROFILES = ("return", "balanced")
BOUNDS = {"validation": ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"),
          "test": ("2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z")}
ROOT_INPUTS = ("summary.csv", "daily_equity.csv", "monthly_returns.csv", "frozen_spec.json",
               "validation_selection.json", "training_completion.json", "validation_completion.json",
               "test_completion.json", "test_started.json")
GATE_KEYS = {
    "base_return_at_least_5pct", "base_max_drawdown_at_most_20pct", "stress_return_nonnegative",
    "beats_equal_hold_return_and_drawdown", "at_least_three_positive_training_seeds",
}


def selection_stub():
    return {"selected_profiles": {name: "balanced" for name in ALGOS},
            "primary_validation_leader": {"algorithm": "crossq", "profile": "balanced"}}


@pytest.fixture(scope="module")
def synthetic_summary():
    rows = []
    def add(split, scenario, algorithm, profile, seed, strategy, run_id, primary):
        if algorithm == "baseline":
            result = 0. if strategy == "cash" else .02 if strategy not in ("sma200_trend", "momentum180") else .03
            dd = 0. if strategy == "cash" else -.30 if strategy not in ("sma200_trend", "momentum180") else -.20
        else:
            result = (.15 if algorithm == "crossq" else .08) if split == "validation" else (.12 if algorithm == "tqc" else .06)
            result -= .04 if profile == "return" else 0.
            result -= .02 if scenario == "stress" else 0.
            dd = -.10
            if split == "test" and scenario == "base" and algorithm == "crossq" and seed is not None:
                result = (.01, -.02, .03, .02, -.01)[seed]
        start, end = BOUNDS[split]
        rows.append({"run_id": run_id, "strategy": strategy, "label": "SYNTHETIC_TEST_FIXTURE_ONLY",
                     "algorithm": algorithm, "profile": profile, "seed": seed, "primary_policy": primary,
                     "scenario": scenario, "split": split, "start": start, "end_exclusive": end,
                     "initial_capital": 10000., "final_equity": 10000. * (1 + result), "net_return": result,
                     "max_drawdown": dd, "drawdown_grain": "1h_close",
                     "observed_hours": (pd.Timestamp(end) - pd.Timestamp(start)).days * 24,
                     "fees": 0. if strategy == "cash" else 10., "average_exposure": 0. if strategy == "cash" else .50,
                     "slippage_cost": 0. if strategy == "cash" else 2., "turnover": 1.,
                     "net_liquidation_return": result - .0006, "sharpe_daily": None, "calmar": None})
    for split in BOUNDS:
        for scenario in ("base", "stress"):
            for algorithm in ALGOS:
                for profile in PROFILES if split == "validation" else ("balanced",):
                    for seed in (*range(5), None):
                        key = "ensemble" if seed is None else str(seed)
                        add(split, scenario, algorithm, profile, seed, algorithm,
                            f"{split}__{scenario}__{algorithm}__{profile}__{key}", seed is None)
            for name in BASELINES:
                add(split, scenario, "baseline", "baseline", None, name, f"{split}__{scenario}__{name}", False)
    frame = pd.DataFrame(rows)
    assert len(frame) == 172 and frame.split.eq("validation").sum() == 110 and frame.split.eq("test").sum() == 62
    return frame


def curves_for(summary):
    curves, monthly = [], []
    for row in summary.itertuples(index=False):
        dates = pd.date_range(row.start, row.end_exclusive, freq="D")
        # A closed-form early drawdown followed by recovery. These are fixture
        # marks, not claimed market results or policy inference.
        values = np.r_[10000., np.linspace(10000. * (1 + row.max_drawdown), row.final_equity, len(dates) - 1)]
        dd = values / np.maximum.accumulate(values) - 1
        curves.extend({"run_id": row.run_id, "timestamp": stamp.isoformat(), "equity": value,
                       "net_return": value / 10000. - 1, "drawdown": drawdown, "seed": row.seed,
                       "split": row.split, "scenario": row.scenario, "strategy": row.strategy}
                      for stamp, value, drawdown in zip(dates, values, dd))
        day_values = pd.Series(values[1:], index=dates[:-1])
        month_ends = day_values.groupby(day_values.index.strftime("%Y-%m"), sort=True).last()
        last = 10000.
        for month, value in month_ends.items():
            monthly.append({"run_id": row.run_id, "month": month, "return": value / last - 1,
                            "seed": row.seed, "split": row.split, "scenario": row.scenario, "strategy": row.strategy})
            last = value
    return pd.DataFrame(curves), pd.DataFrame(monthly)


@pytest.fixture
def one_run(synthetic_summary):
    frame = synthetic_summary.loc[synthetic_summary.run_id.eq("test__base__crossq__balanced__ensemble")].copy()
    return frame, *curves_for(frame)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.clean_json(value), ensure_ascii=False, allow_nan=False), encoding="utf-8")


def gate_only_root(tmp_path):
    for name in ROOT_INPUTS:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    audit = {"passed": True, "status": "passed", "test_complete": True, "failure_count": 0, "failures": [],
             "run_count": 172, "runs_checked": 172,
             "input_sha256": {name: report.sha(tmp_path / name) for name in ROOT_INPUTS}}
    write_json(tmp_path / "independent_audit.json", audit)
    return audit


def test_missing_test_completion_cannot_create_snapshot_or_report(tmp_path):
    gate_only_root(tmp_path)
    (tmp_path / "test_completion.json").unlink()
    with pytest.raises(FileNotFoundError, match="test_completion"):
        report.make_reviewed_snapshot(tmp_path)
    assert not (tmp_path / "report").exists()
    assert not list(tmp_path.glob("*.html"))


@pytest.mark.parametrize("kwargs", [
    {"band_validation_root": "validation"},
    {"band_audit_path": "audit.json"},
    {"band_annual_root": "annual"},
    {"band_annual_root": "annual", "band_validation_root": "validation"},
])
def test_partial_followup_inputs_cannot_start_report_reads(tmp_path, kwargs):
    with pytest.raises(ValueError, match="execution-band|Execution-band"):
        report.make_reviewed_snapshot(tmp_path, **kwargs)
    assert not (tmp_path / "report").exists()
    assert not list(tmp_path.glob("*.html"))


@pytest.mark.parametrize("change", [
    {"status": "partial_test_pending", "test_complete": False},
    {"passed": False}, {"failure_count": 1, "failures": ["ledger mismatch"]}, {"run_count": 110},
])
def test_validation_only_or_failed_audit_is_rejected(tmp_path, change):
    audit = gate_only_root(tmp_path)
    audit.update(change)
    path = tmp_path / "validation_audit.json"
    write_json(path, audit)
    with pytest.raises(ValueError, match="final passed 172-run"):
        report.make_reviewed_snapshot(tmp_path, audit_path=path)


def test_changed_audited_file_bytes_are_rejected_before_reading_results(tmp_path):
    gate_only_root(tmp_path)
    (tmp_path / "summary.csv").write_text("changed result", encoding="utf-8")
    with pytest.raises(ValueError, match="Audited evidence changed: summary.csv"):
        report.make_reviewed_snapshot(tmp_path)


def test_changed_audit_input_hash_is_rejected(tmp_path):
    audit = gate_only_root(tmp_path)
    audit["input_sha256"]["monthly_returns.csv"] = "0" * 64
    write_json(tmp_path / "independent_audit.json", audit)
    with pytest.raises(ValueError, match="Audited evidence changed: monthly_returns.csv"):
        report.make_reviewed_snapshot(tmp_path)


def test_missing_audited_input_cannot_be_claimed_complete(tmp_path):
    audit = gate_only_root(tmp_path)
    del audit["input_sha256"]["test_started.json"]
    write_json(tmp_path / "independent_audit.json", audit)
    with pytest.raises(ValueError, match="bind every root input"):
        report.make_reviewed_snapshot(tmp_path)


def test_synthetic_172_run_set_is_accepted_without_dropping_seeds(synthetic_summary):
    ids = report._verify_summary(synthetic_summary, selection_stub())
    assert len(ids) == 172
    assert sum(identity[-1] for identity in ids.values()) == 24


@pytest.mark.parametrize("field,value,message", [
    ("start", "2025-10-24T00:00:00Z", "window mismatch"),
    ("end_exclusive", "2026-10-22T00:00:00Z", "window mismatch"),
    ("initial_capital", 50000., "10000 USDT"),
    ("seed", 4., "Seed or actual ensemble"),
    ("primary_policy", True, "Seed or actual ensemble"),
    ("observed_hours", 8759, "Complete hourly"),
    ("max_drawdown", np.nan, "Missing primary financial"),
])
def test_wrong_window_capital_seed_or_hour_coverage_is_rejected(synthetic_summary, field, value, message):
    changed = synthetic_summary.copy(deep=True)
    index = changed.index[changed.run_id.eq("test__base__crossq__balanced__0")][0]
    changed.loc[index, field] = value
    with pytest.raises(ValueError, match=message):
        report._verify_summary(changed, selection_stub())


def test_one_missing_seed_and_one_duplicate_cannot_pass_same_row_count(synthetic_summary):
    changed = synthetic_summary.copy(deep=True)
    changed.iloc[-1] = changed.iloc[0]
    assert len(changed) == 172
    with pytest.raises(ValueError, match="exactly 110 validation and 62 test"):
        report._verify_summary(changed, selection_stub())


def test_curve_and_monthly_closed_form_fixture_reconcile(one_run):
    summary, curves, months = one_run
    report._verify_history(curves, months, summary)
    assert len(curves) == 366
    assert np.prod(1 + months["return"]) - 1 == pytest.approx(.06)


@pytest.mark.parametrize("change,message", [
    ("final_equity", "Curve final equity"),
    ("initial_equity", "Curve initial equity"),
    ("plotted_return", "Plotted curve returns"),
    ("missing_boundary", "initial point and every daily boundary"),
    ("monthly_return", "Monthly returns do not reconcile"),
    ("monthly_duplicate", "Monthly window coverage"),
])
def test_misleading_curves_or_monthly_returns_are_rejected(one_run, change, message):
    summary, curves, months = one_run
    if change == "final_equity":
        curves.loc[curves.index[-1], "equity"] += 50.
        curves.loc[curves.index[-1], "net_return"] = curves.equity.iloc[-1] / 10000. - 1
    elif change == "initial_equity":
        curves.loc[curves.index[0], "equity"] = 50000.
    elif change == "plotted_return":
        curves.loc[curves.index[-1], "net_return"] += .10
    elif change == "missing_boundary":
        curves = curves.drop(curves.index[12])
    elif change == "monthly_return":
        months.loc[months.index[-1], "return"] += .01
    elif change == "monthly_duplicate":
        months.loc[months.index[-1], "month"] = months.month.iloc[0]
    with pytest.raises(ValueError, match=message):
        report._verify_history(curves, months, summary)


def test_research_gate_names_and_values_match_current_independent_auditor(synthetic_summary):
    selected = selection_stub()
    audited = _research_target_result(synthetic_summary, selected)
    assert set(audited["gates"]) == GATE_KEYS
    assert audited["gates"] == {name: True for name in GATE_KEYS}
    assert audited["all_predeclared_primary_targets_met"] is True
    rows = report._research_targets(synthetic_summary, selected, {"research_targets": audited})
    assert {row["gate"]: row["met"] for row in rows if row["category"] == "事前研究条件"} == audited["gates"]
    assert {row["gate"]: row["met"] for row in rows if row["category"] == "简单规则对照"} == audited["dominates_simple_trend_controls"]


@pytest.mark.parametrize("change,gate,expected", [
    ("return_below", "base_return_at_least_5pct", False),
    ("return_boundary", "base_return_at_least_5pct", True),
    ("drawdown_above", "base_max_drawdown_at_most_20pct", False),
    ("drawdown_boundary", "base_max_drawdown_at_most_20pct", True),
    ("negative_stress", "stress_return_nonnegative", False),
    ("zero_stress", "stress_return_nonnegative", True),
    ("two_positive_seeds", "at_least_three_positive_training_seeds", False),
    ("tied_hold_return", "beats_equal_hold_return_and_drawdown", False),
])
def test_gate_boundaries_use_exact_values_and_do_not_round_to_pass(synthetic_summary, change, gate, expected):
    frame = synthetic_summary.copy(deep=True)
    primary = frame.run_id.eq("test__base__crossq__balanced__ensemble")
    if change.startswith("return_"):
        frame.loc[primary, "net_return"] = .05 - 1e-12 if change == "return_below" else .05
    elif change.startswith("drawdown_"):
        frame.loc[primary, "max_drawdown"] = -.20 - 1e-12 if change == "drawdown_above" else -.20
    elif change in ("negative_stress", "zero_stress"):
        frame.loc[frame.run_id.eq("test__stress__crossq__balanced__ensemble"), "net_return"] = -1e-12 if change == "negative_stress" else 0.
    elif change == "two_positive_seeds":
        frame.loc[frame.run_id.eq("test__base__crossq__balanced__0"), "net_return"] = 0.
    else:
        frame.loc[frame.run_id.eq("test__base__equal_hold"), "net_return"] = float(frame.loc[primary, "net_return"].iloc[0])
    audit = _research_target_result(frame, selection_stub())
    assert audit["gates"][gate] is expected
    rows = report._research_targets(frame, selection_stub(), {"research_targets": audit})
    assert next(row for row in rows if row["gate"] == gate)["met"] is expected


@pytest.mark.parametrize("gate", sorted(GATE_KEYS))
def test_individual_audit_gate_cannot_be_forged(synthetic_summary, gate):
    audited = _research_target_result(synthetic_summary, selection_stub())
    audited["gates"][gate] = not audited["gates"][gate]
    with pytest.raises(ValueError, match="performance checks differ"):
        report._research_targets(synthetic_summary, selection_stub(), {"research_targets": audited})


@pytest.mark.parametrize("flag", ["all_predeclared_primary_targets_met", "stronger_claim_supported"])
def test_aggregate_success_flags_cannot_disagree_with_gate_rows(synthetic_summary, flag):
    audited = _research_target_result(synthetic_summary, selection_stub())
    audited[flag] = False
    with pytest.raises(ValueError, match="Aggregate research target|Stronger performance claim"):
        report._research_targets(synthetic_summary, selection_stub(), {"research_targets": audited})


def test_false_aggregate_pass_is_rejected_even_when_individual_failures_are_preserved(synthetic_summary):
    changed = synthetic_summary.copy(deep=True)
    changed.loc[changed.run_id.eq("test__base__crossq__balanced__ensemble"), "net_return"] = -.03
    audited = _research_target_result(changed, selection_stub())
    assert audited["all_predeclared_primary_targets_met"] is False
    audited["all_predeclared_primary_targets_met"] = True
    with pytest.raises(ValueError, match="Aggregate research target"):
        report._research_targets(changed, selection_stub(), {"research_targets": audited})


@pytest.mark.parametrize("metric", ["base_net_return", "base_max_drawdown", "stress_net_return"])
def test_audit_headline_metric_cannot_differ_from_actual_summary(synthetic_summary, metric):
    audited = _research_target_result(synthetic_summary, selection_stub())
    audited[metric] += .01
    with pytest.raises(ValueError, match="metric differs"):
        report._research_targets(synthetic_summary, selection_stub(), {"research_targets": audited})


@pytest.mark.parametrize("source,expected", [
    (r"H:\PycharmProjects\hummingbot\report app", "/mnt/h/PycharmProjects/hummingbot/report app"),
    ("C:/Users/sunny/Research", "/mnt/c/Users/sunny/Research"),
    ("C:/", "/mnt/c/"),
    ("/home/sunny/tools/node", "/home/sunny/tools/node"),
])
def test_wsl_paths_use_posix_separators(source, expected):
    assert report.wsl_path(source) == expected
    assert "\\" not in report.wsl_path(source)


@pytest.mark.parametrize("source", [r"C:relative\node", "relative/path", r"\\server\share\node", r"/mnt/c/Users\sunny/node"])
def test_ambiguous_or_mixed_wsl_paths_are_rejected(source):
    with pytest.raises(ValueError, match="drive-absolute|forward separators"):
        report.wsl_path(source)


@pytest.mark.skipif(os.name != "nt", reason="Windows host mapping is platform-specific")
@pytest.mark.parametrize("source,expected", [("/mnt/h/", "H:/"), ("/mnt/h", "H:/"), ("/mnt/c/Users/sunny", "C:/Users/sunny")])
def test_wsl_returned_paths_map_to_windows_including_drive_root(source, expected):
    assert report.host_path(source) == Path(expected)


def test_wsl_sdk_command_converts_both_path_objects_and_strings(tmp_path, monkeypatch):
    plugin = tmp_path / "official sdk"
    (plugin / "scripts").mkdir(parents=True)
    (plugin / "scripts/data-app.mjs").write_text("// test SDK entry marker", encoding="utf-8")
    observed = []
    def run(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"test-only"}\n', stderr="")
    monkeypatch.setattr(report.subprocess, "run", run)
    runtime = report.DataRuntime(plugin_root=plugin, build_mode="wsl", wsl_node="/home/sunny/research-tools/node")
    runtime.run("data-app.mjs", "build", "--project-dir", tmp_path / "report app", "--output", r"H:\Research\report app\index.html", cwd=tmp_path)
    command, kwargs = observed[0]
    assert command[:6] == ["wsl.exe", "--distribution", "Ubuntu-26.04", "--exec", "/home/sunny/research-tools/node", report.wsl_path(plugin / "scripts/data-app.mjs")]
    assert command[command.index("--project-dir") + 1] == report.wsl_path(tmp_path / "report app")
    assert command[command.index("--output") + 1] == "/mnt/h/Research/report app/index.html"
    assert all("\\" not in value for value in command)
    assert kwargs["cwd"] == tmp_path


def test_wsl_mode_refuses_a_windows_executable_path(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/data-app.mjs").write_text("// marker", encoding="utf-8")
    with pytest.raises(ValueError, match="Linux absolute path"):
        report.DataRuntime(plugin_root=tmp_path, build_mode="wsl", wsl_node=r"C:\tools\node.exe")


@pytest.fixture
def complete_study_root(tmp_path, synthetic_summary):
    spec = {"train": ["2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], **{name: list(value) for name, value in BOUNDS.items()},
            "algorithms": list(ALGOS), "baseline_names": list(BASELINES), "seeds": list(range(5)),
            "capital": 10000., "train_episode_days": 32, "timesteps_per_model": 32768,
            "dataset_sha256": hashlib.sha256(b"SYNTHETIC_TEST_FIXTURE_ONLY").hexdigest()}
    write_json(tmp_path / "frozen_spec.json", spec)
    spec_hash = report.sha(tmp_path / "frozen_spec.json")
    training, training_hashes = [], {}
    for algorithm in ALGOS:
        for profile in PROFILES:
            for seed in range(5):
                relative = f"training/{algorithm}__{profile}__seed{seed}"
                folder = tmp_path / relative
                folder.mkdir(parents=True)
                (folder / "model.zip").write_bytes(b"SYNTHETIC_TEST_FIXTURE_NOT_A_TRAINED_CHECKPOINT")
                record = {"algorithm": algorithm, "profile": profile, "seed": seed,
                          "actual_timesteps": 32768, "elapsed_seconds": 1., "model_sha256": report.sha(folder / "model.zip"),
                          "parameter_sha256": "a" * 64, "device": {"visible_gpu_uuid": "test-fixture-only", "name": "test-fixture-only"},
                          "training_window_quality": {"requested_train_days": 1827, "episode_days": 32,
                                                      "covered_training_days": 1403, "eligible_windows": 1217, "excluded_windows": 579}}
                write_json(folder / "record.json", record)
                training.append(record)
                for name in ("model.zip", "record.json"):
                    training_hashes[relative + "/" + name] = report.sha(folder / name)
    write_json(tmp_path / "training_completion.json", {"records": training})
    training_hashes["training_completion.json"] = report.sha(tmp_path / "training_completion.json")
    selected = recompute_selection(synthetic_summary)
    assert selected["primary_validation_leader"] == selection_stub()["primary_validation_leader"]
    selected.update(frozen_spec_sha256=spec_hash, training_source_sha256=training_hashes)
    write_json(tmp_path / "validation_selection.json", selected)
    for name in ("validation_completion.json", "test_completion.json", "test_started.json"):
        write_json(tmp_path / name, {"synthetic_fixture_only": True})
    curves, months = curves_for(synthetic_summary)
    for name, frame in (("summary.csv", synthetic_summary), ("daily_equity.csv", curves), ("monthly_returns.csv", months)):
        frame.to_csv(tmp_path / name, index=False, float_format="%.17g")
    audited = {"passed": True, "status": "passed", "test_complete": True, "failure_count": 0, "failures": [],
               "run_count": 172, "runs_checked": 172, "checks_total": 1, "frozen_spec_sha256": spec_hash,
               "input_sha256": {name: report.sha(tmp_path / name) for name in ROOT_INPUTS},
               "runs": {name: {} for name in synthetic_summary.run_id}, "audit_completed_at": "2026-10-30T00:00:00Z",
               "training_calendar": {"requested_train_days": 1827, "episode_days": 32, "covered_days": 1403,
                                     "eligible_windows": 1217, "excluded_windows": 579},
               "research_targets": _research_target_result(synthetic_summary, selected)}
    write_json(tmp_path / "independent_audit.json", audited)
    return tmp_path


def test_complete_fixture_builds_only_an_in_memory_snapshot_and_keeps_validation_leader(complete_study_root, synthetic_summary):
    snapshot, evidence = report.make_reviewed_snapshot(complete_study_root)
    assert snapshot["report"]["primaryLeader"] == {"algorithm": "crossq", "profile": "balanced"}
    assert snapshot["report"]["scope"] == "fixed_btc_eth_usdt"
    assert snapshot["report"]["researchTargets"]["all_predeclared_primary_targets_met"] is True
    assert len(snapshot["queries"]["strategy_summary"]["rows"]) == 172
    assert len(snapshot["queries"]["training_records"]["rows"]) == 40
    assert len(snapshot["queries"]["seed_statistics"]["rows"]) == 24
    assert evidence["numeric_results_changed"] is False
    # TQC has the best synthetic test return. It must not replace the earlier
    # validation choice, and nothing here writes an HTML or reviewed report.
    assert synthetic_summary.loc[synthetic_summary.run_id.eq("test__base__tqc__balanced__ensemble"), "net_return"].iloc[0] > snapshot["report"]["primaryBaseReturn"]
    assert not (complete_study_root / "report").exists()
    assert not list(complete_study_root.rglob("*.html"))


def test_explicit_local_audit_provenance_keeps_original_audit_bytes_and_hash_separate(complete_study_root):
    root = complete_study_root
    original_path = root / "independent_audit.json"
    original_bytes = original_path.read_bytes()
    original_sha = report.sha(original_path)
    local_path = root / "independent_audit.local.json"
    local = json.loads(original_bytes)
    local["audit_completed_at"] = "2026-10-30T01:00:00Z"
    local["scope"] = "Synthetic test fixture representing a separate local re-audit"
    write_json(local_path, local)
    local_sha = report.sha(local_path)
    assert local_sha != original_sha
    snapshot, evidence = report.make_reviewed_snapshot(root, audit_path=local_path)
    identity = snapshot["report"]["auditEvidence"]
    assert identity["primary_audit_file"] == "independent_audit.local.json"
    assert identity["primary_audit_sha256"] == local_sha
    assert identity["primary_audit_external_to_study"] is False
    assert identity["retained_original_audit_file"] == "independent_audit.json"
    assert identity["retained_original_audit_sha256"] == original_sha
    assert evidence["audit_source_file"] == "independent_audit.local.json"
    assert evidence["direct_input_sha256"]["independent_audit.local.json"] == local_sha
    assert evidence["direct_input_sha256"]["independent_audit.json"] == original_sha
    for query in snapshot["queries"].values():
        assert "independent_audit.local.json" in query["source"]["files"]
        assert query["source"]["fileHashes"]["independent_audit.local.json"] == local_sha
        assert query["source"]["auditEvidence"]["primary_audit_file"] == "independent_audit.local.json"
    audit_source = snapshot["queries"]["audit_evidence"]["source"]
    assert "independent_audit.json" in audit_source["files"]
    assert audit_source["fileHashes"]["independent_audit.json"] == original_sha
    assert "independent_audit.json" not in snapshot["queries"]["research_targets"]["source"]["files"]
    assert snapshot["queries"]["audit_evidence"]["rows"][0]["primary_audit_sha256"] == local_sha
    assert original_path.read_bytes() == original_bytes
    assert not (root / "report").exists()
    assert not list(root.rglob("*.html"))


def test_explicit_external_audit_is_named_accurately_without_redirecting_source_root(complete_study_root):
    root = complete_study_root
    original_sha = report.sha(root / "independent_audit.json")
    external = root.parent / (root.name + "-explicit-local-audit.json")
    audited = report.read_json(root / "independent_audit.json")
    audited["audit_completed_at"] = "2026-10-30T02:00:00Z"
    write_json(external, audited)
    snapshot, evidence = report.make_reviewed_snapshot(root, audit_path=external)
    name = external.resolve().as_posix()
    assert evidence["audit_source_file"] == name
    assert snapshot["report"]["auditEvidence"]["primary_audit_external_to_study"] is True
    for query in snapshot["queries"].values():
        assert name in query["source"]["files"]
        assert query["source"]["fileHashes"][name] == report.sha(external)
    assert evidence["direct_input_sha256"]["independent_audit.json"] == original_sha
    assert evidence["direct_input_sha256"]["summary.csv"] == report.sha(root / "summary.csv")
    assert not (root / "report").exists()


def test_external_audit_cannot_use_input_hashes_to_escape_the_study_root(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    audit = gate_only_root(root)
    escaped = tmp_path / "outside.csv"
    escaped.write_bytes(b"not a study input")
    audit["input_sha256"]["../outside.csv"] = report.sha(escaped)
    external_audit = tmp_path / "explicit-audit.json"
    write_json(external_audit, audit)
    with pytest.raises(ValueError, match="root-relative"):
        report.make_reviewed_snapshot(root, audit_path=external_audit)


@pytest.mark.parametrize("collision", [False, True])
def test_followup_queries_cannot_replace_first_study_results(complete_study_root, monkeypatch, collision):
    from scripts import btceth_rl_report_execution_band as followup_reader
    from scripts import btceth_rl_report_extension as extension_reader

    original, _ = report.make_reviewed_snapshot(complete_study_root)
    query_id = "strategy_summary" if collision else "execution_band_summary"
    # Reader validation has its own financial fixtures; this isolates the app's
    # merge boundary and never creates market results or a delivered report.
    payload = dict(queries={query_id: {"rows": [{"synthetic_fixture_only": True}], "source": {}}},
                   targets={"goal_performance_achieved": False}, selection={}, audit={"passed": True},
                   query_run_count=38, external_input_sha256={})
    monkeypatch.setattr(followup_reader, "load_execution_band", lambda *args, **kwargs: deepcopy(payload))
    monkeypatch.setattr(extension_reader, "augment_snapshot", lambda snapshot, evidence, **kwargs: (snapshot, evidence))
    kwargs = dict(extension_root=complete_study_root / "cap-fixture",
                  band_annual_root=complete_study_root / "annual-fixture",
                  band_validation_root=complete_study_root / "validation-fixture")
    if collision:
        with pytest.raises(ValueError, match="cannot replace"):
            report.make_reviewed_snapshot(complete_study_root, **kwargs)
    else:
        updated, evidence = report.make_reviewed_snapshot(complete_study_root, **kwargs)
        for identity, query in original["queries"].items():
            assert updated["queries"][identity] == query
        assert updated["report"]["primaryLeader"] == original["report"]["primaryLeader"]
        assert updated["report"]["researchTargets"] == original["report"]["researchTargets"]
        assert updated["report"]["executionBand"]["targets"]["goal_performance_achieved"] is False
        assert updated["report"]["executionBand"]["freshHoldoutSuccess"] is False
        assert evidence["runs"] == 172
        assert evidence["additional_unique_execution_band_runs"] == 12
    assert not (complete_study_root / "report").exists()


@pytest.mark.parametrize("collision", [False, True])
def test_td3_queries_preserve_original_results_and_failed_goal(complete_study_root, monkeypatch, collision):
    from scripts import btceth_rl_report_td3 as td3_reader

    original, _ = report.make_reviewed_snapshot(complete_study_root)
    query_id = "strategy_summary" if collision else "td3_summary"
    payload = dict(queries={query_id: {"rows": [{"synthetic_fixture_only": True}], "source": {}}},
                   targets={"goal_performance_achieved": False}, selection={"candidate": {"family": "raw", "profile": "return"}},
                   audit={"passed": True}, external_input_sha256={})
    monkeypatch.setattr(td3_reader, "load_td3", lambda *args, **kwargs: deepcopy(payload))
    if collision:
        with pytest.raises(ValueError, match="TD3 queries cannot replace"):
            report.make_reviewed_snapshot(complete_study_root, td3_root=complete_study_root / "td3-fixture")
    else:
        updated, evidence = report.make_reviewed_snapshot(complete_study_root, td3_root=complete_study_root / "td3-fixture")
        for identity, query in original["queries"].items():
            assert updated["queries"][identity] == query
        assert updated["report"]["primaryLeader"] == original["report"]["primaryLeader"]
        assert updated["report"]["researchTargets"] == original["report"]["researchTargets"]
        assert updated["report"]["td3Followup"]["targets"]["goal_performance_achieved"] is False
        assert updated["report"]["td3Followup"]["freshHoldoutSuccess"] is False
        assert updated["report"]["td3Followup"]["trainingBudgetMultiple"] == 2
        assert evidence["runs"] == 172
        assert evidence["td3_scope_runs"] == 26
        assert evidence["additional_unique_td3_runs"] == 12
    assert not (complete_study_root / "report").exists()


def test_td3_audit_cannot_redirect_to_an_unspecified_study(tmp_path):
    with pytest.raises(ValueError, match="explicit TD3 study root"):
        report.make_reviewed_snapshot(tmp_path, td3_audit_path=tmp_path / "audit.json")


def test_td3_incomplete_reader_cannot_create_a_reviewed_report(complete_study_root, monkeypatch):
    from scripts import btceth_rl_report_td3 as td3_reader

    def reject(*args, **kwargs):
        raise ValueError("Complete independent TD3 annual26 required")
    monkeypatch.setattr(td3_reader, "load_td3", reject)
    with pytest.raises(ValueError, match="annual26 required"):
        report.make_reviewed_snapshot(complete_study_root, td3_root=complete_study_root / "td3-fixture")
    assert not (complete_study_root / "report").exists()


def test_td3_cannot_replace_the_identity_of_an_existing_report_source(complete_study_root, monkeypatch):
    from scripts import btceth_rl_report_extension as extension_reader
    from scripts import btceth_rl_report_td3 as td3_reader

    existing = (complete_study_root / "frozen_spec.json").resolve().as_posix()
    expected = report.sha(Path(existing))
    def earlier(snapshot, evidence, **kwargs):
        evidence["external_input_sha256"] = {existing: expected}
        return snapshot, evidence
    monkeypatch.setattr(extension_reader, "augment_snapshot", earlier)
    payload = dict(queries={"td3_summary": {"rows": [], "source": {}}}, targets={}, selection={}, audit={},
                   external_input_sha256={existing: "0" * 64})
    monkeypatch.setattr(td3_reader, "load_td3", lambda *args, **kwargs: deepcopy(payload))
    with pytest.raises(ValueError, match="cannot replace an existing source identity"):
        report.make_reviewed_snapshot(complete_study_root,
            extension_root=complete_study_root / "cap-fixture", td3_root=complete_study_root / "td3-fixture")
    assert report.sha(Path(existing)) == expected
    assert not (complete_study_root / "report").exists()


def _droq_integration_payload(*, goal=False):
    names = ("summary", "equity", "monthly", "targets", "validation_selection", "training", "audit")
    chosen = {"algorithm": "droq", "family": "raw", "profile": "balanced"}
    return dict(queries={"droq_" + name: {"rows": [{"synthetic_fixture_only": True}], "source": {}} for name in names},
        targets={"goal_performance_achieved": goal}, selection={"candidate": chosen}, audit={"passed": True},
        external_input_sha256={}, query_run_count=26, new_run_count=12, baseline_reference_run_count=14,
        validation_run_count=24, training_model_count=5, original_scopes_unchanged=True,
        fresh_holdout_success=False, environment_step_budget_multiple=1, compute_budget_comparable=False,
        droqFollowup={"chosen": chosen, "q_inner_rounds_per_model": 555360})


@pytest.mark.parametrize("goal", [False, True])
def test_droq_cannot_replace_original_queries_leader_or_failed_goals(complete_study_root, monkeypatch, goal):
    from scripts import btceth_rl_report_droq as droq_reader
    original, _ = report.make_reviewed_snapshot(complete_study_root)
    payload = _droq_integration_payload(goal=goal)
    monkeypatch.setattr(droq_reader, "load_droq", lambda *args, **kwargs: deepcopy(payload))
    updated, evidence = report.make_reviewed_snapshot(complete_study_root, droq_root=complete_study_root / "droq-fixture")
    for identity, query in original["queries"].items():
        assert updated["queries"][identity] == query
    assert updated["report"]["primaryLeader"] == original["report"]["primaryLeader"]
    assert updated["report"]["researchTargets"] == original["report"]["researchTargets"]
    new = updated["report"]["droqFollowup"]
    assert new["targets"]["goal_performance_achieved"] is goal
    assert new["chosen"] == payload["selection"]["candidate"]
    assert new["freshHoldoutSuccess"] is False and new["computeBudgetComparable"] is False
    assert new["trainingModels"] == 5 and new["validationAccounts"] == 24
    assert evidence["runs"] == 172 and evidence["droq_scope_runs"] == 26
    assert evidence["additional_unique_droq_runs"] == 12
    assert not (complete_study_root / "report").exists()


@pytest.mark.parametrize("damage", ["collision", "training_count", "query_scope", "compute_claim"])
def test_droq_integration_requires_separate_exact_scope(complete_study_root, monkeypatch, damage):
    from scripts import btceth_rl_report_droq as droq_reader
    payload = _droq_integration_payload()
    if damage == "collision":
        payload["queries"]["strategy_summary"] = payload["queries"].pop("droq_summary")
    elif damage == "training_count":
        payload["training_model_count"] = 10
    elif damage == "query_scope":
        payload["queries"].pop("droq_audit")
    else:
        payload["compute_budget_comparable"] = True
    monkeypatch.setattr(droq_reader, "load_droq", lambda *args, **kwargs: deepcopy(payload))
    with pytest.raises(ValueError, match="DroQ"):
        report.make_reviewed_snapshot(complete_study_root, droq_root=complete_study_root / "droq-fixture")
    assert not (complete_study_root / "report").exists()


def test_droq_audit_cannot_redirect_to_an_unspecified_study(tmp_path):
    with pytest.raises(ValueError, match="explicit DroQ study root"):
        report.make_reviewed_snapshot(tmp_path, droq_audit_path=tmp_path / "audit.json")


def test_droq_incomplete_reader_cannot_create_a_reviewed_report(complete_study_root, monkeypatch):
    from scripts import btceth_rl_report_droq as droq_reader
    def reject(*args, **kwargs):
        raise ValueError("Complete independent DroQ annual26 required")
    monkeypatch.setattr(droq_reader, "load_droq", reject)
    with pytest.raises(ValueError, match="annual26 required"):
        report.make_reviewed_snapshot(complete_study_root, droq_root=complete_study_root / "droq-fixture")
    assert not (complete_study_root / "report").exists()


def test_droq_cannot_replace_an_existing_source_identity(complete_study_root, monkeypatch):
    from scripts import btceth_rl_report_extension as extension_reader
    from scripts import btceth_rl_report_droq as droq_reader
    existing = (complete_study_root / "frozen_spec.json").resolve().as_posix()
    expected = report.sha(Path(existing))
    def earlier(snapshot, evidence, **kwargs):
        evidence["external_input_sha256"] = {existing: expected}
        return snapshot, evidence
    monkeypatch.setattr(extension_reader, "augment_snapshot", earlier)
    payload = _droq_integration_payload()
    payload["external_input_sha256"] = {existing: "0" * 64}
    monkeypatch.setattr(droq_reader, "load_droq", lambda *args, **kwargs: deepcopy(payload))
    with pytest.raises(ValueError, match="DroQ evidence cannot replace"):
        report.make_reviewed_snapshot(complete_study_root, extension_root=complete_study_root / "cap-fixture",
                                     droq_root=complete_study_root / "droq-fixture")
    assert report.sha(Path(existing)) == expected
    assert not (complete_study_root / "report").exists()
