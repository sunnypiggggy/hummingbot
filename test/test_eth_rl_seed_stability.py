"""Population/orchestration tests; fixtures are never research performance data."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import backtest_eth_rl_seed_stability as job

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "results/backtests/single_pair_rl_six_20261030"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def expanded(source):
    result = deepcopy(source)
    result.update(symbols=list(job.SYMBOLS), algorithms=list(job.ALGORITHMS), seeds=list(job.SEEDS), training_model_count=60,
        seed_extension=deepcopy(job.SEED_EXTENSION), runtime_overlay=deepcopy(job.RUNTIME_OVERLAY),
        recipes={a: deepcopy(source["recipes"][a]) for a in job.ALGORITHMS},
        timesteps_per_model={a: source["timesteps_per_model"][a] for a in job.ALGORITHMS},
        training_window_quality={"ETHUSDT": deepcopy(source["training_window_quality"]["ETHUSDT"])})
    return result


@pytest.fixture
def source_spec():
    return job.read(RETAINED / "protocol.json")


def test_exact_thirty_seed_contract(source_spec):
    value = expanded(source_spec)
    job.compatible(source_spec, value)
    assert len(job.SEEDS) == 30 and job.NEW_SEEDS == tuple(range(5, 30))
    assert 2 * 30 == job.MODEL_COUNT and 2 * 25 == job.NEW_COUNT
    assert 2 * 30 * 2 + 4 * 2 == job.ACCOUNT_COUNT


@pytest.mark.parametrize("change", [
    lambda x: x["costs"]["base"].update(fee=.002),
    lambda x: x["environment"].update(rebalance_band=.01),
    lambda x: x["environment"].update(drawdown_penalty=0),
    lambda x: x.update(capital_per_account=5000),
    lambda x: x.update(capital_shared_between_pairs=True),
    lambda x: x.update(seeds=list(range(29))),
    lambda x: x.update(symbols=["BTCUSDT"]),
    lambda x: x["timesteps_per_model"].update(td3=32768),
    lambda x: x["recipes"]["sac"].update(batch_size=512),
    lambda x: x.update(dataset_manifest_sha256="0"*64),
    lambda x: x["training_window_quality"]["ETHUSDT"].update(applied=False),
])
def test_incompatible_contracts_rejected(source_spec, change):
    value = expanded(source_spec)
    change(value)
    with pytest.raises(ValueError):
        job.compatible(source_spec, value)


@pytest.fixture
def real_import(tmp_path, monkeypatch, source_spec):
    """Only market/runtime validation is substituted; real model trees copy/hash."""
    from scripts import backtest_single_pair_rl as core
    monkeypatch.setattr(job, "core_module", lambda *args: core)
    monkeypatch.setattr(job, "_validate_source", lambda *args: deepcopy(source_spec))
    output = tmp_path / "expanded"
    audit_source = tmp_path / "audit_entry.py"
    audit_source.write_text("# test contract entry only; never executed\n", encoding="utf-8")
    result = job.prepare(RETAINED, output, tmp_path / "not_loaded_dataset", audit_script=audit_source)
    return output, result


def test_real_original_ten_trees_are_byte_identical(real_import):
    output, result = real_import
    assert result == {"retained": 10, "pending": 50, "models": 60, "accounts_per_phase": 128}
    imported = job.validate_import(output, RETAINED)
    assert len(imported["retained_models"]) == 10 and len(imported["new_models"]) == 50
    assert job.sha(output / "retained_protocol.json") == job.sha(RETAINED / "protocol.json")
    assert len(list((output / "training").rglob("record.json"))) == 10
    for item in imported["retained_models"]:
        old = RETAINED / item["source_directory"]
        new = output / item["destination_directory"]
        assert job.model_files(old) == job.model_files(new) == item["file_sha256"]
        assert job.read(new / "record.json")["completed_at"] == item["source_completed_at"]


def test_retained_record_tamper_is_rejected(real_import):
    output, _ = real_import
    path = output / "training/ETHUSDT/sac/seed0/record.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes changed"):
        job.validate_import(output, RETAINED)


def test_retained_protocol_tamper_is_rejected(real_import):
    output, _ = real_import
    path = output / "retained_protocol.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="protocol bytes changed"):
        job.validate_import(output, RETAINED)


def test_existing_destination_is_preserved(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        job.prepare(RETAINED, RETAINED, tmp_path / "unused")
    target = tmp_path / "exists"
    target.mkdir()
    marker = target / "keep"
    marker.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        job.prepare(RETAINED, target, tmp_path / "unused")
    assert marker.read_bytes() == b"unchanged"


def test_each_worker_expands_only_population_whitelists(monkeypatch):
    from scripts import backtest_single_pair_rl as core
    from scripts import single_pair_rl_models as models
    before = {a: deepcopy(models.recipe(a)) for a in job.ALGORITHMS}
    monkeypatch.setattr(core, "SYMBOLS", core.SYMBOLS)
    monkeypatch.setattr(core, "SEEDS", core.SEEDS)
    monkeypatch.setattr(models, "SEEDS", models.SEEDS)
    returned, adapter = job.apply_runtime_overlay(ROOT)
    assert returned is core and adapter is models
    assert core.SYMBOLS == ("ETHUSDT",) and core.SEEDS == tuple(range(30)) and models.SEEDS == tuple(range(30))
    assert {a: models.recipe(a) for a in job.ALGORITHMS} == before
    assert models.STEPS["td3"] == 65536 and models.STEPS["sac"] == 32768
    assert job.sha(ROOT / "scripts/single_pair_rl_models.py") == job.read(RETAINED / "protocol.json")["sources"]["scripts/single_pair_rl_models.py"]


def test_fit_dispatches_only_fifty_new_identities(tmp_path, monkeypatch, source_spec):
    spec = expanded(source_spec)
    write(tmp_path / "protocol.json", spec)
    for algorithm in job.ALGORITHMS:
        for seed in job.OLD_SEEDS:
            write(tmp_path / f"training/ETHUSDT/{algorithm}/seed{seed}/record.json", {})
    calls = []
    core = SimpleNamespace(ROOT=ROOT,
        validate_training_record=lambda out, symbol, a, seed, protocol: {"symbol": symbol, "algorithm": a, "seed": seed},
        run_parallel=lambda out, tasks, fn, phase, workers: calls.append((tasks, fn, phase, workers)),
        write=lambda p, v: write(p, v))
    monkeypatch.setattr(job, "check_protocol", lambda *args, **kwargs: spec)
    monkeypatch.setattr(job, "validate_import", lambda *args: {})
    monkeypatch.setattr(job, "apply_runtime_overlay", lambda *args: (core, None))
    monkeypatch.setattr(job, "training_counter_audit", lambda *args: {"passed": True, "models_checked": 60})
    result = job.fit(tmp_path, tmp_path / "not_loaded", RETAINED, max_workers=28)
    assert result["passed"] is True
    tasks, function, phase, workers = calls[0]
    assert len(tasks) == 50 and function is job.train_one and phase == "fit" and workers == 28
    assert {(t["algorithm"], t["seed"]) for t in tasks} == {(a, s) for a in job.ALGORITHMS for s in range(5, 30)}
    assert all(t["symbol"] == "ETHUSDT" for t in tasks)
    assert job.read(tmp_path / "fit_complete.json")["model_count"] == 60


def test_fresh_training_rejects_original_seed(monkeypatch):
    called = []
    core = SimpleNamespace(train_one=lambda task: called.append(task))
    monkeypatch.setattr(job, "_task_scope", lambda task: (core, None))
    with pytest.raises(ValueError, match="must not be retrained"):
        job.train_one({"seed": 4})
    assert called == []


def test_worker_rejects_other_coin_and_seed30(tmp_path, monkeypatch):
    write(tmp_path / "protocol.json", {"test": "scope fixture, not market data"})
    task = {"output": str(tmp_path), "symbol": "BTCUSDT", "algorithm": "sac", "seed": 29,
            "protocol_sha256": job.sha(tmp_path / "protocol.json"), "core_root": str(ROOT)}
    with pytest.raises(ValueError, match="ETH study"):
        job._task_scope(task)
    monkeypatch.setattr(job, "apply_runtime_overlay", lambda *args: (None, None))
    task.update(symbol="ETHUSDT", seed=30)
    with pytest.raises(ValueError, match="seed/algorithm"):
        job._task_scope(task)


def test_training_counter_audit_checks_native_sac_td3_and_fragments(tmp_path, monkeypatch, source_spec):
    spec = expanded(source_spec)
    spec["frozen_at"] = "2026-10-01T00:00:00+00:00"
    write(tmp_path / "protocol.json", spec)
    episodes = {}
    records = {}
    for algorithm in job.ALGORITHMS:
        steps = spec["timesteps_per_model"][algorithm]
        updates = steps - 512
        optimizers = {"actor": updates if algorithm == "sac" else updates // 2, "critic": updates}
        if algorithm == "sac":
            optimizers["entropy"] = updates
        episode = {"start": "2020-01-01T00:00:00+00:00", "end_exclusive": "2020-02-02T00:00:00+00:00",
                   "completed_until": "2020-02-02T00:00:00+00:00", "completed_steps": 32, "terminated": False, "truncated": True}
        episodes[algorithm] = [episode] * (steps // 32)
        for seed in job.SEEDS:
            record = {"started_at": "2026-10-31T00:00:00+00:00", "training": {"training_episode_history": episodes[algorithm],
                "counters": {"gradient_update_rounds": updates, "optimizer_steps": deepcopy(optimizers),
                             "replay_transitions": steps, "replay_add_calls": steps}}}
            records[(algorithm, seed)] = record
            directory = tmp_path / f"training/ETHUSDT/{algorithm}/seed{seed}"
            write(directory / "record.json", {"identity": [algorithm, seed], "fixture_only": True})
            if seed in job.NEW_SEEDS:
                write(directory / "runtime_overlay.json", {"symbol": "ETHUSDT", "algorithm": algorithm, "seed": seed,
                    "runtime_overlay": job.RUNTIME_OVERLAY, "actual_model_seeds": list(job.SEEDS),
                    "actual_core_seeds": list(job.SEEDS), "actual_core_symbols": list(job.SYMBOLS),
                    "driver_source_sha256": job.sha(job.__file__), "record_sha256": job.sha(directory / "record.json")})
    core = SimpleNamespace(TRAIN=tuple(source_spec["train"]), write=lambda p, v: write(p, v),
        validate_training_record=lambda output, symbol, algorithm, seed, protocol: records[(algorithm, seed)])
    result = job.training_counter_audit(tmp_path, core)
    assert result["passed"] is True and result["models_checked"] == 60
    assert sum(r["retained"] for r in result["models"]) == 10
    records[("td3", 29)]["training"]["counters"]["optimizer_steps"]["actor"] += 1
    with pytest.raises(ValueError, match="counter/time/worker-scope"):
        job.training_counter_audit(tmp_path, core)
    assert job.read(tmp_path / "training_counter_audit.json")["failure_count"] == 1


def test_pipeline_failure_preserves_phase_and_never_runs_later_steps(tmp_path, monkeypatch):
    manifest = {"driver_relative_path": "extension_source/" + job.DRIVER_NAME,
                "audit_entry_relative_path": "extension_source/" + job.AUDIT_NAME}
    write(tmp_path / "extension_manifest.json", manifest)
    core = SimpleNamespace(write=lambda p, v: write(p, v))
    monkeypatch.setattr(job, "check_protocol", lambda *args, **kwargs: {})
    monkeypatch.setattr(job, "validate_import", lambda *args: {})
    monkeypatch.setattr(job, "core_module", lambda *args: core)
    calls = []
    monkeypatch.setattr(job.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="fit"):
        job.pipeline(tmp_path, tmp_path / "data", RETAINED)
    assert len(calls) == 1 and "fit" in calls[0]
    assert job.read(tmp_path / "pipeline_status.json")["status"] == "failed"
    assert (tmp_path / "fit.log").is_file() and not (tmp_path / "validation.log").exists()


def test_validation_is_descriptive_fixed_comparison_never_seed_selection(tmp_path, monkeypatch):
    import pandas as pd
    spec = {"costs": {"base": {}, "stress": {}}, "baselines": ["cash", "buy_hold", "sma200_trend", "momentum180"]}
    write(tmp_path / "protocol.json", spec)
    rows = [{"symbol": "ETHUSDT", "algorithm": a, "seed": seed, "scenario": c, "initial_capital": 10000,
             "split": "validation", "net_return": 0, "max_drawdown": 0}
            for a in job.ALGORITHMS for seed in job.SEEDS for c in spec["costs"]]
    rows += [{"symbol": "ETHUSDT", "algorithm": a, "seed": None, "scenario": c, "initial_capital": 10000,
              "split": "validation", "net_return": 0, "max_drawdown": 0}
             for a in spec["baselines"] for c in spec["costs"]]
    pd.DataFrame(rows).to_csv(tmp_path / "validation_summary.csv", index=False)
    audit = {"passed": True, "failure_count": 0, "accounts_checked": 128, "repeats_checked": 128,
             "protocol_sha256": job.sha(tmp_path / "protocol.json"), "summary_sha256": job.sha(tmp_path / "validation_summary.csv")}
    write(tmp_path / "validation_audit.json", audit)
    write(tmp_path / "validation_seed_contract_audit.json", {"fixture_only": True})
    monkeypatch.setattr(job, "check_phase_audits", lambda *args: (audit, {}))
    result = job.describe_validation(tmp_path)
    assert result["algorithm_selection"] is False and result["seed_selection"] is False
    saved = job.read(tmp_path / "validation_selection.json")
    assert saved["selection"]["ETHUSDT"]["selection_role"] == "compatibility_anchor_only"
    assert saved["selection"]["ETHUSDT"]["seed"] == 0
    assert all(len(r["seed_scores"]) == 30 for r in saved["selection"]["ETHUSDT"]["all_candidates"])
    write(tmp_path / "test_started.json", {})
    with pytest.raises(ValueError, match="Never change"):
        job.describe_validation(tmp_path)


@pytest.mark.parametrize("damage", ["contract_false", "wrong_count", "wrong_protocol", "wrong_wrapper"])
def test_both_financial_and_extension_audits_are_mandatory(tmp_path, damage):
    write(tmp_path / "protocol.json", {"fixture_only": True})
    (tmp_path / "validation_summary.csv").write_text("fixture\n", encoding="utf-8")
    write(tmp_path / "extension_manifest.json", {"extension_sources": {job.AUDIT_NAME: "a"*64}})
    financial = {"phase": "validation", "passed": True, "failure_count": 0, "failures": [],
                 "accounts_checked": 128, "repeats_checked": 128,
                 "protocol_sha256": job.sha(tmp_path / "protocol.json"),
                 "summary_sha256": job.sha(tmp_path / "validation_summary.csv")}
    write(tmp_path / "validation_audit.json", financial)
    contract = {"phase": "validation", "passed": True, "failure_count": 0, "failures": [],
                "core_financial_audit_passed": True, "core_financial_audit_sha256": job.sha(tmp_path / "validation_audit.json"),
                "wrapper_source_sha256": "a"*64, "inputs_sha256": {"protocol.json": job.sha(tmp_path / "protocol.json"),
                "validation_audit.json": job.sha(tmp_path / "validation_audit.json")}}
    write(tmp_path / "validation_seed_contract_audit.json", contract)
    job.check_phase_audits(tmp_path, "validation")
    if damage == "contract_false":
        contract["passed"] = False
    elif damage == "wrong_count":
        financial["accounts_checked"] = 127
        write(tmp_path / "validation_audit.json", financial)
    elif damage == "wrong_protocol":
        contract["inputs_sha256"]["protocol.json"] = "b"*64
    else:
        contract["wrapper_source_sha256"] = "c"*64
    write(tmp_path / "validation_seed_contract_audit.json", contract)
    with pytest.raises(ValueError):
        job.check_phase_audits(tmp_path, "validation")
