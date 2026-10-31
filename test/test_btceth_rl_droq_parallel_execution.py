"""Synthetic gates/two-card account parity only; no SSH/GPU/market/checkpoint."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import btceth_rl_droq_parallel_execution as execution
from scripts import btceth_rl_droq_parallel_eval as parallel
from scripts import btceth_rl_droq as original
from scripts import audit_btceth_rl_droq as old_audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    execution.write_new(path, value)


def change(path, **values):
    obj = execution.read(path); obj.update(values); path.write_text(__import__('json').dumps(obj), encoding="utf-8")


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    output, main, source, extra = (tmp_path / name for name in ("study", "main", "source", "extension"))
    for folder in (output, main, source, extra): folder.mkdir()
    dataset = tmp_path / "not_a_pickle"; dataset.write_bytes(b"SYNTHETIC ONLY")
    (source / "financial.py").write_bytes(b"not imported")
    (main / "baseline_reference").write_bytes(b"immutable synthetic reference")
    for name in ("btceth_rl_droq_parallel_eval.py", "btceth_rl_droq_parallel_audit.py"):
        (extra / name).write_bytes(("SYNTHETIC UNIMPORTED " + name).encode())
    monkeypatch.setattr(execution, "PARALLEL_SHA", execution.sha(extra / "btceth_rl_droq_parallel_eval.py"))
    audit_sha = execution.sha(extra / "btceth_rl_droq_parallel_audit.py")
    put(output / "protocol.json", {"synthetic_not_authority": True}); put(output / "protocol_receipt.json", {})
    put(output / "training_completion.json", {"explicitly_synthetic": True})
    training = {"training_completion.json": execution.sha(output / "training_completion.json")}
    gpus = [dict(index=i, uuid=f"SYNTHETIC-GPU-{i}", name="Tesla V100 mocked", memory_total_mib=32768) for i in range(2)]
    spec = dict(source_sha256={"financial.py": execution.sha(source / "financial.py")}, physical_gpus=gpus,
        family_gpu_assignments={"raw": gpus[0]["uuid"], "cap075": gpus[1]["uuid"]})
    frames = {}
    for family in ("raw", "cap075"):
        rows = []
        for scenario in ("base", "stress"):
            for key in (*map(str, range(5)), "ensemble"):
                # A tie intentionally preserves raw. No invented market evidence.
                ret = .1 if scenario == "base" else .05
                rows.append(dict(run_id=f"validation__{scenario}__{family}__droq__balanced__{key}", algorithm="droq", family=family, profile="balanced", scenario=scenario,
                    split="validation", seed=np.nan if key == "ensemble" else int(key), primary_policy=key == "ensemble", initial_capital=10000.,
                    start=original.VALIDATION[0], end_exclusive=original.VALIDATION[1], net_return=ret, max_drawdown=-.1,
                    final_equity=10000 * (1 + ret), fees=1., rebalance_band=.025, exposure_multiplier=original.BUDGETS[family]))
        frame = pd.DataFrame(rows); path = output / f"validation/{family}/balanced/summary.csv"; path.parent.mkdir(parents=True)
        frame.to_csv(path, index=False); frames[family] = frame
    process = dict(status="completed", completed_cases=24, source_sha256=execution.PARALLEL_SHA, protocol_sha256=execution.sha(output / "protocol.json"), completed_at="2026-10-30T06:30:00Z")
    put(output / "parallel_validation_process.json", process)
    validation = {"parallel_validation_process.json", *(f"validation/{family}/balanced/summary.csv" for family in frames)}
    required = set(training) | validation | {"protocol.json", "protocol_receipt.json"}
    audit = dict(passed=True, status="partial_test_pending", validation_complete=True, test_complete=False, failures=[], failure_count=0,
        training_model_count=5, validation_run_count=24, run_count=24, repeat_count=24, audited_at="2026-10-30T06:40:00Z",
        source_sha256=spec["source_sha256"], main_input_sha256={"baseline_reference": execution.sha(main / "baseline_reference")},
        input_sha256={n: execution.sha(output / n) for n in required}, extension_source_sha256={name: execution.sha(extra / name) for name in ("btceth_rl_droq_parallel_eval.py", "btceth_rl_droq_parallel_audit.py")},
        runs={row["run_id"]: dict(hours=7080, reconstructed=True) for frame in frames.values() for row in frame.to_dict("records")})
    put(output / execution.VALIDATION_AUDIT, audit)
    fake = SimpleNamespace(protocol=lambda *a: spec, training_files=lambda *a: training, candidate=original.candidate, no_test=original.no_test,
        now=lambda: "2026-10-30T06:50:00Z", utc=original.utc, TEST=original.TEST)
    independent = SimpleNamespace(validation_files=lambda: validation)
    monkeypatch.setattr(execution, "dependencies", lambda *a: (fake, parallel, old_audit, independent))
    return SimpleNamespace(output=output, main=main, source=source, extra=extra, dataset=dataset, audit_sha=audit_sha, fake=fake, spec=spec)


def args(state):
    return state.main, state.dataset, state.output, state.source, state.extra, state.audit_sha


def test_selection_real_original_scores_raw_tie_and_complete_byte_seal(sealed):
    selected = execution.select(*args(sealed))
    assert selected["candidate"]["family"] == "raw" and len(selected["candidates"]) == 2
    assert selected["seed_selection"] is False and selected["selection_uses_test"] is False
    assert selected["audit_file"] == execution.VALIDATION_AUDIT and selected["execution_mode"] == "explicit_parallel_adapter"
    assert execution.frozen_selection(*args(sealed))[2] == selected
    with pytest.raises(ValueError): execution.select(*args(sealed))


@pytest.mark.parametrize("damage", ["partial", "bad_count", "bool_count", "failed", "model", "baseline", "source", "extension", "summary", "fake_runs"])
def test_no_selection_without_all_financial_and_source_prerequisites(sealed, damage):
    p = sealed.output / execution.VALIDATION_AUDIT
    if damage == "partial": change(p, validation_run_count=23)
    elif damage == "bad_count": change(p, repeat_count=23)
    elif damage == "bool_count": change(p, failure_count=False)
    elif damage == "failed": change(p, passed=False)
    elif damage == "model": (sealed.output / "training_completion.json").write_bytes(b"changed")
    elif damage == "baseline": (sealed.main / "baseline_reference").write_bytes(b"changed")
    elif damage == "source": (sealed.source / "financial.py").write_bytes(b"changed")
    elif damage == "extension": (sealed.extra / "btceth_rl_droq_parallel_audit.py").write_bytes(b"changed")
    elif damage == "summary": (sealed.output / "validation/raw/balanced/summary.csv").write_bytes(b"changed")
    else: change(p, runs={})
    with pytest.raises((ValueError, OSError)): execution.select(*args(sealed))
    assert not (sealed.output / "selection.json").exists()


def authorized(sealed):
    execution.select(*args(sealed)); annex = execution.prepare_annual(*args(sealed))
    auth = dict(annual_evaluation_authorized=True, parallel_case_evaluation_authorized=True, no_live_trading=True, financial_parameters_fixed=True,
        selection_sha256=execution.sha(sealed.output / "selection.json"), resource_annex_sha256=execution.sha(sealed.output / "parallel_annual_resource_annex.json"),
        resource_receipt_sha256=execution.sha(sealed.output / "parallel_annual_resource_receipt.json"), execution_source_sha256=execution.sha(execution.__file__),
        parallel_source_sha256=execution.PARALLEL_SHA, validation_audit_sha256=execution.sha(sealed.output / execution.VALIDATION_AUDIT), protocol_sha256=execution.sha(sealed.output / "protocol.json"),
        authorized_at="2026-10-30T06:51:00Z", **{f: True for f in execution.FLAGS})
    put(sealed.output / "annual_authorization.json", auth)
    sealed.fake.now = lambda: "2026-10-30T06:52:00Z"
    return annex


def test_two_cards_exact6_each_and_both_ensemble_accounts_on_card1(sealed):
    annex = authorized(sealed); mapping = annex["case_gpu_assignments"]
    assert len(mapping) == 12 and list(mapping.values()).count("SYNTHETIC-GPU-0") == list(mapping.values()).count("SYNTHETIC-GPU-1") == 6
    assert all(gpu == "SYNTHETIC-GPU-1" for run_id, gpu in mapping.items() if run_id.endswith("ensemble"))
    assert annex["original_family_gpu_assignments"] == sealed.spec["family_gpu_assignments"]
    assert execution.annual_gate(*args(sealed), require_new=True)[3] == annex


@pytest.mark.parametrize("damage", ["no_auth", "early_auth", "future_auth", "wrong_selection", "source", "changed_map", "future_marker", "known_year", "seed_swap"])
def test_no_annual_before_sealed_explicit_exact_approval(sealed, damage):
    authorized(sealed); p = sealed.output / "annual_authorization.json"
    if damage == "no_auth": p.unlink()
    elif damage == "early_auth": change(p, authorized_at="2026-10-30T06:00:00Z")
    elif damage == "future_auth": change(p, authorized_at="2026-10-30T06:53:00Z")
    elif damage == "wrong_selection": change(p, selection_sha256="0" * 64)
    elif damage == "source": change(p, execution_source_sha256="0" * 64)
    elif damage == "changed_map":
        path = sealed.output / "parallel_annual_resource_annex.json"; v = execution.read(path)
        v["case_gpu_assignments"][next(iter(v["case_gpu_assignments"]))] = "OTHER"
        path.write_text(__import__('json').dumps(v))
    elif damage == "future_marker": put(sealed.output / "test_started.json", {})
    elif damage == "known_year": change(p, not_fresh_holdout=False)
    else: change(sealed.output / "selection.json", candidate={"family": "cap075"})
    with pytest.raises((ValueError, OSError)): execution.annual_gate(*args(sealed), require_new=True)
    assert not (sealed.output / "test").exists()


@pytest.mark.parametrize("returncode", [0, 7])
def test_supervisor_preserves_true_pid_exit_and_refuses_restart(sealed, monkeypatch, returncode):
    authorized(sealed)
    import subprocess
    class Child:
        pid = 987654
        def wait(self):
            if returncode == 0: put(sealed.output / "test_completion.json", {"synthetic_complete": True})
            return returncode
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Child())
    if returncode:
        with pytest.raises(ValueError): execution.supervise_test(*args(sealed))
    else: execution.supervise_test(*args(sealed))
    process = execution.read(sealed.output / "parallel_test_process.json")
    assert process["child_pid"] == process["coordinator_pid"] == 987654 and process["exit_code"] == returncode
    assert process["status"] == ("completed" if returncode == 0 else "failed")
    before = (sealed.output / "parallel_test_process.json").read_bytes()
    with pytest.raises(ValueError): execution.supervise_test(*args(sealed))
    assert before == (sealed.output / "parallel_test_process.json").read_bytes()


@pytest.mark.parametrize("damage", [None, "duplicate", "wrong_card"])
def test_merge_keeps_exact_sorted12_tables_and_two_card_truth(sealed, damage):
    annex = authorized(sealed); results = []
    for index, plan in enumerate(parallel.plan_cases("test", "raw")):
        run_id = f"test__{plan['scenario']}__raw__droq__balanced__{plan['key']}"
        relative = f"test/raw/balanced/{plan['scenario']}/{plan['key']}"; directory = sealed.output / relative; directory.mkdir(parents=True)
        device = dict(visible_gpu_uuid=annex["case_gpu_assignments"][run_id], name="SYNTHETIC no CUDA", device="cuda:0")
        proof = dict(run_id=run_id, passed=True, device=device, worker_pid=1000+index, execution_source_sha256=execution.sha(execution.__file__))
        put(directory / "replay.json", proof)
        # This fixture checks schema/ordering, never purports to be financial data.
        row = dict(run_id=run_id, synthetic_only=True)
        frame = pd.DataFrame([dict(run_id=run_id, synthetic_only=True)])
        results.append(dict(identity=plan, row=row, history=frame.copy(), months=frame.copy(), proof=proof,
            case_relative_path=relative, case_files_sha256={relative + "/replay.json": execution.sha(directory / "replay.json")}))
    if damage == "duplicate": results[-1] = results[0]
    elif damage == "wrong_card":
        result = results[0]; result["proof"]["device"]["visible_gpu_uuid"] = "UNKNOWN"
        path = sealed.output / result["case_relative_path"] / "replay.json"; path.write_text(__import__('json').dumps(result["proof"]))
        result["case_files_sha256"] = {result["case_relative_path"] + "/replay.json": execution.sha(path)}
    writer = SimpleNamespace(GROUP_FILES=original.GROUP_FILES, PROFILE="balanced", BUDGETS=original.BUDGETS, COSTS=original.COSTS, SEEDS=original.SEEDS,
        canonical_csv=original.canonical_csv, verify_pins=execution.verify_pins, now=sealed.fake.now)
    if damage:
        with pytest.raises(ValueError): execution.merge_annual(writer, parallel, sealed.output, "raw", list(reversed(results)), annex)
        assert not (sealed.output / "test_completion.json").exists()
    else:
        complete = execution.merge_annual(writer, parallel, sealed.output, "raw", list(reversed(results)), annex)
        group = execution.read(sealed.output / "test/raw/balanced/completion.json")
        assert complete["case_count"] == complete["repeat_count"] == 12 and len(group["case_workers"]) == 12
        assert group["device"] is None and group["worker_pid"] is None and len(group["devices"]) == 2
        actual = pd.read_csv(sealed.output / "test/raw/balanced/summary.csv")
        assert list(actual.run_id) == [result["row"]["run_id"] for result in results]


@pytest.mark.timeout(180)
def test_real_f146_financial_bytes_identical_when_ensemble_device_changes(tmp_path, monkeypatch):
    import torch
    from test.test_btceth_rl_droq_evaluation_flow import ConstantSyntheticPolicy
    assert not torch.cuda.is_initialized()
    old_output, new_output, main = (tmp_path / name for name in ("legacy", "two_gpu", "main"))
    for folder in (old_output, new_output, main): folder.mkdir()
    dataset = tmp_path / "not_a_pickle"; dataset.write_bytes(b"No market payload")
    start, end = map(pd.Timestamp, original.TEST)
    hours = pd.date_range(start-pd.Timedelta(days=210), end, freq="h", inclusive="left")
    prices = pd.DataFrame({"BTCUSDT": 100., "ETHUSDT": 200.}, index=hours)
    data = {"hourly_" + name: prices.copy() for name in ("open", "high", "low", "close")}; data["hourly_volume"] = prices * 0 + 100
    models = {seed: ConstantSyntheticPolicy(seed, action) for seed, action in enumerate([(-1., -1.), (1., -1.), (1., 1.), (.5, 0.), (0., .5)])}
    spec = dict(physical_gpus=[dict(index=i, uuid=f"SYNTHETIC-GPU-{i}", name="V100 synthetic") for i in range(2)], family_gpu_assignments={"raw": "SYNTHETIC-GPU-0", "cap075": "SYNTHETIC-GPU-1"})
    active = {"uuid": "SYNTHETIC-GPU-0"}
    for output in (old_output, new_output):
        put(output / "protocol.json", {"explicit_synthetic_only": True})
        for seed, model in models.items(): put(output / "training" / original.job_id(seed) / "record.json", dict(seed=seed, parameter_sha256=original.parameter_digest(model)))
    run_id = "test__base__raw__droq__balanced__ensemble"
    put(new_output / "parallel_annual_resource_annex.json", dict(case_gpu_assignments={run_id: "SYNTHETIC-GPU-1"}))
    put(new_output / "annual_authorization.json", {"synthetic_only_not_authority": True}); put(new_output / "test_started.json", {"synthetic_only_not_marker": True})
    monkeypatch.setattr(original, "protocol", lambda *a: spec)
    monkeypatch.setattr(parallel, "get_runner", lambda *a: original)
    monkeypatch.setattr(execution, "load_module", lambda *a: parallel)
    monkeypatch.setattr(original.shared, "torch_setup", lambda *a: dict(visible_gpu_uuid=active["uuid"], capability=[7, 0], device="cuda:0", name="V100 mocked, no CUDA"))
    monkeypatch.setattr(original.shared, "load_dataset", lambda *a: data)
    monkeypatch.setattr(original, "load_policy", lambda path, **kw: models[int(Path(path).name.rsplit("seed", 1)[1])])
    common = dict(main=str(main), dataset=str(dataset), family="raw", scenario="base", key="ensemble", split="test", source_root=str(Path.cwd()))
    parallel.case_worker(dict(common, output=str(old_output), protocol_sha256=execution.sha(old_output / "protocol.json"), source_sha256=parallel.sha(parallel.__file__)))
    active["uuid"] = "SYNTHETIC-GPU-1"
    result = execution.annual_case_worker(dict(common, output=str(new_output), extension_root=str(tmp_path), gpu_uuid=active["uuid"], execution_source_sha256=execution.sha(execution.__file__),
        protocol_sha256=execution.sha(new_output / "protocol.json"), resource_annex_sha256=execution.sha(new_output / "parallel_annual_resource_annex.json"),
        authorization_sha256=execution.sha(new_output / "annual_authorization.json"), test_marker_sha256=execution.sha(new_output / "test_started.json")))
    relative = "test/raw/balanced/base/ensemble"
    for name in original.CASE_FILES:
        if name != "replay.json": assert (old_output / relative / name).read_bytes() == (new_output / relative / name).read_bytes(), name
    assert result["proof"]["device"]["visible_gpu_uuid"] == "SYNTHETIC-GPU-1" and result["proof"]["execution_mode"] == execution.MODE
    assert not torch.cuda.is_initialized()
