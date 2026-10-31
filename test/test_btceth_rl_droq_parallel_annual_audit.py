"""Pure two-GPU process/group evidence validation; no annual data execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import audit_btceth_rl_droq as bf
from scripts import btceth_rl_droq_parallel_annual_audit as annual
from scripts.crypto_rl_reporting import canonical_csv


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def stamp(n):
    return f"2026-10-30T01:00:{n:02d}+00:00"


@pytest.fixture
def evidence(tmp_path):
    out, main = tmp_path / "study", tmp_path / "main"
    out.mkdir(); main.mkdir()
    inputs = bf.Inputs(out, main, Path(__file__).resolve().parents[1], bf.Checks())
    family = "cap075"
    devices = {gpu: dict(visible_gpu_uuid=gpu, device="cuda:0", capability=[7, 0], name="Tesla V100 mocked only")
               for gpu in ("GPU-0", "GPU-1")}
    plans = [(cost, key) for cost in ("base", "stress") for key in ("0", "1", "2", "3", "4", "ensemble")]
    assignments = {f"test__{cost}__{family}__droq__balanced__{key}": "GPU-0" if key in ("0", "2", "4") else "GPU-1" for cost, key in plans}
    annex = dict(case_gpu_assignments=assignments)
    selection = dict(candidate=dict(family=family, profile="balanced", algorithm="droq"))
    put(out / "protocol.json", {"synthetic_only": True})
    put(out / "selection.json", selection)
    put(out / "parallel_annual_resource_annex.json", annex)
    put(out / "annual_authorization.json", {"synthetic_authorization_fixture_only": True})
    put(out / "parallel_test.owner.json", dict(pid=10, source_sha256=annual.EXECUTION_SHA, token="a" * 32, at=stamp(0)))
    put(out / "test_started.json", dict(supervisor_pid=10, at=stamp(1)))
    (out / "parallel_test_runner.log").write_bytes(b"Synthetic actual-child terminal fixture")
    physical = [dict(index=i, uuid=f"GPU-{i}", name="V100", memory_total_mib=16384,
                     memory_used_mib=10, utilization_percent=0) for i in range(2)]
    spec = dict(physical_gpus=physical, allowed_gpu_rows=[])
    resource = dict(at=stamp(3), physical_gpus=physical,
        command=["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        unfiltered_stdout="", rows=[])
    put(out / "parallel_test_resource_preflight.json", resource)
    proofs, rows, histories, monthly, expected = [], [], [], [], {}
    for i, (cost, key) in enumerate(plans):
        run_id = f"test__{cost}__{family}__droq__balanced__{key}"
        proof = dict(run_id=run_id, worker_pid=100 + i, device=devices[assignments[run_id]],
            assigned_gpu_uuid=assignments[run_id], execution_mode=annual.MODE,
            execution_source_sha256=annual.EXECUTION_SHA,
            resource_annex_sha256=annual.sha(out / "parallel_annual_resource_annex.json"))
        put(out / f"test/{family}/balanced/{cost}/{key}/replay.json", proof)
        proofs.append(proof)
        rows.append(dict(run_id=run_id, algorithm="droq", family=family, profile="balanced", scenario=cost,
            split="test", seed=None if key == "ensemble" else int(key), primary_policy=key == "ensemble",
            initial_capital=10000., start=bf.TEST[0], end_exclusive=bf.TEST[1], net_return=0., max_drawdown=0.,
            final_equity=10000., fees=0., rebalance_band=.025, exposure_multiplier=.75))
        history = pd.DataFrame(dict(timestamp=[bf.TEST[0]], equity=[10000.], net_return=[0.], drawdown=[0.]))
        histories.append(history.assign(run_id=run_id))
        monthly.append(dict(run_id=run_id, month="2025-10", **{"return": 0.}))
        expected[run_id] = (dict(run_id=run_id, net_return=0., max_drawdown=0., fees=0., final_equity=10000.), history, {"2025-10": 0.})
    directory = out / f"test/{family}/balanced"
    for name, frame in (("summary.csv", pd.DataFrame(rows)), ("daily_equity.csv", pd.concat(histories)), ("monthly_returns.csv", pd.DataFrame(monthly))):
        (directory / name).write_bytes(canonical_csv(frame))
    workers = [{k: p[k] for k in ("run_id", "worker_pid", "device")} for p in proofs]
    group = dict(at=stamp(4), protocol_sha256=annual.sha(out / "protocol.json"), case_count=12, repeat_count=12,
        repeats=proofs, device=None, devices=[devices[g] for g in sorted(devices)], execution_mode=annual.MODE,
        worker_pid=None, coordinator_pid=11, worker_pids=list(range(100, 112)), case_workers=workers,
        case_gpu_assignments=assignments, execution_source_sha256=annual.EXECUTION_SHA,
        resource_annex_sha256=annual.sha(out / "parallel_annual_resource_annex.json"))
    put(directory / "completion.json", group)
    complete = dict(at=stamp(5), protocol_sha256=annual.sha(out / "protocol.json"), selection_sha256=annual.sha(out / "selection.json"),
        candidate=selection["candidate"], case_count=12, repeat_count=12, execution_mode=annual.MODE,
        execution_source_sha256=annual.EXECUTION_SHA, resource_annex_sha256=group["resource_annex_sha256"],
        coordinator_pid=11, workers=[{k: group[k] for k in ("device", "devices", "case_workers", "worker_pids")}],
        case_workers=workers, case_gpu_assignments=assignments)
    put(out / "test_completion.json", complete)
    execution = SimpleNamespace(PARALLEL_SHA="synthetic parallel pin", ORIGINAL_SHA="synthetic original pin")
    process = dict(scope="explicit_droq_two_gpu_selected_annual", phase="test", status="completed", exit_code=0,
        source_sha256=annual.EXECUTION_SHA, parallel_source_sha256=execution.PARALLEL_SHA, original_runner_sha256=execution.ORIGINAL_SHA,
        total_cases=12, max_workers_per_gpu=6, case_gpu_assignments=assignments, supervisor_pid=10, child_pid=11, coordinator_pid=11,
        command=["python", "-B", "-u", "/fixture/btceth_rl_droq_parallel_execution.py", "--phase", "_test", "--audit-source-sha", annual.BASE_SHA],
        log_relative_path="parallel_test_runner.log", log_sha256=annual.sha(out / "parallel_test_runner.log"),
        started_at=stamp(2), completed_at=stamp(6))
    for field, name in (("protocol_sha256", "protocol.json"), ("selection_sha256", "selection.json"),
        ("authorization_sha256", "annual_authorization.json"), ("test_marker_sha256", "test_started.json"),
        ("resource_annex_sha256", "parallel_annual_resource_annex.json"), ("completion_sha256", "test_completion.json")):
        process[field] = annual.sha(out / name)
    put(out / "parallel_test_process.json", process)
    return SimpleNamespace(inputs=inputs, out=out, process=process, complete=complete, annex=annex,
        execution=execution, spec=spec, selection=selection, expected=expected, family=family)


def test_true_two_gpu_process_and_case_group_preserve_original_financial_checks(evidence):
    e = evidence
    process, complete = annual.verify_annual_process(e.inputs, bf, e.spec, e.selection, e.annex, e.execution)
    summary, daily, months, cases = annual.audit_group(e.inputs, bf, process, complete, e.annex, e.expected, family=e.family)
    assert len(summary) == len(cases) == 12 and e.inputs.checks.failed == 0
    assert {v["device"]["visible_gpu_uuid"] for v in complete["case_workers"]} == {"GPU-0", "GPU-1"}


@pytest.mark.parametrize("change", ["exit_bool", "source", "coordinator", "assignment", "log"])
def test_reject_process_identity_source_assignment_or_real_log_tampering(evidence, change):
    e = evidence; value = dict(e.process)
    if change == "exit_bool": value["exit_code"] = True
    elif change == "source": value["source_sha256"] = "changed"
    elif change == "coordinator": value["coordinator_pid"] = value["supervisor_pid"]
    elif change == "assignment": value["case_gpu_assignments"] = {key: "GPU-0" for key in value["case_gpu_assignments"]}
    else: (e.out / "parallel_test_runner.log").write_bytes(b"Changed original child log")
    put(e.out / "parallel_test_process.json", value)
    with pytest.raises(ValueError):
        annual.verify_annual_process(e.inputs, bf, e.spec, e.selection, e.annex, e.execution)


def test_case_gpu_relabel_cannot_pass_even_when_both_cards_are_original(evidence):
    e = evidence
    path = e.out / "test/cap075/balanced/base/0/replay.json"
    value = json.loads(path.read_text()); value["device"]["visible_gpu_uuid"] = "GPU-1"
    put(path, value)
    with pytest.raises(ValueError):
        annual.audit_group(e.inputs, bf, e.process, e.complete, e.annex, e.expected, family=e.family)


def test_declared_annual_inventory_has_12_repeat_proofs_and_no_fake_test_terminal():
    names = annual.test_files("cap075")
    assert len([name for name in names if name.endswith("/replay.json")]) == 12
    assert "parallel_test_process.json" in names and "test_process.json" not in names
