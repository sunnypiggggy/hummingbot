"""Pure byte/JSON schedule fixtures, never SSH/CUDA/Torch/financial execution."""
from pathlib import Path

import pytest

from scripts import btceth_rl_droq_scheduling as schedule
from scripts import btceth_rl_droq_scheduling_audit as audit
from test.test_btceth_rl_droq_scheduling import put, state


@pytest.fixture
def cohort(state, monkeypatch):
    monkeypatch.setattr(audit, "PROTOCOL_SHA256", schedule.PROTOCOL_SHA256)
    monkeypatch.setattr(audit, "SOURCES", state.p["source_sha256"])
    receipt = schedule.read(state.output / "protocol_receipt.json")
    put(state.output / "protocol_receipt.json", {**receipt, "sealed_at": "2026-10-30T00:30:00Z"})
    for seed in (0, 1, 2, 3):
        path = state.output / f"training/droq__balanced__seed{seed}/started.json"
        started = schedule.read(path)
        started["task"].update(profile="balanced")
        started["device"] = dict(visible_gpu_uuid=state.p["seed_gpu_assignments"][str(seed)])
        put(path, started)
    _, annex = schedule.readiness(state.output, state.main, state.dataset, state.source)
    annex["prepared_at"] = "2026-10-30T03:00:00Z"
    annex["resource_snapshot"]["at"] = "2026-10-30T02:59:59Z"
    snap = "preflight/scheduling/original_fit_process.initial.json"
    put(state.output / snap, state.fit)
    annex["original_fit_initial_sha256"] = schedule.sha(state.output / snap)
    annex["original_fit_initial_snapshot_file"] = snap
    perf = state.output.parent / "performance"; perf.mkdir()
    for name in audit.PERFORMANCE_FILES:
        (perf / name).write_bytes(("SYNTHETIC SOURCE ONLY " + name).encode())
    fake_performance = {name: schedule.sha(perf / name) for name in audit.PERFORMANCE_FILES}
    monkeypatch.setattr(audit, "PERFORMANCE_FILES", fake_performance)
    annex["performance_adapter"] = dict(path=str(perf / "droq_execution_adapter.py"), sha256=fake_performance["droq_execution_adapter.py"], files={str(perf / k): v for k, v in fake_performance.items()})
    put(state.output / "scheduling_annex.json", annex)
    put(state.output / "accelerated_fit.owner.json", dict(token="a" * 32, supervisor_pid=800, at="2026-10-30T03:00:00Z", source_sha256=audit.SCHEDULER_SHA256))
    command = ["/synthetic/python", "-u", "/synthetic/btceth_rl_droq_scheduling.py", "--worker", "--output-dir", str(state.output), "--reference-main", str(state.main), "--dataset", str(state.dataset), "--source-root", str(state.source), "--owner-token", "a" * 32]
    process = dict(phase="accelerated_fit", status="running", supervisor_pid=800, child_pid=900, command=command,
        started_at="2026-10-30T03:00:01Z", source_sha256=audit.SCHEDULER_SHA256, protocol_sha256=schedule.PROTOCOL_SHA256,
        annex_sha256=schedule.sha(state.output / "scheduling_annex.json"), log_relative_path="accelerated_seed4.log")
    put(state.output / "accelerated_fit_process.json", process)
    state.annex, state.process, state.performance = annex, process, perf
    return state


def verify(state):
    return audit.verify_accelerated_fit(state.output, state.main, state.dataset, state.source, performance_root=state.performance)


def finish(state):
    for seed in (2, 3, 4):
        state.model(seed)
        folder = state.output / f"training/droq__balanced__seed{seed}"
        record = schedule.read(folder / "record.json")
        record["completed_at"] = "2026-10-30T04:30:00Z" if seed == 4 else "2026-10-30T04:00:00Z"
        put(folder / "record.json", record)
        if seed == 4:
            started = schedule.read(folder / "started.json")
            started.update(at="2026-10-30T03:00:03Z", device=dict(visible_gpu_uuid=state.p["seed_gpu_assignments"]["4"]))
            started["task"]["profile"] = "balanced"
            put(folder / "started.json", started)
    log = state.output / "fit.log"
    collision_path = str(state.output).replace("\\", "/") + "/training/droq__balanced__seed4"
    log.write_text("concurrent.futures.process._RemoteTraceback: \nFileExistsError: [Errno 17] File exists: '" + collision_path + "'\n")
    original = {**state.fit, "status": "failed", "exit_code": 1, "completed_at": "2026-10-30T04:40:00Z", "log_sha256": schedule.sha(log)}
    put(state.output / "fit_process.json", original)
    worker = dict(status="completed", pid=900, ppid=800, seed=4, started_at="2026-10-30T03:00:02Z", completed_at="2026-10-30T04:30:01Z",
        protocol_sha256=schedule.PROTOCOL_SHA256, annex_sha256=schedule.sha(state.output / "scheduling_annex.json"),
        record_sha256=schedule.sha(state.output / "training/droq__balanced__seed4/record.json"), parameter_sha256="4" * 64,
        performance_adapter=state.annex["performance_adapter"], performance_install_count=1,
        performance_provenance=dict(scope="explicit_new_worker_execution_adapter", sources=audit.PERFORMANCE_FILES, live_attach=False, install_before_first_store=True,
            pinned_core_sha256="16fec92b1de1ce5942599f57079bc3a002b2869429515b4cd68606bda8fb52db", pinned_algorithm_sha256="1417451e076f59584e1d5f3cdea1c47366c6625dff5591e3d864dd7c88152cf7"))
    put(state.output / "accelerated_seed4_worker.json", worker)
    records = [schedule.read(state.output / f"training/droq__balanced__seed{seed}/record.json") for seed in range(5)]
    put(state.output / "training_completion.json", dict(at="2026-10-30T04:40:01Z", model_count=5, records=records, protocol_sha256=schedule.PROTOCOL_SHA256))
    names = {f"training/droq__balanced__seed{seed}/{leaf}" for seed in range(5) for leaf in (*audit.LEAVES, "started.json")}
    complete = dict(at="2026-10-30T04:40:02Z", protocol_sha256=schedule.PROTOCOL_SHA256, annex_sha256=schedule.sha(state.output / "scheduling_annex.json"),
        model_count=5, records=records, training_completion_sha256=schedule.sha(state.output / "training_completion.json"), record_file_sha256=schedule.hashes(state.output, names),
        original_0_1_hashes_preserved=True, no_signals=True,
        original_fit_terminal=dict(file="fit_process.json", sha256=schedule.sha(state.output / "fit_process.json"), status="failed", exit_code=1, log_sha256=schedule.sha(log), collision_verified=True, collision_path=collision_path),
        cohort_started_at=original["started_at"], cohort_completed_at="2026-10-30T04:40:03Z", worker_proof_sha256=schedule.sha(state.output / "accelerated_seed4_worker.json"),
        source_sha256=state.p["source_sha256"], scheduler_source_sha256=audit.SCHEDULER_SHA256)
    put(state.output / "accelerated_fit_completion.json", complete)
    (state.output / "accelerated_seed4.log").write_bytes(b"SYNTHETIC WORKER EXIT0\n")
    put(state.output / "accelerated_fit_process.json", {**state.process, "status": "completed", "exit_code": 0, "worker_exit_code": 0,
        "worker_completed_at": "2026-10-30T04:30:02Z", "completed_at": "2026-10-30T04:40:04Z",
        "log_sha256": schedule.sha(state.output / "accelerated_seed4.log"), "completion_sha256": schedule.sha(state.output / "accelerated_fit_completion.json")})


def edit(state, name, change):
    path = state.output / name; value = schedule.read(path); change(value); put(path, value)


def test_running_is_pending_and_not_financial_or_training_completion_passage(cohort):
    value = verify(cohort)
    assert value["status"] == "pending" and value["passed"] is False and value["training_model_count"] == 2
    assert value["derived_cohort_envelope"] is None and value["financial_audit_passed"] is False and value["tensor_payload_loaded"] is False
    assert not (cohort.output / "training_completion.json").exists()


def test_complete_schedule_preserves_failed_original_and_returns_only_derived_time_bounds(cohort):
    finish(cohort); before = {p: p.read_bytes() for p in cohort.output.rglob("*") if p.is_file()}
    result = verify(cohort)
    assert result["passed"] is True, result["failures"]
    envelope = result["derived_cohort_envelope"]
    assert envelope["kind"] == "derived_training_time_bounds_only" and envelope["original_terminal_relabelled"] is False
    assert envelope["original_terminal_status"] == "failed" and envelope["accelerated_terminal_status"] == "completed"
    assert envelope["started_at"] == cohort.fit["started_at"] and envelope["completed_at"] == "2026-10-30T04:40:04Z"
    assert len([n for n in result["input_sha256"] if n.startswith("training/")]) == 25
    assert before == {p: p.read_bytes() for p in cohort.output.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["wrong_scheduler", "changed0", "restart2", "wrong_gpu", "unknown_process", "source", "data", "manifest", "count", "date"])
def test_schedule_or_model_tamper_rejected_even_before_completion(cohort, damage):
    if damage == "wrong_scheduler": edit(cohort, "scheduling_annex.json", lambda v: v.update(scheduler_source_sha256="0" * 64))
    elif damage == "changed0": (cohort.output / "training/droq__balanced__seed0/model_tensors.pt").write_bytes(b"changed")
    elif damage == "restart2": edit(cohort, "training/droq__balanced__seed2/started.json", lambda v: v.update(pid=999))
    elif damage == "wrong_gpu": edit(cohort, "training/droq__balanced__seed1/record.json", lambda v: v["device"].update(visible_gpu_uuid="OTHER"))
    elif damage == "unknown_process": edit(cohort, "scheduling_annex.json", lambda v: v["resource_snapshot"].update(unfiltered_stdout=v["resource_snapshot"]["unfiltered_stdout"] + "GPU-0, 9999\n"))
    elif damage == "source": (cohort.source / "scripts/btceth_rl_droq.py").write_bytes(b"changed")
    elif damage == "data": cohort.dataset.write_bytes(b"changed")
    elif damage == "manifest": (cohort.output / "sources/scripts/btceth_rl_droq.py").write_bytes(b"changed")
    elif damage == "count": edit(cohort, "training/droq__balanced__seed0/record.json", lambda v: v["actual_counters"].update(q_inner_rounds=1))
    else: edit(cohort, "training/droq__balanced__seed0/record.json", lambda v: v["training_episode_history"][0].update(end_exclusive="2026-01-01T00:00:00Z"))
    value = verify(cohort)
    assert value["status"] == "failed" and not value["passed"] and value["failures"] and value["derived_cohort_envelope"] is None


@pytest.mark.parametrize("damage", ["exit7", "bool_exit", "original_success", "other_failure", "wrong_pid", "cohort_time", "partial", "worker_log", "missing_install", "performance_source", "completion_hash", "owner_token"])
def test_false_completion_exit_failure_identity_time_or_hash_never_pass(cohort, damage):
    finish(cohort)
    if damage == "exit7": edit(cohort, "accelerated_fit_process.json", lambda v: v.update(worker_exit_code=7))
    elif damage == "bool_exit": edit(cohort, "accelerated_fit_process.json", lambda v: v.update(worker_exit_code=False))
    elif damage == "original_success": edit(cohort, "fit_process.json", lambda v: v.update(status="completed", exit_code=0))
    elif damage == "other_failure":
        log = cohort.output / "fit.log"; log.write_text(log.read_text() + "RuntimeError: unrelated failure\n")
        edit(cohort, "fit_process.json", lambda v: v.update(log_sha256=schedule.sha(log)))
    elif damage == "wrong_pid": edit(cohort, "accelerated_seed4_worker.json", lambda v: v.update(pid=901))
    elif damage == "cohort_time": edit(cohort, "accelerated_fit_process.json", lambda v: v.update(completed_at="2026-10-30T03:00:00Z"))
    elif damage == "partial": (cohort.output / "training/droq__balanced__seed3/record.json").unlink()
    elif damage == "worker_log": (cohort.output / "accelerated_seed4.log").write_bytes(b"changed")
    elif damage == "missing_install": edit(cohort, "accelerated_seed4_worker.json", lambda v: v.update(performance_install_count=0))
    elif damage == "performance_source": (cohort.performance / "polyak_cuda_graph.py").write_bytes(b"changed")
    elif damage == "completion_hash": edit(cohort, "accelerated_fit_process.json", lambda v: v.update(completion_sha256="0" * 64))
    else: edit(cohort, "accelerated_fit.owner.json", lambda v: v.update(token="b" * 32))
    value = verify(cohort)
    assert value["status"] == "failed" and not value["passed"] and value["derived_cohort_envelope"] is None


def test_runtime_paths_may_be_remote_without_relabelling_local_evidence(cohort):
    finish(cohort)
    path = cohort.output / "accelerated_fit_process.json"; process = schedule.read(path)
    process["command"][process["command"].index("--output-dir") + 1] = "/remote/study"
    put(path, process)
    log = cohort.output / "fit.log"; log.write_text("FileExistsError: [Errno 17] File exists: '/remote/study/training/droq__balanced__seed4'\n")
    edit(cohort, "fit_process.json", lambda v: v.update(log_sha256=schedule.sha(log)))
    edit(cohort, "accelerated_fit_completion.json", lambda v: v["original_fit_terminal"].update(sha256=schedule.sha(cohort.output / "fit_process.json"), log_sha256=schedule.sha(log), collision_path="/remote/study/training/droq__balanced__seed4"))
    edit(cohort, "accelerated_fit_process.json", lambda v: v.update(completion_sha256=schedule.sha(cohort.output / "accelerated_fit_completion.json")))
    assert verify(cohort)["passed"] is True


def test_source_symlink_rejected_without_writes(cohort, monkeypatch):
    target = cohort.source / "scripts/btceth_rl_droq.py"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    value = verify(cohort)
    assert value["status"] == "failed" and "Links" in value["failures"][0]


def test_no_pending_success_when_actual_accelerated_supervisor_failed(cohort):
    edit(cohort, "accelerated_fit_process.json", lambda v: v.update(status="failed", exit_code=1))
    result = verify(cohort)
    assert result["status"] == "failed" and not result["passed"]


def reseal(cohort):
    """Recompute outer byte links so identity checks, rather than stale SHA, fail."""
    complete = cohort.output / "accelerated_fit_completion.json"
    value = schedule.read(complete)
    value["worker_proof_sha256"] = schedule.sha(cohort.output / "accelerated_seed4_worker.json")
    original = schedule.read(cohort.output / "fit_process.json")
    value["original_fit_terminal"].update(sha256=schedule.sha(cohort.output / "fit_process.json"), log_sha256=original["log_sha256"])
    put(complete, value)
    edit(cohort, "accelerated_fit_process.json", lambda v: v.update(completion_sha256=schedule.sha(complete)))


@pytest.mark.parametrize("damage", ["identity", "installation", "unrelated_error", "early_worker", "late_model", "canonical_before_original_exit", "outer_scope"])
def test_forged_but_rehashed_identity_or_time_is_independently_rejected(cohort, damage):
    finish(cohort)
    if damage == "identity": edit(cohort, "accelerated_seed4_worker.json", lambda v: v.update(pid=901))
    elif damage == "installation": edit(cohort, "accelerated_seed4_worker.json", lambda v: v.update(performance_install_count=0))
    elif damage == "unrelated_error":
        log = cohort.output / "fit.log"; log.write_text(log.read_text() + "RuntimeError: actual unrelated failure\n")
        edit(cohort, "fit_process.json", lambda v: v.update(log_sha256=schedule.sha(log)))
    elif damage == "early_worker": edit(cohort, "accelerated_seed4_worker.json", lambda v: v.update(started_at="2026-10-30T02:59:59Z"))
    elif damage == "late_model": edit(cohort, "accelerated_seed4_worker.json", lambda v: v.update(completed_at="2026-10-30T03:00:04Z"))
    elif damage == "canonical_before_original_exit":
        edit(cohort, "training_completion.json", lambda v: v.update(at="2026-10-30T04:39:59Z"))
        edit(cohort, "accelerated_fit_completion.json", lambda v: v.update(training_completion_sha256=schedule.sha(cohort.output / "training_completion.json")))
    else: edit(cohort, "accelerated_fit_completion.json", lambda v: v.update(no_signals=False))
    reseal(cohort)
    value = verify(cohort)
    assert value["status"] == "failed" and value["derived_cohort_envelope"] is None


def test_failed_original_unknown_error_is_failed_while_independent_worker_still_running(cohort):
    log = cohort.output / "fit.log"; log.write_text("RuntimeError: unrelated original failure\n")
    edit(cohort, "fit_process.json", lambda v: v.update(status="failed", exit_code=1, completed_at="2026-10-30T03:30:00Z", log_sha256=schedule.sha(log)))
    assert verify(cohort)["status"] == "failed"


def test_complete_original_collision_with_worker_in_progress_is_still_pending(cohort):
    log = cohort.output / "fit.log"
    log.write_text("FileExistsError: [Errno 17] File exists: '" + str(cohort.output).replace("\\", "/") + "/training/droq__balanced__seed4'\n")
    edit(cohort, "fit_process.json", lambda v: v.update(status="failed", exit_code=1, completed_at="2026-10-30T03:30:00Z", log_sha256=schedule.sha(log)))
    result = verify(cohort)
    assert result["status"] == "pending" and result["passed"] is False
