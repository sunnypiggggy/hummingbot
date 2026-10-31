"""Metadata/control-flow proofs; no signals, SSH, model fitting or inference."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continue_sac_validation_optimization as controller


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    top, dataset = tmp_path / "study", tmp_path / "data/dataset.pkl.gz"
    dataset.parent.mkdir()
    dataset.write_bytes(b"metadata unit fixture only, not market data")
    write(dataset.parent / "manifest.json", {"fixture": True})
    top.mkdir()
    old_names = ("scripts/backtest_sac_validation_optimization.py", "scripts/sac_validation_optimization_models.py", controller.OLD_AUDITOR)
    for name in old_names:
        target = top / "extension_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# old immutable source metadata fixture " + name, encoding="utf-8")
    oracle = top / "frozen_source/scripts/audit_single_pair_rl.py"
    oracle.parent.mkdir(parents=True)
    oracle.write_text("# oracle metadata fixture; never executed", encoding="utf-8")
    new_audit = top / controller.NEW_AUDITOR
    new_audit.parent.mkdir(parents=True)
    new_audit.write_text("# v2 metadata fixture; never executed", encoding="utf-8")
    (top / controller.CONTROLLER).write_bytes(Path(controller.__file__).read_bytes())
    write(top / "protocol.json", {"dataset_sha256": controller.sha(dataset),
        "dataset_manifest_sha256": controller.sha(dataset.parent / "manifest.json"),
        "sources": {"scripts/audit_single_pair_rl.py": controller.sha(oracle)}})
    write(top / "protocol_receipt.json", {"protocol_sha256": controller.sha(top / "protocol.json")})
    write(top / "search_plan.json", {"fixture": True})
    write(top / "search_plan_receipt.json", {"sha256": controller.sha(top / "search_plan.json")})
    extension = {"extension_sources": {name: controller.sha(top / "extension_source" / name) for name in old_names}}
    write(top / "extension_manifest.json", extension)
    command = ["python", "-B", str(top / controller.DRIVER), "tuning-fit", "--output", str(top), "--dataset", str(dataset)]
    write(top / "pipeline_launch.json", {"pid": 1234, "command": command[:3] + ["pipeline"] + command[4:]})
    handoff = {"requested_at_utc": "2026-10-31T04:23:25+00:00", "reason_code": controller.REASON,
        "old_controller_pid": 1234, "old_controller_start_ticks": 11, "active_tuning_fit_pid": 1235,
        "active_tuning_fit_start_ticks": 22, "active_tuning_fit_command": command + [""],
        "original_pipeline_launch_sha256": controller.sha(top / "pipeline_launch.json"),
        "protocol_sha256": controller.sha(top / "protocol.json"), "original_auditor_sha256": extension["extension_sources"][controller.OLD_AUDITOR],
        "training_restarted": False, "all_frozen_inputs_preserved": True}
    write(top / "controller_handoff_requested.json", handoff)
    write(top / "controller_handoff_confirmed.json", {**handoff, "active_fit_still_alive": True})
    revision = {"schema_version": 1, "revision_id": controller.REVISION_ID, "reason_code": controller.REASON,
        "created_at_utc": "2026-10-31T04:24:00+00:00", "original_protocol_sha256": controller.sha(top / "protocol.json"),
        "search_plan_sha256": controller.sha(top / "search_plan.json"), "original_extension_manifest_sha256": controller.sha(top / "extension_manifest.json"),
        "original_audit_source_sha256": extension["extension_sources"][controller.OLD_AUDITOR],
        "revised_audit_source_path": controller.NEW_AUDITOR, "revised_audit_source_sha256": controller.sha(new_audit),
        "controller_source_path": controller.CONTROLLER, "controller_source_sha256": controller.sha(top / controller.CONTROLLER),
        "original_pipeline_launch_sha256": controller.sha(top / "pipeline_launch.json"),
        "financial_oracle_source_sha256": controller.sha(oracle), "permitted_metadata_normalization": controller.NORMALIZATION,
        "financial_math_changed": False, "model_recipe_changed": False, "training_restarted": False}
    write(top / "audit_revision.json", revision)
    write(top / "audit_revision_receipt.json", {"sha256": controller.sha(top / "audit_revision.json"), "at_utc": "2026-10-31T04:24:01+00:00"})
    launch = {"pid": os.getpid(), "command": ["python", "-B", str(Path(controller.__file__).resolve()), "--output", str(top), "--dataset", str(dataset)],
        "original_pipeline_launch_sha256": revision["original_pipeline_launch_sha256"], "inherited_tuning_fit_pid": 1235,
        "inherited_tuning_fit_start_ticks": 22, "audit_revision_sha256": controller.sha(top / "audit_revision.json"),
        "controller_source_sha256": revision["controller_source_sha256"], "revised_audit_source_sha256": revision["revised_audit_source_sha256"], "training_restarted": False}
    write(top / "pipeline_launch_revision.json", launch)
    monkeypatch.setattr(controller, "process_identity", lambda pid: None)
    return top, dataset, revision, {**handoff, "active_fit_still_alive": True}


def test_only_exact_versioned_metadata_correction_is_accepted(sealed):
    top, dataset, revision, handoff = sealed
    actual, confirmed = controller.verify_revision(top, dataset)
    assert actual == revision and confirmed["active_tuning_fit_pid"] == handoff["active_tuning_fit_pid"]
    assert controller.wait_for_launch(top, revision, handoff)["pid"] == os.getpid()


@pytest.mark.parametrize("name", ["protocol.json", "search_plan.json", "pipeline_launch.json", "extension_manifest.json",
    "extension_source/" + controller.OLD_AUDITOR, controller.NEW_AUDITOR, controller.CONTROLLER])
def test_original_and_revision_bytes_are_never_silently_replaced(sealed, name):
    top, dataset, _, _ = sealed
    path = top / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        controller.verify_revision(top, dataset)


def test_live_original_controller_blocks_duplicate_orchestration(sealed, monkeypatch):
    top, dataset, _, _ = sealed
    monkeypatch.setattr(controller, "process_identity", lambda pid: {"state": "S", "start_ticks": "11"})
    with pytest.raises(ValueError, match="still active"):
        controller.verify_revision(top, dataset)


def test_inherited_fit_waits_without_any_training_launch(sealed, monkeypatch):
    top, dataset, revision, handoff = sealed
    monkeypatch.setattr(controller, "verify_revision", lambda *args: (revision, handoff))
    states = iter([{ "pid": 1235, "state": "R", "start_ticks": "22", "command": controller.normalized(handoff["active_tuning_fit_command"])}, None])
    monkeypatch.setattr(controller, "process_identity", lambda pid: next(states))
    calls = []
    monkeypatch.setattr(controller.time, "sleep", lambda interval: calls.append(interval))
    monkeypatch.setattr(controller, "validate_finished_tuning", lambda top: {"fresh_tasks": 40})
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: pytest.fail("Inherited fit must never be relaunched"))
    assert controller.wait_for_existing_fit(top, dataset, handoff, interval=30) == {"fresh_tasks": 40}
    assert calls == [30] and controller.read(top / "pipeline_status.json")["phase"] == "inherited-tuning-fit"


def test_reused_fit_pid_is_rejected_without_touching_it(sealed, monkeypatch):
    top, dataset, revision, handoff = sealed
    monkeypatch.setattr(controller, "verify_revision", lambda *args: (revision, handoff))
    monkeypatch.setattr(controller, "process_identity", lambda pid: {"pid": 1235, "state": "S", "start_ticks": "999", "command": ["other-program"]})
    with pytest.raises(ValueError, match="reused"):
        controller.wait_for_existing_fit(top, dataset, handoff)


def make_tuning_metadata(top):
    extension = controller.read(top / "extension_manifest.json")
    mapping = {}
    for candidate in controller.CANDIDATES:
        child = top / "tuning" / candidate
        write(child / "protocol.json", {"fixture_candidate": candidate})
        records, rows = [], []
        for symbol in controller.SYMBOLS:
            for seed in controller.SEEDS:
                directory = child / f"training/{symbol}/sac/seed{seed}"
                counters = {"environment_steps": 32768, "gradient_update_rounds": 32256,
                    "optimizer_steps": {"actor": 32256, "critic": 32256, "entropy": 32256},
                    "replay_add_calls": 32768, "replay_transitions": 32768}
                write(directory / "checkpoint/fixture.json", {"not_a_native_model": True})
                record = {"symbol": symbol, "algorithm": "sac", "seed": seed, "candidate_id": candidate, "stage": "tuning",
                    "training": {"actual_steps": 32768, "counters": counters}, "files": {"checkpoint/fixture.json": controller.sha(directory / "checkpoint/fixture.json")}}
                write(directory / "record.json", record)
                records.append(record)
                rows.append({"symbol": symbol, "seed": seed, "candidate_id": candidate, "training_origin": "fresh",
                    "record_sha256": controller.sha(directory / "record.json"), "counters": counters})
        write(child / "fit_complete.json", {"model_count": 10, "records": records})
        write(child / "training_counter_audit.json", {"passed": True, "failure_count": 0, "failures": [], "models_checked": 10,
            "new_models": 10, "retained_models": 0, "models": rows, "core_protocol_sha256": controller.sha(child / "protocol.json"),
            "parent_protocol_sha256": controller.sha(top / "protocol.json"),
            "extension_source_sha256": extension["extension_sources"]["scripts/backtest_sac_validation_optimization.py"]})
        mapping["tuning/" + candidate] = controller.sha(child / "training_counter_audit.json")
    write(top / "tuning_fit_status.json", {"completed": 40, "total": 40, "failed": []})
    write(top / "tuning_fit_complete.json", {"fresh_tasks": 40, "child_counter_audit_sha256": mapping})


def test_full_inherited_work_must_bind_forty_records_counters_and_files(sealed):
    top, _, _, _ = sealed
    make_tuning_metadata(top)
    assert controller.validate_finished_tuning(top)["fresh_tasks"] == 40
    p = top / "tuning/c03_lr1e4_gamma099/training/ETHUSDT/sac/seed104/record.json"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        controller.validate_finished_tuning(top)


def test_missing_terminal_fit_never_causes_retraining(sealed):
    top, _, _, _ = sealed
    with pytest.raises(FileNotFoundError):
        controller.validate_finished_tuning(top)


def pipeline_fixture(sealed, monkeypatch, *, fail_first=False):
    top, dataset, revision, handoff = sealed
    monkeypatch.setattr(controller, "verify_revision", lambda *args: (revision, handoff))
    monkeypatch.setattr(controller, "wait_for_launch", lambda *args: {})
    waited = []
    monkeypatch.setattr(controller, "wait_for_existing_fit", lambda *args, **kwargs: waited.append(True))
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if fail_first:
            return SimpleNamespace(returncode=1)
        if "--phase" in command:
            phase = command[command.index("--phase") + 1]
            if phase == "tuning-validation":
                cid = command[command.index("--candidate") + 1]; target = top / "tuning" / cid / "validation_optimization_audit.json"
            else:
                target = top / {"selection": "selection_audit.json", "final-validation": "final/validation_optimization_audit.json",
                                "final-test": "final/test_optimization_audit.json", "completion": "completion_contract_audit.json"}[phase]
            write(target, {"passed": True, "failure_count": 0, "audit_source_sha256": revision["revised_audit_source_sha256"],
                "inputs_sha256": {"protocol.json": controller.sha(top / "protocol.json")}})
        else:
            phase = command[3]
            name = {"tuning-validation": "tuning_validation_complete.json", "select": "selection.json", "final-prepare": "final_prepared.json",
                    "final-fit": "final_fit_complete.json", "final-validation": "final_validation_complete.json", "final-test": "final_test_complete.json"}[phase]
            write(top / name, {"control_flow_fixture_only": True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(controller.subprocess, "run", execute)
    return top, dataset, calls, waited


def test_continuation_only_runs_old_remaining_phases_and_new_auditor(sealed, monkeypatch):
    top, dataset, calls, waited = pipeline_fixture(sealed, monkeypatch)
    result = controller.continue_pipeline(top, dataset)
    assert result["completed"] is True and waited == [True]
    assert all("tuning-fit" not in command and "pipeline" not in command for command in calls)
    assert sum("final-fit" in command for command in calls) == 1
    assert all(command[2] == str(top / controller.NEW_AUDITOR) for command in calls if "--phase" in command)
    assert controller.read(top / "pipeline_status.json")["status"] == "completed"
    assert not (top / "tuning-fit.log").exists() and (top / "audit_revision_logs/tuning-validation.log").is_file()


def test_failed_continuation_phase_preserves_log_and_never_runs_later_work(sealed, monkeypatch):
    top, dataset, calls, waited = pipeline_fixture(sealed, monkeypatch, fail_first=True)
    with pytest.raises(RuntimeError, match="tuning-validation"):
        controller.continue_pipeline(top, dataset)
    assert waited == [True] and len(calls) == 1
    assert controller.read(top / "pipeline_status.json")["status"] == "failed"
    assert (top / "audit_revision_logs/tuning-validation.log").is_file()


def test_unowned_launch_cannot_overwrite_active_status(sealed, monkeypatch):
    top, dataset, _, _ = sealed
    write(top / "pipeline_status.json", {"status": "running", "owner": "existing"})
    prior = (top / "pipeline_status.json").read_bytes()
    launch = controller.read(top / "pipeline_launch_revision.json"); launch["pid"] = os.getpid() + 1
    write(top / "pipeline_launch_revision.json", launch)
    monkeypatch.setattr(controller, "_STATUS_OWNED", False)
    monkeypatch.setattr(controller.sys, "argv", ["prog", "--output", str(top), "--dataset", str(dataset)])
    with pytest.raises(ValueError):
        controller.main()
    assert (top / "pipeline_status.json").read_bytes() == prior
