"""Fallback-only orchestration boundaries; no training, inference, SSH or GPU."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import finish_sac_fallback_audit_v3 as finish


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def passed(top, source, **extra):
    return {"passed": True, "failure_count": 0, "failures": [], "audit_source_sha256": source,
        "protocol_sha256": finish.sha(top / "protocol.json"),
        "inputs_sha256": {"protocol.json": finish.sha(top / "protocol.json")}, **extra}


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "_STATUS_OWNED", False)
    top = tmp_path / "study"; top.mkdir()
    data = tmp_path / "data.pkl.gz"; data.write_bytes(b"not market data: contract fixture only")
    baseline = {"fixture_recipe": "unchanged"}
    write(top / "protocol.json", {"base_recipe": baseline})
    write(top / "search_plan.json", {"fixture": True})
    write(top / "extension_manifest.json", {"extension_sources": {"scripts/backtest_sac_validation_optimization.py": "frozen-runner-sha"}})
    parent_path = top / finish.PARENT_CONTROLLER; parent_path.parent.mkdir(parents=True)
    parent_path.write_text("# pinned parent fixture, never executed", encoding="utf-8")
    write(top / "audit_revision.json", {"controller_source_sha256": finish.sha(parent_path),
        "revised_audit_source_sha256": finish.PARENT_AUDITOR_SHA256})
    write(top / "pipeline_launch_revision.json", {"pid": 12345})
    audit_path = top / finish.AUDITOR; audit_path.parent.mkdir(parents=True)
    audit_path.write_text("# independent v3 auditor fixture", encoding="utf-8")
    (top / finish.FINISHER).write_bytes(Path(finish.__file__).read_bytes())
    selection = {"at": "2026-10-31T04:49:08+00:00", "uses_test": False, "uses_final_validation": False,
        "seed_selection": False, "algorithm_selection": False, "selection": {s: {"candidate_id": "c00_baseline",
            "fallback_to_baseline": True, "selected_recipe": baseline} for s in finish.SYMBOLS}}
    write(top / "selection.json", selection)
    for name in ["tuning/" + c + "/validation_optimization_audit.json" for c in finish.CANDIDATES] + ["selection_audit.json"]:
        write(top / name, passed(top, finish.PARENT_AUDITOR_SHA256))
    write(top / finish.FAILED_AUDIT, {"passed": False, "failure_count": 1, "phase": "final-validation",
        "audit_source_sha256": finish.PARENT_AUDITOR_SHA256, "failures": ["AttributeError: 'str' object has no attribute 'get'"]})
    spec = {"recipe_ids_by_symbol": {s: "c00_baseline" for s in finish.SYMBOLS},
        "recipes_by_symbol": {s: baseline for s in finish.SYMBOLS}, "seeds": list(range(30)), "symbols": list(finish.SYMBOLS),
        "algorithms": ["sac"], "capital_per_account": 10000., "capital_shared_between_pairs": False}
    write(top / "final/protocol.json", spec)
    write(top / "final_prepared.json", {"retained_models": 60, "new_models": 0,
        "selection_sha256": finish.sha(top / "selection.json"), "final_protocol_sha256": finish.sha(top / "final/protocol.json")})
    records, rows, entries = [], [], []
    for symbol in finish.SYMBOLS:
        for seed in range(30):
            tail = f"training/{symbol}/sac/seed{seed}"
            source, destination = "references/" + symbol + "/" + tail, "final/" + tail
            record = {"symbol": symbol, "algorithm": "sac", "seed": seed, "training": {"counters": {"fixture": True}}}
            for directory in (source, destination):
                write(top / directory / "record.json", record)
            files = {"record.json": finish.sha(top / source / "record.json")}
            entries.append({"symbol": symbol, "seed": seed, "source_directory": source, "destination_directory": destination,
                "origin": "retained_baseline_fallback", "file_sha256": files})
            rows.append({"symbol": symbol, "seed": seed, "record_sha256": files["record.json"], "retained": True,
                "training_origin": "retained_baseline_fallback", "counters": record["training"]["counters"]})
            records.append(record)
    write(top / "final/import_manifest.json", {"selection_sha256": finish.sha(top / "selection.json"), "entries": entries})
    write(top / "final/fit_complete.json", {"model_count": 60, "records": records})
    write(top / "final/training_counter_audit.json", {"passed": True, "failure_count": 0, "failures": [], "models_checked": 60,
        "retained_models": 60, "new_models": 0, "models": rows, "core_protocol_sha256": finish.sha(top / "final/protocol.json"),
        "parent_protocol_sha256": finish.sha(top / "protocol.json"), "extension_source_sha256": "frozen-runner-sha"})
    write(top / "final_fit_complete.json", {"fresh_tasks": 0,
        "child_counter_audit_sha256": {"final": finish.sha(top / "final/training_counter_audit.json")}})
    write(top / "final_validation_complete.json", {"fresh_tasks": 0})
    revision = {"schema_version": 1, "revision_id": finish.REVISION_ID, "reason_code": finish.REASON,
        "created_at_utc": "2026-10-31T05:00:00+00:00", "original_protocol_sha256": finish.sha(top / "protocol.json"),
        "search_plan_sha256": finish.sha(top / "search_plan.json"), "parent_audit_revision_sha256": finish.sha(top / "audit_revision.json"),
        "parent_auditor_source_sha256": finish.PARENT_AUDITOR_SHA256, "revised_audit_source_path": finish.AUDITOR,
        "revised_audit_source_sha256": finish.sha(audit_path), "finisher_source_path": finish.FINISHER,
        "finisher_source_sha256": finish.sha(top / finish.FINISHER), "failed_final_validation_audit_sha256": finish.sha(top / finish.FAILED_AUDIT),
        "selection_sha256": finish.sha(top / "selection.json"), "selection_audit_sha256": finish.sha(top / "selection_audit.json"),
        "permitted_change": finish.PERMITTED_CHANGE, "financial_math_changed": False, "model_recipe_changed": False,
        "selection_changed": False, "training_restarted": False}
    write(top / "audit_import_revision.json", revision)
    write(top / "audit_import_revision_receipt.json", {"sha256": finish.sha(top / "audit_import_revision.json"), "at_utc": "2026-10-31T05:00:01+00:00"})
    calls = []
    parent = SimpleNamespace(verify_revision=lambda *args: calls.append("verify-parent"),
        validate_finished_tuning=lambda *args: calls.append("verify-forty"), process_identity=lambda pid: None)
    original_load = finish.load_source
    monkeypatch.setattr(finish, "load_source", lambda path, name: parent if name == "_sac_fallback_parent_controller" else original_load(path, name))
    write(top / "pipeline_launch_import_revision.json", {"pid": os.getpid(), "command": ["python", "-B", str(Path(finish.__file__).resolve())],
        "audit_import_revision_sha256": finish.sha(top / "audit_import_revision.json"), "finisher_source_sha256": revision["finisher_source_sha256"],
        "revised_audit_source_sha256": revision["revised_audit_source_sha256"],
        "parent_pipeline_launch_revision_sha256": finish.sha(top / "pipeline_launch_revision.json"), "training_restarted": False})
    return top, data, revision, parent, calls


def test_exact_fallback_contract_reuses_parent_source_and_forty_model_proofs(sealed):
    top, data, revision, _, calls = sealed
    assert finish.verify_contract(top, data) == revision
    assert calls == ["verify-parent", "verify-forty"]
    assert finish.wait_for_launch(top, revision)["pid"] == os.getpid()


@pytest.mark.parametrize("name", ["selection.json", "selection_audit.json", finish.FAILED_AUDIT,
    "audit_revision.json", "protocol.json", "search_plan.json", finish.AUDITOR, finish.FINISHER])
def test_every_old_and_new_sealed_binding_is_checked_before_work(sealed, name):
    top, data, *_ = sealed
    p = top / name; p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(ValueError): finish.verify_contract(top, data)


def reseal_revision(top):
    revision = finish.read(top / "audit_import_revision.json")
    revision["selection_sha256"] = finish.sha(top / "selection.json")
    write(top / "audit_import_revision.json", revision)
    write(top / "audit_import_revision_receipt.json", {"sha256": finish.sha(top / "audit_import_revision.json"), "at_utc": "2026-10-31T05:00:01+00:00"})


def test_nonbaseline_selection_is_rejected_even_if_receipt_is_consistent(sealed):
    top, data, *_ = sealed
    selection = finish.read(top / "selection.json"); selection["selection"]["ETHUSDT"]["candidate_id"] = "c01_lr1e4"
    write(top / "selection.json", selection); reseal_revision(top)
    with pytest.raises(ValueError, match="two-coin c00"): finish.verify_contract(top, data)


def test_new_model_or_incomplete_population_is_never_accepted_as_fallback(sealed):
    top, *_ = sealed
    p = top / "final/training_counter_audit.json"; value = finish.read(p); value["new_models"] = 1; write(p, value)
    with pytest.raises(ValueError, match="zero new"): finish.verify_retained_final(top)


def test_retained_native_checkpoint_bytes_cannot_change(sealed):
    top, *_ = sealed
    p = top / "final/training/ETHUSDT/sac/seed29/record.json"; p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"): finish.verify_retained_final(top)


def test_live_parent_blocks_duplicate_controller_without_signals(sealed):
    top, data, _, parent, _ = sealed
    parent.process_identity = lambda pid: {"state": "S"}
    with pytest.raises(ValueError, match="alongside"): finish.verify_contract(top, data)


def test_wrong_launch_has_no_status_ownership(sealed):
    top, _, revision, *_ = sealed
    p = top / "pipeline_launch_import_revision.json"; value = finish.read(p); value["pid"] = -1; write(p, value)
    with pytest.raises(ValueError, match="exact fallback"): finish.wait_for_launch(top, revision)
    assert not (top / "pipeline_status.json").exists()


def test_new_gate_points_to_v3_validation_without_replacing_failed_canonical(sealed):
    top, _, revision, *_ = sealed
    failed_sha = finish.sha(top / finish.FAILED_AUDIT)
    write(top / "final/validation_audit.json", {"financial": "fixture"})
    write(top / finish.VALIDATION_AUDIT, passed(top, revision["revised_audit_source_sha256"], core_financial_audit_passed=True))
    compatible, pins, shared = finish.annual_bindings(top, revision)
    assert pins["validation_optimization_audit_sha256"] == finish.sha(top / finish.VALIDATION_AUDIT)
    assert compatible["validation_optimization_audit_path"] == shared["validation_optimization_audit_path"] == finish.VALIDATION_AUDIT
    assert compatible["selection"] == {s: {"algorithm": "sac", "seed": 0} for s in finish.SYMBOLS}
    assert finish.sha(top / finish.FAILED_AUDIT) == failed_sha


def test_failed_or_tampered_audit_never_opens_annual(sealed, monkeypatch):
    top, data, revision, *_ = sealed
    write(top / finish.VALIDATION_AUDIT, passed(top, revision["revised_audit_source_sha256"], passed=False))
    monkeypatch.setattr(finish, "verify_contract", lambda *args: revision)
    monkeypatch.setattr(finish, "load_source", lambda *args: pytest.fail("Do not import original annual helpers before passed gate"))
    with pytest.raises(ValueError, match="successful"): finish.import_annual(top, data, revision)
    assert not (top / "final/test_started.json").exists()


def test_fallback_pipeline_invokes_only_new_auditor_and_import_no_fitting(sealed, monkeypatch):
    top, data, revision, *_ = sealed
    monkeypatch.setattr(finish, "verify_contract", lambda *args: revision)
    monkeypatch.setattr(finish, "wait_for_launch", lambda *args: {})
    imported, executed = [], []
    monkeypatch.setattr(finish, "import_annual", lambda *args: imported.append(True))
    def execute(command, **kwargs):
        executed.append(command); phase = command[command.index("--phase") + 1]
        marker = {"final-validation": finish.VALIDATION_AUDIT, "final-test": finish.TEST_AUDIT,
            "completion": "completion_contract_audit.json"}[phase]
        write(top / marker, passed(top, revision["revised_audit_source_sha256"], core_financial_audit_passed=True))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(finish.subprocess, "run", execute)
    result = finish.finish(top, data, workers=8)
    assert result["new_models"] == result["fresh_annual_accounts"] == 0 and imported == [True]
    assert len(executed) == 3 and all(c[2] == str(top / finish.AUDITOR) for c in executed)
    assert all(not any(x in c for x in ("fit", "pipeline", "final-test")) or "--phase" in c for c in executed)
    assert finish.read(top / "pipeline_status.json")["status"] == "completed"


def test_audit_failure_preserves_log_stops_before_annual_and_keeps_phase(sealed, monkeypatch):
    top, data, revision, *_ = sealed
    monkeypatch.setattr(finish, "verify_contract", lambda *args: revision)
    monkeypatch.setattr(finish, "wait_for_launch", lambda *args: {})
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    monkeypatch.setattr(finish, "import_annual", lambda *args: pytest.fail("Annual must remain closed"))
    failed_sha = finish.sha(top / finish.FAILED_AUDIT)
    with pytest.raises(RuntimeError, match="final-validation"): finish.finish(top, data)
    assert finish.read(top / "pipeline_status.json")["phase"] == "fallback-v3:final-validation"
    assert finish.read(top / "pipeline_status.json")["status"] == "failed"
    assert (top / "audit_import_revision_logs/final-validation.log").exists()
    assert finish.sha(top / finish.FAILED_AUDIT) == failed_sha


def test_existing_failed_versioned_marker_is_preserved_not_overwritten(sealed, monkeypatch):
    top, data, revision, *_ = sealed
    monkeypatch.setattr(finish, "verify_contract", lambda *args: revision)
    write(top / finish.VALIDATION_AUDIT, passed(top, revision["revised_audit_source_sha256"], passed=False))
    before = finish.sha(top / finish.VALIDATION_AUDIT)
    monkeypatch.setattr(finish.subprocess, "run", lambda *args, **kwargs: pytest.fail("No repeated audit over existing marker"))
    with pytest.raises(ValueError): finish.run_audit(top, data, revision, "final-validation", finish.VALIDATION_AUDIT)
    assert finish.sha(top / finish.VALIDATION_AUDIT) == before


def test_partial_annual_import_is_preserved_without_replaying_or_recopying(sealed, monkeypatch):
    top, data, revision, *_ = sealed
    write(top / "final/validation_audit.json", {"fixture": True})
    write(top / finish.VALIDATION_AUDIT, passed(top, revision["revised_audit_source_sha256"], core_financial_audit_passed=True))
    write(top / "final/test_started.json", {"incomplete_attempt": True})
    monkeypatch.setattr(finish, "verify_contract", lambda *args: revision)
    monkeypatch.setattr(finish, "load_source", lambda *args: SimpleNamespace(import_final_cases=lambda *args: pytest.fail("Preserve partial import")))
    with pytest.raises(FileExistsError, match="partial annual"): finish.import_annual(top, data, revision)
    assert finish.read(top / "final/test_started.json") == {"incomplete_attempt": True}


def test_original_import_case_and_frozen_csv_copy_136_accounts_then_reference_without_replay(sealed, monkeypatch):
    from scripts import backtest_sac_validation_optimization as original
    top, data, revision, *_ = sealed
    child = top / "final"
    reporting = Path(original.__file__).parent / "crypto_rl_reporting.py"
    target = top / "frozen_source/scripts/crypto_rl_reporting.py"; target.parent.mkdir(parents=True)
    target.write_bytes(reporting.read_bytes())
    protocol = finish.read(top / "protocol.json"); protocol["sources"] = {"scripts/crypto_rl_reporting.py": finish.sha(target)}
    write(top / "protocol.json", protocol)
    spec = {"seeds": list(range(30)), "costs": {"base": {}, "stress": {}},
        "recipe_ids_by_symbol": {s: "c00_baseline" for s in finish.SYMBOLS}}
    monkeypatch.setattr(original, "check_child", lambda *args: spec)
    monkeypatch.setattr(original, "_selection_gate", lambda *args: None)
    for forbidden in ("fit", "train_one", "evaluate", "evaluate_one"):
        monkeypatch.setattr(original, forbidden, lambda *args, **kwargs: pytest.fail("Fallback never fits or replays"))
    for symbol in finish.SYMBOLS:
        for algorithm, seed in [("sac", n) for n in range(30)] + [(a, None) for a in original.BASELINES]:
            for scenario in spec["costs"]:
                for kind in ("evaluation", "repeat"):
                    directory = top / "references" / symbol / original._case_directory("test", symbol, algorithm, seed, scenario, kind)
                    directory.mkdir(parents=True)
                    for name in original.RESULT_FILES:
                        if name == "summary.json":
                            write(directory / name, {"run_id": f"test__{scenario}__{symbol}__{algorithm}__{seed}", "net_return": .123})
                        else: (directory / name).write_bytes(b"identical pre-existing financial fixture\n")
    write(top / "final/validation_audit.json", {"arithmetic_fixture": True})
    write(top / finish.VALIDATION_AUDIT, passed(top, revision["revised_audit_source_sha256"], core_financial_audit_passed=True))
    monkeypatch.setattr(original, "import_annual_references", lambda root: write(top / "reference_annual_import.json",
        {"selection_sha256": finish.sha(top / "selection.json"), "files_sha256": {}}))
    imported = []
    native_import = original.import_final_cases
    runner = SimpleNamespace(BASELINES=original.BASELINES, check_child=original.check_child, _case_result=original._case_result,
        import_final_cases=lambda *args: (imported.append(True), native_import(*args)))
    original_load = finish.load_source
    monkeypatch.setattr(finish, "load_source", lambda path, name: runner if name == "_sac_fallback_frozen_runner" else original_load(path, name))
    monkeypatch.setattr(finish, "verify_contract", lambda *args: revision)
    assert finish.import_annual(top, data, revision) == {"fresh_accounts": 0, "accounts": 136, "referenced_existing": False}
    manifest = finish.read(top / "final/test_import_manifest.json")
    assert len(manifest["entries"]) == 272 and all(len(e["file_sha256"]) == 10 for e in manifest["entries"])
    completion = finish.read(top / "final/test_complete.json")
    assert len(completion["results"]) == 136 and completion["repeat_equal"] is True
    assert (top / "final/test_summary.csv").read_bytes().count(b"\n") == 137
    before = finish.sha(top / "final/test_started.json")
    assert finish.import_annual(top, data, revision) == {"fresh_accounts": 0, "accounts": 136, "referenced_existing": True}
    assert imported == [True] and finish.sha(top / "final/test_started.json") == before
    altered = child / original._case_directory("test", "ETHUSDT", "sac", 29, "stress", "repeat") / "fills.csv"
    altered.write_bytes(b"changed repeat")
    with pytest.raises(ValueError, match="changed"): finish.import_annual(top, data, revision)
