"""Fake byte evidence only: no SSH, policy inference or financial reports."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from scripts import collect_btceth_rl_exposure_cap as c


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_tree(tmp_path, *, full=False):
    main, root, source, data = tmp_path / "main", tmp_path / "cap", tmp_path / "repo", tmp_path / "data/dataset.pkl.gz"
    main.mkdir(); root.mkdir(); source.mkdir(); data.parent.mkdir()
    data.write_bytes(b"TEST_ONLY_NOT_MARKET_DATA")
    dump(data.parent / "manifest.json", {"synthetic": True})
    names = c.SOURCE_NAMES | {"scripts/audit_btceth_rl_exposure_cap.py", "scripts/audit_btceth_rl.py"}
    src = {}
    for name in names:
        p = source / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(("SYNTHETIC " + name).encode())
        src[name] = c.sha(p)
    spec = {"dataset_sha256": c.sha(data), "dataset_manifest_sha256": c.sha(data.parent / "manifest.json"), "code_sha256": {n: src[n] for n in c.FINANCIAL}}
    dump(main / "frozen_spec.json", spec); dump(main / "goal_candidate_protocol.json", {"synthetic": True})
    models, training = [], {}
    for profile in ("return", "balanced"):
        for seed in range(5):
            folder = f"training/tqc__{profile}__seed{seed}"
            model = main / folder / "model.zip"; model.parent.mkdir(parents=True); model.write_bytes(f"TEST_ONLY {profile} {seed}".encode())
            record = dict(algorithm="tqc", profile=profile, seed=seed, model_relative_path=folder + "/model.zip", model_sha256=c.sha(model), parameter_sha256=str(seed) * 64, frozen_spec_sha256=c.sha(main / "frozen_spec.json"))
            dump(main / folder / "record.json", record)
            for filename in ("record.json", "model.zip"):
                training[folder + "/" + filename] = c.sha(main / folder / filename)
            models.append(dict(profile=profile, seed=seed, record_file=folder + "/record.json", model_file=folder + "/model.zip", record_sha256=c.sha(main / folder / "record.json"), model_sha256=record["model_sha256"], parameter_sha256=record["parameter_sha256"]))
    protocol = dict(multiplier=.75, capital=10000., profiles=["return", "balanced"], seeds=list(range(5)), original_frozen_spec_sha256=c.sha(main / "frozen_spec.json"), original_goal_protocol_sha256=c.sha(main / "goal_candidate_protocol.json"), source_sha256={n: src[n] for n in c.SOURCE_NAMES}, dataset_sha256=spec["dataset_sha256"], dataset_manifest_sha256=spec["dataset_manifest_sha256"], training_source_sha256=training, models=models)
    dump(root / "protocol.json", protocol)
    dump(root / "protocol_receipt.json", {"protocol_file": "protocol.json", "protocol_sha256": c.sha(root / "protocol.json")})
    for name in c.SOURCE_NAMES:
        p = root / "sources" / name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source / name, p)
    required = {"protocol.json", "protocol_receipt.json", "validation_completion.json", "summary.csv", "daily_equity.csv", "monthly_returns.csv"} | c.group_names("validation", ("return", "balanced"))
    for name in required - {"protocol.json", "protocol_receipt.json"}:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"TEST_ONLY")
    dump(root / "validation_completion.json", dict(protocol_sha256=c.sha(root / "protocol.json"), case_count=24, repeat_count=24))
    runs = {f"validation__{cost}__cap075__tqc__{profile}__{key}": {"reconstructed": True, "hours": 7080} for cost in ("base", "stress") for profile in ("return", "balanced") for key in (*map(str, range(5)), "ensemble")}
    main_inputs = {"frozen_spec.json": c.sha(main / "frozen_spec.json"), **training}
    if full:
        required |= {"selection.json", "selection_receipt.json", "test_started.json", "test_completion.json"} | c.group_names("test", ("balanced",))
        for name in required:
            p = root / name
            if not p.exists(): p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"TEST_ONLY")
        dump(root / "selection.json", {"candidate": {"family": "cap075", "algorithm": "tqc", "profile": "balanced"}})
        dump(root / "test_completion.json", dict(protocol_sha256=c.sha(root / "protocol.json"), case_count=12, repeat_count=12))
        dump(main / "independent_audit.json", dict(passed=True, status="passed", test_complete=True, run_count=172, failures=[]))
        main_inputs["independent_audit.json"] = c.sha(main / "independent_audit.json")
        runs.update({f"test__{cost}__cap075__tqc__balanced__{key}": {"reconstructed": True, "hours": 8760} for cost in ("base", "stress") for key in (*map(str, range(5)), "ensemble")})
    phase = "test" if full else "validate"
    (root / (phase + ".log")).write_bytes(b"TEST_ONLY_SUCCESS_LOG")
    dump(root / (phase + "_process.json"), dict(phase=phase, status="completed", exit_code=0, started_at="2026-10-30T01:00:00+00:00", completed_at="2026-10-30T02:00:00+00:00", child_pid=2147483600, supervisor_pid=2147483601, protocol_sha256=c.sha(root / "protocol.json"), runner_sha256=src["scripts/btceth_rl_exposure_cap.py"], log_relative_path=phase + ".log", log_sha256=c.sha(root / (phase + ".log")), command=["python", "runner", "--phase", phase, "--output-dir", str(root.resolve()), "--main-output", str(main.resolve()), "--dataset", str(data.resolve())]))
    audit = dict(passed=True, status="passed" if full else "passed_validation_only", test_complete=full, run_count=36 if full else 24, failure_count=0, failures=[], checks=123, dataset_sha256=c.sha(data), runs=runs, input_sha256={n: c.sha(root / n) for n in required}, main_input_sha256=main_inputs, source_sha256=src, main_final_audit_path="independent_audit.json" if full else None, test_financial_reads_enabled=full, research_targets={"evaluated": full})
    dump(root / ("independent_audit.json" if full else "independent_audit.validation.json"), audit)
    return root, main, data, source


@pytest.mark.parametrize("full", [False, True])
def test_distinct_terminal24_and36_scopes_require_real_byte_evidence(tmp_path, full):
    root, main, data, source = fixture_tree(tmp_path, full=full)
    proof = c.terminal_guard(root, main=main, dataset=data, source_root=source, validation_only=not full)
    assert proof["mode"] == ("full" if full else "validation_only")


@pytest.mark.parametrize("target,field,value", [("process", "status", "running"), ("process", "exit_code", False), ("audit", "run_count", 23), ("audit", "test_complete", True)])
def test_unfinished_or_scope_masquerading_proofs_are_rejected(tmp_path, target, field, value):
    root, main, data, source = fixture_tree(tmp_path)
    path = root / ("validate_process.json" if target == "process" else "independent_audit.validation.json")
    item = c.read_json(path); item[field] = value; dump(path, item)
    with pytest.raises(ValueError):
        c.terminal_guard(root, main=main, dataset=data, source_root=source, validation_only=True)


@pytest.mark.parametrize("which", ["log", "main", "model", "source", "input"])
def test_changed_log_main_model_source_or_audited_input_blocks_export(tmp_path, which):
    root, main, data, source = fixture_tree(tmp_path)
    path = {"log": root / "validate.log", "main": main / "frozen_spec.json", "model": main / "training/tqc__return__seed0/model.zip", "source": source / "scripts/btceth_rl_exposure_cap.py", "input": root / "validation/balanced/base/ensemble/member_targets.csv"}[which]
    path.write_bytes(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        c.export_snapshot(root, main=main, dataset=data, source_root=source, validation_only=True, export_parent=tmp_path / "exports")
    assert not (tmp_path / "exports").exists()


def test_unique_archive_round_trip_preserves_remote_bytes_and_restores_guard(tmp_path):
    root, main, data, source = fixture_tree(tmp_path)
    before, guard = c.transport.inventory(root), c.transport.terminal_guard
    one = c.export_snapshot(root, main=main, dataset=data, source_root=source, validation_only=True, export_parent=tmp_path / "exports")
    two = c.export_snapshot(root, main=main, dataset=data, source_root=source, validation_only=True, export_parent=tmp_path / "exports")
    assert one["export_directory"] != two["export_directory"]
    assert c.transport.inventory(root) == before and c.transport.terminal_guard is guard
    manifest = c.read_json(one["manifest_path"])
    extracted = tmp_path / "extract"
    entries = c.transport.verify_extract(one["archive_path"], manifest, extracted)
    # Existing prepared bytes can be identical, while all other conflicts are
    # checked by the reused merge before any new result gets copied.
    local = tmp_path / "local"
    for name in ("protocol.json", "protocol_receipt.json", "sources/scripts/btceth_rl_exposure_cap.py"):
        p = local / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((root / name).read_bytes())
    merged = c.transport.merge_verified_files(extracted, entries, local)
    assert len(merged["identical_files_preserved"]) == 3
    assert c.sha(local / "independent_audit.validation.json") == c.sha(root / "independent_audit.validation.json")


def test_remote_source_mutation_during_archive_is_detected_and_guard_restored(tmp_path, monkeypatch):
    root, main, data, source = fixture_tree(tmp_path)
    original_inventory, original_guard = c.transport.inventory, c.transport.terminal_guard
    calls = 0
    def inventory(path):
        nonlocal calls
        calls += 1
        if calls == 2: (root / "summary.csv").write_bytes(b"mutated while exporting")
        return original_inventory(path)
    monkeypatch.setattr(c.transport, "inventory", inventory)
    with pytest.raises(RuntimeError, match="changed during snapshot"):
        c.export_snapshot(root, main=main, dataset=data, source_root=source, validation_only=True, export_parent=tmp_path / "exports")
    assert c.transport.terminal_guard is original_guard


def test_validation_snapshot_cannot_be_merged_into_prepared_or_existing_tree(tmp_path):
    root, main, data, source = fixture_tree(tmp_path)
    for target in (root, root / "nested", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            c.collect(output=target, reference_output=root, main_output=main, dataset_path=data, source_root=source, validation_only=True)


def test_malicious_tar_and_existing_conflict_are_rejected_before_merge(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../escape"); member.size = 1; tar.addfile(member, io.BytesIO(b"x"))
    manifest = dict(schema_version=1, files=[{"path": "safe", "size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}], file_count=1, total_bytes=1, archive_sha256=c.sha(archive), archive_size=archive.stat().st_size)
    with pytest.raises(ValueError):
        c.transport.verify_extract(archive, manifest, tmp_path / "extract_bad")
    root, main, data, source = fixture_tree(tmp_path)
    exported = c.export_snapshot(root, main=main, dataset=data, source_root=source, validation_only=True, export_parent=tmp_path / "exports")
    extracted = tmp_path / "extract_good"
    entries = c.transport.verify_extract(exported["archive_path"], c.read_json(exported["manifest_path"]), extracted)
    local = tmp_path / "local"; local.mkdir(); (local / "protocol.json").write_bytes(b"different existing preparation")
    with pytest.raises(FileExistsError, match="overwritten"):
        c.transport.merge_verified_files(extracted, entries, local)
    assert not (local / "summary.csv").exists()


@pytest.mark.parametrize("wire_change", [None, "audit_file", "audit_sha256"])
def test_mock_download_merges_prepared_bytes_and_preserves_remote_and_local_audits(tmp_path, monkeypatch, wire_change):
    root, main, data, source = fixture_tree(tmp_path, full=True)
    remote_root, remote_main, remote_data = "/home/research/cap", "/home/research/main", "/home/research/data/dataset.pkl.gz"
    process = c.read_json(root / "test_process.json")
    process["command"] = ["python", "runner", "--phase", "test", "--output-dir", remote_root, "--main-output", remote_main, "--dataset", remote_data]
    dump(root / "test_process.json", process)
    proof = c.terminal_guard(root, main=main, dataset=data, source_root=source,
        process_root=remote_root, process_main=remote_main, process_dataset=remote_data)
    files = c.transport.inventory(root)
    archive, manifest_path = tmp_path / "payload.tar.gz", tmp_path / "manifest.json"
    with tarfile.open(archive, "w:gz") as tar:
        for item in files:
            body = (root / item["path"]).read_bytes()
            info = tarfile.TarInfo(item["path"]); info.size = len(body); tar.addfile(info, io.BytesIO(body))
    manifest = dict(schema_version=1, created_at="2026-10-30T00:00:00Z", mode="full", remote_root=remote_root,
        files=files, file_count=len(files), total_bytes=sum(f["size"] for f in files), terminal_guard=proof,
        archive_name="experiment_snapshot.tar.gz", archive_sha256=c.sha(archive), archive_size=archive.stat().st_size)
    dump(manifest_path, manifest)
    export = "/home/research/exports/full-/20261030T000000Z-" + "a" * 32
    response = dict(status="complete", mode="full", remote_root=remote_root, export_directory=export,
        archive_path=export + "/experiment_snapshot.tar.gz", manifest_path=export + "/snapshot_manifest.json",
        archive_sha256=c.sha(archive), archive_size=archive.stat().st_size, manifest_sha256=c.sha(manifest_path),
        frozen_spec_sha256=proof["frozen_spec_sha256"], protocol_sha256=proof["protocol_sha256"],
        audit_file=proof["audit_file"], audit_sha256=proof["audit_sha256"],
        reused_transport_sha256=c.sha(Path(c.transport.__file__)))
    calls = []
    def remote_call(alias, python, args, content, *, destination=None):
        calls.append(args)
        assert args[args.index("--main-output") + 1] == remote_main
        assert args[args.index("--dataset") + 1] == remote_data
        if destination is None:
            if wire_change == "audit_file": return {**response, "audit_file": "wrong_audit.json"}
            if wire_change == "audit_sha256": return {**response, "audit_sha256": "0" * 64}
            return response
        artifact = args[args.index("--remote-read-file") + 1]
        shutil.copyfile(manifest_path if artifact.endswith("snapshot_manifest.json") else archive, destination)
    monkeypatch.setattr(c.transport, "_remote_call", remote_call)
    local = tmp_path / "prepared_local"; local.mkdir()
    for name in ("protocol.json", "protocol_receipt.json"):
        shutil.copyfile(root / name, local / name)
    shutil.copytree(root / "sources", local / "sources")
    (local / "independent_audit.local.json").write_bytes(b"PREEXISTING_SEPARATE_LOCAL_AUDIT")
    kwargs = dict(output=local, reference_output=local, main_output=main, dataset_path=data, source_root=source,
        remote_root=remote_root, remote_main=remote_main, remote_dataset=remote_data,
        export_parent="/home/research/exports", staging=tmp_path / "new_staging")
    if wire_change:
        with pytest.raises(ValueError, match="audit"):
            c.collect(**kwargs)
        assert not (local / "summary.csv").exists()
        assert not (local / "transfer_receipt.json").exists()
        return
    result = c.collect(**kwargs)
    assert result["status"] == "collected" and len(calls) == 3
    assert c.sha(local / "independent_audit.json") == c.sha(root / "independent_audit.json")
    assert (local / "independent_audit.local.json").read_bytes() == b"PREEXISTING_SEPARATE_LOCAL_AUDIT"
    receipt = c.read_json(local / "transfer_receipt.json")
    exact = {item["path"]: item["sha256"] for item in receipt["exactfiles"]}
    assert exact["independent_audit.json"] == c.sha(root / "independent_audit.json")
    assert "independent_audit.local.json" not in exact
    assert len(receipt["identical_files_preserved"]) == 2 + len(c.SOURCE_NAMES)
    assert not (root / "transfer_receipt.json").exists()


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", "C:/outside.json", "a\\outside.json", "NUL.json", "./audit.json", "audit.csv"])
def test_explicit_validation_audit_cannot_escape_or_change_type(name):
    with pytest.raises(ValueError):
        c.audit_filename(True, name)


def test_failed_default_audit_is_preserved_and_explicit_corrected_export_succeeds(tmp_path):
    root, main, data, source = fixture_tree(tmp_path)
    failed_path = root / c.DEFAULT_VALIDATION_AUDIT
    corrected_name = "independent_audit.validation.corrected.json"
    good = c.read_json(failed_path)
    dump(root / corrected_name, good)
    dump(failed_path, {**good, "passed": False, "status": "failed", "failure_count": 824, "failures": ["ORIGINAL_FAILED_PROOF"]})
    original_failed_sha = c.sha(failed_path)
    with pytest.raises(ValueError, match="Independent overlay audit"):
        c.terminal_guard(root, main=main, dataset=data, source_root=source, validation_only=True)
    guard = c.terminal_guard(root, main=main, dataset=data, source_root=source, validation_only=True, validation_audit_name=corrected_name)
    assert guard["audit_file"] == corrected_name and guard["audit_sha256"] == c.sha(root / corrected_name)
    exported = c.export_snapshot(root, main=main, dataset=data, source_root=source, validation_only=True,
        validation_audit_name=corrected_name, export_parent=tmp_path / "exports")
    assert exported["audit_file"] == corrected_name and exported["audit_sha256"] == guard["audit_sha256"]
    manifest = c.read_json(exported["manifest_path"])
    extracted = tmp_path / "extracted"
    c.transport.verify_extract(exported["archive_path"], manifest, extracted)
    assert c.sha(extracted / c.DEFAULT_VALIDATION_AUDIT) == original_failed_sha == c.sha(failed_path)
    assert c.terminal_guard(extracted, main=main, dataset=data, source_root=source, validation_only=True,
        validation_audit_name=corrected_name, process_root=root, process_main=main, process_dataset=data) == guard


@pytest.mark.parametrize("name", ["missing.json", "protocol.json"])
def test_wrong_explicit_audit_name_does_not_fall_back_to_another_passed_audit(tmp_path, name):
    root, main, data, source = fixture_tree(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        c.terminal_guard(root, main=main, dataset=data, source_root=source, validation_only=True, validation_audit_name=name)


def test_corrected_audit_cannot_bypass_input_sha_and_cannot_replace_full_scope(tmp_path):
    root, main, data, source = fixture_tree(tmp_path)
    corrected = c.read_json(root / c.DEFAULT_VALIDATION_AUDIT)
    corrected["input_sha256"]["summary.csv"] = "0" * 64
    dump(root / "independent_audit.validation.corrected.json", corrected)
    with pytest.raises(ValueError, match="Audited extension input changed"):
        c.terminal_guard(root, main=main, dataset=data, source_root=source, validation_only=True,
            validation_audit_name="independent_audit.validation.corrected.json")
    with pytest.raises(ValueError, match="cannot replace the full36"):
        c.audit_filename(False, "independent_audit.validation.corrected.json")
