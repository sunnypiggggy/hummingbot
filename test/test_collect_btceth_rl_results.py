"""Fake terminal studies and archives; no SSH, training or production access."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import tarfile

import pytest

from scripts import collect_btceth_rl_results as collector


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    collector.write_new_json(path, value)


def terminal_study(tmp_path, *, validation=False, spec=None):
    root = tmp_path / "experiment"
    root.mkdir()
    spec = spec or {"dataset_sha256": "a" * 64, "code_sha256": {"scripts/model.py": "b" * 64}}
    dump(root / "frozen_spec.json", spec)
    protocol_hash = collector.sha(root / "frozen_spec.json")
    names = ["validation_selection.json", "training_completion.json", "summary.csv", "daily_equity.csv",
             "monthly_returns.csv", "validation_completion.json"]
    if not validation:
        names += ["test_started.json", "test_completion.json"]
    for name in names:
        if name.endswith("_completion.json"):
            dump(root / name, {"frozen_spec_sha256": protocol_hash})
        else:
            (root / name).write_bytes(b"original bytes\n")
    phase = "fit" if validation else "test"
    (root / (phase + ".log")).write_bytes(b"research terminal success\n")
    dump(root / (phase + "_process.json"),
         {"phase": phase, "status": "completed", "exit_code": 0,
          "frozen_spec_sha256": protocol_hash, "log_relative_path": phase + ".log",
          "log_sha256": collector.sha(root / (phase + ".log"))})
    inputs = {name: collector.sha(root / name) for name in ["frozen_spec.json", *names]}
    dump(root / ("validation_audit.json" if validation else "independent_audit.json"),
         {"passed": True, "status": "partial_test_pending" if validation else "passed",
          "test_complete": not validation, "failures": [], "checks_total": 100,
          "run_count": 110 if validation else 172, "frozen_spec_sha256": protocol_hash,
          "dataset_sha256": spec["dataset_sha256"], "source_sha256": spec["code_sha256"],
          "input_sha256": inputs})
    return root


def fake_archive(tmp_path, members, *, expected=None):
    path = tmp_path / "snapshot.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        for name, body, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind in (tarfile.REGTYPE, tarfile.AREGTYPE):
                info.size = len(body)
                stream.addfile(info, io.BytesIO(body))
            else:
                info.linkname = "../../outside"
                stream.addfile(info)
    ordinary = [(name, body) for name, body, kind in members if kind in (tarfile.REGTYPE, tarfile.AREGTYPE)]
    content = expected if expected is not None else dict(ordinary)
    files = [{"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name, body in content.items()]
    manifest = {"schema_version": 1, "files": files, "file_count": len(files),
                "total_bytes": sum(item["size"] for item in files),
                "archive_sha256": collector.sha(path), "archive_size": path.stat().st_size}
    return path, manifest


def local_reference(tmp_path):
    reference = tmp_path / "formal_output"
    source = tmp_path / "source"
    data = tmp_path / "data" / "dataset.pkl.gz"
    data.parent.mkdir()
    data.write_bytes(b"frozen prepared dataset")
    dump(data.parent / "manifest.json", {"dataset_sha256": collector.sha(data)})
    script = source / "scripts" / "runner.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"frozen source\n")
    spec = {"dataset_sha256": collector.sha(data),
            "dataset_manifest_sha256": collector.sha(data.parent / "manifest.json"),
            "code_sha256": {"scripts/runner.py": collector.sha(script)}}
    dump(reference / "frozen_spec.json", spec)
    return reference, data, source, spec


def test_full_and_validation_terminal_guards_require_distinct_finished_scopes(tmp_path):
    full = terminal_study(tmp_path)
    proof = collector.terminal_guard(full)
    assert proof["mode"] == "full"
    other = tmp_path / "second"
    other.mkdir()
    validation = terminal_study(other, validation=True)
    assert collector.terminal_guard(validation, validation_only=True)["mode"] == "validation_only"
    with pytest.raises(ValueError):
        collector.terminal_guard(validation)


@pytest.mark.parametrize("field,value", [("status", "running"), ("status", "failed"), ("exit_code", 1), ("exit_code", False)])
def test_running_failed_or_nonzero_process_cannot_be_collected(tmp_path, field, value):
    root = terminal_study(tmp_path)
    path = root / "test_process.json"
    proof = collector.read_json(path)
    proof[field] = value
    path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError, match="process"):
        collector.terminal_guard(root)


@pytest.mark.parametrize("field,value", [("passed", False), ("test_complete", False),
                                        ("status", "partial_test_pending"), ("run_count", 171),
                                        ("failures", ["bad cash"])])
def test_partial_or_failed_audit_cannot_be_presented_as_complete(tmp_path, field, value):
    root = terminal_study(tmp_path)
    path = root / "independent_audit.json"
    audited = collector.read_json(path)
    audited[field] = value
    path.write_text(json.dumps(audited), encoding="utf-8")
    with pytest.raises(ValueError, match="audit"):
        collector.terminal_guard(root)


def test_audited_input_or_terminal_log_change_is_detected(tmp_path):
    root = terminal_study(tmp_path)
    (root / "summary.csv").write_bytes(b"changed after independent audit")
    with pytest.raises(ValueError, match="Audited input changed"):
        collector.terminal_guard(root)
    other = tmp_path / "second"
    other.mkdir()
    root = terminal_study(other)
    (root / "test.log").write_bytes(b"changed process log")
    with pytest.raises(ValueError, match="log"):
        collector.terminal_guard(root)


def test_export_is_new_outside_experiment_and_preserves_all_original_bytes(tmp_path):
    root = terminal_study(tmp_path)
    before = collector.inventory(root)
    first = collector.export_snapshot(root, export_parent=tmp_path / "exports")
    second = collector.export_snapshot(root, export_parent=tmp_path / "exports")
    assert first["export_directory"] != second["export_directory"]
    assert collector.inventory(root) == before
    assert not Path(first["export_directory"]).is_relative_to(root)
    manifest = collector.read_json(first["manifest_path"])
    extracted = tmp_path / "extracted"
    entries = collector.verify_extract(first["archive_path"], manifest, extracted)
    assert collector.inventory(extracted) == before
    assert len(entries) == len(before)
    assert collector.terminal_guard(extracted) == manifest["terminal_guard"]


def test_export_inside_experiment_is_rejected_before_archive_creation(tmp_path):
    root = terminal_study(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        collector.export_snapshot(root, export_parent=root / "exports")
    assert not (root / "exports").exists()


@pytest.mark.parametrize("bad_name", ["../outside", "/absolute", "C:/outside", "a\\outside", "NUL", "COM1.txt", "a/../outside", "./outside"])
def test_unsafe_tar_names_cannot_escape_staging(tmp_path, bad_name):
    archive, manifest = fake_archive(tmp_path, [(bad_name, b"bad", tarfile.REGTYPE)], expected={"safe.txt": b"bad"})
    with pytest.raises(ValueError):
        collector.verify_extract(archive, manifest, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_links_hardlinks_and_nonregular_tar_members_are_rejected(tmp_path, kind):
    archive, manifest = fake_archive(tmp_path, [("file.txt", b"", kind)], expected={"file.txt": b""})
    with pytest.raises(ValueError, match="nonregular"):
        collector.verify_extract(archive, manifest, tmp_path / "extract")


def test_duplicate_or_case_colliding_archive_members_are_rejected(tmp_path):
    archive, manifest = fake_archive(tmp_path, [("file.txt", b"a", tarfile.REGTYPE), ("file.txt", b"a", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="duplicate"):
        collector.verify_extract(archive, manifest, tmp_path / "extract")
    manifest["files"] = [{"path": name, "size": 1, "sha256": hashlib.sha256(b"a").hexdigest()} for name in ("File.txt", "file.txt")]
    manifest["file_count"], manifest["total_bytes"] = 2, 2
    with pytest.raises(ValueError, match="colliding"):
        collector.manifest_entries(manifest)


def test_archive_or_member_sha_mismatch_never_reaches_merge(tmp_path):
    archive, manifest = fake_archive(tmp_path, [("file.txt", b"original", tarfile.REGTYPE)])
    bad = deepcopy(manifest)
    bad["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="archive SHA256"):
        collector.verify_extract(archive, bad, tmp_path / "bad_archive")
    assert not (tmp_path / "bad_archive").exists()
    bad = deepcopy(manifest)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="member SHA256"):
        collector.verify_extract(archive, bad, tmp_path / "bad_member")


def test_wrong_size_missing_files_and_expansion_limit_are_rejected(tmp_path):
    archive, manifest = fake_archive(tmp_path, [("file.txt", b"original", tarfile.REGTYPE)])
    bad = deepcopy(manifest)
    bad["files"][0]["size"] += 1
    bad["total_bytes"] += 1
    with pytest.raises(ValueError, match="wrong size"):
        collector.verify_extract(archive, bad, tmp_path / "wrong_size")
    with pytest.raises(ValueError, match="expansion"):
        collector.verify_extract(archive, manifest, tmp_path / "too_large", max_total_bytes=1)
    bad = deepcopy(manifest)
    bad["files"].append({"path": "absent.txt", "size": 0, "sha256": hashlib.sha256(b"").hexdigest()})
    bad["file_count"] += 1
    with pytest.raises(ValueError, match="missing"):
        collector.verify_extract(archive, bad, tmp_path / "missing")


def verified_source(tmp_path):
    archive, manifest = fake_archive(tmp_path, [("frozen_spec.json", b"same frozen protocol", tarfile.REGTYPE),
                                               ("evaluation/result.csv", b"original raw remote bytes\n", tarfile.REGTYPE)])
    extracted = tmp_path / "verified"
    entries = collector.verify_extract(archive, manifest, extracted)
    return extracted, entries


def test_conflict_is_found_before_copying_any_new_result_file(tmp_path):
    extracted, entries = verified_source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "frozen_spec.json").write_bytes(b"different preexisting protocol")
    with pytest.raises(FileExistsError, match="overwritten"):
        collector.merge_verified_files(extracted, entries, output)
    assert (output / "frozen_spec.json").read_bytes() == b"different preexisting protocol"
    assert not (output / "evaluation").exists()


def test_identical_protocol_is_preserved_and_operational_extras_are_retained(tmp_path):
    extracted, entries = verified_source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "frozen_spec.json").write_bytes(b"same frozen protocol")
    (output / "work_status.json").write_bytes(b"local status only")
    before = (output / "frozen_spec.json").stat().st_mtime_ns
    merged = collector.merge_verified_files(extracted, entries, output)
    assert merged["identical_files_preserved"] == ["frozen_spec.json"]
    assert merged["copied_files"] == ["evaluation/result.csv"]
    assert (output / "frozen_spec.json").stat().st_mtime_ns == before
    assert (output / "work_status.json").read_bytes() == b"local status only"
    assert (output / "evaluation/result.csv").read_bytes() == b"original raw remote bytes\n"


def test_unannounced_old_financial_files_cannot_mix_into_new_study(tmp_path):
    extracted, entries = verified_source(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "old-summary.csv").write_bytes(b"other experiment")
    with pytest.raises(FileExistsError, match="mix studies"):
        collector.merge_verified_files(extracted, entries, output)
    assert not (output / "evaluation").exists()


def test_staging_tampering_is_detected_before_merge(tmp_path):
    extracted, entries = verified_source(tmp_path)
    (extracted / "frozen_spec.json").write_bytes(b"changed staging bytes")
    with pytest.raises(ValueError, match="Staging files changed"):
        collector.merge_verified_files(extracted, entries, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_ssh_paths_are_arguments_and_shell_quoted_not_json_shell_programs():
    path = "/home/sunny/research/space '; $(not-a-command)/experiment"
    command = collector.ssh_command(collector.DEFAULT_ALIAS, collector.DEFAULT_REMOTE_PYTHON,
                                    ["--remote-export-worker", "--remote-root", path])
    assert shlex.split(command[-1]) == [collector.DEFAULT_REMOTE_PYTHON, "-", "--remote-export-worker", "--remote-root", path]
    assert command[-2] == collector.DEFAULT_ALIAS
    with pytest.raises(ValueError):
        collector.ssh_command("-oProxyCommand=unsafe", collector.DEFAULT_REMOTE_PYTHON, [])


def test_local_sources_and_data_are_checked_without_copying_or_loading_dataset(tmp_path):
    reference, dataset, source, _ = local_reference(tmp_path)
    checked = collector.verify_local_inputs(reference, dataset, source_root=source)
    assert checked["dataset_sha256"] == collector.sha(dataset)
    dataset.write_bytes(b"modified local data")
    with pytest.raises(ValueError, match="dataset"):
        collector.verify_local_inputs(reference, dataset, source_root=source)


def test_validation_destination_cannot_pollute_formal_output_even_before_ssh(tmp_path, monkeypatch):
    reference, dataset, source, _ = local_reference(tmp_path)
    monkeypatch.setattr(collector, "_remote_call", lambda *a, **kw: pytest.fail("SSH must not be called"))
    for output in (reference, reference / "validation_snapshot", reference.parent):
        with pytest.raises(ValueError, match="separate"):
            collector.collect(output=output, reference_output=reference, dataset_path=dataset,
                              source_root=source, validation_only=True)


def test_complete_fake_ssh_collection_preserves_records_and_emits_audit_compatible_receipt(tmp_path, monkeypatch):
    from scripts.audit_btceth_rl import Checks, _verify_receipt
    reference, dataset, source, spec = local_reference(tmp_path)
    (reference / "work_status.json").write_bytes(b"keep existing local status")
    remote_parent = tmp_path / "remote"
    remote_parent.mkdir()
    remote = terminal_study(remote_parent, spec=spec)
    original_record = b'{"model_relative_path":"training/model.zip","task":"/home/sunny/original/path"}\n'
    record_path = remote / "training" / "one-job" / "record.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_bytes(original_record)
    exported = collector.export_snapshot(remote, export_parent=tmp_path / "exports")
    manifest_path, archive_path = Path(exported["manifest_path"]), Path(exported["archive_path"])
    manifest = collector.read_json(manifest_path)
    manifest["remote_root"] = collector.DEFAULT_REMOTE_ROOT
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    remote_directory = (PurePosixPath(collector.DEFAULT_REMOTE_ROOT).parent.parent / "btceth_rl_exports"
                        / "full-" / Path(exported["export_directory"]).name)
    proof = dict(exported, remote_root=collector.DEFAULT_REMOTE_ROOT, export_directory=str(remote_directory),
                 archive_path=str(remote_directory / "experiment_snapshot.tar.gz"),
                 manifest_path=str(remote_directory / "snapshot_manifest.json"),
                 manifest_sha256=collector.sha(manifest_path))
    originals = {proof["manifest_path"]: manifest_path, proof["archive_path"]: archive_path}
    calls = []

    def fake_remote(alias, python, arguments, code, *, destination=None):
        calls.append(list(arguments))
        assert alias == collector.DEFAULT_ALIAS
        assert python == collector.DEFAULT_REMOTE_PYTHON
        assert b"standard-library-only" in code
        if "--remote-export-worker" in arguments:
            return proof
        remote_path = arguments[arguments.index("--remote-read-file") + 1]
        with Path(destination).open("xb") as stream:
            stream.write(originals[remote_path].read_bytes())
        return None

    monkeypatch.setattr(collector, "_remote_call", fake_remote)
    result = collector.collect(output=reference, reference_output=reference, dataset_path=dataset,
                               source_root=source, staging=tmp_path / "staging-one")
    assert result["status"] == "collected" and result["mode"] == "full"
    assert len(calls) == 3
    assert (reference / "training/one-job/record.json").read_bytes() == original_record
    assert (reference / "work_status.json").read_bytes() == b"keep existing local status"
    receipt_path = reference / "transfer_receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    checks = Checks()
    _verify_receipt(reference, checks)
    assert checks.failed == 0
    # Recollecting identical immutable evidence does not replace the existing
    # receipt or result bytes; this attempt gets its own new staging proof.
    again = collector.collect(output=reference, reference_output=reference, dataset_path=dataset,
                              source_root=source, staging=tmp_path / "staging-two")
    assert again["copied_files"] == 0
    assert receipt_path.read_bytes() == receipt_bytes
    assert (tmp_path / "staging-two/collection_receipt.json").is_file()
    with pytest.raises(FileExistsError, match="new staging"):
        collector.collect(output=reference, reference_output=reference, dataset_path=dataset,
                          source_root=source, staging=tmp_path / "staging-two")
    # A second local audit is a derived proof. Its explicit destination must
    # leave the transferred remote audit and receipt byte-for-byte intact.
    import sys
    from scripts import audit_btceth_rl as audit_module
    original_audit = (reference / "independent_audit.json").read_bytes()
    local_report = {"status": "passed", "passed": True, "test_complete": True,
                    "checks": 10, "checks_total": 10, "failure_count": 0,
                    "runs_checked": 172, "fixture_only": True}
    monkeypatch.setattr(audit_module, "audit", lambda *a, **kw: local_report)
    monkeypatch.setattr(sys, "argv", ["audit_btceth_rl.py", "--output-dir", str(reference),
                                    "--audit-file", str(reference / "independent_audit.local.json")])
    with pytest.raises(SystemExit) as stopped:
        audit_module.main()
    assert stopped.value.code == 0
    assert (reference / "independent_audit.local.json").is_file()
    assert (reference / "independent_audit.json").read_bytes() == original_audit
    assert receipt_path.read_bytes() == receipt_bytes
    derived_checks = Checks()
    _verify_receipt(reference, derived_checks)
    assert derived_checks.failed == 0


def test_source_change_during_export_prevents_a_completed_manifest(tmp_path, monkeypatch):
    root = terminal_study(tmp_path)
    original = tarfile.TarFile.addfile
    changed = False

    def mutate_once(self, *args, **kwargs):
        nonlocal changed
        result = original(self, *args, **kwargs)
        if not changed:
            changed = True
            (root / "summary.csv").write_bytes(b"source changed during archive")
        return result

    monkeypatch.setattr(tarfile.TarFile, "addfile", mutate_once)
    with pytest.raises(RuntimeError, match="changed (before|during) snapshot"):
        collector.export_snapshot(root, export_parent=tmp_path / "exports")
    assert list((tmp_path / "exports").rglob("export_failure.json"))
    assert not list((tmp_path / "exports").rglob("snapshot_manifest.json"))


def test_linked_source_directory_is_rejected_without_following_it(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    (root / "linked").mkdir(parents=True)
    actual = collector.is_link
    monkeypatch.setattr(collector, "is_link", lambda path: Path(path) == root / "linked" or actual(path))
    with pytest.raises(ValueError, match="linked directory"):
        collector.inventory(root)
