"""Collector safety with small labelled evidence and fully mocked networking.

The remote export PROGRAM runs locally against temporary synthetic files.
No SSH, remote process, learner, or financial result is created or modified.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest

from scripts import collect_single_pair_rl as collector


def digest(body):
    return hashlib.sha256(body).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def remote_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_remote") / "study"
    root.mkdir()
    spec = {"symbols": ["BTCUSDT", "ETHUSDT"], "algorithms": ["sac", "tqc"], "seeds": list(range(5)),
            "costs": {"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}},
            "baselines": ["cash", "buy_hold", "sma200_trend", "momentum180"], "ensemble": False,
            "capital_per_account": 10000, "capital_shared_between_pairs": False, "synthetic_fixture_only": True}
    write_json(root / "protocol.json", spec)
    write_json(root / "pipeline_status.json", {"status": "completed", "synthetic_fixture_only": True})
    write_json(root / "pipeline_launch.json", {"pid": 987654321, "synthetic_fixture_only": True})
    names = ["equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json",
             "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv"]
    for split in ("validation", "test"):
        # Their hashes, not economic contents, are under test in the transport.
        (root / (split + "_summary.csv")).write_bytes(b"synthetic_summary_only\n")
        write_json(root / (split + "_complete.json"), {"synthetic_fixture_only": True})
        inputs = {name: collector.sha(root / name) for name in ("protocol.json", split + "_summary.csv", split + "_complete.json")}
        identities = [(symbol, algorithm, seed, cost) for symbol in spec["symbols"] for algorithm in spec["algorithms"] for seed in spec["seeds"] for cost in spec["costs"]]
        identities += [(symbol, algorithm, None, cost) for symbol in spec["symbols"] for algorithm in spec["baselines"] for cost in spec["costs"]]
        cases = []
        for symbol, algorithm, seed, cost in identities:
            directory = f"evaluation/{split}/{symbol}/{algorithm}/{'baseline' if seed is None else 'seed'+str(seed)}/{cost}"
            hashes = {}
            for prefix in (directory, directory.replace("evaluation/", "repeat/", 1)):
                for name in names:
                    path = root / prefix / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(("SYNTHETIC-NONFINANCIAL:" + prefix + "/" + name).encode())
                    hashes[path.relative_to(root).as_posix()] = collector.sha(path)
            cases.append({"relative_path": directory, "passed": True, "failure_count": 0, "failures": [], "checks_total": 1,
                          "inputs_sha256": hashes})
        assert len(cases) == 56
        write_json(root / (split + "_audit.json"), {"phase": split, "passed": True, "failure_count": 0, "failures": [], "checks_total": 56,
            "expected_accounts": 56, "accounts_checked": 56, "accounts_count": 56, "repeats_checked": 56, "cases": cases,
            "protocol_sha256": collector.sha(root / "protocol.json"), "summary_sha256": collector.sha(root / (split + "_summary.csv")), "inputs_sha256": inputs})
    return root


@pytest.fixture
def remote_root(remote_files):
    controls = [remote_files / name for name in ("protocol.json", "pipeline_status.json", "pipeline_launch.json", "validation_audit.json", "test_audit.json")]
    controls.append(remote_files / "evaluation/validation/BTCUSDT/sac/seed0/base/fills.csv")
    original = {path: path.read_bytes() for path in controls}
    yield remote_files
    for path, payload in original.items():
        if path.read_bytes() != payload:
            path.write_bytes(payload)


def local_export(root):
    source = collector.REMOTE_EXPORT.replace("REMOTE_ROOT", repr(str(root)))
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(source, "<synthetic-remote-export>", "exec"), {})
    return json.loads(output.getvalue())


def test_remote_export_accepts_only_complete_bound_56_case_phases(remote_root):
    receipt = local_export(remote_root)
    assert receipt["archive_sha256"] == collector.sha(receipt["archive"])
    assert receipt["protocol_sha256"] == collector.sha(remote_root / "protocol.json")
    assert len(receipt["files"]) > 2000
    with tarfile.open(receipt["archive"], "r:gz") as stream:
        members = stream.getmembers()
        assert len(members) == len(receipt["files"])
        assert all(member.isfile() for member in members)
    assert all(collector.sha(remote_root / name) == expected for name, expected in receipt["files"].items())


@pytest.mark.parametrize("damage", ["truthy_passed", "wrong_phase", "wrong_count", "missing_case", "duplicate_case", "missing_phase_input", "missing_case_input", "truthy_case", "wrong_hash"])
def test_remote_export_rejects_stale_false_or_incomplete_audits(remote_root, damage):
    path = remote_root / "validation_audit.json"
    audit = json.loads(path.read_text())
    if damage == "truthy_passed": audit["passed"] = "false"
    elif damage == "wrong_phase": audit["phase"] = "test"
    elif damage == "wrong_count": audit["accounts_count"] = 55
    elif damage == "missing_case": audit["cases"].pop()
    elif damage == "duplicate_case": audit["cases"][-1] = audit["cases"][0]
    elif damage == "missing_phase_input": audit["inputs_sha256"].pop("protocol.json")
    elif damage == "missing_case_input": audit["cases"][0]["inputs_sha256"].pop(next(iter(audit["cases"][0]["inputs_sha256"])))
    elif damage == "truthy_case": audit["cases"][0]["passed"] = "true"
    else: audit["summary_sha256"] = "0" * 64
    write_json(path, audit)
    with pytest.raises(AssertionError):
        local_export(remote_root)


def test_remote_export_refuses_a_still_active_supervisor(remote_root, monkeypatch):
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: True if path.as_posix() == "/proc/987654321" else original(path))
    with pytest.raises(AssertionError, match="supervisor"):
        local_export(remote_root)


@pytest.mark.parametrize("name", ["validation_audit.json", "evaluation/validation/BTCUSDT/sac/seed0/base/fills.csv"])
def test_remote_export_rebinds_audited_bytes_after_initial_hash_checks(remote_root, monkeypatch, name):
    original = Path.rglob
    changed = []
    def race(path, pattern, *args, **kwargs):
        if path == remote_root and not changed:
            target = remote_root / name
            target.write_bytes(target.read_bytes() + b"\nRACE-CHANGE")
            changed.append(True)
        return original(path, pattern, *args, **kwargs)
    monkeypatch.setattr(Path, "rglob", race)
    with pytest.raises(AssertionError, match="bindings changed"):
        local_export(remote_root)
    assert changed


def test_remote_export_refuses_bound_evidence_changed_after_tar_member_was_written(remote_root, tmp_path, monkeypatch):
    copied = tmp_path / "post_archive_race"
    shutil.copytree(remote_root, copied)
    original_add = tarfile.TarFile.add
    changed = []
    def add_then_change(stream, name, arcname=None, *args, **kwargs):
        result = original_add(stream, name, arcname=arcname, *args, **kwargs)
        if arcname == "validation_summary.csv" and not changed:
            path = copied / "validation_summary.csv"
            path.write_bytes(path.read_bytes() + b"POST-ARCHIVE-RACE")
            changed.append(True)
        return result
    monkeypatch.setattr(tarfile.TarFile, "add", add_then_change)
    with pytest.raises(AssertionError, match="changed during export"):
        local_export(copied)
    assert changed


def test_remote_export_never_replaces_an_existing_remote_receipt(remote_root, tmp_path, monkeypatch):
    copied = tmp_path / "remote_receipt_preservation"
    shutil.copytree(remote_root, copied)
    original_open = tarfile.open
    planted = []
    def plant_existing_receipt(name, mode="r", *args, **kwargs):
        if mode == "x:gz" and not planted:
            archive = Path(name)
            receipt = archive.with_suffix(archive.suffix + ".json")
            receipt.write_bytes(b"PRESERVE-EXISTING-REMOTE-RECEIPT")
            planted.append(receipt)
        return original_open(name, mode, *args, **kwargs)
    monkeypatch.setattr(tarfile, "open", plant_existing_receipt)
    with pytest.raises((FileExistsError, AssertionError)):
        local_export(copied)
    assert len(planted) == 1 and planted[0].read_bytes() == b"PRESERVE-EXISTING-REMOTE-RECEIPT"


def mock_transport(tmp_path, monkeypatch, bodies, *, kinds=None, expected_hash=None):
    archive = tmp_path / "synthetic_remote.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, body in bodies.items():
            member = tarfile.TarInfo(name)
            kind = (kinds or {}).get(name, tarfile.REGTYPE)
            member.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../outside"
                stream.addfile(member)
            else:
                member.size = len(body)
                stream.addfile(member, io.BytesIO(body))
    receipt = {"archive": "/synthetic/study_export.tar.gz", "archive_sha256": expected_hash or collector.sha(archive),
               "files": {name: digest(body) for name, body in bodies.items()}, "remote_root": "/synthetic/study", "no_production_change": True}
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh":
            return SimpleNamespace(stdout=json.dumps(receipt), returncode=0)
        assert command[0] == "scp"
        shutil.copyfile(archive, command[-1])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(collector.subprocess, "run", run)
    return receipt, calls


def test_partial_local_output_resumes_identical_files_without_replacing_them(tmp_path, monkeypatch):
    output = tmp_path / "local"
    output.mkdir()
    original = output / "protocol.json"
    original.write_bytes(b"SAME")
    timestamp = original.stat().st_mtime_ns
    receipt, calls = mock_transport(tmp_path, monkeypatch, {"protocol.json": b"SAME", "training/record.json": b"NEW"})
    result = collector.collect("test-only-no-network", "/synthetic/study", output, "/synthetic/python")
    assert result["files_verified"] == 2
    assert original.read_bytes() == b"SAME" and original.stat().st_mtime_ns == timestamp
    assert (output / "training/record.json").read_bytes() == b"NEW"
    assert len(calls) == 2
    recorded = json.loads((output / "transport/study_export.tar.gz.receipt.json").read_text())
    assert recorded["files"] == receipt["files"]


def test_partial_local_output_refuses_different_evidence_and_preserves_it(tmp_path, monkeypatch):
    output = tmp_path / "local"
    output.mkdir()
    (output / "protocol.json").write_bytes(b"USER-LOCAL-DIFFERENCE")
    mock_transport(tmp_path, monkeypatch, {"protocol.json": b"REMOTE"})
    with pytest.raises(FileExistsError, match="changed local evidence"):
        collector.collect("test-only", "/synthetic/study", output, "/synthetic/python")
    assert (output / "protocol.json").read_bytes() == b"USER-LOCAL-DIFFERENCE"


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "C:drive.txt", "folder\\file.txt", "a//b.txt", "./alias.txt"])
def test_local_extraction_rejects_escape_and_noncanonical_paths(tmp_path, monkeypatch, name):
    mock_transport(tmp_path, monkeypatch, {name: b"NO"})
    with pytest.raises(ValueError):
        collector.collect("test-only", "/synthetic/study", tmp_path / "local", "/synthetic/python")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_local_extraction_rejects_tar_links(tmp_path, monkeypatch, kind):
    mock_transport(tmp_path, monkeypatch, {"link": b""}, kinds={"link": kind})
    with pytest.raises(ValueError, match="regular evidence"):
        collector.collect("test-only", "/synthetic/study", tmp_path / "local", "/synthetic/python")


@pytest.mark.parametrize("location", ["root", "ancestor", "transport", "archive", "receipt"])
def test_local_root_ancestors_transport_archive_and_receipt_reject_links_before_writing(tmp_path, monkeypatch, location):
    output = tmp_path / "outer/local"
    output.mkdir(parents=True)
    _, calls = mock_transport(tmp_path, monkeypatch, {"protocol.json": b"DATA"})
    candidates = {"root": output, "ancestor": output.parent, "transport": output / "transport",
                  "archive": output / "transport/study_export.tar.gz", "receipt": output / "transport/study_export.tar.gz.receipt.json"}
    linked = candidates[location].absolute()
    original = Path.is_symlink
    # Explicit filesystem-metadata boundary avoids requiring Windows symlink
    # privilege; all collection and ancestor traversal code remains real.
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.absolute() == linked or original(path))
    with pytest.raises(ValueError, match="link"):
        collector.collect("test-only", "/synthetic/study", output, "/synthetic/python")
    assert not any(command[0] == "scp" for command in calls)
    assert not (output / "protocol.json").exists()


def test_existing_transfer_receipt_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "local"
    (output / "transport").mkdir(parents=True)
    old = output / "transport/study_export.tar.gz.receipt.json"
    old.write_bytes(b"PRESERVE-OLD-RECEIPT")
    _, calls = mock_transport(tmp_path, monkeypatch, {"protocol.json": b"DATA"})
    with pytest.raises(FileExistsError):
        collector.collect("test-only", "/synthetic/study", output, "/synthetic/python")
    assert old.read_bytes() == b"PRESERVE-OLD-RECEIPT"
    assert not any(command[0] == "scp" for command in calls)


def test_local_link_rejection_also_checks_junction_metadata(tmp_path, monkeypatch):
    root = tmp_path / "junction_parent"
    root.mkdir()
    monkeypatch.setattr(Path, "is_junction", lambda path: path.absolute() == root.absolute(), raising=False)
    with pytest.raises(ValueError, match="link"):
        collector.reject_links(root / "nested/output")


def test_transferred_archive_hash_is_checked_before_extracting_evidence(tmp_path, monkeypatch):
    mock_transport(tmp_path, monkeypatch, {"protocol.json": b"DATA"}, expected_hash="0" * 64)
    output = tmp_path / "local"
    with pytest.raises(ValueError, match="archive hash"):
        collector.collect("test-only", "/synthetic/study", output, "/synthetic/python")
    assert not (output / "protocol.json").exists()
