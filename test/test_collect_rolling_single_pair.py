"""Nonfinancial temporary export fixtures and fully mocked SSH/SCP only.

These fixtures are transport contracts, never trained models or economic
results. No incomplete real rolling study is exported by these tests.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import tarfile
from types import SimpleNamespace

import pytest

from scripts import collect_rolling_single_pair as collector


def digest(body):
    return hashlib.sha256(body).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def remote_fixture(tmp_path):
    root = tmp_path / "research" / collector.STUDY
    root.mkdir(parents=True)
    dataset = root.parent / "synthetic_market/dataset.pkl.gz"
    dataset.parent.mkdir()
    dataset.write_bytes(b"NONFINANCIAL TRANSPORT TEST, NOT A MARKET DATASET")
    market_manifest = dataset.with_name("manifest.json")
    write(market_manifest, {"nonfinancial_transport_fixture": True})
    source = root / "source/scripts/audit_rolling_single_pair.py"
    source.parent.mkdir(parents=True)
    source.write_text("import hashlib,json\n"
                      "def verify_completed_study(root):\n"
                      " gate=json.loads((root/'completion_gate.json').read_text())\n"
                      " assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==value for name,value in gate['inputs_sha256'].items())\n"
                      " return {'passed':True,'no_training_inference_or_repair':True,'inputs_verified':len(gate['inputs_sha256'])+len(gate['external_inputs_sha256']),"
                      "'completion_gate_sha256':hashlib.sha256((root/'completion_gate.json').read_bytes()).hexdigest()}\n",
                      encoding="utf-8")
    runner = root / "source/scripts/backtest_rolling_single_pair.py"
    runner.write_text("# NONFINANCIAL TRANSPORT FIXTURE; never executed\n", encoding="utf-8")
    control = root / "control_source/rolling_single_pair_remote_ops.py"
    control.parent.mkdir()
    control.write_text("# NONFINANCIAL CONTROL FIXTURE; never executed\n", encoding="utf-8")
    write(root / "inventory.json", {"nonfinancial_transport_fixture": True})
    spec = dict(test=["2025-10-23T00:00:00Z", "2026-10-23T00:00:00Z"], seeds=list(range(5)),
                capital_per_account=10000., capital_shared_between_pairs=False, ensemble=False,
                selection_uses_test=False, fresh_holdout=False, test_year_already_observed=True,
                cadences={"fixed": None, "every_4w": 28, "every_week": 7},
                costs={"base": {"fee": .001, "slippage": .0002}, "stress": {"fee": .002, "slippage": .0005}},
                train_days=1827, lag_days=1, preheat_days=200,
                selected={"BTCUSDT": ["td3_default", "sac_default", "sac_gamma099"],
                          "ETHUSDT": ["sac_default", "tqc_default", "sac_gamma099"]},
                expected_models=1590, expected_accounts=180,
                sources={"scripts/audit_rolling_single_pair.py": collector.sha(source),
                         "scripts/backtest_rolling_single_pair.py": collector.sha(runner)},
                dataset_sha256=collector.sha(dataset), dataset_manifest_sha256=collector.sha(market_manifest),
                nonfinancial_transport_fixture=True)
    write(root / "protocol.json", spec)
    write(root / "protocol_receipt.json", {"protocol_sha256": collector.sha(root / "protocol.json")})
    (root / "summary.csv").write_text("NONFINANCIAL_TRANSPORT_FIXTURE_ONLY\n", encoding="utf-8")
    status = {"status": "completed", "phase": "complete", "pid": 987654321, "no_production_change": True}
    write(root / "pipeline_status.json", status)
    launch_dir = root / "control/launches/nonfinancial-fixture"
    launch_dir.mkdir(parents=True)
    python = "/synthetic/native/python"
    plan = dict(phase="pipeline", at="2026-10-31T00:00:00Z",
                argv=[python, "-B", str(runner), "pipeline", "--output", str(root), "--dataset", str(dataset), "--max-workers", "32"],
                source_and_selection_sha256={p.relative_to(root).as_posix(): collector.sha(p) for p in (source, runner, control, root / "inventory.json")},
                control_source_sha256=collector.sha(control), protocol_sha256=collector.sha(root / "protocol.json"),
                no_production_change=True)
    write(launch_dir / "launch_plan.json", plan)
    launch = dict(plan, pid=status["pid"], start_ticks=12345, state="R",
                  launch_plan_sha256=collector.sha(launch_dir / "launch_plan.json"),
                  log_path="control/launches/nonfinancial-fixture/stdout.log")
    write(launch_dir / "launched.json", launch)
    (launch_dir / "stdout.log").write_text("NONFINANCIAL FIXTURE\n", encoding="utf-8")
    complete = dict(passed=True, failure_count=0, failures=[], checks_total=1,
                    expected_models=1590, models_checked=1590,
                    expected_accounts=180, accounts_checked=180, repeats_checked=180,
                    protocol_sha256=collector.sha(root / "protocol.json"), summary_sha256=collector.sha(root / "summary.csv"),
                    source_sha256=collector.sha(source),
                    inputs_sha256={name: collector.sha(root / name) for name in ("protocol.json", "summary.csv", source.relative_to(root).as_posix())},
                    external_inputs_sha256={str(dataset): collector.sha(dataset), str(market_manifest): collector.sha(market_manifest)},
                    nonfinancial_transport_fixture=True)
    write(root / "completion_gate.json", complete)
    program = collector.export_program(root, python, collector.sha(source))
    return root, dataset, source, complete, launch_dir, program


def execute(program):
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(program, "<NONFINANCIAL-transport-fixture>", "exec"), {})
    return json.loads(output.getvalue())


def test_remote_fixture_checks_current_gate_launch_and_exact_snapshot(remote_fixture):
    root, dataset, source, _, launch_dir, program = remote_fixture
    result = execute(program)
    assert result["archive_sha256"] == collector.sha(result["archive"])
    assert result["completion_gate_sha256"] == collector.sha(root / "completion_gate.json")
    assert result["verifier_source_sha256"] == collector.sha(source)
    assert result["pipeline_launch_receipt_sha256"] == collector.sha(launch_dir / "launched.json")
    assert result["external_inputs_sha256"] == result["external_verified_sha256"]
    assert result["external_rechecked_after_archive"] is True
    with tarfile.open(result["archive"], "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(result["files"])
        assert {member.name for member in members} == set(result["files"])
        for member in members:
            assert member.isfile()
            assert digest(archive.extractfile(member).read()) == result["files"][member.name]


@pytest.mark.parametrize("damage", ["running", "truthy_gate", "failed_gate", "wrong_model_count", "missing_repeat", "source_changed",
                                     "summary_changed", "market_changed", "wrong_launch_pid", "duplicate_launch", "wrong_plan",
                                     "wrong_argv", "wrong_launch_phase", "changed_control_source"])
def test_remote_refuses_unfinished_wrong_generation_or_unbound_process(remote_fixture, damage):
    root, dataset, source, complete, launch_dir, program = remote_fixture
    if damage == "running":
        write(root / "pipeline_status.json", {"status": "running"})
    elif damage in ("truthy_gate", "failed_gate", "wrong_model_count", "missing_repeat"):
        if damage == "truthy_gate": complete["passed"] = "true"
        elif damage == "failed_gate": complete.update(passed=False, failure_count=1, failures=["fixture failure"])
        elif damage == "wrong_model_count": complete["models_checked"] = 1589
        else: complete["repeats_checked"] = 179
        write(root / "completion_gate.json", complete)
    elif damage == "source_changed":
        source.write_text("# changed frozen verifier")
    elif damage == "summary_changed":
        (root / "summary.csv").write_text("CHANGED")
    elif damage == "market_changed":
        dataset.write_bytes(b"CHANGED")
    elif damage == "changed_control_source":
        (root / "control_source/rolling_single_pair_remote_ops.py").write_text("# changed control")
    elif damage == "duplicate_launch":
        duplicate = root / "control/launches/duplicate"
        shutil.copytree(launch_dir, duplicate)
    else:
        launch = json.loads((launch_dir / "launched.json").read_text())
        if damage == "wrong_launch_pid": launch["pid"] += 1
        elif damage == "wrong_launch_phase": launch["phase"] = "preflight"
        elif damage == "wrong_plan": launch["launch_plan_sha256"] = "0" * 64
        else: launch["argv"] = ["/wrong/python"]
        write(launch_dir / "launched.json", launch)
    with pytest.raises(AssertionError):
        execute(program)


def test_exact_pipeline_pid_must_have_exited_without_killing_it(remote_fixture, monkeypatch):
    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: True if path.as_posix() == "/proc/987654321" else original(path))
    with pytest.raises(AssertionError, match="exact pipeline PID"):
        execute(remote_fixture[-1])


def test_post_archive_external_market_race_is_rejected(remote_fixture, monkeypatch):
    _, dataset, _, _, _, program = remote_fixture
    original = tarfile.open
    class ChangeAfterClose:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self.stream.__enter__()
        def __exit__(self, *args):
            result = self.stream.__exit__(*args)
            dataset.write_bytes(b"EXTERNAL INPUT RACE AFTER ARCHIVE")
            return result
    monkeypatch.setattr(tarfile, "open", lambda *args, **kwargs: ChangeAfterClose(original(*args, **kwargs)))
    with pytest.raises(AssertionError, match="External frozen market"):
        execute(program)


def test_post_archive_member_race_and_old_failed_history_are_preserved(remote_fixture, monkeypatch):
    root, _, _, _, _, program = remote_fixture
    old = root / "audit_failed_OLD.json"
    old.write_bytes(b'{"passed":false,"fixture":true}')
    preserved = old.read_bytes()
    original = tarfile.TarFile.add
    changed = []
    def change_after_add(stream, name, arcname=None, *args, **kwargs):
        result = original(stream, name, arcname=arcname, *args, **kwargs)
        if arcname == "summary.csv" and not changed:
            (root / "summary.csv").write_bytes(b"RACE AFTER MEMBER WRITE")
            changed.append(True)
        return result
    monkeypatch.setattr(tarfile.TarFile, "add", change_after_add)
    with pytest.raises(AssertionError, match="changed during export"):
        execute(program)
    assert old.read_bytes() == preserved and changed


def transport_fixture(tmp_path, monkeypatch, *, body_changes=None, special_members=None):
    """Mock the trusted SSH attestation; exercise local transport only."""
    body_changes = body_changes or {}
    basename = collector.STUDY + "_audited_FIXTURE.tar.gz"
    launch = "control/launches/FIXTURE/launched.json"
    plan = "control/launches/FIXTURE/launch_plan.json"
    bodies = {"protocol.json": b"NONFINANCIAL_PROTOCOL", "completion_gate.json": b"NONFINANCIAL_GATE",
              "source/" + collector.AUDITOR_NAME: (collector.ROOT / collector.AUDITOR_NAME).read_bytes(),
              launch: b"NONFINANCIAL_LAUNCH", plan: b"NONFINANCIAL_PLAN", "new/result.txt": b"NEW RESULT"}
    bodies.update(body_changes)
    archive_path = tmp_path / "mock_remote.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in bodies.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        for member in special_members or []:
            archive.addfile(member, io.BytesIO(b"X") if member.isfile() else None)
    receipt = dict(schema_version=1, study=collector.STUDY, remote_root=collector.REMOTE_ROOT,
                   archive="/home/sunny/research/" + basename, archive_sha256=collector.sha(archive_path),
                   files={name: digest(content) for name, content in bodies.items()},
                   protocol_sha256=digest(bodies["protocol.json"]), completion_gate_sha256=digest(bodies["completion_gate.json"]),
                   verifier_source_sha256=collector.AUDITOR_SHA256, pipeline_pid=987654321, pipeline_start_ticks=12345,
                   pipeline_launch_receipt=launch, pipeline_launch_receipt_sha256=digest(bodies[launch]), launch_plan_sha256=digest(bodies[plan]),
                   inputs_sha256={"protocol.json": digest(bodies["protocol.json"])},
                   external_inputs_sha256={"/home/sunny/research/synthetic/dataset.pkl.gz": "a" * 64,
                                          "/home/sunny/research/synthetic/manifest.json": "b" * 64},
                   external_verified_sha256={"/home/sunny/research/synthetic/dataset.pkl.gz": "a" * 64,
                                             "/home/sunny/research/synthetic/manifest.json": "b" * 64},
                   external_rechecked_after_archive=True, no_training_inference_or_repair=True, no_production_change=True)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh": return SimpleNamespace(stdout=json.dumps(receipt) + "\n", returncode=0)
        assert command[0] == "scp"
        shutil.copyfile(archive_path, command[-1])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(collector.subprocess, "run", run)
    return receipt, calls, bodies, archive_path


def test_local_extracts_exactly_and_retains_received_attestation(tmp_path, monkeypatch):
    receipt, calls, bodies, _ = transport_fixture(tmp_path, monkeypatch)
    output = tmp_path / collector.STUDY
    output.mkdir()
    original = output / "protocol.json"
    original.write_bytes(bodies["protocol.json"])
    before = original.stat().st_mtime_ns
    result = collector.collect("synthetic-only", collector.REMOTE_ROOT, output, "/synthetic/python")
    assert result["files_verified"] == len(bodies)
    assert original.stat().st_mtime_ns == before
    saved = json.loads(Path(result["receipt"]).read_text())
    assert saved["archive"] == Path(receipt["archive"]).name
    assert saved["remote_archive"] == receipt["archive"]
    assert saved["external_inputs_sha256"] == saved["external_verified_sha256"]
    assert saved["all_archive_members_hashed_before_materialization"] is True
    assert len(calls) == 2
    assert all((output / name).read_bytes() == body for name, body in bodies.items())
    assert collector.sha(output / saved["remote_export_receipt_file"]) == saved["remote_export_receipt_sha256"]


def test_changed_local_evidence_is_preserved_before_any_new_file_is_written(tmp_path, monkeypatch):
    _, _, _, _ = transport_fixture(tmp_path, monkeypatch)
    output = tmp_path / collector.STUDY
    output.mkdir()
    (output / "protocol.json").write_bytes(b"USER CHANGES MUST SURVIVE")
    with pytest.raises(FileExistsError, match="changed local evidence"):
        collector.collect("synthetic", collector.REMOTE_ROOT, output, "/synthetic/python")
    assert (output / "protocol.json").read_bytes() == b"USER CHANGES MUST SURVIVE"
    assert not (output / "new/result.txt").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:drive", "a\\b", "a//b", "./alias", "a/../b"])
def test_receipt_member_names_cannot_escape_output(tmp_path, monkeypatch, name):
    receipt, calls, _, _ = transport_fixture(tmp_path, monkeypatch)
    receipt["files"][name] = "a" * 64
    with pytest.raises(ValueError):
        collector.collect("synthetic", collector.REMOTE_ROOT, tmp_path / collector.STUDY, "/synthetic/python")
    assert not any(command[0] == "scp" for command in calls)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE])
def test_tar_nonregular_members_are_rejected_before_materialization(tmp_path, monkeypatch, kind):
    member = tarfile.TarInfo("forbidden")
    member.type, member.linkname = kind, "../outside"
    receipt, _, _, _ = transport_fixture(tmp_path, monkeypatch, special_members=[member])
    receipt["files"][member.name] = "a" * 64
    output = tmp_path / collector.STUDY
    with pytest.raises(ValueError, match="regular"):
        collector.collect("synthetic", collector.REMOTE_ROOT, output, "/synthetic/python")
    assert not (output / "protocol.json").exists()


def test_duplicate_tar_member_is_rejected(tmp_path, monkeypatch):
    member = tarfile.TarInfo("protocol.json")
    member.size = 1
    transport_fixture(tmp_path, monkeypatch, special_members=[member])
    with pytest.raises(ValueError, match="unique"):
        collector.collect("synthetic", collector.REMOTE_ROOT, tmp_path / collector.STUDY, "/synthetic/python")


def test_even_skipped_identical_file_requires_correct_archive_body_sha(tmp_path, monkeypatch):
    receipt, _, bodies, archive = transport_fixture(tmp_path, monkeypatch)
    output = tmp_path / collector.STUDY
    output.mkdir()
    (output / "protocol.json").write_bytes(bodies["protocol.json"])
    # Keep the declared file digest and local file correct, but substitute a
    # different archived body under a newly attested whole-archive digest.
    with tarfile.open(archive, "w:gz") as stream:
        for name, body in bodies.items():
            content = b"CORRUPT BODY" if name == "protocol.json" else body
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    receipt["archive_sha256"] = collector.sha(archive)
    with pytest.raises(ValueError, match="member SHA256"):
        collector.collect("synthetic", collector.REMOTE_ROOT, output, "/synthetic/python")
    assert (output / "protocol.json").read_bytes() == bodies["protocol.json"]


def test_previous_receipts_are_never_overwritten(tmp_path, monkeypatch):
    receipt, calls, _, _ = transport_fixture(tmp_path, monkeypatch)
    output = tmp_path / collector.STUDY
    target = output / "transport" / (Path(receipt["archive"]).name + ".receipt.json")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"OLD RECEIPT MUST SURVIVE")
    with pytest.raises(FileExistsError):
        collector.collect("synthetic", collector.REMOTE_ROOT, output, "/synthetic/python")
    assert target.read_bytes() == b"OLD RECEIPT MUST SURVIVE"
    assert not any(command[0] == "scp" for command in calls)


def test_junction_metadata_is_rejected_even_without_symlink_flag(tmp_path, monkeypatch):
    target = tmp_path / "junction"
    target.mkdir()
    monkeypatch.setattr(Path, "is_junction", lambda p: p.absolute() == target.absolute(), raising=False)
    with pytest.raises(ValueError, match="junction"):
        collector.reject_links(target / collector.STUDY)


def test_collector_refuses_any_other_study_before_network(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collector.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="named remote"):
        collector.collect("synthetic", "/other/study", tmp_path / collector.STUDY, "/synthetic/python")
    with pytest.raises(ValueError, match="named local"):
        collector.collect("synthetic", collector.REMOTE_ROOT, tmp_path / "other", "/synthetic/python")
    assert calls == []


@pytest.mark.parametrize("damage", ["wrong_verifier", "unverified_external", "unknown_bound_input", "wrong_archive_parent", "wrong_source_binding", "false_production_contract"])
def test_receipt_cannot_downgrade_frozen_gate_or_transfer_scope(tmp_path, monkeypatch, damage):
    receipt, calls, _, _ = transport_fixture(tmp_path, monkeypatch)
    if damage == "wrong_verifier": receipt["verifier_source_sha256"] = "0" * 64
    elif damage == "unverified_external": receipt["external_rechecked_after_archive"] = False
    elif damage == "unknown_bound_input": receipt["inputs_sha256"]["missing.txt"] = "a" * 64
    elif damage == "wrong_archive_parent": receipt["archive"] = "/other/" + Path(receipt["archive"]).name
    elif damage == "wrong_source_binding": receipt["files"]["source/" + collector.AUDITOR_NAME] = "0" * 64
    else: receipt["no_production_change"] = "true"
    with pytest.raises(ValueError):
        collector.collect("synthetic", collector.REMOTE_ROOT, tmp_path / collector.STUDY, "/synthetic/python")
    assert not any(command[0] == "scp" for command in calls)


def test_missing_tar_member_refuses_complete_materialization(tmp_path, monkeypatch):
    receipt, _, _, _ = transport_fixture(tmp_path, monkeypatch)
    receipt["files"]["declared_but_missing.txt"] = "a" * 64
    output = tmp_path / collector.STUDY
    with pytest.raises(ValueError, match="complete archive membership"):
        collector.collect("synthetic", collector.REMOTE_ROOT, output, "/synthetic/python")
    assert not (output / "protocol.json").exists()


def test_whole_archive_sha_is_checked_before_extracting(tmp_path, monkeypatch):
    receipt, _, _, _ = transport_fixture(tmp_path, monkeypatch)
    receipt["archive_sha256"] = "0" * 64
    output = tmp_path / collector.STUDY
    with pytest.raises(ValueError, match="Transferred archive SHA256"):
        collector.collect("synthetic", collector.REMOTE_ROOT, output, "/synthetic/python")
    assert not (output / "protocol.json").exists()


def test_windows_reparse_attribute_is_checked_on_old_python(tmp_path, monkeypatch):
    target = tmp_path / "reparse_parent"
    target.mkdir()
    original = Path.lstat
    def lstat(path, *args, **kwargs):
        if path.absolute() == target.absolute():
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    monkeypatch.setattr(Path, "is_junction", lambda path: False, raising=False)
    with pytest.raises(ValueError, match="reparse point"):
        collector.reject_links(target / collector.STUDY)


def test_remote_rejects_symlink_before_following_or_archiving(remote_fixture, monkeypatch):
    root, _, source, _, _, program = remote_fixture
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.absolute() == source.absolute() or original(path))
    with pytest.raises(AssertionError, match="Linked evidence"):
        execute(program)
    assert not list(root.parent.glob(collector.STUDY + "_audited_*.tar.gz"))


def test_remote_never_replaces_existing_export_receipt_even_on_race(remote_fixture, monkeypatch):
    original = tarfile.open
    planted = []
    def create_old_receipt(path, mode="r", *args, **kwargs):
        if mode == "x:gz":
            receipt = Path(path).with_suffix(Path(path).suffix + ".json")
            receipt.write_bytes(b"OLD REMOTE RECEIPT")
            planted.append(receipt)
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(tarfile, "open", create_old_receipt)
    with pytest.raises(FileExistsError):
        execute(remote_fixture[-1])
    assert len(planted) == 1 and planted[0].read_bytes() == b"OLD REMOTE RECEIPT"
