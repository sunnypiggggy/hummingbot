"""One-shot delivery safety; waits/SSH/subprocess/HTTP are explicit mocks.

No collector, report builder, server, native browser, or learner is launched.
The official split-build manifest format is a labelled tiny synthetic fixture.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import finish_single_pair_rl_six as worker


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def preview_fixture(output):
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "report.html").write_bytes(b"SYNTHETIC-PORTABLE-REPORT")
    (report / "reviewed.json").write_bytes(b'{"synthetic_fixture_only":true}\n')
    dist = report / "report_app/dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_bytes(b"SYNTHETIC-INDEX-PREVIEW")
    snapshot = b'{"synthetic_fixture_only":true,"queries":{}}\n'
    snapshot_sha = hashlib.sha256(snapshot).hexdigest()
    snapshot_name = "snapshot." + snapshot_sha + ".json"
    (dist / snapshot_name).write_bytes(snapshot)
    runtime_sha = "a" * 64
    manifest = {"version": 1, "kind": "separate-data-v1", "html": {"path": "index.html", "sha256": worker.sha(dist / "index.html"), "bytes": (dist / "index.html").stat().st_size},
                "snapshot": {"path": snapshot_name, "sha256": snapshot_sha, "bytes": len(snapshot)},
                "runtimeSha256": runtime_sha, "sourceSnapshotSha256": snapshot_sha}
    write_json(dist / "data-app-build.json", manifest)
    build = {"html_sha256": worker.sha(report / "report.html"), "reviewed_sha256": worker.sha(report / "reviewed.json"),
             "runtime": {"htmlSha256": manifest["html"]["sha256"], "snapshotSha256": snapshot_sha, "runtimeSha256": runtime_sha}}
    write_json(report / "report_build.json", build)
    return dist, build


def progress(state="completed", alive=False, failed=0):
    return {"pipeline": {"status": state}, "supervisor_alive": alive,
            "algorithms": {name: {"started": 10, "completed": 10, "failed": failed if name == "droq" else 0}
                           for name in ("sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq")}}


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in worker.SOURCE_NAMES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("SYNTHETIC-EXECUTION-BOUNDARY:" + name).encode())
    monkeypatch.setattr(worker, "ROOT", source)
    output, retained = tmp_path / "new_study", tmp_path / "retained_study"
    output.mkdir(); retained.mkdir()
    (retained / "immutable.bin").write_bytes(b"RETAINED-EVIDENCE")
    args = SimpleNamespace(output=output, retained_study=retained, host="no-network-fixture", remote_python="/fixture/python",
        remote_root="/fixture/remote", poll_interval=15, wsl_node="/fixture/node",
        preview_dir=tmp_path / "existing_server_root/six", preview_url="http://127.0.0.1:8774/six/index.html?view=1")
    calls, sleeps, urls, snapshots = [], [], [], [progress()]
    failure = {"collect": False, "build": False, "http": False, "readback": False}
    def snapshot(*values):
        result = snapshots.pop(0)
        if isinstance(result, Exception): raise result
        return result
    monkeypatch.setattr(worker, "remote_snapshot", snapshot)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: sleeps.append(seconds))
    def run(command, **kwargs):
        name = Path(command[2]).name
        calls.append(name)
        assert name in ("collect_single_pair_rl.py", "single_pair_rl_six_report_app.py"), "No training/restart/server command is allowed"
        if name == "collect_single_pair_rl.py":
            if failure["collect"]: raise subprocess.CalledProcessError(1, command)
        else:
            if failure["build"]: raise subprocess.CalledProcessError(1, command)
            preview_fixture(output)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, "run", run)
    def old_readback(*values):
        if failure["readback"]: raise ValueError("Retained replay bytes changed")
        return {"passed": True, "cases_checked": 224, "files_checked": 2240, "synthetic_boundary_only": True}
    monkeypatch.setattr(worker, "retained_replay_readback", old_readback)
    monkeypatch.setattr(worker, "summary_readback", lambda path: [{"symbol": "BTCUSDT", "algorithm": "sac", "synthetic_boundary_only": True}])
    class Response:
        status = 200
        url = args.preview_url
        def read(self): return b"WRONG-PREVIEW" if failure["http"] else (args.preview_dir / "index.html").read_bytes()
        def geturl(self): return self.url
        def __enter__(self): return self
        def __exit__(self, *values): return False
    monkeypatch.setattr(worker.urllib.request, "urlopen", lambda url, **kwargs: (urls.append(url), Response())[1])
    return SimpleNamespace(args=args, output=output, retained=retained, source=source, calls=calls, sleeps=sleeps, urls=urls, snapshots=snapshots, failure=failure)


def test_waits_until_pipeline_complete_and_supervisor_exit_then_delivers_once(delivery):
    p = delivery
    p.snapshots[:] = [progress("running", True), progress("completed", True), progress("completed", False)]
    worker.run(p.args)
    assert p.sleeps == [15, 15]
    assert p.calls == ["collect_single_pair_rl.py", "single_pair_rl_six_report_app.py"]
    assert p.urls == [p.args.preview_url]
    status = worker.read(p.output / "work_status.json")
    assert status["task_status"] == "numerical_complete_visual_review_pending"
    assert status["numerical_delivery_complete"] is True and status["visual_review_pending"] is True
    assert status["models_completed"] == 60
    assert (p.retained / "immutable.bin").read_bytes() == b"RETAINED-EVIDENCE"
    attempt = next((p.output / "delivery").iterdir())
    result = worker.read(attempt / "completed.json")
    assert result["visual_review_pending"] is True and result["unchanged_retained_files"] == 2240


@pytest.mark.parametrize("state,failed", [("failed", 0), ("running", 1)])
def test_remote_failure_stops_without_collect_build_restart_or_completion(delivery, state, failed):
    p = delivery
    p.snapshots[:] = [progress(state, True, failed)]
    with pytest.raises(RuntimeError, match="no automatic restart"):
        worker.run(p.args)
    assert p.calls == [] and p.sleeps == [] and p.urls == []
    assert worker.read(p.output / "work_status.json")["task_status"] == "delivery_failed"
    assert not list((p.output / "delivery").glob("*/completed.json"))


def test_ten_consecutive_status_failures_stop_without_restart(delivery):
    p = delivery
    p.snapshots[:] = [subprocess.CalledProcessError(1, ["mock-ssh"]) for _ in range(10)]
    with pytest.raises(RuntimeError, match="Ten consecutive"):
        worker.run(p.args)
    assert len(p.sleeps) == 9 and p.calls == []
    assert worker.read(p.output / "work_status.json")["numerical_delivery_complete"] is False


@pytest.mark.parametrize("phase", ["collect", "readback", "build", "http"])
def test_failed_dependency_never_claims_delivery_complete_or_restarts(delivery, phase):
    p = delivery
    p.failure[phase] = True
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        worker.run(p.args)
    value = worker.read(p.output / "work_status.json")
    assert value["task_status"] == "delivery_failed" and value["numerical_delivery_complete"] is False
    assert not list((p.output / "delivery").glob("*/completed.json"))
    assert p.calls.count("collect_single_pair_rl.py") == 1
    assert p.calls.count("single_pair_rl_six_report_app.py") <= 1


def test_changed_launch_pinned_source_stops_before_any_external_command(delivery, monkeypatch):
    p = delivery
    def changed(*values):
        source = p.source / worker.SOURCE_NAMES[-1]
        source.write_bytes(source.read_bytes() + b"CHANGED")
        return progress()
    monkeypatch.setattr(worker, "remote_snapshot", changed)
    with pytest.raises(ValueError, match="source changed"):
        worker.run(p.args)
    assert p.calls == [] and p.urls == []


@pytest.mark.parametrize("relationship", ["same", "output_under_retained", "retained_under_output"])
def test_overlapping_studies_are_rejected_before_any_write_or_wait(delivery, relationship):
    p = delivery
    if relationship == "same": p.args.output = p.retained
    elif relationship == "output_under_retained": p.args.output = p.retained / "nested"
    else: p.args.retained_study = p.output / "nested"
    before = {str(path): path.read_bytes() for path in p.retained.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="disjoint"):
        worker.run(p.args)
    assert p.calls == [] and p.sleeps == [] and p.urls == []
    assert {str(path): path.read_bytes() for path in p.retained.rglob("*") if path.is_file()} == before
    assert not (p.output / "delivery").exists()


@pytest.mark.parametrize("relationship", ["same", "destination_child", "source_child"])
def test_preview_copy_refuses_overlapping_roots(tmp_path, relationship):
    source = tmp_path / "source"
    source.mkdir(); (source / "index.html").write_bytes(b"ORIGINAL")
    if relationship == "same": destination = source
    elif relationship == "destination_child": destination = source / "nested"
    else: destination, source = source, source / "nested"
    with pytest.raises(ValueError, match="disjoint"):
        worker.publish_preview(source, destination)
    assert (tmp_path / "source/index.html").read_bytes() == b"ORIGINAL"


def test_preview_copy_preserves_a_different_existing_destination(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir(); destination.mkdir()
    (source / "index.html").write_bytes(b"NEW")
    (destination / "index.html").write_bytes(b"PRESERVE-USER-COPY")
    with pytest.raises(FileExistsError): worker.publish_preview(source, destination)
    assert (destination / "index.html").read_bytes() == b"PRESERVE-USER-COPY"


@pytest.mark.parametrize("damage", ["index", "snapshot", "manifest_hash", "snapshot_path", "bytes"])
def test_preview_requires_actual_official_index_snapshot_bytes_and_build_binding(tmp_path, damage):
    dist, build = preview_fixture(tmp_path)
    manifest = worker.read(dist / "data-app-build.json")
    if damage == "index": (dist / "index.html").write_bytes(b"CHANGED")
    elif damage == "snapshot": (dist / manifest["snapshot"]["path"]).write_bytes(b"CHANGED")
    elif damage == "manifest_hash": manifest["html"]["sha256"] = "0" * 64
    elif damage == "snapshot_path": manifest["snapshot"]["path"] = "../snapshot.json"
    else: manifest["html"]["bytes"] += 1
    write_json(dist / "data-app-build.json", manifest)
    with pytest.raises(ValueError): worker.verify_built_preview(dist, build)


def test_delivery_attempt_directory_cannot_follow_a_link(delivery, monkeypatch):
    p = delivery
    linked = p.output / "delivery"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.absolute() == linked.absolute() or original(path))
    with pytest.raises(ValueError, match="link"):
        worker.run(p.args)
    assert p.calls == [] and p.urls == []
