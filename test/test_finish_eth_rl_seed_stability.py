"""ETH delivery orchestration only; no waits, network, build or learner runs.

The old-study readback test hashes existing immutable evidence against itself.
Other tests use labelled toy build files and mocked process/HTTP boundaries.
General link, archive and SDK-preview safety belongs to the common 22-test suite.
"""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import finish_eth_rl_seed_stability as worker


ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def fake_official_build(output):
    report, dist = output / "report", output / "report/report_app/dist"
    dist.mkdir(parents=True, exist_ok=False)
    (report / "report.html").write_bytes(b"SYNTHETIC NONFINANCIAL PORTABLE REPORT")
    reviewed = {"synthetic_fixture_only": True, "queries": {"toy_contract_rows": {"rows": [{"synthetic_only": True}]}}}
    dump(report / "reviewed.json", reviewed)
    (dist / "index.html").write_bytes(b"SYNTHETIC NONFINANCIAL PREVIEW INDEX")
    snapshot = b'{"synthetic_fixture_only":true,"queries":{}}\n'
    digest = hashlib.sha256(snapshot).hexdigest()
    name = "snapshot." + digest + ".json"
    (dist / name).write_bytes(snapshot)
    runtime = "a" * 64
    manifest = {"version": 1, "kind": "separate-data-v1",
                "html": {"path": "index.html", "sha256": worker.common.sha(dist / "index.html"), "bytes": (dist / "index.html").stat().st_size},
                "snapshot": {"path": name, "sha256": digest, "bytes": len(snapshot)},
                "runtimeSha256": runtime, "sourceSnapshotSha256": digest}
    dump(dist / "data-app-build.json", manifest)
    dump(report / "report_build.json", {"html_sha256": worker.common.sha(report / "report.html"),
        "reviewed_sha256": worker.common.sha(report / "reviewed.json"),
        "runtime": {"htmlSha256": manifest["html"]["sha256"], "snapshotSha256": digest, "runtimeSha256": runtime}})


def progress(state="completed", *, alive=False, failed=0):
    return {"pipeline": {"status": state}, "supervisor_alive": alive,
            "algorithms": {a: {"completed": 30, "failed": failed if a == "td3" else 0} for a in ("sac", "td3")}}


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    source = tmp_path / "synthetic_source"
    for name in worker.SOURCE_NAMES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# SYNTHETIC EXECUTION-BOUNDARY SOURCE " + name, encoding="utf-8")
    monkeypatch.setattr(worker, "ROOT", source)
    monkeypatch.setattr(worker.common, "ROOT", source)
    output, retained = tmp_path / "new_study", tmp_path / "retained_study"
    output.mkdir()
    retained.mkdir()
    (retained / "immutable.bin").write_bytes(b"PRESERVE ORIGINAL NONFINANCIAL FIXTURE")
    args = SimpleNamespace(output=output, retained_study=retained, host="NO_NETWORK_FIXTURE", remote_python="/fixture/python",
        remote_root="/fixture/no_remote_call", poll_interval=15, wsl_node="/fixture/node",
        preview_dir=tmp_path / "existing_server/eth-seed-stability",
        preview_url="http://127.0.0.1:8774/eth-seed-stability/index.html?view=1")
    calls, sleeps, urls, snapshots = [], [], [], [progress()]
    failures = {"collect": False, "build": False}
    def snapshot(*unused):
        return snapshots.pop(0)
    monkeypatch.setattr(worker.common, "remote_snapshot", snapshot)
    monkeypatch.setattr(worker.time, "sleep", lambda value: sleeps.append(value))
    def run(command, **unused):
        name = Path(command[2]).name
        calls.append(name)
        assert name in ("collect_eth_rl_seed_stability.py", "eth_rl_seed_stability_report_app.py"), "No train/restart/server command allowed"
        if name == "collect_eth_rl_seed_stability.py":
            if failures["collect"]:
                raise subprocess.CalledProcessError(1, command)
        else:
            if failures["build"]:
                raise subprocess.CalledProcessError(1, command)
            fake_official_build(output)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(worker.subprocess, "run", run)
    monkeypatch.setattr(worker, "retained_readback", lambda *unused: {"passed": True, "cases_checked": 112, "files_checked": 1120, "synthetic_boundary_only": True})
    class Response:
        status = 200
        def read(self):
            return (args.preview_dir / "index.html").read_bytes()
        def __enter__(self):
            return self
        def __exit__(self, *unused):
            return False
    monkeypatch.setattr(worker.urllib.request, "urlopen", lambda url, **unused: (urls.append(url), Response())[1])
    return SimpleNamespace(args=args, output=output, retained=retained, calls=calls, sleeps=sleeps,
                           urls=urls, snapshots=snapshots, failures=failures)


def test_completed_supervisor_must_exit_before_once_only_numeric_delivery(delivery):
    p = delivery
    p.snapshots[:] = [progress("completed", alive=True), progress("completed", alive=False)]
    worker.run(p.args)
    assert p.sleeps == [15]
    assert p.calls == ["collect_eth_rl_seed_stability.py", "eth_rl_seed_stability_report_app.py"]
    assert p.urls == [p.args.preview_url]
    status = worker.common.read(p.output / "work_status.json")
    assert status["task_status"] == "numerical_complete_visual_review_pending"
    assert status["numerical_delivery_complete"] is True and status["visual_review_pending"] is True
    assert status["models_completed"] == 60
    result = worker.common.read(next((p.output / "delivery").glob("*/completed.json")))
    assert result["original_cases_unchanged"] == 112 and result["original_files_unchanged"] == 1120
    assert result["query_counts"] == {"toy_contract_rows": 1}


@pytest.mark.parametrize("state,failed", [("failed", 0), ("running", 1)])
def test_remote_failures_never_collect_restart_or_claim_completion(delivery, state, failed):
    p = delivery
    p.snapshots[:] = [progress(state, alive=True, failed=failed)]
    with pytest.raises(RuntimeError, match="no automatic restart"):
        worker.run(p.args)
    assert p.calls == [] and p.sleeps == [] and p.urls == []
    assert worker.common.read(p.output / "work_status.json")["numerical_delivery_complete"] is False
    assert not list((p.output / "delivery").glob("*/completed.json"))


@pytest.mark.parametrize("phase", ["collect", "build"])
def test_collect_or_build_failures_never_become_numeric_completion(delivery, phase):
    p = delivery
    p.failures[phase] = True
    with pytest.raises(subprocess.CalledProcessError):
        worker.run(p.args)
    assert p.calls.count("collect_eth_rl_seed_stability.py") == 1
    assert p.calls.count("eth_rl_seed_stability_report_app.py") == (phase == "build")
    assert p.urls == []
    status = worker.common.read(p.output / "work_status.json")
    assert status["task_status"] == "delivery_failed" and status["numerical_delivery_complete"] is False
    assert not list((p.output / "delivery").glob("*/completed.json"))


def test_actual_old_eth_only_replay_selection_is_112_accounts_and_1120_files():
    retained = ROOT / "results/backtests/single_pair_rl_six_20261030"
    result = worker.retained_readback(retained, retained)
    assert result["passed"] is True and result["cases_checked"] == 112 and result["files_checked"] == 1120
    assert all("/ETHUSDT/" in name and not any("/" + a + "/" in name for a in ("tqc", "droq", "crossq", "recurrent_ppo"))
               for name in result["files_sha256"])
    for kind in ("evaluation", "repeat"):
        for split in ("validation", "test"):
            selected = [name for name in result["files_sha256"] if name.startswith(kind + "/" + split + "/") and name.endswith("/summary.json")]
            assert len(selected) == 28


def test_existing_different_preview_is_preserved_without_completion(delivery):
    p = delivery
    p.args.preview_dir.mkdir(parents=True)
    path = p.args.preview_dir / "index.html"
    original = b"PRESERVE EXISTING USER PREVIEW"
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        worker.run(p.args)
    assert path.read_bytes() == original and p.urls == []
    assert worker.common.read(p.output / "work_status.json")["numerical_delivery_complete"] is False
