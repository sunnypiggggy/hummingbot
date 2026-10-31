"""BTC/ETH combined delivery boundaries, not training or real network calls.

Existing immutable BTC replay files are read only against themselves. Other
cases use toy SDK build files and mocked process/loopback responses. The common
worker's separate suite owns general archive/path/SDK-manifest safety.
"""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

from scripts import finish_btc_rl_seed_stability as worker


ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def fake_official_build(output):
    report, dist = output / "report", output / "report/report_app/dist"
    dist.mkdir(parents=True, exist_ok=False)
    (report / "report.html").write_bytes(b"SYNTHETIC NONFINANCIAL COMBINED PORTABLE REPORT")
    dump(report / "reviewed.json", {"synthetic_fixture_only": True, "queries": {"toy_contract_rows": {"rows": [{"synthetic_only": True}]}}})
    (dist / "index.html").write_bytes(b"SYNTHETIC NONFINANCIAL COMBINED PREVIEW")
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
def make_delivery(tmp_path, monkeypatch):
    def make(name):
        base = tmp_path / name
        source = base / "synthetic_source"
        for filename in worker.SOURCE_NAMES:
            path = source / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# SYNTHETIC EXECUTION-BOUNDARY SOURCE " + filename, encoding="utf-8")
        monkeypatch.setattr(worker, "ROOT", source)
        monkeypatch.setattr(worker.common, "ROOT", source)
        output, retained, eth = base / "new_btc", base / "retained_six", base / "complete_eth"
        for path in (output, retained, eth):
            path.mkdir()
        (retained / "immutable.bin").write_bytes(b"PRESERVE ORIGINAL STUDY")
        (eth / "immutable.bin").write_bytes(b"PRESERVE COMPLETED ETH; NO RETRAIN")
        args = SimpleNamespace(output=output, retained_study=retained, eth_study=eth, host="NO_NETWORK_FIXTURE",
            remote_python="/fixture/python", remote_root="/fixture/no_remote_call", poll_interval=15, wsl_node="/fixture/node",
            preview_dir=base / "existing_server/btceth-stability",
            preview_url="http://127.0.0.1:8774/btceth-stability/index.html?view=1")
        calls, commands, sleeps, urls, proxies, snapshots = [], [], [], [], [], [progress()]
        failures = {"collect": False, "build": False, "http": None}
        monkeypatch.setattr(worker.common, "remote_snapshot", lambda *unused: snapshots.pop(0))
        monkeypatch.setattr(worker.time, "sleep", lambda value: sleeps.append(value))
        def run(command, **unused):
            filename = Path(command[2]).name
            calls.append(filename); commands.append(command)
            assert filename in ("collect_btc_rl_seed_stability.py", "btceth_seed_stability_report_app.py"), "No training/restart/server/ETH-run command permitted"
            if filename == "collect_btc_rl_seed_stability.py":
                if failures["collect"]: raise subprocess.CalledProcessError(1, command)
            else:
                if failures["build"]: raise subprocess.CalledProcessError(1, command)
                fake_official_build(output)
            return SimpleNamespace(returncode=0)
        monkeypatch.setattr(worker.subprocess, "run", run)
        monkeypatch.setattr(worker, "retained_readback", lambda *unused: {"passed": True, "cases_checked": 112, "files_checked": 1120, "synthetic_boundary_only": True})
        class Response:
            status = 200
            def read(self):
                return b"WRONG HTML SHA" if failures["http"] == "wrong_sha" else (args.preview_dir / "index.html").read_bytes()
            def __enter__(self): return self
            def __exit__(self, *unused): return False
        def open_url(url, **unused):
            urls.append(url)
            if failures["http"] == "connection": raise urllib.error.URLError("SYNTHETIC LOOPBACK UNAVAILABLE")
            if failures["http"] == "503": raise urllib.error.HTTPError(url, 503, "SYNTHETIC 503", {}, None)
            return Response()
        def build_opener(handler):
            proxies.append(dict(handler.proxies))
            return SimpleNamespace(open=open_url)
        monkeypatch.setattr(worker.urllib.request, "build_opener", build_opener)
        def forbid_system_proxy(*unused):
            raise AssertionError("Do not install/change a global proxy opener")
        monkeypatch.setattr(worker.urllib.request, "install_opener", forbid_system_proxy)
        return SimpleNamespace(args=args, output=output, retained=retained, eth=eth, calls=calls, commands=commands,
            sleeps=sleeps, urls=urls, proxies=proxies, snapshots=snapshots, failures=failures)
    return make


def test_waits_for_supervisor_exit_then_combines_three_roots_without_eth_training(make_delivery):
    p = make_delivery("normal")
    p.snapshots[:] = [progress(alive=True), progress(alive=False)]
    worker.run(p.args)
    assert p.sleeps == [15]
    assert p.calls == ["collect_btc_rl_seed_stability.py", "btceth_seed_stability_report_app.py"]
    command = p.commands[1]
    for flag, root in (("--study-root", p.output), ("--eth-study-root", p.eth), ("--retained-study-root", p.retained)):
        assert command[command.index(flag) + 1] == str(root)
    assert p.proxies == [{}] and p.urls == [p.args.preview_url]
    status = worker.common.read(p.output / "work_status.json")
    assert status["numerical_delivery_complete"] is True and status["visual_review_pending"] is True
    assert status["delivery_status"] == "completed" and status["preview_http_verified"] is True
    result = worker.common.read(next((p.output / "delivery").glob("*/completed.json")))
    assert result["total_compared_models"] == 120 and result["eth_study_root"] == str(p.eth)
    assert result["original_cases_unchanged"] == 112 and result["original_files_unchanged"] == 1120
    assert (p.eth / "immutable.bin").read_bytes() == b"PRESERVE COMPLETED ETH; NO RETRAIN"


def test_remote_or_model_failure_stops_without_collect_restart_or_completion(make_delivery):
    for label, state, failed in (("phase", "failed", 0), ("model", "running", 1)):
        p = make_delivery(label)
        p.snapshots[:] = [progress(state, alive=True, failed=failed)]
        with pytest.raises(RuntimeError, match="no automatic restart"): worker.run(p.args)
        assert p.calls == [] and p.urls == [] and p.sleeps == []
        assert worker.common.read(p.output / "work_status.json")["numerical_delivery_complete"] is False
        assert not list((p.output / "delivery").glob("*/completed.json"))


def test_collect_or_combined_build_failure_does_not_claim_numeric_completion(make_delivery):
    for phase in ("collect", "build"):
        p = make_delivery(phase); p.failures[phase] = True
        with pytest.raises(subprocess.CalledProcessError): worker.run(p.args)
        assert p.calls.count("collect_btc_rl_seed_stability.py") == 1
        assert p.calls.count("btceth_seed_stability_report_app.py") == (phase == "build")
        assert p.urls == []
        assert worker.common.read(p.output / "work_status.json")["task_status"] == "delivery_failed"
        assert not list((p.output / "delivery").glob("*/completed.json"))


def test_real_old_btc_only_replay_selection_is_112_accounts_and_1120_files():
    retained = ROOT / "results/backtests/single_pair_rl_six_20261030"
    result = worker.retained_readback(retained, retained)
    assert result["passed"] is True and result["cases_checked"] == 112 and result["files_checked"] == 1120
    assert all("/BTCUSDT/" in name and "/ETHUSDT/" not in name and not any("/" + a + "/" in name for a in ("tqc", "droq", "crossq", "recurrent_ppo")) for name in result["files_sha256"])


def test_all_three_studies_must_be_mutually_disjoint_before_any_write(make_delivery):
    for label in ("btc_eth", "old_eth", "btc_old", "eth_child_btc"):
        p = make_delivery(label)
        if label == "btc_eth": p.args.eth_study = p.output
        elif label == "old_eth": p.args.eth_study = p.retained
        elif label == "btc_old": p.args.output = p.retained
        else: p.args.eth_study = p.output / "nested_eth"
        with pytest.raises(ValueError, match="disjoint"): worker.run(p.args)
        assert p.calls == [] and p.urls == [] and p.sleeps == []
        assert not (p.output / "delivery").exists()


def test_connection_and_503_preserve_verified_numerical_delivery_and_offline_html(make_delivery):
    for error in ("connection", "503"):
        p = make_delivery(error); p.failures["http"] = error
        worker.run(p.args)
        status = worker.common.read(p.output / "work_status.json")
        assert status["numerical_delivery_complete"] is True and status["all_financial_results_ready"] is True
        assert status["preview_http_verified"] is False and status["delivery_status"] == "numerical_complete_preview_unavailable"
        assert status["visual_review_pending"] is True and status["preview_error"]
        assert p.calls == ["collect_btc_rl_seed_stability.py", "btceth_seed_stability_report_app.py"] and p.proxies == [{}]
        assert (p.output / "report/report.html").is_file()
        assert list((p.output / "delivery").glob("*/completed.json")) and not list((p.output / "delivery").glob("*/failed.json"))


def test_http200_with_wrong_html_sha_remains_a_hard_delivery_failure(make_delivery):
    p = make_delivery("wrong_sha"); p.failures["http"] = "wrong_sha"
    with pytest.raises(ValueError, match="exact report"): worker.run(p.args)
    status = worker.common.read(p.output / "work_status.json")
    assert status["numerical_delivery_complete"] is False and status["delivery_status"] == "failed"
    assert not list((p.output / "delivery").glob("*/completed.json"))


def test_existing_different_preview_is_preserved(make_delivery):
    p = make_delivery("existing_preview"); p.args.preview_dir.mkdir(parents=True)
    target = p.args.preview_dir / "index.html"; original = b"PRESERVE USER PREVIEW"; target.write_bytes(original)
    with pytest.raises(FileExistsError): worker.run(p.args)
    assert target.read_bytes() == original and p.urls == []
    assert worker.common.read(p.output / "work_status.json")["numerical_delivery_complete"] is False
