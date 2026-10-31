"""Delivery-control fixtures are nonfinancial and all external operations mocked."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from scripts import finish_rolling_single_pair as worker


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def completed_progress():
    return {"pipeline": {"status": "completed", "phase": "complete"}, "models_completed": 1590,
        "failed_model_attempts": 0, "supervisor_alive": False, "markers": {
            "fit_complete.json": {"models": 1590}, "evaluation_complete.json": {"accounts": 180},
            "completion_gate.json": {"passed": True, "failure_count": 0}}}


def transport_fixture(output):
    """This is a control receipt fixture, not a valid financial study or archive."""
    write(output / "protocol.json", {"sources": {"scripts/audit_rolling_single_pair.py": "f" * 64}, "nonfinancial_fixture": True})
    (output / "summary.csv").write_text("nonfinancial-control-fixture\n", encoding="utf-8")
    write(output / "accounts_manifest.json", {"nonfinancial_fixture": True})
    inputs = {n: worker.common.sha(output / n) for n in ("protocol.json", "summary.csv", "accounts_manifest.json")}
    external = {"/fixture-market/archive.pkl.gz": "e" * 64}
    gate = {"passed": True, "failure_count": 0, "failures": [], "expected_accounts": 180,
        "accounts_checked": 180, "repeats_checked": 180, "expected_models": 1590, "models_checked": 1590,
        "external_inputs_sha256": external, "inputs_sha256": inputs, "nonfinancial_fixture": True}
    write(output / "completion_gate.json", gate)
    transport = output / "transport"
    transport.mkdir()
    archive = transport / "control-fixture-only.tar"
    archive.write_bytes(b"nonfinancial control receipt fixture; not a real archive")
    receipt = {"archive": archive.name, "archive_sha256": worker.common.sha(archive),
        "protocol_sha256": inputs["protocol.json"], "completion_gate_sha256": worker.common.sha(output / "completion_gate.json"),
        "verifier_source_sha256": "f" * 64, "external_inputs_sha256": external, "external_verified_sha256": deepcopy(external),
        "external_rechecked_after_archive": True, "files": {**inputs, "completion_gate.json": worker.common.sha(output / "completion_gate.json")}}
    write(transport / "control-fixture-only.tar.receipt.json", receipt)
    return gate, receipt


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    output = tmp_path / "new-study"
    preview = tmp_path / "existing-private-preview"
    output.mkdir()
    preview.mkdir()
    (preview / "index.html").write_text("original root must remain unchanged", encoding="utf-8")
    old = preview / "old-user-report"
    old.mkdir()
    (old / "index.html").write_text("original child must remain unchanged", encoding="utf-8")
    args = Namespace(output=output, host=worker.ops.HOST, remote_root=worker.ops.REMOTE_ROOT,
        remote_python=worker.ops.REMOTE_PYTHON, wsl_node="/nonfinancial-fixture/node", preview_dir=preview / worker.CHILD_NAME,
        preview_url=worker.PREVIEW_URL, poll_interval=15)
    monkeypatch.setattr(worker, "STUDY_ROOT", output)
    monkeypatch.setattr(worker, "PREVIEW_ROOT", preview)
    monkeypatch.setattr(worker, "SOURCE_NAMES", ("scripts/finish_rolling_single_pair.py",))
    monkeypatch.setattr(worker, "remote_snapshot", completed_progress)
    sleep_calls = []
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[2].endswith("collect_rolling_single_pair.py"):
            transport_fixture(output)
        elif command[2].endswith("all_algorithms_rolling_report_app.py"):
            report = output / "report"
            queries = {k: {"rows": [None] * n} for k, n in worker.QUERY_COUNTS.items()}
            queries["historical_results"] = {"rows": [{"nonfinancial_fixture": True}]}
            write(report / "reviewed.json", {"queries": queries, "nonfinancial_fixture": True})
            (report / "report.html").write_text("nonfinancial fixture offline HTML", encoding="utf-8")
            dist = report / "report_app/dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text("nonfinancial fixture preview", encoding="utf-8")
            write(report / "report_build.json", {"html_sha256": worker.common.sha(report / "report.html"), "reviewed_sha256": worker.common.sha(report / "reviewed.json")})
        else:
            raise AssertionError("Unexpected external command; training/server operations are forbidden")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", run)
    monkeypatch.setattr(worker.common, "verify_built_preview", lambda directory, build: {"index.html": worker.common.sha(directory / "index.html")})
    handlers = []

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return (args.preview_dir / "index.html").read_bytes()

    def opener(*values):
        handlers.extend(values)
        return SimpleNamespace(open=lambda *a, **k: Response())
    monkeypatch.setattr(worker.urllib.request, "build_opener", opener)
    return args, calls, handlers, sleep_calls


def test_complete_control_delivery_does_not_claim_visual_review(delivery):
    args, calls, handlers, sleep_calls = delivery
    root_before = (args.preview_dir.parent / "index.html").read_bytes()
    child_before = (args.preview_dir.parent / "old-user-report/index.html").read_bytes()
    result = worker.run(args)
    assert result["numerical_complete"] and result["numerical_delivery_complete"] and result["visual_review_pending"]
    assert result["preview_http_verified"] and result["original_preview_entries_unchanged"]
    status = worker.common.read(args.output / "work_status.json")
    assert status["delivery_status"] == "completed" and status["visual_review_pending"]
    assert status["task_status"] == "numerical_complete_visual_review_pending"
    assert len(calls) == 2 and not sleep_calls
    assert calls[0][2].endswith("collect_rolling_single_pair.py") and calls[1][2].endswith("all_algorithms_rolling_report_app.py")
    assert all(h.proxies == {} for h in handlers)
    assert (args.preview_dir.parent / "index.html").read_bytes() == root_before
    assert (args.preview_dir.parent / "old-user-report/index.html").read_bytes() == child_before
    assert len(list((args.output / "delivery").glob("*/completed.json"))) == 1


def test_unavailable_loopback_retains_numeric_offline_artifact(delivery, monkeypatch):
    args, _, _, _ = delivery
    def unavailable(*a, **k): raise urllib.error.URLError("fixture connection refused")
    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=unavailable))
    worker.run(args)
    status = worker.common.read(args.output / "work_status.json")
    assert status["numerical_complete"] and status["numerical_delivery_complete"]
    assert status["delivery_status"] == "numerical_complete_preview_unavailable"
    assert not status["preview_http_verified"] and status["visual_review_pending"]


def test_http200_wrong_report_is_not_success(delivery, monkeypatch):
    args, _, _, _ = delivery
    class Wrong:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b"different report"
    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: Wrong()))
    with pytest.raises(ValueError, match="exact report"):
        worker.run(args)
    status = worker.common.read(args.output / "work_status.json")
    assert status["delivery_status"] == "failed" and status["numerical_complete"]
    assert not status["numerical_delivery_complete"] and status["visual_review_pending"]
    assert list((args.output / "delivery").glob("*/failed.json"))
    assert not list((args.output / "delivery").glob("*/completed.json"))


@pytest.mark.parametrize("failure", ["model", "pipeline", "incomplete_models", "incomplete_accounts", "failed_gate"])
def test_failed_or_incomplete_research_is_never_collected_or_restarted(delivery, monkeypatch, failure):
    args, calls, _, _ = delivery
    progress = completed_progress()
    if failure == "model": progress["failed_model_attempts"] = 1
    elif failure == "pipeline": progress["pipeline"]["status"] = "failed"
    elif failure == "incomplete_models": progress["models_completed"] = 1589
    elif failure == "incomplete_accounts": progress["markers"]["evaluation_complete.json"]["accounts"] = 179
    else: progress["markers"]["completion_gate.json"]["passed"] = False
    monkeypatch.setattr(worker, "remote_snapshot", lambda: progress)
    with pytest.raises((ValueError, RuntimeError)):
        worker.run(args)
    assert calls == []
    assert worker.common.read(args.output / "work_status.json")["delivery_status"] == "failed"


def test_completed_status_waits_until_recorded_process_exits(delivery, monkeypatch):
    args, calls, _, sleeps = delivery
    active = completed_progress()
    active["supervisor_alive"] = True
    samples = iter([active, completed_progress()])
    monkeypatch.setattr(worker, "remote_snapshot", lambda: next(samples))
    worker.run(args)
    assert sleeps == [15] and len(calls) == 2


def test_missing_recorded_pipeline_three_observations_fail_without_restart(delivery, monkeypatch):
    args, calls, _, sleeps = delivery
    progress = completed_progress()
    progress["pipeline"]["status"] = "running"
    monkeypatch.setattr(worker, "remote_snapshot", lambda: progress)
    with pytest.raises(RuntimeError, match="without completion"):
        worker.run(args)
    assert calls == [] and sleeps == [15, 15]


def test_ten_status_errors_preserve_attempt_and_do_not_collect(delivery, monkeypatch):
    args, calls, _, sleeps = delivery
    def broken(): raise RuntimeError("fixture status transport failure")
    monkeypatch.setattr(worker, "remote_snapshot", broken)
    with pytest.raises(RuntimeError, match="Ten consecutive"):
        worker.run(args)
    assert calls == [] and len(sleeps) == 9
    error_log = next((args.output / "delivery").glob("*/connection_errors.jsonl"))
    assert len(error_log.read_text(encoding="utf-8").splitlines()) == 10


def test_source_change_stops_delivery_without_mutating_research(delivery, monkeypatch):
    args, calls, _, _ = delivery
    def changed(*a): raise ValueError("fixture source changed")
    monkeypatch.setattr(worker.common, "verify_sources", changed)
    with pytest.raises(ValueError, match="source changed"):
        worker.run(args)
    assert calls == [] and list((args.output / "delivery").glob("*/failed.json"))


def test_missing_launch_source_still_records_failed_attempt(delivery, monkeypatch):
    args, calls, _, _ = delivery
    monkeypatch.setattr(worker, "SOURCE_NAMES", ("scripts/nonexistent_delivery_source_fixture.py",))
    with pytest.raises(FileNotFoundError):
        worker.run(args)
    assert calls == [] and list((args.output / "delivery").glob("*/failed.json"))


@pytest.mark.parametrize("change", ["public_url", "root_preview", "old_child", "other_study", "other_host", "other_remote", "long_poll"])
def test_scope_cannot_expand_to_old_reports_other_hosts_or_public_server(delivery, change):
    args, calls, _, _ = delivery
    if change == "public_url": args.preview_url = worker.PREVIEW_URL.replace("127.0.0.1", "0.0.0.0")
    elif change == "root_preview": args.preview_dir = args.preview_dir.parent
    elif change == "old_child": args.preview_dir = args.preview_dir.parent / "old-user-report"
    elif change == "other_study": args.output = args.output.parent / "old-user-study"
    elif change == "other_host": args.host = "unrelated-host"
    elif change == "other_remote": args.remote_root = "/unrelated/study"
    else: args.poll_interval = 600
    with pytest.raises(ValueError):
        worker.run(args)
    assert calls == []


def test_preview_child_conflict_preserves_existing_child(delivery):
    args, _, _, _ = delivery
    args.preview_dir.mkdir()
    (args.preview_dir / "index.html").write_text("existing different delivery", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different existing preview"):
        worker.run(args)
    assert (args.preview_dir / "index.html").read_text(encoding="utf-8") == "existing different delivery"


def test_delivery_lease_is_exclusive_and_released_on_exit(delivery):
    args, _, _, _ = delivery
    with worker.delivery_lease(args.output):
        with pytest.raises(RuntimeError, match="Another delivery worker"):
            with worker.delivery_lease(args.output):
                raise AssertionError("Unreachable concurrent lease")
    with worker.delivery_lease(args.output):
        pass


@pytest.mark.parametrize("change", ["external_after", "not_rechecked", "archive", "gate", "verifier", "file_pin", "account_count", "model_count"])
def test_transport_receipt_requires_matching_current_gate_and_market(tmp_path, change):
    gate, receipt = transport_fixture(tmp_path)
    path = tmp_path / "transport/control-fixture-only.tar.receipt.json"
    if change == "external_after": receipt["external_verified_sha256"] = {"/different-market": "0" * 64}
    elif change == "not_rechecked": receipt["external_rechecked_after_archive"] = False
    elif change == "archive": receipt["archive_sha256"] = "0" * 64
    elif change == "gate": receipt["completion_gate_sha256"] = "0" * 64
    elif change == "verifier": receipt["verifier_source_sha256"] = "0" * 64
    elif change == "file_pin": receipt["files"]["summary.csv"] = "0" * 64
    elif change == "account_count": gate["accounts_checked"] = 179
    else: gate["models_checked"] = 1589
    write(path, receipt)
    write(tmp_path / "completion_gate.json", gate)
    with pytest.raises(ValueError):
        worker.transport_readback(tmp_path)


def test_status_action_is_only_read_operation_and_normalizes_markers(monkeypatch):
    requests = []
    def status(action, payload, **kwargs):
        requests.append((action, payload, kwargs))
        return {"at": "unit", "remote_root": worker.ops.REMOTE_ROOT, "models": {"started": 12, "completed": 3, "failed": 0},
            "markers": {"pipeline_status.json": {"status": "running", "phase": "fit"}}, "active_recorded_phases": [{"pid": 123}]}
    monkeypatch.setattr(worker.ops, "remote_request", status)
    snapshot = worker.remote_snapshot()
    assert requests == [("status", {}, {"timeout": 45})]
    assert snapshot["models_completed"] == 3 and snapshot["supervisor_alive"]
    assert snapshot["pipeline"]["phase"] == "fit"
