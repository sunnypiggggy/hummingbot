"""Delivery-control tests use explicitly nonfinancial artifacts and mocked I/O."""
from argparse import Namespace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from scripts import finish_sac_validation_optimization as worker


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    args = Namespace(output=tmp_path / "optimization", baseline_btc=tmp_path / "original-btc",
        baseline_eth=tmp_path / "original-eth", retained_study=tmp_path / "original-six",
        host="fixture-host", remote_root="/fixture/optimization", remote_python="/fixture/python",
        wsl_node="/fixture/node", preview_dir=tmp_path / "preview" / "sac-test",
        preview_url="http://127.0.0.1:8774/sac-test/index.html?view=1", poll_interval=15)
    for p in (args.output, args.baseline_btc, args.baseline_eth, args.retained_study):
        p.mkdir()
    monkeypatch.setattr(worker, "SOURCE_NAMES", ("scripts/finish_sac_validation_optimization.py",))
    monkeypatch.setattr(worker, "remote_snapshot", lambda *a: {"pipeline": {"status": "completed", "phase": "completed"},
        "supervisor_alive": False, "models_completed": 100, "failed_model_attempts": 0, "stages": {}})
    calls = []

    def subprocess_run(command, **kwargs):
        calls.append(command)
        if command[2].endswith("collect_sac_validation_optimization.py"):
            write(args.output / "completion_contract_audit.json", {"nonfinancial_fixture": True})
        elif command[2].endswith("sac_validation_optimization_report_app.py"):
            report = args.output / "report"
            write(report / "reviewed.json", {"queries": {"account_summary": {"rows": [{"fixture_id": i} for i in range(256)]}}})
            (report / "report.html").write_text("nonfinancial fixture offline HTML", encoding="utf-8")
            dist = report / "report_app/dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text("nonfinancial fixture preview", encoding="utf-8")
            write(report / "report_build.json", {"html_sha256": worker.common.sha(report / "report.html"),
                "reviewed_sha256": worker.common.sha(report / "reviewed.json")})
        else:
            raise AssertionError("Unexpected external command")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", subprocess_run)
    monkeypatch.setattr(worker.common, "verify_built_preview", lambda directory, build: {"index.html": worker.common.sha(directory / "index.html")})
    handlers_seen = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return (args.preview_dir / "index.html").read_bytes()

    def opener(*handlers):
        handlers_seen.extend(handlers)
        return SimpleNamespace(open=lambda *a, **k: Response())

    monkeypatch.setattr(worker.urllib.request, "build_opener", opener)
    return args, calls, handlers_seen


def test_complete_delivery_preserves_numerical_and_visual_separation(delivery):
    args, calls, handlers = delivery
    worker.run(args)
    status = worker.common.read(args.output / "work_status.json")
    assert status["numerical_delivery_complete"] is True and status["visual_review_pending"] is True
    assert status["preview_http_verified"] is True and status["delivery_status"] == "completed"
    assert len(calls) == 2
    command = calls[1]
    for flag, value in (("--baseline-btc-root", args.baseline_btc), ("--baseline-eth-root", args.baseline_eth),
        ("--retained-study-root", args.retained_study), ("--study-root", args.output)):
        assert command[command.index(flag) + 1] == str(value.resolve())
    assert all(h.proxies == {} for h in handlers)
    assert len(list((args.output / "delivery").glob("*/completed.json"))) == 1


def test_offline_completion_is_retained_when_loopback_unavailable(delivery, monkeypatch):
    args, calls, _ = delivery

    def unavailable(*a, **kw):
        raise urllib.error.URLError("fixture 503")

    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=unavailable))
    worker.run(args)
    status = worker.common.read(args.output / "work_status.json")
    assert status["numerical_delivery_complete"] is True
    assert status["preview_http_verified"] is False
    assert status["delivery_status"] == "numerical_complete_preview_unavailable"
    assert status["visual_review_pending"] is True and len(calls) == 2


def test_http200_wrong_artifact_is_not_success(delivery, monkeypatch):
    args, _, _ = delivery

    class Wrong:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b"a different report"

    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **k: Wrong()))
    with pytest.raises(ValueError, match="exact report"):
        worker.run(args)
    assert not list((args.output / "delivery").glob("*/completed.json"))
    assert list((args.output / "delivery").glob("*/failed.json"))


def test_remote_failure_never_restarts_or_collects(delivery, monkeypatch):
    args, calls, _ = delivery
    monkeypatch.setattr(worker, "remote_snapshot", lambda *a: {"pipeline": {"status": "failed", "phase": "tuning"},
        "models_completed": 7, "failed_model_attempts": 1, "supervisor_alive": False})
    with pytest.raises(RuntimeError, match="no automatic restart"):
        worker.run(args)
    assert calls == []
    assert worker.common.read(args.output / "work_status.json")["delivery_status"] == "failed"


def test_source_change_stops_delivery_without_retraining(delivery, monkeypatch):
    args, calls, _ = delivery

    def changed(*a):
        raise ValueError("fixture pinned source changed")

    monkeypatch.setattr(worker.common, "verify_sources", changed)
    with pytest.raises(ValueError, match="source changed"):
        worker.run(args)
    assert calls == []


def test_overlapping_original_study_is_rejected(delivery):
    args, calls, _ = delivery
    args.baseline_btc = args.output
    with pytest.raises(ValueError, match="disjoint"):
        worker.run(args)
    assert calls == []


def test_wrong_preview_destination_is_rejected(delivery):
    args, calls, _ = delivery
    args.preview_url = "http://0.0.0.0:8774/sac-test/index.html"
    with pytest.raises(ValueError, match="loopback"):
        worker.run(args)
    assert calls == []
