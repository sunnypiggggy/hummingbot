"""Wait for the authorized rolling study, collect its audited evidence and report.

This control process does not start/restart training, change selection or sources,
create servers, change SSH configuration, publish externally, or trade. Actual
financial completion and human/browser visual review are separate states.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import finish_single_pair_rl_six as common
from scripts import rolling_single_pair_remote_ops as ops

STUDY = "single_pair_rolling_all_algorithms_20261031"
STUDY_ROOT = ROOT / "results/backtests" / STUDY
PREVIEW_ROOT = ROOT / "results/backtests/btceth_rl_20261029/report/report_app/dist"
CHILD_NAME = "single-pair-rolling-all-algorithms-20261031"
PREVIEW_URL = "http://127.0.0.1:8774/" + CHILD_NAME + "/index.html?view=1"
EXPECTED_MODELS, EXPECTED_ACCOUNTS = 1590, 180
SOURCE_NAMES = ("scripts/finish_rolling_single_pair.py", "scripts/rolling_single_pair_remote_ops.py",
    "scripts/finish_single_pair_rl_six.py", "scripts/collect_rolling_single_pair.py", "scripts/collect_single_pair_rl.py",
    "scripts/all_algorithms_rolling_report_app.py", "scripts/all_algorithms_rolling_report_app.jsx",
    "scripts/all_algorithms_rolling_report_app.css", "scripts/btceth_rl_report_app.py", "scripts/crypto_rl_reporting.py")
QUERY_COUNTS = {"account_summary": 180, "seed_statistics": 36, "matched_seed_results": 120,
    "matched_statistics": 24, "equity_history": 65880, "monthly_returns": 2340,
    "validation_candidates": 18, "candidate_selection": 6, "sac_replication": 2,
    "expanded_seed_results": 16, "experiment_families": 18, "framework_reviews": 5,
    "foundation_provenance": 3, "training_records": 1590, "audit_evidence": 1, "protocol": 1}


@contextmanager
def delivery_lease(output):
    """OS releases this lock on exit/crash; its harmless evidence file is kept."""
    folder = common.reject_links(output / "delivery")
    folder.mkdir(parents=True, exist_ok=True)
    path = common.reject_links(folder / ".worker.lock")
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another delivery worker owns this study; preserve its progress") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def remote_snapshot():
    """The existing control's status action is read-only and validates PID receipts."""
    raw = ops.remote_request("status", {}, timeout=45)
    if raw.get("remote_root") != ops.REMOTE_ROOT:
        raise ValueError("Status belongs to another research root")
    markers = raw.get("markers", {})
    phase = markers.get("pipeline_status.json", {"status": "not_started", "phase": "waiting_for_pipeline"})
    models = raw.get("models", {})
    failed = models.get("failed", 0)
    evaluation = markers.get("evaluation_status.json", {})
    return {"captured_at_utc": raw.get("at"), "pipeline": phase, "markers": markers,
        "models_completed": models.get("completed", 0), "models_started": models.get("started", 0),
        "failed_model_attempts": failed, "evaluation_completed": evaluation.get("completed", 0),
        "supervisor_alive": bool(raw.get("active_recorded_phases", [])),
        "active_recorded_phases": raw.get("active_recorded_phases", []), "resource_status": {
            k: raw[k] for k in ("available_cpus", "load_average", "gpus", "gpu_processes", "mem_available_kb") if k in raw}}


def ready_to_collect(progress):
    phase = progress["pipeline"]
    if progress["failed_model_attempts"] or phase["status"] == "failed":
        raise RuntimeError("The research failed; no automatic restart, repair or replacement")
    if phase["status"] != "completed" or progress["supervisor_alive"]:
        return False
    markers = progress["markers"]
    gate = markers.get("completion_gate.json", {})
    if (progress["models_completed"] != EXPECTED_MODELS
            or markers.get("fit_complete.json", {}).get("models") != EXPECTED_MODELS
            or markers.get("evaluation_complete.json", {}).get("accounts") != EXPECTED_ACCOUNTS
            or gate.get("passed") is not True or gate.get("failure_count") != 0):
        raise ValueError("Completed status lacks exact1590-model/180-account passed completion evidence")
    return True


def validate_arguments(args):
    output = common.reject_links(args.output).resolve()
    target = common.reject_links(args.preview_dir).resolve()
    parent = common.reject_links(PREVIEW_ROOT).resolve()
    if output != common.reject_links(STUDY_ROOT).resolve():
        raise ValueError("Only the named new independent rolling study may be delivered")
    if (args.host, args.remote_root, args.remote_python) != (ops.HOST, ops.REMOTE_ROOT, ops.REMOTE_PYTHON):
        raise ValueError("Use the existing authorized V100 host, study and Python environment")
    url = urllib.parse.urlsplit(args.preview_url)
    if (url.scheme != "http" or url.netloc != "127.0.0.1:8774" or url.fragment
            or url.path != "/" + CHILD_NAME + "/index.html" or url.query not in ("", "view=1")
            or target != parent / CHILD_NAME):
        raise ValueError("Use only the named child on the existing loopback preview service")
    if not parent.is_dir() or not (parent / "index.html").is_file():
        raise ValueError("The existing preview root must already exist; this worker cannot create a server")
    if not 15 <= args.poll_interval <= 60:
        raise ValueError("Use a bounded15–60 second status interval")
    return output, target, parent


def protect_existing_preview(root):
    """Record the root files and each existing child's delivered entry points."""
    result = {}
    for entry in root.iterdir():
        common.reject_links(entry)
        if entry.name == CHILD_NAME:
            continue
        if entry.is_file():
            result[entry.relative_to(root).as_posix()] = common.sha(entry)
        elif entry.is_dir():
            for name in ("index.html", "data-app-build.json"):
                path = entry / name
                if path.is_file():
                    common.reject_links(path)
                    result[path.relative_to(root).as_posix()] = common.sha(path)
    return result


def verify_preview_protection(root, pins):
    for name, digest in pins.items():
        path = common.reject_links(root / name)
        if not path.is_file() or common.sha(path) != digest:
            raise ValueError("An original preview entry changed; preserve evidence: " + name)


def transport_readback(output):
    """Bind successful transport to the same final gate and remote market bytes."""
    gate = common.read(output / "completion_gate.json")
    protocol = common.read(output / "protocol.json")
    if (gate.get("passed") is not True or gate.get("failure_count") != 0 or gate.get("failures") != []
            or any(gate.get(k) != EXPECTED_ACCOUNTS for k in ("expected_accounts", "accounts_checked", "repeats_checked"))
            or any(gate.get(k) != EXPECTED_MODELS for k in ("expected_models", "models_checked"))):
        raise ValueError("Full local1590-model/180-account completion gate required")
    protocol_hash, gate_hash = common.sha(output / "protocol.json"), common.sha(output / "completion_gate.json")
    auditor = protocol.get("sources", {}).get("scripts/audit_rolling_single_pair.py")
    external = gate.get("external_inputs_sha256")
    if not auditor or not isinstance(external, dict) or not external:
        raise ValueError("Frozen auditor and exact external-market inputs required")
    matches = []
    transport = common.reject_links(output / "transport")
    for path in transport.glob("*.receipt.json"):
        common.reject_links(path)
        receipt = common.read(path)
        if (receipt.get("protocol_sha256") != protocol_hash or receipt.get("completion_gate_sha256") != gate_hash
                or receipt.get("verifier_source_sha256") != auditor
                or receipt.get("external_inputs_sha256") != external or receipt.get("external_verified_sha256") != external
                or receipt.get("external_rechecked_after_archive") is not True):
            continue
        name = Path(str(receipt.get("archive", ""))).name
        if not name or name in (".", "..") or "\\" in name or ":" in name:
            continue
        archive = common.reject_links(transport / name)
        if not archive.is_file() or common.sha(archive) != receipt.get("archive_sha256"):
            continue
        files = receipt.get("files")
        if (not isinstance(files, dict) or files.get("protocol.json") != protocol_hash or files.get("completion_gate.json") != gate_hash
                or any(files.get(n) != h for n, h in gate.get("inputs_sha256", {}).items())):
            continue
        matches.append({"path": str(path.relative_to(output)), "sha256": common.sha(path), "archive_sha256": receipt["archive_sha256"], "completion_gate_sha256": gate_hash})
    if not matches:
        raise ValueError("No successful transport receipt matches current gate, external market and archive")
    return gate, matches


def run(args):
    output, target, preview_root = validate_arguments(args)
    output.mkdir(parents=True, exist_ok=True)
    with delivery_lease(output):
        return deliver(args, output, target, preview_root)


def deliver(args, output, target, preview_root):
    attempt = common.reject_links(output / "delivery" / (common.datetime.now(common.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]))
    attempt.mkdir(parents=True, exist_ok=False)
    errors, missing_process_observations, numerical_ready = 0, 0, False
    try:
        pins = {name: common.sha(common.reject_links(ROOT / name)) for name in SOURCE_NAMES}
        preview_pins = protect_existing_preview(preview_root)
        common.write_new(attempt / "launch.json", {"at": common.now(), "pid": os.getpid(), "command": sys.argv,
            "source_sha256": pins, "remote_root": args.remote_root, "preview_root_files_sha256": preview_pins,
            "no_training_or_selection": True, "no_new_server": True, "visual_review_claimed": False, "no_production_change": True})
        common.status(output, task_status="running", delivery_status="running", delivery_worker_pid=os.getpid(),
            delivery_attempt=str(attempt.relative_to(output)), delivery_error=None, numerical_complete=False,
            numerical_delivery_complete=False, all_financial_results_ready=False, visual_review_pending=True)
        while True:
            common.verify_sources(pins)
            verify_preview_protection(preview_root, preview_pins)
            try:
                progress = remote_snapshot()
                errors = 0
            except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
                errors += 1
                with (attempt / "connection_errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": common.now(), "consecutive": errors, "error": str(exc)}) + "\n")
                if errors >= 10:
                    raise RuntimeError("Ten consecutive status failures; preserve all remote training progress") from exc
                time.sleep(args.poll_interval)
                continue
            with (attempt / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, ensure_ascii=False) + "\n")
            phase = progress["pipeline"]
            common.status(output, latest_remote_snapshot=progress, remote_phase=phase,
                models_completed=progress["models_completed"], failed_model_attempts=progress["failed_model_attempts"])
            print(json.dumps({"at": common.now(), "models_completed": progress["models_completed"], "phase": phase.get("phase", phase["status"]), "failed": progress["failed_model_attempts"]}), flush=True)
            if ready_to_collect(progress):
                break
            if phase["status"] == "running" and not progress["supervisor_alive"]:
                missing_process_observations += 1
                if missing_process_observations >= 3:
                    raise RuntimeError("Recorded pipeline exited without completion; preserve all evidence and do not restart")
            else:
                missing_process_observations = 0
            time.sleep(args.poll_interval)
        common.verify_sources(pins)
        common.status(output, task_status="collecting_audited_results")
        with (attempt / "collect.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/collect_rolling_single_pair.py"), "--host", args.host,
                "--remote-root", args.remote_root, "--remote-python", args.remote_python, "--output", str(output)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        common.verify_sources(pins)
        gate, receipts = transport_readback(output)
        common.write_new(attempt / "transport_readback.json", {"at": common.now(), "passed": True, "receipts": receipts, "accounts": EXPECTED_ACCOUNTS, "models": EXPECTED_MODELS})
        common.status(output, task_status="building_audited_report", all_financial_results_ready=True)
        with (attempt / "report_build.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/all_algorithms_rolling_report_app.py"), "--study-root", str(output),
                "--build-mode", "wsl", "--wsl-node", args.wsl_node], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        common.verify_sources(pins)
        build = common.read(output / "report/report_build.json")
        if common.sha(output / "report/report.html") != build["html_sha256"] or common.sha(output / "report/reviewed.json") != build["reviewed_sha256"]:
            raise ValueError("Official HTML or reviewed snapshot changed")
        snapshot = common.read(output / "report/reviewed.json")
        counts = {name: len(q["rows"]) for name, q in snapshot["queries"].items()}
        if set(counts) != set(QUERY_COUNTS) | {"historical_results"} or counts["historical_results"] <= 0 or any(counts.get(name) != n for name, n in QUERY_COUNTS.items()):
            raise ValueError("Complete rolling comparison populations required; missing outcomes cannot become zero")
        numerical_ready = True
        common.status(output, task_status="delivering_private_preview", numerical_complete=True)
        expected = common.verify_built_preview(output / "report/report_app/dist", build)
        actual = common.publish_preview(output / "report/report_app/dist", target)
        if any(actual.get(name) != digest for name, digest in expected.items()):
            raise ValueError("Private preview differs from the official build")
        verify_preview_protection(preview_root, preview_pins)
        http_verified, preview_error = False, None
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(args.preview_url, timeout=10) as response:
                if response.status != 200 or hashlib.sha256(response.read()).hexdigest() != expected["index.html"]:
                    raise ValueError("Existing preview service is not serving this exact report")
                http_verified = True
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            preview_error = type(error).__name__ + ": " + str(error)
        common.verify_sources(pins)
        verify_preview_protection(preview_root, preview_pins)
        result = {"at": common.now(), "numerical_complete": True, "numerical_delivery_complete": True, "visual_review_pending": True,
            "query_counts": counts, "report_build_sha256": common.sha(output / "report/report_build.json"), "completion_gate_sha256": common.sha(output / "completion_gate.json"),
            "transport_receipts": receipts, "preview_files_sha256": actual, "preview_http_verified": http_verified, "preview_error": preview_error,
            "original_preview_entries_unchanged": True, "no_training_or_selection": True, "no_new_server": True, "no_production_change": True}
        common.write_new(attempt / "completed.json", result)
        common.status(output, task_status="numerical_complete_visual_review_pending", numerical_complete=True, numerical_delivery_complete=True,
            all_financial_results_ready=True, visual_review_pending=True, delivery_status="completed" if http_verified else "numerical_complete_preview_unavailable",
            preview_http_verified=http_verified, preview_error=preview_error, report_url=args.preview_url, portable_html="report/report.html", delivery_completed_at=result["at"], no_production_change=True)
        print(json.dumps({k: v for k, v in result.items() if k != "preview_files_sha256"}), flush=True)
        return result
    except BaseException as error:
        common.write_new(attempt / "failed.json", {"at": common.now(), "type": type(error).__name__, "error": str(error), "all_evidence_preserved": True})
        common.status(output, task_status="delivery_failed", delivery_status="failed", delivery_error=str(error), numerical_complete=numerical_ready,
            numerical_delivery_complete=False, visual_review_pending=True, delivery_attempt=str(attempt.relative_to(output)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", default=ops.HOST)
    parser.add_argument("--remote-root", default=ops.REMOTE_ROOT)
    parser.add_argument("--remote-python", default=ops.REMOTE_PYTHON)
    parser.add_argument("--wsl-node", required=True)
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_ROOT / CHILD_NAME)
    parser.add_argument("--preview-url", default=PREVIEW_URL)
    parser.add_argument("--poll-interval", type=int, default=60)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
