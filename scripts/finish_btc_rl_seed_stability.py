"""Continuous, one-run delivery after the authorized BTC seed study finishes.

No training, retraining, scheduler, new server, external message or production
change. The original studies and every failed/completed record are preserved.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import finish_single_pair_rl_six as common

SOURCE_NAMES = ('scripts/finish_btc_rl_seed_stability.py', 'scripts/finish_single_pair_rl_six.py', 'scripts/collect_btc_rl_seed_stability.py', 'scripts/collect_single_pair_rl.py', 'scripts/backtest_btc_rl_seed_stability.py', 'scripts/audit_btc_rl_seed_stability.py', 'scripts/btc_rl_seed_stability_report_app.py', 'scripts/eth_rl_seed_stability_report_app.py', 'scripts/eth_rl_seed_stability_report_app.jsx', 'scripts/eth_rl_seed_stability_report_app.css', 'scripts/btceth_seed_stability_report_app.py', 'scripts/btceth_seed_stability_report_app.jsx', 'scripts/btceth_seed_stability_report_app.css', 'scripts/single_pair_rl_six_report_app.py', 'scripts/btceth_rl_report_app.py')


def retained_readback(output, retained):
    pins, case_count = {}, 0
    for split in ("validation", "test"):
        for kind in ("evaluation", "repeat"):
            selected = []
            for path in (retained / kind / split / "BTCUSDT").rglob("summary.json"):
                row = common.read(path)
                if row["algorithm"] in ("sac", "td3", "cash", "buy_hold", "sma200_trend", "momentum180"):
                    selected.append(path.parent)
            if len(selected) != 28:
                raise ValueError("Require exactly twenty retained learned and eight baseline cost accounts per phase/repeat")
            for directory in selected:
                case_count += 1
                for filename in common.RESULT_FILES:
                    name = (directory.relative_to(retained) / filename).as_posix()
                    expected = common.sha(common.reject_links(retained / name))
                    if common.sha(common.reject_links(output / name)) != expected:
                        raise ValueError("Original BTC policy/baseline replay changed: " + name)
                    pins[name] = expected
    return {"passed": True, "cases_checked": case_count, "files_checked": len(pins), "files_sha256": pins,
            "scope": "original BTC SAC/TD3 seeds0..4 and four baselines; both phases and repetitions"}


def run(args):
    output = common.reject_links(args.output).resolve()
    retained = common.reject_links(args.retained_study).resolve()
    eth_study = common.reject_links(args.eth_study).resolve()
    if output.is_relative_to(eth_study) or eth_study.is_relative_to(output) or retained.is_relative_to(eth_study) or eth_study.is_relative_to(retained):
        raise ValueError("BTC, ETH and original studies must be disjoint")
    if output.is_relative_to(retained) or retained.is_relative_to(output):
        raise ValueError("Studies must be disjoint")
    url = urllib.parse.urlsplit(args.preview_url)
    if (url.scheme != "http" or url.netloc != "127.0.0.1:8774" or url.fragment
            or url.path != "/" + Path(args.preview_dir).name + "/index.html"):
        raise ValueError("Use the new child directory on the existing loopback service")
    delivery = common.reject_links(output / "delivery")
    attempt = common.reject_links(delivery / common.datetime.now(common.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    attempt.mkdir(parents=True, exist_ok=False)
    pins = {name: common.sha(ROOT / name) for name in SOURCE_NAMES}
    common.write_new(attempt / "launch.json", {"at": common.now(), "pid": os.getpid(), "source_sha256": pins,
        "command": sys.argv, "remote_root": args.remote_root, "no_production_change": True,
        "no_new_server": True, "visual_review_claimed": False})
    errors = 0
    try:
        common.status(output, task_status="running", delivery_status="running", delivery_worker_pid=os.getpid(),
            delivery_attempt=str(attempt.relative_to(output)), delivery_error=None, numerical_delivery_complete=False,
            all_financial_results_ready=False, visual_review_pending=True)
        while True:
            common.verify_sources(pins)
            try:
                progress = common.remote_snapshot(args.host, args.remote_python, args.remote_root)
                errors = 0
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                errors += 1
                with (attempt / "connection_errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": common.now(), "consecutive": errors, "error": str(exc)}) + "\n")
                if errors >= 10:
                    raise RuntimeError("Ten consecutive status connection failures; keep all remote progress") from exc
                time.sleep(args.poll_interval)
                continue
            with (attempt / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, ensure_ascii=False) + "\n")
            phase = progress["pipeline"]
            completed = sum(row["completed"] for row in progress["algorithms"].values())
            failed = sum(row["failed"] for row in progress["algorithms"].values())
            common.status(output, task_status="running", models_completed=completed, failed_model_attempts=failed,
                          remote_phase=phase, latest_remote_snapshot=progress)
            print(json.dumps({"at": common.now(), "models_completed": completed, "phase": phase.get("phase", phase["status"]), "failed": failed}), flush=True)
            if phase["status"] == "failed" or failed:
                raise RuntimeError("Remote research failed; no automatic restart or overwritten evidence")
            if phase["status"] == "completed" and not progress["supervisor_alive"]:
                break
            time.sleep(args.poll_interval)
        common.verify_sources(pins)
        common.status(output, task_status="collecting_audited_results")
        with (attempt / "collect.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/collect_btc_rl_seed_stability.py"), "--host", args.host,
                            "--remote-root", args.remote_root, "--remote-python", args.remote_python, "--output", str(output)],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        comparison = retained_readback(output, retained)
        common.write_new(attempt / "retained_replay_readback.json", comparison)
        common.verify_sources(pins)
        common.status(output, task_status="building_audited_report", all_financial_results_ready=True)
        with (attempt / "report_build.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/btceth_seed_stability_report_app.py"),
                "--study-root", str(output), "--eth-study-root", str(eth_study), "--retained-study-root", str(retained), "--build-mode", "wsl", "--wsl-node", args.wsl_node],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        common.verify_sources(pins)
        build = common.read(output / "report/report_build.json")
        if build["html_sha256"] != common.sha(output / "report/report.html") or build["reviewed_sha256"] != common.sha(output / "report/reviewed.json"):
            raise ValueError("Audited HTML/reviewed bytes changed")
        expected = common.verify_built_preview(output / "report/report_app/dist", build)
        actual = common.publish_preview(output / "report/report_app/dist", args.preview_dir)
        if any(actual.get(name) != digest for name, digest in expected.items()):
            raise ValueError("Published preview differs from official build receipt")
        preview_http_verified, preview_error = False, None
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(args.preview_url, timeout=10) as response:
                if response.status != 200 or hashlib.sha256(response.read()).hexdigest() != expected["index.html"]:
                    raise ValueError("Existing service is not serving this exact report")
                preview_http_verified = True
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            preview_error = type(exc).__name__ + ": " + str(exc)
        reviewed = common.read(output / "report/reviewed.json")
        result = {"at": common.now(), "numerical_delivery_complete": True, "visual_review_pending": True,
            "original_cases_unchanged": comparison["cases_checked"], "original_files_unchanged": comparison["files_checked"],
            "preview_http_verified": preview_http_verified, "preview_error": preview_error,
            "eth_study_root": str(eth_study), "total_compared_models": 120,
            "query_counts": {name: len(value["rows"]) for name, value in reviewed["queries"].items()},
            "report_build_sha256": common.sha(output / "report/report_build.json"), "preview_files_sha256": actual,
            "no_production_change": True}
        common.write_new(attempt / "completed.json", result)
        common.status(output, task_status="numerical_complete_visual_review_pending", models_completed=60,
            all_financial_results_ready=True, numerical_delivery_complete=True, visual_review_pending=True,
            delivery_status="completed" if preview_http_verified else "numerical_complete_preview_unavailable", preview_http_verified=preview_http_verified, preview_error=preview_error, delivery_completed_at=result["at"], report_url=args.preview_url,
            portable_html="report/report.html", no_production_change=True)
        print(json.dumps({k: v for k, v in result.items() if k != "preview_files_sha256"}), flush=True)
    except BaseException as exc:
        common.write_new(attempt / "failed.json", {"at": common.now(), "error": str(exc), "type": type(exc).__name__, "all_evidence_preserved": True})
        common.status(output, task_status="delivery_failed", delivery_status="failed", numerical_delivery_complete=False,
                      delivery_error=str(exc), delivery_evidence=str(attempt.relative_to(output)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-study", type=Path, required=True)
    parser.add_argument("--eth-study", type=Path, required=True)
    parser.add_argument("--host", default="v100-192-168-8-120")
    parser.add_argument("--remote-root", default="/home/sunny/research/btc_rl_seed_stability_20261031")
    parser.add_argument("--remote-python", default="/home/sunny/research-envs/btceth-rl-20261029/bin/python")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--wsl-node", required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--preview-url", required=True)
    arguments = parser.parse_args()
    if not 15 <= arguments.poll_interval <= 60:
        parser.error("Status interval must be 15..60 seconds")
    run(arguments)
