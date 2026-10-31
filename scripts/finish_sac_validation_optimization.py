"""Collect and deliver the already-running validation-only SAC search.

No training, restart, source editing, selection, server creation or trading is
performed here. Numerical completion and actual browser review remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
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

SOURCE_NAMES = (
    "scripts/finish_sac_validation_optimization.py", "scripts/finish_single_pair_rl_six.py",
    "scripts/collect_sac_validation_optimization.py", "scripts/collect_single_pair_rl.py",
    "scripts/backtest_sac_validation_optimization.py", "scripts/sac_validation_optimization_models.py",
    "scripts/audit_sac_validation_optimization.py", "scripts/sac_validation_optimization_report_app.py",
    "scripts/sac_validation_optimization_report_app.jsx", "scripts/sac_validation_optimization_report_app.css",
    "scripts/btc_rl_seed_stability_report_app.py", "scripts/eth_rl_seed_stability_report_app.py",
    "scripts/eth_rl_seed_stability_report_app.jsx", "scripts/eth_rl_seed_stability_report_app.css",
    "scripts/single_pair_rl_six_report_app.py", "scripts/btceth_rl_report_app.py",
    "scripts/backtest_single_pair_rl.py", "scripts/single_pair_rl_env.py", "scripts/single_pair_rl_models.py",
    "scripts/audit_single_pair_rl.py", "scripts/backtest_btceth_rl.py", "scripts/btceth_rl_env.py",
    "scripts/crypto_rl_portfolio_env.py", "scripts/crypto_rl_reporting.py", "scripts/crypto_rl_data.py",
    "scripts/btceth_rl_td3.py", "scripts/btceth_rl_droq.py", "scripts/backtest_btc_rl_seed_stability.py",
    "scripts/audit_btc_rl_seed_stability.py", "scripts/backtest_eth_rl_seed_stability.py",
    "scripts/audit_eth_rl_seed_stability.py", "scripts/extend_single_pair_rl.py",
    "scripts/audit_sac_validation_optimization_v2.py", "scripts/continue_sac_validation_optimization.py",
    "scripts/audit_sac_validation_optimization_v3.py", "scripts/finish_sac_fallback_audit_v3.py",
)


def remote_snapshot(host, remote_python, remote_root):
    code = '''
import json,pathlib,datetime,hashlib
r=pathlib.Path(REMOTE_ROOT)
read=lambda p:json.loads(p.read_text())
value={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pipeline':read(r/'pipeline_status.json'),'stages':{}}
for stage in ['tuning/c00_baseline','tuning/c01_lr1e4','tuning/c02_gamma099','tuning/c03_lr1e4_gamma099','final']:
    ds=list((r/stage/'training').glob('*/sac/seed*'))
    value['stages'][stage]={'started':len(ds),'completed':sum((p/'record.json').exists() for p in ds),'failed':sum((p/'failed.json').exists() for p in ds)}
launch=read(r/'pipeline_launch.json')
if (r/'pipeline_launch_revision.json').exists():
    original=launch;launch=read(r/'pipeline_launch_revision.json')
    assert launch['original_pipeline_launch_sha256']==hashlib.sha256((r/'pipeline_launch.json').read_bytes()).hexdigest()
    assert launch['training_restarted'] is False
if (r/'pipeline_launch_import_revision.json').exists():
    launch=read(r/'pipeline_launch_import_revision.json')
    assert launch['parent_pipeline_launch_revision_sha256']==hashlib.sha256((r/'pipeline_launch_revision.json').read_bytes()).hexdigest()
    assert launch['training_restarted'] is False
value['supervisor_alive']=(pathlib.Path('/proc')/str(launch['pid'])).exists()
value['active_controller_pid']=launch['pid']
value['models_completed']=sum(x['completed'] for x in value['stages'].values())
value['failed_model_attempts']=sum(x['failed'] for x in value['stages'].values())
print(json.dumps(value))
'''.replace("REMOTE_ROOT", repr(str(remote_root)))
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
        shlex.quote(remote_python) + " -"], input=code, capture_output=True, text=True, encoding="utf-8", timeout=45, check=True)
    return json.loads(result.stdout)


def run(args):
    output = common.reject_links(args.output).resolve()
    btc = common.reject_links(args.baseline_btc).resolve()
    eth = common.reject_links(args.baseline_eth).resolve()
    retained = common.reject_links(args.retained_study).resolve()
    roots = (output, btc, eth, retained)
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError("Optimization and all original studies must remain disjoint")
    url = urllib.parse.urlsplit(args.preview_url)
    if (url.scheme != "http" or url.netloc != "127.0.0.1:8774" or url.fragment
            or url.path != "/" + Path(args.preview_dir).name + "/index.html"):
        raise ValueError("Use a new child of the existing loopback preview service")
    attempt = common.reject_links(output / "delivery" / common.datetime.now(common.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    attempt.mkdir(parents=True, exist_ok=False)
    pins = {name: common.sha(ROOT / name) for name in SOURCE_NAMES}
    common.write_new(attempt / "launch.json", {"at": common.now(), "pid": os.getpid(), "command": sys.argv,
        "source_sha256": pins, "remote_root": args.remote_root, "no_training_or_selection": True,
        "no_new_server": True, "visual_review_claimed": False, "no_production_change": True})
    errors = 0
    try:
        common.status(output, task_status="running", delivery_status="running", delivery_worker_pid=os.getpid(),
            delivery_attempt=str(attempt.relative_to(output)), delivery_error=None, numerical_delivery_complete=False,
            all_financial_results_ready=False, visual_review_pending=True)
        while True:
            common.verify_sources(pins)
            try:
                progress = remote_snapshot(args.host, args.remote_python, args.remote_root)
                errors = 0
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                errors += 1
                with (attempt / "connection_errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": common.now(), "consecutive": errors, "error": str(exc)}) + "\n")
                if errors >= 10:
                    raise RuntimeError("Ten consecutive status failures; preserve remote progress") from exc
                time.sleep(args.poll_interval)
                continue
            with (attempt / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress) + "\n")
            phase, failed = progress["pipeline"], progress["failed_model_attempts"]
            common.status(output, latest_remote_snapshot=progress, remote_phase=phase,
                models_completed=progress["models_completed"], failed_model_attempts=failed)
            print(json.dumps({"at": common.now(), "models_completed": progress["models_completed"],
                "phase": phase.get("phase", phase["status"]), "failed": failed}), flush=True)
            if failed or phase["status"] == "failed":
                raise RuntimeError("The research failed; no automatic restart or replacement")
            if phase["status"] == "completed" and not progress["supervisor_alive"]:
                break
            time.sleep(args.poll_interval)
        common.verify_sources(pins)
        common.status(output, task_status="collecting_audited_results")
        with (attempt / "collect.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/collect_sac_validation_optimization.py"),
                "--host", args.host, "--remote-root", args.remote_root, "--remote-python", args.remote_python,
                "--output", str(output)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        common.verify_sources(pins)
        common.status(output, task_status="building_audited_report", all_financial_results_ready=True)
        with (attempt / "report_build.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/sac_validation_optimization_report_app.py"),
                "--study-root", str(output), "--baseline-btc-root", str(btc), "--baseline-eth-root", str(eth),
                "--retained-study-root", str(retained), "--build-mode", "wsl", "--wsl-node", args.wsl_node],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        common.verify_sources(pins)
        build = common.read(output / "report/report_build.json")
        if common.sha(output / "report/report.html") != build["html_sha256"] or common.sha(output / "report/reviewed.json") != build["reviewed_sha256"]:
            raise ValueError("Official HTML or reviewed snapshot changed")
        expected = common.verify_built_preview(output / "report/report_app/dist", build)
        actual = common.publish_preview(output / "report/report_app/dist", args.preview_dir)
        if any(actual.get(name) != digest for name, digest in expected.items()):
            raise ValueError("Preview differs from the official build")
        http_verified, preview_error = False, None
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(args.preview_url, timeout=10) as response:
                if response.status != 200 or hashlib.sha256(response.read()).hexdigest() != expected["index.html"]:
                    raise ValueError("Preview service is not serving this exact report")
                http_verified = True
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            preview_error = type(exc).__name__ + ": " + str(exc)
        snapshot = common.read(output / "report/reviewed.json")
        query_counts = {name: len(query["rows"]) for name, query in snapshot["queries"].items()}
        if query_counts.get("account_summary") != 256:
            raise ValueError("The before/after annual comparison requires exactly 256 actual cost accounts")
        result = {"at": common.now(), "numerical_delivery_complete": True, "visual_review_pending": True,
            "query_counts": query_counts, "report_build_sha256": common.sha(output / "report/report_build.json"),
            "completion_contract_audit_sha256": common.sha(output / "completion_contract_audit.json"),
            "preview_files_sha256": actual, "preview_http_verified": http_verified, "preview_error": preview_error,
            "no_training_or_selection": True, "no_production_change": True}
        common.write_new(attempt / "completed.json", result)
        common.status(output, task_status="numerical_complete_visual_review_pending", numerical_delivery_complete=True,
            all_financial_results_ready=True, visual_review_pending=True,
            delivery_status="completed" if http_verified else "numerical_complete_preview_unavailable",
            preview_http_verified=http_verified, preview_error=preview_error, report_url=args.preview_url,
            portable_html="report/report.html", delivery_completed_at=result["at"], no_production_change=True)
        print(json.dumps({k: v for k, v in result.items() if k != "preview_files_sha256"}), flush=True)
    except BaseException as exc:
        common.write_new(attempt / "failed.json", {"at": common.now(), "type": type(exc).__name__, "error": str(exc), "all_evidence_preserved": True})
        common.status(output, task_status="delivery_failed", delivery_status="failed", delivery_error=str(exc), visual_review_pending=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-btc", type=Path, required=True)
    parser.add_argument("--baseline-eth", type=Path, required=True)
    parser.add_argument("--retained-study", type=Path, required=True)
    parser.add_argument("--host", default="v100-192-168-8-120")
    parser.add_argument("--remote-root", default="/home/sunny/research/sac_validation_optimization_20261031")
    parser.add_argument("--remote-python", default="/home/sunny/research-envs/btceth-rl-20261029/bin/python")
    parser.add_argument("--wsl-node", required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--preview-url", required=True)
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()
    if not 15 <= args.poll_interval <= 60:
        parser.error("Use a 15–60 second status interval")
    run(args)


if __name__ == "__main__":
    main()
