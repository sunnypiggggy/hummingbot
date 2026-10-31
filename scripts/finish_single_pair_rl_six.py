"""One continuous, audit-gated delivery job for the authorized six-RL run.

This process waits for the already running research pipeline. It does not train,
restart, change a protocol, schedule later jobs, start servers, or trade. Numerical
completion and pending visual review are reported separately.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ("scripts/finish_single_pair_rl_six.py", "scripts/collect_single_pair_rl.py",
                "scripts/single_pair_rl_six_report_app.py", "scripts/single_pair_rl_six_report_app.jsx",
                "scripts/single_pair_rl_six_report_app.css", "scripts/btceth_rl_report_app.py")
RESULT_FILES = ("equity_1h.csv.gz", "fills.csv", "weights.csv", "boundary_equity.csv", "metadata.json",
                "fingerprints.json", "actions.json", "summary.json", "daily_equity.csv", "monthly_returns.csv")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def status(output, **updates):
    path = output / "work_status.json"
    value = read(path) if path.exists() else {}
    value.update(captured_at_utc=now(), **updates)
    temporary = path.with_suffix(".delivery-tmp.json")
    write_new(temporary, value)
    os.replace(temporary, path)


def reject_links(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or hasattr(p, "is_junction") and p.is_junction() for p in (path, *path.parents)):
        raise ValueError("Delivery paths cannot cross a link")
    return path


def verify_sources(pins):
    for name, expected in pins.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Delivery source changed after launch: " + name)


def remote_snapshot(host, remote_python, remote_root):
    source = """
import json,pathlib,datetime
r=pathlib.Path(REMOTE_ROOT)
value={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pipeline':json.loads((r/'pipeline_status.json').read_text()), 'algorithms':{}}
for algorithm in ['sac','tqc','crossq','recurrent_ppo','td3','droq']:
    ds=list((r/'training').glob('*/'+algorithm+'/seed*'))
    value['algorithms'][algorithm]={'started':len(ds),'completed':sum((p/'record.json').exists() for p in ds),'failed':sum((p/'failed.json').exists() for p in ds)}
launch=json.loads((r/'pipeline_launch.json').read_text())
value['supervisor_alive']=(pathlib.Path('/proc')/str(launch['pid'])).exists()
print(json.dumps(value))
""".replace("REMOTE_ROOT", repr(remote_root))
    completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                                shlex.quote(remote_python) + " -"], input=source,
                               capture_output=True, text=True, encoding="utf-8", timeout=45, check=True)
    return json.loads(completed.stdout)


def retained_replay_readback(output, retained):
    """Require old policies and fixed baselines to reproduce all serialized files."""
    pins = {}
    accounts = 0
    for split in ("validation", "test"):
        for kind in ("evaluation", "repeat"):
            old_root = retained / kind / split
            cases = sorted(p.parent for p in old_root.rglob("summary.json"))
            if len(cases) != 56:
                raise ValueError("The retained two-algorithm study must contain exactly 56 cases per phase/repetition")
            for directory in cases:
                row = read(directory / "summary.json")
                if row["algorithm"] not in ("sac", "tqc", "cash", "buy_hold", "sma200_trend", "momentum180"):
                    raise ValueError("Unexpected retained algorithm")
                accounts += 1
                for filename in RESULT_FILES:
                    name = (directory.relative_to(retained) / filename).as_posix()
                    expected = sha(reject_links(retained / name))
                    if sha(reject_links(output / name)) != expected:
                        raise ValueError("Unchanged policy/baseline failed byte-identical replay: " + name)
                    pins[name] = expected
    return {"passed": True, "cases_checked": accounts, "files_checked": len(pins), "files_sha256": pins,
            "scope": "56 original accounts per phase and repetition; all serialized financial/action/metadata files identical"}


def summary_readback(output):
    with (output / "test_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(s, a, str(seed), c) for s in ("BTCUSDT", "ETHUSDT") for a in ("sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq")
                for seed in range(5) for c in ("base", "stress")}
    expected.update((s, a, "", c) for s in ("BTCUSDT", "ETHUSDT") for a in ("cash", "buy_hold", "sma200_trend", "momentum180") for c in ("base", "stress"))
    if len(rows) != 136 or {(r["symbol"], r["algorithm"], r["seed"], r["scenario"]) for r in rows} != expected:
        raise ValueError("Annual summary lacks the full declared account population")
    highlights = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        for algorithm in ("sac", "tqc", "crossq", "recurrent_ppo", "td3", "droq"):
            primary = {r["scenario"]: r for r in rows if r["symbol"] == symbol and r["algorithm"] == algorithm and r["seed"] == "0"}
            if len(primary) != 2:
                raise ValueError("Exactly two cost accounts per seed0 are required")
            value = {"symbol": symbol, "algorithm": algorithm, "seed": 0, "initial_capital": 10000}
            for scenario, row in primary.items():
                if float(row["initial_capital"]) != 10000 or row["ensemble"] != "False" or row["independent_capital"] != "True":
                    raise ValueError("Independent 10000 USDT identity changed")
                for key in ("net_return", "max_drawdown", "fees", "slippage_cost", "net_liquidation_return"):
                    value[scenario + "_" + key] = float(row[key])
            group = [r for r in rows if r["symbol"] == symbol and r["algorithm"] == algorithm and r["scenario"] == "base"]
            value["positive_base_seeds"] = sum(float(r["net_return"]) > 0 for r in group)
            value["seed_denominator"] = 5
            highlights.append(value)
    return highlights


def publish_preview(source, destination):
    source = reject_links(source).resolve()
    destination = reject_links(destination).resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Preview source and destination must be disjoint directories")
    files = {}
    for path in source.rglob("*"):
        reject_links(path)
        if path.is_file():
            files[path.relative_to(source).as_posix()] = sha(path)
    if not files or "index.html" not in files:
        raise ValueError("A complete preview is required")
    if destination.exists():
        actual = {}
        for path in destination.rglob("*"):
            reject_links(path)
            if path.is_file():
                actual[path.relative_to(destination).as_posix()] = sha(path)
        if actual != files:
            raise FileExistsError("Preserve a different existing preview")
    else:
        shutil.copytree(source, destination)
    if any(sha(destination / name) != expected for name, expected in files.items()):
        raise ValueError("Preview transfer differs from the audited build")
    return files


def verify_built_preview(directory, build):
    directory = reject_links(directory)
    manifest = read(directory / "data-app-build.json")
    runtime = build["runtime"]
    if (manifest["version"] != 1 or manifest["kind"] != "separate-data-v1"
            or manifest["html"]["path"] != "index.html"
            or manifest["html"]["sha256"] != runtime["htmlSha256"]
            or manifest["snapshot"]["sha256"] != runtime["snapshotSha256"]
            or manifest["sourceSnapshotSha256"] != runtime["snapshotSha256"]
            or manifest["runtimeSha256"] != runtime["runtimeSha256"]
            or manifest["snapshot"]["path"] != "snapshot." + runtime["snapshotSha256"] + ".json"):
        raise ValueError("Preview manifest is not bound to the audited official build")
    expected = {}
    for key in ("html", "snapshot"):
        entry = manifest[key]
        path = reject_links(directory / entry["path"])
        if sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Preview bytes differ from the audited official build")
        expected[entry["path"]] = entry["sha256"]
    expected["data-app-build.json"] = sha(directory / "data-app-build.json")
    return expected


def run(args):
    output, retained = reject_links(args.output).resolve(), reject_links(args.retained_study).resolve()
    if output.is_relative_to(retained) or retained.is_relative_to(output):
        raise ValueError("New and retained studies must be disjoint directories")
    preview_url = urllib.parse.urlsplit(args.preview_url)
    if (preview_url.scheme != "http" or preview_url.netloc != "127.0.0.1:8774" or preview_url.fragment
            or preview_url.path != "/" + Path(args.preview_dir).name + "/index.html"):
        raise ValueError("Use the new child directory on the already verified loopback preview service")
    delivery_root = reject_links(output / "delivery")
    attempt = reject_links(delivery_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    attempt.mkdir(parents=True, exist_ok=False)
    pins = {name: sha(ROOT / name) for name in SOURCE_NAMES}
    write_new(attempt / "launch.json", {"at": now(), "pid": os.getpid(), "source_sha256": pins,
              "command": sys.argv, "remote_root": args.remote_root, "no_production_change": True,
              "no_new_server": True, "visual_review_claimed": False})
    errors = 0
    try:
        status(output, delivery_worker_pid=os.getpid(), delivery_attempt=str(attempt.relative_to(output)), task_status="training",
               delivery_status="running", delivery_error=None, numerical_delivery_complete=False,
               all_six_financial_results_ready=False, visual_review_pending=True)
        while True:
            verify_sources(pins)
            try:
                progress = remote_snapshot(args.host, args.remote_python, args.remote_root)
                errors = 0
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                errors += 1
                if errors >= 10:
                    raise RuntimeError("Ten consecutive remote status failures; preserve all training progress") from exc
                with (attempt / "connection_errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": now(), "consecutive": errors, "error": str(exc)}) + "\n")
                time.sleep(args.poll_interval)
                continue
            with (attempt / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress, ensure_ascii=False) + "\n")
            phase = progress["pipeline"]
            completed = sum(r["completed"] for r in progress["algorithms"].values())
            failed = sum(r["failed"] for r in progress["algorithms"].values())
            status(output, task_status="running", models_completed=completed, failed_model_attempts=failed,
                   remote_phase=phase, latest_remote_snapshot=progress, all_six_financial_results_ready=False)
            print(json.dumps({"at": now(), "models_completed": completed, "phase": phase.get("phase", phase["status"]), "failed": failed}), flush=True)
            if phase["status"] == "failed" or failed:
                raise RuntimeError("Remote pipeline/model failure; no automatic restart or evidence overwrite")
            if phase["status"] == "completed" and not progress["supervisor_alive"]:
                break
            time.sleep(args.poll_interval)
        verify_sources(pins)
        status(output, task_status="collecting_audited_results")
        with (attempt / "collect.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/collect_single_pair_rl.py"), "--host", args.host,
                            "--remote-root", args.remote_root, "--remote-python", args.remote_python, "--output", str(output)],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        old_comparison = retained_replay_readback(output, retained)
        write_new(attempt / "retained_replay_readback.json", old_comparison)
        highlights = summary_readback(output)
        verify_sources(pins)
        status(output, task_status="building_audited_report", all_six_financial_results_ready=True)
        with (attempt / "report_build.log").open("x", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/single_pair_rl_six_report_app.py"), "--study-root", str(output),
                            "--retained-study-root", str(retained), "--build-mode", "wsl", "--wsl-node", args.wsl_node],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        verify_sources(pins)
        build = read(output / "report/report_build.json")
        if build["html_sha256"] != sha(output / "report/report.html") or build["reviewed_sha256"] != sha(output / "report/reviewed.json"):
            raise ValueError("Built report hashes changed")
        expected_preview = verify_built_preview(output / "report/report_app/dist", build)
        preview_sha = publish_preview(output / "report/report_app/dist", args.preview_dir)
        if any(preview_sha.get(name) != expected for name, expected in expected_preview.items()):
            raise ValueError("Published preview is not the audited official build")
        with urllib.request.urlopen(args.preview_url, timeout=10) as response:
            if response.status != 200 or hashlib.sha256(response.read()).hexdigest() != expected_preview["index.html"]:
                raise ValueError("Existing preview service does not serve the exact new report")
        result = {"at": now(), "numerical_delivery_complete": True, "visual_review_pending": True,
                  "highlights": highlights, "unchanged_retained_cases": old_comparison["cases_checked"],
                  "unchanged_retained_files": old_comparison["files_checked"], "preview_files_sha256": preview_sha,
                  "report_build_sha256": sha(output / "report/report_build.json"), "no_production_change": True}
        write_new(attempt / "completed.json", result)
        status(output, task_status="numerical_complete_visual_review_pending", models_completed=60,
               all_six_financial_results_ready=True, numerical_delivery_complete=True, visual_review_pending=True,
               delivery_status="completed",
               report_url=args.preview_url, portable_html="report/report.html", delivery_completed_at=result["at"],
               no_production_change=True)
        print(json.dumps({k: v for k, v in result.items() if k != "preview_files_sha256"}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        write_new(attempt / "failed.json", {"at": now(), "error": str(exc), "type": type(exc).__name__, "all_evidence_preserved": True})
        status(output, task_status="delivery_failed", delivery_status="failed", numerical_delivery_complete=False, delivery_error=str(exc),
               delivery_evidence=str(attempt.relative_to(output)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-study", type=Path, required=True)
    parser.add_argument("--host", default="v100-192-168-8-120")
    parser.add_argument("--remote-root", default="/home/sunny/research/single_pair_rl_six_20261030")
    parser.add_argument("--remote-python", default="/home/sunny/research-envs/btceth-rl-20261029/bin/python")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--wsl-node", required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--preview-url", required=True)
    args = parser.parse_args()
    if not 15 <= args.poll_interval <= 60:
        parser.error("Use a bounded 15..60 second status interval")
    run(args)


if __name__ == "__main__":
    main()
