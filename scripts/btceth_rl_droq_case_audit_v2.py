"""CPU-only parallel calls to the unchanged pinned bf33 financial audit_case.

Receipts preserve source/data/case hashes and reconstructed tables so a later
metadata audit can adopt the actual work without redoing account simulation.
No policy, GPU, training, selection or trade execution is constructed here.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

FROZEN_AUDITOR_SHA = "bf33c879ff5eeaa13ee288c561fb5e6f18efb3fddc0da5179e6d646339b8c2bc"
_STATE = {}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_frozen(source_root):
    root = Path(source_root).resolve()
    path = root / "scripts/audit_btceth_rl_droq.py"
    if sha(path) != FROZEN_AUDITOR_SHA:
        raise ValueError("Unchanged bf33 financial auditor required")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    name = "_btceth_droq_bf33_financial"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    if Path(module.__file__).resolve() != path:
        raise ValueError("Foreign financial auditor import cache")
    return module


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        return _clean(value.item())
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_clean(value), stream, indent=2, allow_nan=False); stream.write("\n")


def initialize_worker(source_root, dataset_path, cutoff, records, original_gpus):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    bf = load_frozen(source_root)
    from scripts.crypto_rl_reporting import canonical_csv
    dataset = bf.load_market(dataset_path, cutoff)
    features = bf.prior.independent_market_features(dataset, cutoff=cutoff)
    _STATE.update(bf=bf, dataset=dataset, features=features, records=records,
        source_sha256={name: sha(Path(source_root) / name) for name in bf.SOURCE_FILES},
        dataset_sha256=sha(dataset_path), original_gpus=original_gpus, canonical_csv=canonical_csv)


def audit_case_worker(task):
    bf, dataset, features = (_STATE[k] for k in ("bf", "dataset", "features"))
    directory = Path(task["directory"])
    expected = task["case_sha256"]
    if set(expected) != set(bf.CASE_FILES):
        raise ValueError("All eight immutable account files are required")
    for name, digest in expected.items():
        if sha(directory / name) != digest:
            raise ValueError("Case changed before independent audit: " + name)
    checks = bf.Checks()
    run, history, monthly = bf.audit_case(directory, dataset, task["row"], _STATE["records"], checks,
        split=task["split"], family=task["family"], profile="balanced", features=features,
        gpus=_STATE["original_gpus"])
    for name, digest in expected.items():
        checks.check(sha(directory / name) == digest, "Unchanged independent worker input:" + name)
    output = Path(task["receipt_directory"])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "run.json", run)
    for name, frame in (("history.csv", history),):
        (output / name).write_bytes(_STATE["canonical_csv"](frame))
    if not isinstance(monthly, dict):
        raise TypeError("Original bf33 monthly reconstruction must be a dict")
    write_new(output / "monthly.json", monthly)
    receipt = dict(scope="original_bf33_cpu_financial_case_audit", run_id=task["row"]["run_id"],
        split=task["split"], family=task["family"], directory_relative=task["directory_relative"],
        audited_at=datetime.now(timezone.utc).isoformat(), worker_pid=os.getpid(),
        passed=checks.failed == 0, checks_total=checks.count, failure_count=checks.failed,
        failures=checks.failures, maximum_errors=checks.maximum_errors,
        case_sha256=expected, dataset_sha256=_STATE["dataset_sha256"],
        source_sha256=_STATE["source_sha256"], wrapper_source_sha256=sha(__file__),
        result_sha256={name: sha(output / name) for name in ("run.json", "history.csv", "monthly.json")})
    write_new(output / "receipt.json", receipt)
    print("DROQ_CASE_AUDIT", receipt["run_id"], receipt["checks_total"], receipt["failure_count"], flush=True)
    return dict(run_id=receipt["run_id"], receipt_directory=str(output),
        receipt_sha256=sha(output / "receipt.json"), passed=receipt["passed"])


def run_case_audits(output, main, dataset_path, source_root, *, phase="validation", workers=24,
                    receipt_root, family=None):
    """Standalone financial evidence only; not a full protocol/scheduling audit."""
    if type(workers) is not int or not 1 <= workers <= 30:
        raise ValueError("Use1..30 independent CPU account audit workers")
    if phase not in ("validation", "test") or phase == "test" and family not in ("raw", "cap075"):
        raise ValueError("A fixed validation or selected-family annual audit is required")
    bf = load_frozen(source_root)
    output, receipt_root = Path(output), Path(receipt_root)
    receipt_root.mkdir(parents=True, exist_ok=False)
    protocol = bf.read_json(output / "protocol.json")
    records = {("balanced", seed): bf.read_json(output / f"training/droq__balanced__seed{seed}/record.json")
               for seed in range(5)}
    plans, tasks = [], []
    for current in (bf.FAMILIES if phase == "validation" else (family,)):
        frame = bf.read_csv(output / phase / current / "balanced/summary.csv")
        bf.validate_group_rows(frame, split=phase, family=current, profile="balanced")
        for row in frame.to_dict("records"):
            key = "ensemble" if row["primary_policy"] else str(int(row["seed"]))
            relative = f"{phase}/{current}/balanced/{row['scenario']}/{key}"
            directory = bf.safe_file(output, relative)
            tasks.append(dict(row=row, family=current, split=phase, directory=str(directory),
                directory_relative=relative, case_sha256={name: sha(directory / name) for name in bf.CASE_FILES},
                receipt_directory=str(receipt_root / row["run_id"])))
            plans.append(row["run_id"])
    expected = 24 if phase == "validation" else 12
    if len(tasks) != len(set(plans)) or len(tasks) != expected:
        raise ValueError("All original costs and five seeds plus ensemble are mandatory")
    cutoff = bf.VALIDATION[1] if phase == "validation" else bf.TEST[1]
    records_out = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=initialize_worker, initargs=(str(source_root), str(dataset_path), cutoff, records,
                                                   protocol["physical_gpus"])) as pool:
        futures = [pool.submit(audit_case_worker, task) for task in tasks]
        for future in as_completed(futures):
            records_out.append(future.result())
    by_id = {v["run_id"]: v for v in records_out}
    manifest = dict(scope="parallel_bf33_financial_cases_only_not_full_study_audit",
        phase=phase, case_count=len(records_out), passed=all(v["passed"] for v in records_out),
        workers=workers, source_sha256=sha(__file__), frozen_auditor_sha256=FROZEN_AUDITOR_SHA,
        dataset_sha256=sha(dataset_path), protocol_sha256=sha(output / "protocol.json"),
        at=datetime.now(timezone.utc).isoformat(), records=[by_id[key] for key in plans])
    write_new(receipt_root / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "reference-main", "dataset", "source-root", "receipt-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--phase", choices=("validation", "test"), default="validation")
    p.add_argument("--family", choices=("raw", "cap075"))
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    value = run_case_audits(a.output_dir, a.reference_main, a.dataset, a.source_root,
        phase=a.phase, workers=a.workers, receipt_root=a.receipt_root, family=a.family)
    print(json.dumps({k: v for k, v in value.items() if k != "records"}))
    raise SystemExit(0 if value["passed"] else 1)
