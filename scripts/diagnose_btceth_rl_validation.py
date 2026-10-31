"""Early CPU-only validation ensembles, separate from formal GPU evaluation.

No training/test PNL, seed selection, checkpoint updates or GPU control. All
price matrices are truncated before test starts before any feature calculation.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import shlex
import shutil
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import backtest_btceth_rl as formal
from scripts.crypto_rl_reporting import result_fingerprints, save_result, summarize, write_json

ALGORITHMS = ("recurrent_ppo", "tqc")
PROFILES = ("return", "balanced")
REFERENCES = ("btc25_cash75_hold", "vol20_inverse_vol30")
REFERENCE_FILES = ("summary.csv", "protocol.json", "completion.json", "input_sha_before.json", "independent_audit_source_snapshot.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_setup():
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU diagnostic must not inherit an initialized CUDA context")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(1730)
    np.random.seed(1730)
    torch.manual_seed(1730)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("CUDA must remain masked and uninitialized")
    return {"backend": "cpu", "CUDA_VISIBLE_DEVICES": "", "torch_cuda_initialized": False,
            "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
            "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
              ("torch", "numpy", "pandas", "stable-baselines3", "sb3-contrib", "gymnasium")}}


def load_references(directory, spec):
    directory = Path(directory)
    files = {name: directory / name for name in REFERENCE_FILES}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError("Previously audited validation reference required: " + str(path))
    protocol = json.loads(files["protocol.json"].read_text(encoding="utf-8"))
    completed = json.loads(files["completion.json"].read_text(encoding="utf-8"))
    if (protocol["validation"] != list(formal.VALIDATION) or protocol["capital"] != 10000
            or protocol["costs"] != formal.COSTS or completed.get("passed") is not True
            or completed.get("failures") != 0 or completed.get("test_evaluations") != 0):
        raise ValueError("Reference scope, capital, costs or independent audit differs")
    frozen = json.loads(files["input_sha_before.json"].read_text(encoding="utf-8"))
    for relative, expected in spec["code_sha256"].items():
        matches = [value for name, value in frozen.items() if name.replace("\\", "/").endswith("/" + relative)]
        if matches != [expected]:
            raise ValueError("Reference does not use the same frozen financial source: " + relative)
    frame = pd.read_csv(files["summary.csv"], float_precision="round_trip")
    chosen = frame.loc[frame.strategy.isin(REFERENCES)]
    if len(chosen) != 4 or set(zip(chosen.strategy, chosen.scenario)) != {(a, c) for a in REFERENCES for c in formal.COSTS}:
        raise ValueError("Need both validation costs for both risk-budget references")
    if not (chosen.split.eq("validation") & chosen.initial_capital.eq(10000) & chosen.observed_hours.eq(295 * 24)).all():
        raise ValueError("Reference is not complete295-day validation")
    return frame, files


def availability(formal_output, spec):
    records, status = {}, []
    for algorithm in ALGORITHMS:
        missing, found = [], []
        for profile in PROFILES:
            for seed in formal.SEEDS:
                path = formal_output / "training" / formal.job_id(algorithm, profile, seed) / "record.json"
                if not path.is_file():
                    missing.append(f"{profile}/seed{seed}")
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = formal_output / record["model_relative_path"]
                if (record["algorithm"], record["profile"], record["seed"]) != (algorithm, profile, seed):
                    raise ValueError("Completed training identity differs from its directory")
                if record["frozen_spec_sha256"] != sha(formal_output / "frozen_spec.json") or not checkpoint.is_file() or sha(checkpoint) != record["model_sha256"]:
                    raise ValueError("Completed model/record differs from frozen evidence")
                found.append((profile, seed, path, checkpoint, record))
        complete = len(found) == 10 and not missing
        status.append({"algorithm": algorithm, "complete": complete, "completed_records": len(found), "missing": missing,
                       "decision": "evaluate both five-model profiles" if complete else "skip incomplete algorithm; no invented result"})
        if complete:
            records[algorithm] = found
    return records, status


def diagnostic(formal_output, dataset_path, output, reference_dir):
    import torch
    formal_output, dataset_path, output, reference_dir = map(lambda value: Path(value).resolve(), (formal_output, dataset_path, output, reference_dir))
    if output.exists():
        raise FileExistsError("Preserve the existing early diagnostic directory")
    if output == formal_output or output.is_relative_to(formal_output):
        raise ValueError("Diagnostic output must be outside the formal experiment")
    spec = formal.check_frozen(formal_output, dataset_path)
    if formal.VALIDATION != ("2025-01-01T00:00:00Z", "2025-10-23T00:00:00Z"):
        raise ValueError("Only the original295-day validation window is authorized")
    runtime = cpu_setup()
    if any(runtime["packages"].get(name) != value for name, value in spec.get("runtime_packages", {}).items()):
        raise ValueError("Diagnostic must reuse the formal experiment's pinned package versions")
    records, status = availability(formal_output, spec)
    references, reference_files = load_references(reference_dir, spec)
    protected = {formal_output / "frozen_spec.json": sha(formal_output / "frozen_spec.json"),
                 dataset_path: sha(dataset_path), dataset_path.parent / "manifest.json": sha(dataset_path.parent / "manifest.json")}
    protected.update({ROOT / name: value for name, value in spec["code_sha256"].items()})
    for rows in records.values():
        for _, _, record_path, checkpoint, _ in rows:
            protected[record_path], protected[checkpoint] = sha(record_path), sha(checkpoint)
    protected.update({path: sha(path) for path in reference_files.values()})
    output.mkdir(parents=True)
    snapshots = output / "sources"
    snapshots.mkdir()
    shutil.copyfile(formal_output / "frozen_spec.json", snapshots / "original_frozen_spec.json")
    for path in (Path(__file__).resolve(), *[ROOT / name for name in spec["code_sha256"]]):
        shutil.copyfile(path, snapshots / path.name)
    for name, path in reference_files.items():
        shutil.copyfile(path, snapshots / ("reference_" + name))
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--formal-output", str(formal_output),
               "--dataset", str(dataset_path), "--output-dir", str(output), "--reference-dir", str(reference_dir)]
    (output / "exact_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    write_json(output / "started.json", {"at": now(), "pid": os.getpid(), "command": command, "runtime": runtime,
                                       "availability": status, "source_sha256": sha(Path(__file__).resolve()),
                                       "scope": "CPU-only early validation ensembles; not formal GPU validation, ranking or seed selection",
                                       "validation": list(formal.VALIDATION), "test_evaluations": 0})
    write_json(output / "input_sha_before.json", {str(path): value for path, value in protected.items()})
    started = time.monotonic()
    source_data = formal.load_dataset(dataset_path)
    data = {name: value for name, value in source_data.items() if not name.startswith("hourly_")}
    for field in ("open", "high", "low", "close", "volume"):
        frame = source_data["hourly_" + field]
        data["hourly_" + field] = frame.loc[frame.index < pd.Timestamp(formal.VALIDATION[1])].copy()
    del source_data
    if any(frame.index.max() >= pd.Timestamp(formal.VALIDATION[1]) for name, frame in data.items() if name.startswith("hourly_")):
        raise AssertionError("No test-year price may reach feature/evaluation calculation")
    audit_path = snapshots / "reference_independent_audit_source_snapshot.py"
    module_spec = importlib.util.spec_from_file_location("early_validation_independent_audit", audit_path)
    audit = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = audit
    module_spec.loader.exec_module(audit)
    rows, histories, months, proofs = [], [], [], []
    try:
        for algorithm, found in records.items():
            for profile in PROFILES:
                models, model_records = [], []
                for candidate_profile, seed, record_path, checkpoint, record in found:
                    if candidate_profile != profile:
                        continue
                    model = formal.model_class(algorithm).load(checkpoint, device="cpu")
                    if any(parameter.device.type != "cpu" for parameter in model.policy.parameters()):
                        raise AssertionError("CPU diagnostic loaded a GPU policy")
                    if formal.parameter_digest(model.policy) != record["parameter_sha256"]:
                        raise ValueError("CPU-loaded policy differs from completed training parameters")
                    models.append(model)
                    model_records.append({"seed": seed, "record_path": str(record_path), "record_sha256": sha(record_path),
                                          "model_sha256": record["model_sha256"], "parameter_sha256": record["parameter_sha256"]})
                if len(models) != 5:
                    raise AssertionError("Never evaluate a partial seed ensemble")
                for scenario, costs in formal.COSTS.items():
                    run_id = f"early_validation__{scenario}__{algorithm}__{profile}__ensemble"
                    before = [formal.parameter_digest(model.policy) for model in models]
                    result = formal.evaluate(data, models, algorithm, *formal.VALIDATION, costs, profile)
                    if torch.cuda.is_initialized():
                        raise AssertionError("CPU diagnostic initialized CUDA")
                    fingerprints = save_result(output / "cases" / algorithm / profile / scenario, result)
                    repeat = formal.evaluate(data, models, algorithm, *formal.VALIDATION, costs, profile)
                    if fingerprints != result_fingerprints(repeat):
                        raise AssertionError("CPU diagnostic financial replay is not exact: " + run_id)
                    after = [formal.parameter_digest(model.policy) for model in models]
                    if before != after or before != [r["parameter_sha256"] for r in model_records]:
                        raise AssertionError("Diagnostic prediction changed a trained policy")
                    checked = audit.reconstruct_fills(data, result["fills"], result["equity"], result["metadata"],
                              start=formal.VALIDATION[0], end=formal.VALIDATION[1], **costs, label=run_id)
                    checks = checked["checks"]
                    if checks.failed:
                        raise AssertionError("Independent diagnostic ledger reconstruction failed: " + str(checks.failures))
                    row, daily, monthly = summarize(result, run_id=run_id, strategy=algorithm, seed=None,
                                                   scenario=scenario, split="validation", start=formal.VALIDATION[0],
                                                   end=formal.VALIDATION[1], capital=10000., **costs)
                    for name in ("net_return", "max_drawdown", "fees", "average_exposure", "final_equity"):
                        if not np.isclose(row[name], checked["metrics"][name], atol=1e-9, rtol=1e-10):
                            raise AssertionError("Independent summary reconstruction differs: " + name)
                    row.update(algorithm=algorithm, profile=profile, backend="cpu", primary_policy=True,
                               formal_evaluation=False, test_evaluations=0, repeat_exact=True, independent_audit_passed=True)
                    for name in REFERENCES:
                        reference = references.loc[references.strategy.eq(name) & references.scenario.eq(scenario)].iloc[0]
                        for metric in ("net_return", "max_drawdown", "fees", "average_exposure"):
                            row[f"{metric}_minus_{name}"] = float(row[metric] - reference[metric])
                    daily["algorithm"], daily["profile"], monthly["algorithm"], monthly["profile"] = algorithm, profile, algorithm, profile
                    rows.append(row)
                    histories.append(daily)
                    months.append(monthly)
                    proof = {"run_id": run_id, "model_records": model_records, "parameter_sha256_before": before,
                             "parameter_sha256_after": after, "fingerprints": fingerprints, "repeat_exact": True,
                             "cuda_initialized_before": False, "cuda_initialized_after": torch.cuda.is_initialized(),
                             "independent_checks": checks.count, "independent_failures": checks.failed,
                             "maximum_reconstruction_errors": checks.maximum_errors, "metrics": checked["metrics"]}
                    proofs.append(proof)
                    write_json(output / "cases" / algorithm / profile / scenario / "independent_reconstruction.json", proof)
                    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False, float_format="%.17g")
                    print(f"{run_id} return={row['net_return']:.6%} DD={row['max_drawdown']:.6%} fees={row['fees']:.4f} exposure={row['average_exposure']:.6%}", flush=True)
                del models
        if rows:
            pd.concat(histories, ignore_index=True).to_csv(output / "daily_equity.csv", index=False, float_format="%.17g")
            pd.concat(months, ignore_index=True).to_csv(output / "monthly_returns.csv", index=False, float_format="%.17g")
        for path, expected in protected.items():
            if sha(path) != expected:
                raise AssertionError("Protected formal/reference input changed: " + str(path))
        if torch.cuda.is_initialized():
            raise AssertionError("CUDA became initialized")
        completion = {"passed": True, "completed_at": now(), "elapsed_seconds": time.monotonic() - started,
                      "cases": len(rows), "expected_cases_for_available_algorithms": len(records) * 4,
                      "availability": status, "repeated_cases": len(proofs),
                      "independent_checks": sum(p["independent_checks"] for p in proofs), "independent_failures": 0,
                      "runtime": runtime, "all_inputs_unchanged": True,
                      "price_calculation_end_exclusive": formal.VALIDATION[1],
                      "latest_price_used": str(data["hourly_close"].index.max()),
                      "new_training": 0, "test_evaluations": 0, "formal_output_modified": False,
                      "interpretation": "CPU diagnostic only. Formal frozen GPU validation/selection remains authoritative; CPU/GPU actions may differ slightly. No seeds, profiles, training budget or financial rules were changed."}
        write_json(output / "completion.json", completion)
        write_json(output / "replay_and_audit_proofs.json", proofs)
        write_json(output / "input_sha_after.json", {str(path): sha(path) for path in protected})
        write_json(output / "file_manifest.json", {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "file_manifest.json"})
        print(json.dumps(completion, ensure_ascii=False), flush=True)
        return completion
    except BaseException as error:
        write_json(output / "failure.json", {"at": now(), "type": type(error).__name__, "error": str(error),
                                           "cuda_initialized": torch.cuda.is_initialized(), "completed_cases": len(rows)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-output", type=Path, default=ROOT / "results/backtests/btceth_rl_20261029")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/crypto_rl_btceth_20261029/dataset.pkl.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/backtests/btceth_rl_pretest_reference_20261029/early_validation")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "results/backtests/btceth_rl_pretest_reference_20261029/risk_budget_controls")
    args = parser.parse_args()
    diagnostic(args.formal_output, args.dataset, args.output_dir, args.reference_dir)


if __name__ == "__main__":
    main()
