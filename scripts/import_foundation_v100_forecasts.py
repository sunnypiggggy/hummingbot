"""Import returned V100 forecasts and evaluate with the unchanged local ledger.

The returned directory is read-only. GPU protocol and daily cache contracts are
copied byte-for-byte. Only per-model forecast manifest paths are derived for
local reading; original manifests and their SHA256 remain separate evidence.
This entry point never imports a native model, trains, connects, or places orders.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CPU_ROOT = ROOT / "results/backtests/crypto_foundation_models_20261024"
DATASETS = {"original": ROOT / "data/crypto_rl_top10/dataset.pkl.gz",
            "ex_ada": ROOT / "data/crypto_rl_top10_ex_ada/dataset.pkl.gz"}
CPU_SOURCE_EVIDENCE = (
    "original_research_preservation.json",
    "sources/independent_market_audit.json",
    "sources/implementation_review.json",
    "forecast_diagnostics_source.json",
    "sources/chronos_synchronous_loader_synthetic.json",
    # These aliases already arrive in the GPU bundle; also check that their
    # bytes still equal the original CPU evidence, without rewriting them.
    "policy_precommit.json", "model_provenance.json", "model_provenance_evaluation.json",
    "sources/source_manifest.json",
)
PUBLIC_SOURCE_EVIDENCE = (
    "sources/weights_manifest.json", "sources/availability_evidence.json",
    "sources/original_upload_byte_comparisons.json", "sources/source_archive_manifest.json",
    "sources/official_inference_sources.zip",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def map_remote_path(remote_path, remote_experiment, local_experiment):
    remote_path, remote_experiment = PurePosixPath(remote_path), PurePosixPath(remote_experiment)
    if not remote_path.is_absolute() or ".." in remote_path.parts:
        raise ValueError("Returned artifact paths must be absolute normalized Linux paths")
    relative = remote_path.relative_to(remote_experiment)
    local = Path(local_experiment).resolve().joinpath(*relative.parts)
    if not local.resolve().is_relative_to(Path(local_experiment).resolve()) or local.is_symlink():
        raise ValueError("Returned artifact escaped the new local experiment")
    return local


def derive_model_manifest(original, *, original_sha256, remote_experiment, local_experiment):
    derived = deepcopy(original)
    for entry in derived["forecast_files"]:
        entry["original_remote_path"] = entry["path"]
        entry["path"] = str(map_remote_path(entry["path"], remote_experiment, local_experiment))
    derived["artifact_relocation"] = {"original_remote_manifest_sha256": original_sha256,
                                      "changed_fields": ["forecast_files[].path"],
                                      "prediction_values_changed": False,
                                      "gpu_protocol_and_cache_contracts_changed": False}
    return derived


def copy_cpu_source_evidence(cpu_root, destination):
    """Copy audit/report evidence; it does not describe the GPU runtime.

    The original files stay read-only. Existing GPU-bundle aliases must match
    their original bytes and are never overwritten. Each copy is included in
    both an explicit source chain and the final immutable import file list.
    """
    cpu_root, destination = Path(cpu_root).resolve(), Path(destination).resolve()
    records = []
    names = [*CPU_SOURCE_EVIDENCE,
             *(name for name in PUBLIC_SOURCE_EVIDENCE if (cpu_root / name).is_file())]
    for name in names:
        source, target = cpu_root / name, destination / name
        if not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(cpu_root):
            raise ValueError("Original CPU audit/report evidence is missing or indirect: " + name)
        expected = digest(source)
        if target.exists():
            if target.is_symlink() or digest(target) != expected:
                raise AssertionError("An imported evidence alias differs from the original CPU bytes: " + name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if digest(source) != expected or digest(target) != expected:
            raise AssertionError("Audit/report evidence changed during byte-copy: " + name)
        records.append({"classification": "original_cpu_source_evidence_not_gpu_runtime",
                        "source_path": str(source), "source_sha256": expected,
                        "local_relative_path": name, "local_sha256": expected})
    return records


def assert_frozen_local_sources(control, spec):
    for name, expected in control["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise AssertionError("The original financial/native source changed: " + name)
    for name, path in DATASETS.items():
        if digest(path) != spec["dataset_sha256_by_variant"][name]:
            raise AssertionError("The original local hourly dataset changed: " + name)


def verify_gpu_receipt(returned, cpu_root, prepared_bundle):
    returned, cpu_root = Path(returned).resolve(), Path(cpu_root).resolve()
    prepared_bundle = Path(prepared_bundle).resolve()
    for name in ("migration_control.json", "bundle_manifest.json", "weights_transfer_manifest.json", "relocation_manifest.json"):
        if digest(returned / name) != digest(prepared_bundle / name):
            raise AssertionError("Returned migration evidence differs from the locally prepared bundle: " + name)
    experiment = returned / "experiment"
    if digest(experiment / "gpu_spec_template.json") != digest(prepared_bundle / "experiment/gpu_spec_template.json"):
        raise AssertionError("The returned GPU template differs from the locally prepared experiment")
    control = read(returned / "migration_control.json")
    spec, completion = read(experiment / "frozen_spec.json"), read(experiment / "forecast_completion.json")
    frozen, backend = read(experiment / "inference_frozen_inputs.json"), read(experiment / "backend_preflight.json")
    protocol = digest(experiment / "frozen_spec.json")
    cpu_frozen = read(cpu_root / "inference_frozen_inputs.json")
    if digest(cpu_root / "frozen_spec.json") != control["parent_cpu_protocol_sha256"] or cpu_frozen["protocol_sha256"] != control["parent_cpu_protocol_sha256"]:
        raise AssertionError("Returned forecasts are linked to a different original CPU study")
    if control["source_sha256"] != cpu_frozen["source_sha256"] or frozen["source_sha256"] != control["source_sha256"]:
        raise AssertionError("Returned GPU sources differ from the original seven frozen files")
    if frozen["protocol_sha256"] != protocol or completion["protocol_sha256"] != protocol or completion["parent_cpu_protocol_sha256"] != control["parent_cpu_protocol_sha256"]:
        raise AssertionError("The returned GPU protocol identity is inconsistent")
    if completion.get("status") != "complete" or completion.get("new_models_trained") != 0 or completion.get("financial_evaluations") != 0 or completion.get("cpu_forecasts_reused") is not False:
        raise AssertionError("Require complete fresh GPU forecasts without training or remote financial evaluation")
    if backend.get("status") != "passed" or digest(experiment / "backend_preflight.json") != spec["backend_evidence_sha256"]:
        raise AssertionError("GPU native synthetic preflight is missing or changed")
    from scripts.run_foundation_v100_forecasts import validate_contract
    validate_contract(read(cpu_root / "frozen_spec.json"), spec)
    expected_spec = {**read(prepared_bundle / "experiment/gpu_spec_template.json"),
                     "frozen_at": spec["frozen_at"], "backend_evidence_sha256": digest(experiment / "backend_preflight.json")}
    if spec != expected_spec:
        raise AssertionError("GPU execution changed the locally prepared contract beyond backend freezing")
    assert_frozen_local_sources(control, spec)
    if digest(experiment / "forecast_request_plan.csv") != control["plan_sha256"] or digest(experiment / "market/daily_ohlcva.pkl.gz") != control["market_sha256"]:
        raise AssertionError("The returned causal plan/market bytes differ from the prepared inputs")
    if digest(experiment / "market/manifest.json") != frozen["market_manifest_sha256"]:
        raise AssertionError("The original market manifest changed")
    devices = backend["gpu_probe"]
    if len(devices) != 2 or len({item["uuid"] for item in devices}) != 2 or any("V100" not in item["name"] for item in devices):
        raise AssertionError("The backend did not verify two different V100 GPUs")
    expected_parameters = control["expected_parameter_sha256_by_model"]
    for checked in backend["synthetic_checks"]:
        binding = checked["binding"]
        if binding["visible_devices"] != binding["uuid"] or binding["torch_current_device"] != 0:
            raise AssertionError("A GPU worker was not isolated to its recorded physical UUID")
        if {row["model_key"] for row in checked["models"]} != set(spec["models"]):
            raise AssertionError("Synthetic checks omitted a native model")
        for row in checked["models"]:
            if row["status"] != "passed" or not row["same_gpu_repeat_equal"] or not row["input_unchanged"] or row["parameter_sha256"] != expected_parameters[row["model_key"]]:
                raise AssertionError("Native GPU synthetic/parameter checks did not pass")
    if len(backend["synthetic_checks"]) != 2 or {item["binding"]["uuid"] for item in backend["synthetic_checks"]} != {item["uuid"] for item in devices}:
        raise AssertionError("Both physical GPU workers must pass their synthetic checks")
    from scripts.backtest_crypto_foundation_models import forecast_jobs
    jobs = forecast_jobs(experiment, spec)
    identities = lambda item: (item["model_key"], item["seed"], item["shard_index"], item["shard_count"])
    expected_jobs = {identities(job): devices[index % 2]["uuid"] for index, job in enumerate(jobs)}
    if len(completion["shards"]) != len(jobs) or {identities(row) for row in completion["shards"]} != set(expected_jobs):
        raise AssertionError("Returned GPU shards are missing or duplicated")
    remote_experiment = PurePosixPath(control["remote_root"]) / "experiment"
    for row in completion["shards"]:
        binding = row["gpu_worker_binding"]
        if binding["uuid"] != expected_jobs[identities(row)] or binding["visible_devices"] != binding["uuid"] or binding["torch_current_device"] != 0:
            raise AssertionError("A GPU shard ran on an unexpected physical device")
        path = map_remote_path(row["path"], remote_experiment, experiment)
        if digest(path) != row["sha256"]:
            raise AssertionError("A returned native shard manifest changed")
        shard = read(path)
        if shard["parameter_sha256_before"] != expected_parameters[row["model_key"]] or shard["parameter_sha256_after"] != shard["parameter_sha256_before"]:
            raise AssertionError("Pretrained parameters changed on a native GPU shard")
        for day in shard["day_files"]:
            if digest(map_remote_path(day["path"], remote_experiment, experiment)) != day["sha256"]:
                raise AssertionError("A returned native daily prediction changed")
    if len(completion["models"]) != len(spec["models"]) or {item["model_key"] for item in completion["models"]} != set(spec["models"]):
        raise AssertionError("The returned native model set is incomplete")
    for item in completion["models"]:
        if digest(experiment / "forecasts" / item["model_key"] / "manifest.json") != item["manifest_sha256"]:
            raise AssertionError("The returned native model manifest changed")
    return control, spec, remote_experiment


def import_artifacts(returned, destination, prepared_bundle, cpu_root=CPU_ROOT):
    returned, destination, cpu_root = Path(returned).resolve(), Path(destination).resolve(), Path(cpu_root).resolve()
    if destination.exists() or destination.is_relative_to(returned) or destination.is_relative_to(cpu_root):
        raise FileExistsError("Import into a new separate local results directory")
    control, spec, remote_experiment = verify_gpu_receipt(returned, cpu_root, prepared_bundle)
    source = returned / "experiment"
    allowed = ("frozen_spec.json", "inference_frozen_inputs.json", "backend_preflight.json", "forecast_completion.json",
               "forecast_request_plan.csv", "gpu_spec_template.json", "market", "evidence", "forecasts", "preflight_rechecks")
    for name in allowed:
        path = source / name
        if path.is_symlink() or (path.is_dir() and any(child.is_symlink() for child in path.rglob("*"))):
            raise ValueError("Do not follow symlinks while importing returned artifacts")
    destination.mkdir(parents=True)
    for name in allowed:
        path = source / name
        if path.is_dir():
            shutil.copytree(path, destination / name)
        elif path.is_file():
            shutil.copy2(path, destination / name)
    preserved = destination / "remote_execution_evidence"
    preserved.mkdir()
    for name in ("migration_control.json", "bundle_manifest.json", "weights_transfer_manifest.json", "relocation_manifest.json"):
        shutil.copy2(returned / name, preserved / name)
    relocated = []
    for key in spec["models"]:
        path = destination / "forecasts" / key / "manifest.json"
        original = read(path)
        raw_copy = preserved / "forecasts" / key / "manifest.json"
        raw_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, raw_copy)
        derived = derive_model_manifest(original, original_sha256=digest(path),
                                         remote_experiment=remote_experiment, local_experiment=destination)
        write(path, derived)
        relocated.append({"model_key": key, "original_manifest_path": str(raw_copy), "original_manifest_sha256": digest(raw_copy),
                          "derived_manifest_path": str(path), "derived_manifest_sha256": digest(path)})
    # Useful local aliases are copies of original evidence, not edited GPU metadata.
    for name in ("model_provenance.json", "model_provenance_evaluation.json", "policy_precommit.json"):
        shutil.copy2(destination / "evidence/cpu" / name, destination / name)
    (destination / "sources").mkdir()
    shutil.copy2(destination / "evidence/cpu/sources/source_manifest.json", destination / "sources/source_manifest.json")
    source_evidence = copy_cpu_source_evidence(cpu_root, destination)
    overrides = {}
    for item in read(returned / "relocation_manifest.json")["files"]:
        path = Path(item["original_path"])
        if not path.is_file() or digest(path) != item["sha256"]:
            raise AssertionError("The local official source/weight equivalent is missing or changed")
        overrides[item["remote_path"]] = str(path)
    overrides[spec["model_provenance_path"]] = str(destination / "model_provenance_evaluation.json")
    overrides[spec["backend_evidence_path"]] = str(destination / "backend_preflight.json")
    from scripts.audit_crypto_foundation_models import independent_plan, audit_plan, audit_caches
    from scripts.foundation_model_audit_helpers import Checks
    from scripts.backtest_crypto_rl_top10 import load_dataset
    import pandas as pd
    market = pickle.loads(gzip.decompress((destination / "market/daily_ohlcva.pkl.gz").read_bytes()))
    datasets = {key: load_dataset(path) for key, path in DATASETS.items()}
    plan, batches = independent_plan(market, datasets, pd.Timestamp(spec["validation"][0]), pd.Timestamp(spec["test"][1]))
    review = Checks()
    audit_plan(review, destination, plan)
    _, summaries = audit_caches(review, destination, spec, read(destination / "inference_frozen_inputs.json"), plan, batches)
    receipt = {"status": "passed" if not review.failures else "failed", "checks_total": review.count,
               "checks_failed": len(review.failures), "failures": review.failures, "forecast_caches": summaries,
               "phase": "returned_gpu_cache_import_only_not_financial_audit",
               "auditor_source_sha256": digest(ROOT / "scripts/audit_crypto_foundation_models.py")}
    write(destination / "import_cache_audit.json", receipt)
    if review.failures:
        raise AssertionError("Returned GPU cache audit failed; preserve the new import directory for diagnosis")
    files = [{"path": path.relative_to(destination).as_posix(), "sha256": digest(path)}
             for path in sorted(destination.rglob("*")) if path.is_file()]
    result = {"status": "imported_verified_not_financially_evaluated", "created_at": datetime.now(timezone.utc).isoformat(),
              "parent_cpu_protocol_sha256": control["parent_cpu_protocol_sha256"],
              "prepared_bundle_manifest_sha256": digest(Path(prepared_bundle) / "bundle_manifest.json"),
              "gpu_protocol_sha256": digest(destination / "frozen_spec.json"),
              "source_sha256": control["source_sha256"], "dataset_sha256_by_variant": spec["dataset_sha256_by_variant"],
              "gpu_original_spec_and_contracts_unchanged": True, "prediction_values_changed": False,
              "cpu_gpu_prediction_equality_claimed": False, "model_manifest_relocations": relocated,
              "source_evidence": source_evidence,
              "proof_path_overrides": overrides, "files": files,
              "importer_source_sha256": digest(Path(__file__)), "production_changes": False}
    write(destination / "local_import_manifest.json", result)
    assert_frozen_local_sources(control, spec)
    return result


def verify_imported(destination):
    destination = Path(destination).resolve()
    imported, spec = read(destination / "local_import_manifest.json"), read(destination / "frozen_spec.json")
    if digest(destination / "frozen_spec.json") != imported["gpu_protocol_sha256"]:
        raise AssertionError("The imported GPU protocol changed")
    for item in imported["files"]:
        if digest(destination / item["path"]) != item["sha256"]:
            raise AssertionError("An imported GPU evidence/cache file changed: " + item["path"])
    assert_frozen_local_sources(imported, spec)
    if read(destination / "import_cache_audit.json")["status"] != "passed":
        raise AssertionError("The independent GPU cache import audit did not pass")
    return imported, spec


def evaluate_imported(destination, *, resume=False):
    destination = Path(destination).resolve()
    imported, spec = verify_imported(destination)
    if any((destination / variant).exists() for variant in DATASETS) and not resume:
        raise FileExistsError("Use explicit --resume-evaluation for partial local financial results")
    from scripts.backtest_crypto_foundation_models import case_task, diagnostics
    tasks = [{"output": str(destination), "model_key": key, "variant": variant, "spec": spec}
             for variant in DATASETS for key in spec["models"]]
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(case_task, task) for task in tasks]):
            results.append(future.result())
    diagnostics(destination, spec)
    count = sum(row["runs"] for row in results)
    if count != 200:
        raise AssertionError("Expected all200 unchanged-ledger financial evaluations and repeats")
    write(destination / "numeric_completion.json", {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
          "cases": results, "evaluations": count, "repeat_checks": count, "new_models_trained": 0,
          "pretrained_models": len(spec["models"]), "execution": "returned_native_dual_v100_forecasts_local_original_ledger",
          "gpu_protocol_sha256": imported["gpu_protocol_sha256"],
          "local_import_manifest_sha256": digest(destination / "local_import_manifest.json"), "production_changes": False})
    verify_imported(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned-root", type=Path, help="Read-only returned bundle root containing experiment/ and migration manifests")
    parser.add_argument("--prepared-bundle", type=Path, help="Original locally prepared bundle, used as the independent transfer trust anchor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-root", type=Path, default=CPU_ROOT)
    parser.add_argument("--phase", choices=("import", "evaluate", "all"), default="import")
    parser.add_argument("--resume-evaluation", action="store_true")
    args = parser.parse_args()
    if args.phase != "evaluate":
        if args.returned_root is None or args.prepared_bundle is None:
            parser.error("--returned-root and --prepared-bundle are required for import/all")
        result = import_artifacts(args.returned_root, args.output_dir, args.prepared_bundle, args.cpu_root)
        print(json.dumps({key: result[key] for key in ("status", "gpu_protocol_sha256", "prediction_values_changed")}))
    if args.phase != "import":
        evaluate_imported(args.output_dir, resume=args.resume_evaluation)
        print("GPU FORECASTS EVALUATED LOCALLY; independent full financial audit/report still required", flush=True)


if __name__ == "__main__":
    main()
