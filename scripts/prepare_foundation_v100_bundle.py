"""Prepare an offline, forecast-only dual-V100 bundle without CPU forecasts.

This copies only pinned source and prepared inputs. It never connects, installs,
executes a model, changes the CPU experiment, or includes the large weight files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results/backtests/crypto_foundation_models_20261024"
ENTRYPOINTS = ("scripts/prepare_foundation_v100_bundle.py", "scripts/run_foundation_v100_forecasts.py")


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def portable_name(path):
    return PureWindowsPath(str(path)).name


def old_key(path):
    return str(path).replace("\\", "/").casefold()


def validate_remote_root(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts or ":" in value:
        raise ValueError("Use a new absolute Linux experiment directory, not a filesystem root")
    return path


def create_bundle(source, destination, remote_root, *, repository=ROOT):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    repository = Path(repository).resolve()
    remote_root = validate_remote_root(str(remote_root))
    if destination.exists() or destination.is_relative_to(source):
        raise FileExistsError("Use a new sibling bundle directory; preserve the CPU experiment")
    cpu_spec = read(source / "frozen_spec.json")
    frozen = read(source / "inference_frozen_inputs.json")
    if digest(source / "frozen_spec.json") != frozen["protocol_sha256"]:
        raise AssertionError("Original CPU protocol is not the frozen version")
    if cpu_spec.get("local_fit") is not False or cpu_spec.get("context_days") != 128:
        raise ValueError("Only the existing inference-only 128-day study can be relocated")
    sources = read(source / "sources/source_manifest.json")
    weights = read(source / "sources/weights_manifest.json")
    spec = deepcopy(cpu_spec)
    relocation, lookup, transfers = [], {}, []
    destination.mkdir(parents=True)

    def copy_file(original, relative, expected=None):
        original, relative = Path(original), PurePosixPath(relative)
        actual = digest(original)
        if expected is not None and actual != expected:
            raise AssertionError("Pinned bundle input changed: " + str(original))
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        return actual

    for name, expected in frozen["source_sha256"].items():
        copy_file(repository / name, name, expected)
    for name in ENTRYPOINTS:
        copy_file(repository / name, name)
    # These files retain their original bytes and historical paths as evidence.
    evidence_names = ["frozen_spec.json", "inference_frozen_inputs.json", "policy_precommit.json",
                      "model_provenance.json", "model_provenance_evaluation.json",
                      "sources/source_manifest.json", "sources/weights_manifest.json"]
    for name in evidence_names:
        copy_file(source / name, "experiment/evidence/cpu/" + name)
    copy_file(source / "market/daily_ohlcva.pkl.gz", "experiment/market/daily_ohlcva.pkl.gz")
    copy_file(source / "market/manifest.json", "experiment/market/manifest.json", cpu_spec["market_manifest_sha256"])
    copy_file(source / "forecast_request_plan.csv", "experiment/forecast_request_plan.csv")
    market_manifest = read(source / "market/manifest.json")
    if digest(source / "market/daily_ohlcva.pkl.gz") != market_manifest["market_sha256"]:
        raise AssertionError("Prepared daily market bytes changed")

    vendor_roots = {}
    for row in sources["sources"]:
        relative_root = PurePosixPath("vendor") / portable_name(row["vendor_path"])
        vendor_roots[row["id"]] = remote_root / relative_root
        for entry in row["files"]:
            relative = relative_root / PurePosixPath(entry["path"])
            copy_file(entry["local_path"], relative.as_posix(), entry["sha256"])
            lookup[old_key(entry["local_path"])] = remote_root / relative
            relocation.append({"original_path": entry["local_path"], "relative_path": relative.as_posix(),
                               "remote_path": str(remote_root / relative), "sha256": entry["sha256"], "copied": True})
    weight_roots = {}
    for row in weights["models"]:
        relative_root = PurePosixPath("weights") / portable_name(row["local_directory"])
        weight_roots[row["name"]] = remote_root / relative_root
        for entry in row["files"]:
            relative = relative_root / PurePosixPath(entry["path"])
            item = {"model_key": row["name"], "source_path": entry["local_path"],
                    "relative_path": relative.as_posix(), "remote_path": str(remote_root / relative),
                    "sha256": entry["sha256"], "size_bytes": Path(entry["local_path"]).stat().st_size}
            transfers.append(item)
            lookup[old_key(entry["local_path"])] = remote_root / relative
            relocation.append({"original_path": entry["local_path"], "relative_path": relative.as_posix(),
                               "remote_path": str(remote_root / relative), "sha256": entry["sha256"], "copied": False})
    for entry in spec["pinned_files"]:
        key = old_key(entry["path"])
        if key not in lookup:
            raise AssertionError("Every pinned source/weight requires an explicit relocation")
        entry["path"] = str(lookup[key])
    for key, config in spec["models"].items():
        config["vendor_path"] = str(vendor_roots["kronos" if key.startswith("kronos") else "chronos2"])
        if key == "chronos2":
            config["vendor_path"] += "/src"
        config["weights_path"], config["device"] = str(weight_roots[key]), "cuda:0"
        if key.startswith("kronos"):
            config["tokenizer_path"] = str(weight_roots["kronos_tokenizer"])
    spec.update(frozen_at=None, model_workers=2,
                backend_evidence_path=str(remote_root / "experiment/backend_preflight.json"),
                backend_evidence_sha256=None,
                model_provenance_path=str(remote_root / "experiment/evidence/cpu/model_provenance_evaluation.json"),
                reproduction_argv=["python", "scripts/run_foundation_v100_forecasts.py", "--phase", "forecast"],
                run_scope="forecasts_only; financial evaluation remains on the local unchanged funded-spot ledger",
                gpu_worker_devices=[0, 1],
                gpu_worker_binding="Two one-process executors. Before importing torch, isolate each physical GPU UUID with CUDA_VISIBLE_DEVICES; both use local cuda:0.",
                chronos_loader_adaptation="Retain the predeclared synchronous DataLoader scheduler; official model source is unchanged.")
    spec["migration"] = {"status": "prepared_not_executed", "parent_cpu_protocol_sha256": frozen["protocol_sha256"],
                         "parent_cpu_frozen_inputs_sha256": digest(source / "inference_frozen_inputs.json"),
                         "remote_root": str(remote_root), "cpu_forecasts_copied": False,
                         "cpu_gpu_bitwise_equality_claimed": False,
                         "allowed_changes": ["execution backend and runtime", "absolute path relocation", "two GPU worker scheduling"],
                         "financial_rules_changed": False, "local_fit": False}
    write(destination / "experiment/gpu_spec_template.json", spec)
    write(destination / "relocation_manifest.json", {"status": "prepared_not_executed", "files": relocation,
          "parent_cpu_protocol_sha256": frozen["protocol_sha256"], "remote_root": str(remote_root)})
    write(destination / "weights_transfer_manifest.json", {"status": "not_included_transfer_exact_bytes_separately",
          "files": transfers, "total_bytes": sum(row["size_bytes"] for row in transfers)})
    expected_parameters = {}
    smoke = source / "sources/native_wrapper_synthetic_smoke.json"
    if smoke.is_file():
        for row in read(smoke)["results"]:
            if row["status"] == "passed":
                expected_parameters[row["model"]] = row["parameter_sha256"]
        copy_file(smoke, "experiment/evidence/cpu/native_wrapper_synthetic_smoke.json")
    write(destination / "migration_control.json", {"status": "prepared_not_executed", "remote_root": str(remote_root),
          "source_sha256": frozen["source_sha256"], "parent_cpu_protocol_sha256": frozen["protocol_sha256"],
          "template_sha256": digest(destination / "experiment/gpu_spec_template.json"),
          "plan_sha256": digest(source / "forecast_request_plan.csv"),
          "market_sha256": market_manifest["market_sha256"], "gpu_indices": [0, 1],
          "expected_parameter_sha256_by_model": expected_parameters,
          "cuda_or_torch_wheel_selected": False, "installation_performed": False})
    write(destination / "runtime_requirements_review.json", {"status": "hardware_not_yet_verified_no_installation",
          "python": ">=3.11", "prepared_pickle_runtime": {"pandas": "3.0.3", "numpy": "2.4.6"},
          "common_inference_packages": {"transformers": "4.53.3", "huggingface-hub": "0.33.1",
                "accelerate": "1.10.1", "einops": "0.8.1", "safetensors": "0.6.2"},
          "torch": "Choose a CUDA-enabled build supporting the actual V100 sm70 and installed driver; verify with synthetic kernels before forecasting. No wheel is selected here.",
          "cuda_toolkit_or_driver_changes": False})
    instructions = """# Dual-V100 forecast migration — prepared, not executed

This bundle contains the seven frozen research source files, these two migration
entry points, pinned official source and licenses, the exact prepared daily
market/causal request plan, and original protocol/provenance evidence.

It contains no CPU forecasts, account credentials, trading configuration, large
model weights, historical hourly datasets, or installation script. Transfer the
exact files in weights_transfer_manifest.json separately and keep their paths.
Do not edit or normalize source/weights. Verify the actual host, GPU, Python,
driver and compatible PyTorch runtime before installing anything.

Extract at the exact remote_root in migration_control.json. Run `python
scripts/run_foundation_v100_forecasts.py --phase preflight` to check all input
hashes and run synthetic-only checks on each GPU. Forecasting requires the
explicit `--phase forecast`; after interruption also use `--resume`. No CPU
fallback exists. The launcher never trains, evaluates PNL, or trades.

The two independent workers isolate different physical GPU UUIDs before Torch
initializes. Each sees one local cuda:0. Fixed same-day asset batches, date seeds,
128-day context, weights, and all financial rules are preserved. GPU predictions
are a new experiment; equality with CPU prediction bytes is not assumed.

Return the entire experiment directory plus bundle/relocation/weights manifests
for local reconciliation. The local unchanged ledger performs the 200 financial
evaluations only after all forecast and source hashes/repeats pass.
"""
    (destination / "README_V100_MIGRATION.md").write_text(instructions, encoding="utf-8")
    files = [{"relative_path": path.relative_to(destination).as_posix(), "sha256": digest(path),
              "size_bytes": path.stat().st_size} for path in sorted(destination.rglob("*")) if path.is_file()]
    manifest = {"status": "prepared_not_executed", "created_at": datetime.now(timezone.utc).isoformat(),
                "parent_cpu_protocol_sha256": frozen["protocol_sha256"], "remote_root": str(remote_root),
                "files": files, "payload_bytes": sum(row["size_bytes"] for row in files),
                "weights_included": False, "cpu_forecasts_included": False, "production_changes": False}
    write(destination / "bundle_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True, help="Exact new absolute Linux bundle directory")
    args = parser.parse_args()
    result = create_bundle(args.source_root, args.bundle_dir, args.remote_root)
    print(json.dumps({key: result[key] for key in ("status", "payload_bytes", "weights_included", "remote_root")}))


if __name__ == "__main__":
    main()
