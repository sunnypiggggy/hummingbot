"""Run relocated native forecasts on two isolated V100 workers; no PNL or fit.

No CPU fallback, environment installation, SSH, account access, or order path is
implemented. A new GPU protocol is frozen after synthetic checks and before any
real-market inference. The original seven financial/native files stay unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WORKER = None
COMMON_VERSIONS = {"pandas": "3.0.3", "numpy": "2.4.6", "transformers": "4.53.3",
                   "huggingface-hub": "0.33.1", "accelerate": "1.10.1",
                   "einops": "0.8.1", "safetensors": "0.6.2"}


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


def validate_contract(cpu, gpu):
    allowed = {"frozen_at", "models", "pinned_files", "model_workers", "backend_evidence_path",
               "backend_evidence_sha256", "model_provenance_path", "reproduction_argv", "chronos_loader_adaptation"}
    for key in cpu:
        if key not in allowed and cpu[key] != gpu.get(key):
            raise AssertionError("Migration changed a frozen financial/signal setting: " + key)
    if set(cpu["models"]) != set(gpu["models"]):
        raise AssertionError("Native model set changed")
    for key, original in cpu["models"].items():
        moved = gpu["models"][key]
        for field, value in original.items():
            if field not in {"vendor_path", "weights_path", "tokenizer_path", "device"} and moved.get(field) != value:
                raise AssertionError("Migration changed a native model rule: " + key + "/" + field)
        if moved["device"] != "cuda:0":
            raise AssertionError("Each isolated one-GPU worker must use local cuda:0")
    if Counter(row["sha256"] for row in cpu["pinned_files"]) != Counter(row["sha256"] for row in gpu["pinned_files"]):
        raise AssertionError("Pinned official source/weight bytes changed during relocation")


def validate_bundle(root, experiment):
    root, experiment = Path(root).resolve(), Path(experiment).resolve()
    control, bundle = read(root / "migration_control.json"), read(root / "bundle_manifest.json")
    if root.as_posix() != control["remote_root"] or experiment != root / "experiment":
        raise ValueError("Extract at the declared new remote_root; do not reuse the CPU output directory")
    for entry in bundle["files"]:
        path = root / entry["relative_path"]
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size != entry["size_bytes"] or digest(path) != entry["sha256"]:
            raise AssertionError("Bundle file is missing or changed: " + entry["relative_path"])
    for name, expected in control["source_sha256"].items():
        if digest(root / name) != expected:
            raise AssertionError("Original financial/native source changed: " + name)
    template = read(experiment / "gpu_spec_template.json")
    if digest(experiment / "gpu_spec_template.json") != control["template_sha256"]:
        raise AssertionError("The migration template changed")
    cpu = read(experiment / "evidence/cpu/frozen_spec.json")
    if digest(experiment / "evidence/cpu/frozen_spec.json") != control["parent_cpu_protocol_sha256"]:
        raise AssertionError("The parent CPU protocol evidence changed")
    validate_contract(cpu, template)
    transfers = read(root / "weights_transfer_manifest.json")
    for item in transfers["files"]:
        path = root / item["relative_path"]
        if str(path) != item["remote_path"] or not path.is_file() or path.stat().st_size != item["size_bytes"] or digest(path) != item["sha256"]:
            raise AssertionError("Transfer the exact pinned weight/config bytes first: " + item["relative_path"])
    for item in template["pinned_files"]:
        if digest(item["path"]) != item["sha256"]:
            raise AssertionError("A relocated official file failed SHA256")
    if digest(experiment / "forecast_request_plan.csv") != control["plan_sha256"] or digest(experiment / "market/daily_ohlcva.pkl.gz") != control["market_sha256"]:
        raise AssertionError("Prepared causal market/request inputs changed")
    return control, template


def package_versions():
    if sys.version_info < (3, 11):
        raise RuntimeError("Prepared pandas3.0.3 inputs require a verified Python>=3.11 environment")
    versions = {name: importlib.metadata.version(name) for name in (*COMMON_VERSIONS, "torch")}
    for name, expected in COMMON_VERSIONS.items():
        if versions[name] != expected:
            raise RuntimeError(f"Expected the pinned common runtime {name}=={expected}; no automatic install is performed")
    return versions


def gpu_probe(indices):
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError("Exactly two different V100 physical indices are required")
    raw = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu,driver_version",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    available = {}
    for line in raw.splitlines():
        index, name, uuid, total, free, utilization, driver = [field.strip() for field in line.split(",")]
        available[int(index)] = {"physical_index": int(index), "name": name, "uuid": uuid,
                                 "memory_total_mib": int(total), "memory_free_mib": int(free),
                                 "utilization_gpu_percent": int(utilization), "driver_version": driver}
    selected = [available[index] for index in indices]
    if any("V100" not in item["name"] for item in selected) or len({item["uuid"] for item in selected}) != 2:
        raise RuntimeError("Both selected physical devices must be independently verified V100 GPUs")
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allocated:
        tokens = {part.strip() for part in allocated.split(",")}
        if any(str(item["physical_index"]) not in tokens and item["uuid"] not in tokens for item in selected):
            raise RuntimeError("Requested physical GPUs are outside the existing CUDA_VISIBLE_DEVICES allocation")
    return selected


def initialize_worker(binding):
    global WORKER
    # The launcher itself imports no Torch/NumPy/Pandas before this initializer.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_VISIBLE_DEVICES"] = binding["uuid"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This isolated worker must see exactly one working CUDA device")
    torch.cuda.set_device(0)
    if "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Torch did not initialize a V100 in this worker")
    WORKER = {**binding, "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "torch_current_device": torch.cuda.current_device(), "logical_device": "cuda:0",
              "compute_capability": list(torch.cuda.get_device_capability(0)), "pid": os.getpid()}


def assert_worker():
    import torch
    if WORKER is None or os.environ.get("CUDA_VISIBLE_DEVICES") != WORKER["uuid"] or torch.cuda.current_device() != 0 or torch.cuda.device_count() != 1:
        raise AssertionError("A worker's fixed physical/logical GPU binding changed")


def synthetic_worker(template, control):
    assert_worker()
    import numpy as np
    import pandas as pd
    import torch
    from scripts.foundation_model_native import NativeForecaster
    rng = np.random.default_rng(1729)
    frames = []
    for index in range(10):
        close = (100 + index) * np.exp(np.cumsum(rng.normal(0, .01, 128)))
        opening, volume = np.r_[close[0], close[:-1]], rng.uniform(100, 1000, 128)
        frames.append(pd.DataFrame({"open": opening, "high": np.maximum(opening, close) * 1.01,
                      "low": np.minimum(opening, close) * .99, "close": close, "volume": volume,
                      "amount": volume * close}, index=pd.date_range("2024-01-01", periods=128, freq="D", tz="UTC")))
    frame_hash = lambda: hashlib.sha256(b"".join(frame.to_numpy().tobytes() + frame.index.asi8.tobytes() for frame in frames)).hexdigest()
    before_inputs, rows = frame_hash(), []
    date, keys = frames[0].index[-1] + pd.Timedelta(days=1), [str(index) for index in range(1, 11)]
    for key, config in template["models"].items():
        model = NativeForecaster(key, config, threads=template["threads_per_model"])
        if any(parameter.device.type != "cuda" or parameter.device.index != 0
               for module in model.modules.values() for parameter in module.parameters()):
            raise AssertionError("The native model has parameters outside its isolated GPU")
        torch.cuda.synchronize()
        started = time.monotonic()
        first, first_seed = model.predict(frames, date, seed=19, asset_keys=keys)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        again, again_seed = model.predict(frames, date, seed=19, asset_keys=keys)
        torch.cuda.synchronize()
        if first != again or first_seed != again_seed or frame_hash() != before_inputs:
            raise AssertionError("Synthetic same-GPU inference is not repeatable or changed its input")
        if any(not np.isfinite(value) for item in first for value in item.values()):
            raise AssertionError("Synthetic native output is nonfinite")
        parameters = model.verify_unchanged()
        expected = control.get("expected_parameter_sha256_by_model", {}).get(key)
        if expected is not None and expected != parameters:
            raise AssertionError("GPU loaded parameter bytes differ from the verified CPU weights")
        rows.append({"model_key": key, "synthetic_only": True, "batch": 10, "context_days": 128,
                     "horizon_days": 1, "status": "passed", "same_gpu_repeat_equal": True,
                     "input_unchanged": True, "parameter_sha256": parameters,
                     "first_inference_seconds": elapsed, "prediction_sha256": hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest()})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {"binding": WORKER, "models": rows}


def run_forecast_job(task):
    assert_worker()
    from scripts.backtest_crypto_foundation_models import forecast_task
    result = forecast_task(task)
    assert_worker()
    return {**result, "gpu_worker_binding": WORKER}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "experiment")
    parser.add_argument("--phase", choices=("preflight", "forecast"), default="preflight")
    parser.add_argument("--resume", action="store_true", help="Resume this exact GPU protocol, never CPU day caches")
    args = parser.parse_args()
    experiment = args.experiment_root.resolve()
    control, template = validate_bundle(ROOT, experiment)
    versions = package_versions()
    devices = gpu_probe(control["gpu_indices"])
    context = multiprocessing.get_context("spawn")
    with ExitStack() as stack:
        pools = [stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=context,
                         initializer=initialize_worker, initargs=(binding,))) for binding in devices]
        preflight = [pool.submit(synthetic_worker, template, control) for pool in pools]
        checked = [future.result() for future in preflight]
        for key in template["models"]:
            values = {row["parameter_sha256"] for result in checked for row in result["models"] if row["model_key"] == key}
            if len(values) != 1:
                raise AssertionError("The two GPU workers loaded different model parameters")
        backend = {"status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
                   "hostname": socket.gethostname(), "python": sys.version, "packages": versions,
                   "gpu_probe": devices, "synthetic_checks": checked, "cpu_gpu_forecast_bitwise_equality_claimed": False,
                   "new_models_trained": 0, "no_cpu_fallback": True, "financial_evaluation_performed": False}
        frozen_path = experiment / "frozen_spec.json"
        if frozen_path.exists():
            if args.phase == "forecast" and (experiment / "forecasts").exists() and not args.resume:
                raise ValueError("Use explicit --resume for an existing GPU protocol/cache")
            spec = read(frozen_path)
            previous_backend = read(experiment / "backend_preflight.json")
            if digest(experiment / "backend_preflight.json") != spec["backend_evidence_sha256"]:
                raise AssertionError("The frozen GPU backend evidence changed")
            if previous_backend["packages"] != versions or previous_backend["python"] != sys.version or previous_backend["hostname"] != socket.gethostname():
                raise AssertionError("The frozen GPU runtime changed; prepare a new experiment")
            if [(item["uuid"], item["driver_version"]) for item in previous_backend["gpu_probe"]] != [(item["uuid"], item["driver_version"]) for item in devices]:
                raise AssertionError("The frozen GPU hardware/driver mapping changed")
            expected = {**template, "frozen_at": spec["frozen_at"],
                        "backend_evidence_sha256": digest(experiment / "backend_preflight.json")}
            if spec != expected or read(experiment / "inference_frozen_inputs.json")["protocol_sha256"] != digest(frozen_path):
                raise AssertionError("The GPU protocol differs from the prepared and frozen experiment")
            # New checks are separate evidence, never overwrite the original frozen backend.
            write(experiment / "preflight_rechecks" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json"), backend)
        else:
            if args.resume:
                raise ValueError("No GPU protocol exists to resume")
            write(experiment / "backend_preflight.json", backend)
            spec = dict(template)
            spec.update(frozen_at=datetime.now(timezone.utc).isoformat(), backend_evidence_sha256=digest(experiment / "backend_preflight.json"))
            write(frozen_path, spec)
            write(experiment / "inference_frozen_inputs.json", {"prepared_at": datetime.now(timezone.utc).isoformat(),
                  "protocol_sha256": digest(frozen_path), "source_sha256": control["source_sha256"],
                  "dataset_sha256_by_variant": spec["dataset_sha256_by_variant"],
                  "market_manifest_sha256": digest(experiment / "market/manifest.json"),
                  "source_manifest_sha256": digest(experiment / "evidence/cpu/sources/source_manifest.json"),
                  "parent_cpu_protocol_sha256": control["parent_cpu_protocol_sha256"], "execution": "isolated_dual_v100_forecasts_only"})
        validate_contract(read(experiment / "evidence/cpu/frozen_spec.json"), spec)
        if args.phase == "preflight":
            print("V100 SYNTHETIC PREFLIGHT PASSED; no real-market forecasts run", flush=True)
            return
        from scripts.backtest_crypto_foundation_models import forecast_jobs, merge_forecast_shards
        jobs = forecast_jobs(experiment, spec)
        pending = [pools[index % 2].submit(run_forecast_job, task) for index, task in enumerate(jobs)]
        results = [future.result() for future in as_completed(pending)]
        manifests = merge_forecast_shards(experiment, spec, results)
        for name, expected in control["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise AssertionError("Frozen financial/native source changed during GPU inference")
        write(experiment / "forecast_completion.json", {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": digest(frozen_path), "parent_cpu_protocol_sha256": control["parent_cpu_protocol_sha256"],
              "models": [{"model_key": row["model_key"], "manifest_sha256": digest(experiment / "forecasts" / row["model_key"] / "manifest.json")} for row in manifests],
              "shards": results, "new_models_trained": 0, "financial_evaluations": 0,
              "cpu_forecasts_reused": False, "gpu_worker_devices": devices, "production_changes": False})
        print("V100 FORECASTS COMPLETE; return artifacts for local unchanged-ledger evaluation", flush=True)


if __name__ == "__main__":
    main()
