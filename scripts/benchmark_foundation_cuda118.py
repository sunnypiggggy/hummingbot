"""Bounded synthetic native inference check for an isolated Pascal GPU runtime.

This is a compatibility/speed fixture, not market data or a backtest. It never
modifies drivers, official vendor source, old environments or financial results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/backtests/crypto_foundation_models_20261024/sources"
VENDOR = Path("C:/Users/sunny/.cache/codex-research-vendors/kronos-a5f5aba12d78")
WEIGHTS = Path("C:/Users/sunny/.cache/codex-research-vendors/foundation-model-weights")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "cuda118_smoke.json")
    parser.add_argument("--model", choices=("kronos_small", "kronos_base"), default="kronos_small")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior compatibility evidence; choose a fresh output")
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "executable": sys.executable, "fixture": "Synthetic OHLCVA only; 128 complete daily bars x 10 assets, no financial PNL",
              "status": "failed", "model_key": args.model,
              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    code = 1
    try:
        import numpy as np
        import pandas as pd
        import torch
        result["packages"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "numpy", "pandas", "huggingface-hub", "safetensors", "einops")}
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        result.update(cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                      numpy_bridge=bool(np.array_equal(torch.from_numpy(np.arange(5, dtype=np.float32)).numpy(), np.arange(5, dtype=np.float32))))
        if not result["cuda_available"] or not result["numpy_bridge"]:
            raise AssertionError("CUDA or NumPy bridge is unavailable")
        result.update(device_name=torch.cuda.get_device_name(0), device_capability=list(torch.cuda.get_device_capability(0)),
                      compiled_architectures=torch.cuda.get_arch_list(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32, allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
        torch.manual_seed(1234)
        a, b = torch.randn(256, 256, device="cuda"), torch.randn(256, 256, device="cuda")
        started = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        result["real_cuda_matmul"] = {"shape": list(c.shape), "finite": bool(torch.isfinite(c).all()),
                                      "seconds": time.perf_counter() - started, "sum": float(c.sum())}
        del a, b, c
        torch.cuda.empty_cache()
        manifest = json.loads((OUT / "weights_manifest.json").read_text(encoding="utf-8"))
        wanted = {args.model, "kronos_tokenizer"}
        actual_files = []
        for model in manifest["models"]:
            if model["name"] not in wanted:
                continue
            for item in model["files"]:
                if sha(item["local_path"]) != item["sha256"]:
                    raise AssertionError("A pinned checkpoint byte changed")
                actual_files.append({"path": item["local_path"], "sha256": item["sha256"]})
        result["verified_checkpoint_files"] = actual_files
        sys.path.insert(0, str(VENDOR))
        from model import Kronos, KronosTokenizer, KronosPredictor
        result["native_class"] = {"predictor": KronosPredictor.__module__, "source_sha256": sha(VENDOR / "model/kronos.py")}
        dates = pd.date_range("2024-01-01", periods=128, freq="D")
        t, frames = np.arange(128, dtype=float), []
        for asset in range(10):
            close = (100 + asset * 10) * np.exp(.0004 * t + .01 * np.sin(t / 11 + asset))
            op = close * np.exp(.001 * np.sin(t / 4 + asset))
            volume = 1000 + 200 * np.cos(t / 7 + asset)
            frames.append(pd.DataFrame({"open": op, "high": np.maximum(op, close) * 1.005,
                "low": np.minimum(op, close) * .995, "close": close, "volume": volume,
                "amount": volume * (op + close) / 2}, index=dates))
        x, y = [pd.Series(dates) for _ in frames], [pd.Series([dates[-1] + pd.Timedelta(days=1)]) for _ in frames]
        model_dir = {"kronos_small": WEIGHTS / "kronos_small-901c26c13326", "kronos_base": WEIGHTS / "kronos_base-2b554741eca4"}[args.model]
        tokenizer = KronosTokenizer.from_pretrained(str(WEIGHTS / "kronos_tokenizer-0e0117387f39"), local_files_only=True).eval()
        model = Kronos.from_pretrained(str(model_dir), local_files_only=True).eval()

        def run(predictor, device):
            times, predictions = [], []
            for _ in range(3):
                random.seed(1234)
                np.random.seed(1234)
                torch.manual_seed(1234)
                if device == "cuda:0":
                    torch.cuda.manual_seed_all(1234)
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = predictor.predict_batch(frames, x, y, pred_len=1, T=1., top_p=.9,
                                                     top_k=0, sample_count=1, verbose=False)
                if device == "cuda:0":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                values = np.stack([frame.to_numpy() for frame in output])
                if values.shape != (10, 1, 6) or not np.isfinite(values).all():
                    raise AssertionError("The native synthetic batch is malformed")
                predictions.append(values)
            error = max(float(np.abs(value - predictions[0]).max()) for value in predictions)
            if error != 0:
                raise AssertionError("Same-device native same-seed inference is not exactly repeatable")
            return {"seconds_all_calls": times, "warm_mean_seconds": float(np.mean(times[1:])),
                    "repeat_max_absolute_difference": error, "exact_same_seed_repeat": True,
                    "output_shape": list(predictions[0].shape), "predicted_close": predictions[0][:, 0, 3].tolist()}, predictions[0]

        cpu = KronosPredictor(model, tokenizer, device="cpu", max_context=128)
        result["cpu_native"], cpu_values = run(cpu, "cpu")
        gpu = KronosPredictor(model, tokenizer, device="cuda:0", max_context=128)
        torch.cuda.reset_peak_memory_stats(0)
        result["cuda_native"], gpu_values = run(gpu, "cuda:0")
        result["cuda_native"].update(peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                                     peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20)
        result["cpu_gpu_max_absolute_difference"] = float(np.abs(cpu_values - gpu_values).max())
        result["cpu_gpu_note"] = "Cross-device RNG/numerics may differ; only same-device repeated inference is asserted equal. No forecast is used as a financial result."
        result["warm_speedup_cpu_over_cuda"] = result["cpu_native"]["warm_mean_seconds"] / result["cuda_native"]["warm_mean_seconds"]
        del gpu, cpu, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        result["status"], code = "passed", 0
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "model_key", "packages", "cuda_build", "device_name", "error", "warm_speedup_cpu_over_cuda") if k in result}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
