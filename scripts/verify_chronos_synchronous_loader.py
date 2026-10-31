"""Real-weight, synthetic-only check of Windows synchronous loading schedule."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_crypto_foundation_models import synthetic_frames
from scripts.foundation_model_native import NativeForecaster

OUTPUT = ROOT / "results/backtests/crypto_foundation_models_20261024"


def digest_frames(frames):
    return hashlib.sha256(b"".join(frame.to_numpy().tobytes() + frame.index.asi8.tobytes()
                                  for frame in frames)).hexdigest()


def main():
    source = json.loads((OUTPUT / "sources/source_manifest.json").read_text(encoding="utf-8"))
    weights = json.loads((OUTPUT / "sources/weights_manifest.json").read_text(encoding="utf-8"))
    source = next(row for row in source["sources"] if row["id"] == "chronos2")
    weight = next(row for row in weights["models"] if row["name"] == "chronos2")
    vendor = Path(source["vendor_path"])
    if vendor.name != "src":
        vendor /= "src"
    config = {"device": "cpu", "vendor_path": str(vendor),
              "weights_path": weight["local_directory"], "synchronous_chronos_loader": False}
    frames = synthetic_frames(10)
    for frame in frames:
        frame.index = pd.date_range("2024-01-01", periods=128, freq="D", tz="UTC")
    original_input = digest_frames(frames)
    model = NativeForecaster("chronos2", config, threads=2)
    module = importlib.import_module(model.pipeline.__class__.__module__)
    loader = module.DataLoader
    timings, outputs = {}, {}
    for mode in ("official_worker_loader", "windows_synchronous_loader", "synchronous_repeat"):
        model.config["synchronous_chronos_loader"] = mode != "official_worker_loader"
        started = time.monotonic()
        results = model._chronos_results(frames)
        timings[mode] = time.monotonic() - started
        outputs[mode] = np.stack([value.detach().cpu().numpy() for value in results])
        assert module.DataLoader is loader, "Official loader factory was not restored"
        assert digest_frames(frames) == original_input, "The inference mutated historical inputs"
    values = outputs["official_worker_loader"]
    assert values.shape == (10, 1, 21, 1), f"Unexpected native quantile tensor: {values.shape}"
    assert np.isfinite(values).all()
    for candidate in outputs.values():
        assert np.array_equal(values, candidate), "Loading schedule altered native forecast quantiles"
    date = frames[0].index[-1] + pd.Timedelta(days=1)
    predicted, _ = model.predict(frames, date, seed=0, asset_keys=[str(i) for i in range(1, 11)])
    median = model.pipeline.quantiles.index(.5)
    assert [row["predicted_close"] for row in predicted] == values[:, 0, median, 0].tolist()
    parameter_digest = model.verify_unchanged()
    result = {"status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "synthetic_only": True, "real_market_forecasts_generated": False,
              "model": "chronos2", "source_commit": source["commit"],
              "device": "cpu", "threads": 2, "batch": 10, "context_days": 128, "horizon_days": 1,
              "prediction_shape": list(values.shape), "quantiles": list(model.pipeline.quantiles),
              "all_native_quantiles_bitwise_equal": True, "native_median_wrapper_equal": True,
              "repeat_max_absolute_difference": 0., "historical_inputs_unchanged": True,
              "parameters_unchanged": True, "parameter_sha256": parameter_digest,
              "timings_seconds": timings, "official_loader_restored": True,
              "official_source_modified": False,
              "adaptation": "Windows synchronous loading scheduler: factory kwargs num_workers=0,pin_memory=False only; official model/input ordering/quantiles unchanged",
              "adapter_source_sha256": hashlib.sha256((ROOT / "scripts/foundation_model_native.py").read_bytes()).hexdigest(),
              "validation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = OUTPUT / "sources/chronos_synchronous_loader_synthetic.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
