"""Thin inference-only calls into pinned, unmodified official model sources."""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch


def parameter_digest(modules):
    digest = hashlib.sha256()
    for prefix, module in modules.items():
        for key, tensor in sorted(module.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update((prefix + "." + key).encode())
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def date_seed(model_key, seed, date, asset_keys):
    """One fixed same-day batch, reused by both historical pool variants."""
    key = f"{model_key}|{seed}|{pd.Timestamp(date).isoformat()}|" + ",".join(asset_keys)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % (2**31)


class NativeForecaster:
    def __init__(self, model_key, config, *, threads=4):
        self.model_key, self.config = model_key, config
        self.device = config.get("device", "cpu")
        if self.device.startswith("cuda"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            if not torch.cuda.is_available():
                raise RuntimeError("The predeclared CUDA inference device is unavailable")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        vendor = Path(config["vendor_path"])
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        if model_key.startswith("kronos"):
            module = importlib.import_module("model")
            if not Path(module.__file__).resolve().is_relative_to(vendor.resolve()):
                raise AssertionError("Unpinned Kronos import")
            self.tokenizer = module.KronosTokenizer.from_pretrained(config["tokenizer_path"])
            self.model = module.Kronos.from_pretrained(config["weights_path"])
            self.tokenizer.eval()
            self.model.eval()
            self.predictor = module.KronosPredictor(self.model, self.tokenizer, device=self.device, max_context=128)
            self.modules = {"tokenizer": self.tokenizer, "model": self.model}
        else:
            module = importlib.import_module("chronos")
            if not Path(module.__file__).resolve().is_relative_to(vendor.resolve()):
                raise AssertionError("Unpinned Chronos import")
            self.pipeline = module.Chronos2Pipeline.from_pretrained(
                config["weights_path"], device_map=self.device, torch_dtype=torch.float32)
            self.pipeline.model.eval()
            self.modules = {"model": self.pipeline.model}
            if .5 not in self.pipeline.quantiles:
                raise AssertionError("The native median forecast is unavailable")
        self.initial_parameter_sha256 = parameter_digest(self.modules)

    def _chronos_results(self, frames):
        """Native quantiles; optional Windows-only synchronous loading scheduler.

        The official source is untouched. Its one-worker DataLoader otherwise
        spawns a Python interpreter for each one-day call on Windows. This
        process-local factory changes loading, not input ordering or inference.
        """
        inputs = [{"target": frame.close.to_numpy(dtype=np.float32, copy=True),
                   "past_covariates": {field: frame[field].to_numpy(dtype=np.float32, copy=True)
                                       for field in ("open", "high", "low", "volume", "amount")}}
                  for frame in frames]
        module = importlib.import_module(self.pipeline.__class__.__module__)
        synchronous = self.config.get("synchronous_chronos_loader", False)
        original_loader = getattr(module, "DataLoader", None)
        if synchronous:
            if original_loader is None:
                raise AssertionError("Pinned Chronos loader factory is unavailable")
            def synchronous_loader(*args, **kwargs):
                return original_loader(*args, **{**kwargs, "num_workers": 0, "pin_memory": False})
            module.DataLoader = synchronous_loader
        try:
            with torch.inference_mode():
                return self.pipeline.predict(inputs, prediction_length=1, context_length=128,
                                             batch_size=64, predict_batches_jointly=False)
        finally:
            if synchronous:
                module.DataLoader = original_loader

    def predict(self, frames, date, *, seed=0, asset_keys):
        if not frames:
            return [], None
        if len(frames) != len(asset_keys) or asset_keys != sorted(asset_keys, key=int):
            raise ValueError("Same-day forecast batch must have fixed stable asset ordering")
        seed_value = date_seed(self.model_key, seed, date, asset_keys)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        with torch.inference_mode():
            if self.model_key.startswith("kronos"):
                x_times = [pd.Series(frame.index.tz_localize(None)) for frame in frames]
                y_times = [pd.Series([pd.Timestamp(date).tz_localize(None)]) for _ in frames]
                results = self.predictor.predict_batch(
                    frames, x_times, y_times, pred_len=1, T=1., top_p=.9, top_k=0,
                    sample_count=1, verbose=False)
                predictions = []
                for result in results:
                    if len(result) != 1:
                        raise AssertionError("Kronos returned the wrong forecast horizon")
                    predictions.append({"predicted_" + str(k): float(v) for k, v in result.iloc[0].items()})
            else:
                results = self._chronos_results(frames)
                median = self.pipeline.quantiles.index(.5)
                predictions = [{"predicted_close": float(result[0, median, 0])} for result in results]
        if len(predictions) != len(frames) or any(not np.isfinite(row["predicted_close"]) for row in predictions):
            raise AssertionError("Native forecasts are missing or nonfinite")
        return predictions, seed_value

    def verify_unchanged(self):
        current = parameter_digest(self.modules)
        if current != self.initial_parameter_sha256:
            raise AssertionError("Pretrained weights changed during inference")
        return current
