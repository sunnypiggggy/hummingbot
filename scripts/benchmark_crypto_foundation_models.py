"""Synthetic-only CPU import/inference compatibility checks for pinned weights."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'results/backtests/crypto_foundation_models_20261024/sources'
CACHE = Path('C:/Users/sunny/.cache/codex-research-vendors')
NAMES = {'kronos_small': 'kronos_small-901c26c13326', 'kronos_base': 'kronos_base-2b554741eca4',
         'chronos2': 'chronos2-95a9710e2596'}


def synthetic_frames(count):
    rng = np.random.default_rng(1729)
    frames = []
    for index in range(count):
        close = (100 + index) * np.exp(np.cumsum(rng.normal(0, .01, 128)))
        opening = np.r_[close[0], close[:-1]]
        volume = rng.uniform(100, 1000, 128)
        frames.append(pd.DataFrame({'open': opening, 'high': np.maximum(opening, close) * 1.01,
            'low': np.minimum(opening, close) * .99, 'close': close, 'volume': volume,
            'amount': volume * close}))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(NAMES), required=True)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    frames = synthetic_frames(args.batch)
    times = pd.Series(pd.date_range('2024-01-01', periods=128, freq='D', tz='UTC'))
    future_times = pd.Series([times.iloc[-1] + pd.Timedelta(days=1)])
    source_specs = json.loads((SOURCES / 'source_manifest.json').read_text(encoding='utf-8'))['sources']
    sources = {x['id']: x for x in source_specs}
    folder = CACHE / 'foundation-model-weights' / NAMES[args.model]
    if args.model.startswith('kronos'):
        sys.path.insert(0, sources['kronos']['vendor_path'])
        from model import Kronos, KronosTokenizer, KronosPredictor
        tokenizer = KronosTokenizer.from_pretrained(CACHE / 'foundation-model-weights/kronos_tokenizer-0e0117387f39', local_files_only=True)
        model = Kronos.from_pretrained(folder, local_files_only=True)
        model.eval()
        tokenizer.eval()
        predictor = KronosPredictor(model, tokenizer, device='cpu', max_context=128)

        def forward():
            result = predictor.predict_batch(frames, [times] * args.batch, [future_times] * args.batch,
                pred_len=1, T=1., top_k=0, top_p=.9, sample_count=1, verbose=False)
            return np.stack([frame.to_numpy() for frame in result])
        model_params = sum(x.numel() for x in model.parameters())
    else:
        sys.path.insert(0, str(Path(sources['chronos2']['vendor_path']) / 'src'))
        from chronos import Chronos2Pipeline
        predictor = Chronos2Pipeline.from_pretrained(folder, device_map='cpu', torch_dtype=torch.float32, local_files_only=True)
        predictor.model.eval()
        inputs = [{'target': frame.close.to_numpy(), 'past_covariates': {key: frame[key].to_numpy()
                   for key in ('open', 'high', 'low', 'volume', 'amount')}} for frame in frames]

        def forward():
            result = predictor.predict(inputs, prediction_length=1, context_length=128,
                                       batch_size=64, predict_batches_jointly=False)
            return np.stack([value.detach().cpu().numpy() for value in result])
        model_params = sum(x.numel() for x in predictor.model.parameters())
    duration, outputs = [], []
    for iteration in range(2):
        torch.manual_seed(42)
        np.random.seed(42)
        started = time.monotonic()
        with torch.inference_mode():
            outputs.append(forward())
        duration.append(time.monotonic() - started)
    assert np.isfinite(outputs[0]).all()
    assert np.array_equal(outputs[0], outputs[1]), 'Same seed + inputs must reproduce inference exactly'
    result = {'status': 'passed', 'inputs': 'SYNTHETIC ONLY; no real training/validation/test price observations',
              'model': args.model, 'batch': args.batch, 'context_days': 128, 'horizon_days': 1,
              'threads': args.threads, 'elapsed_seconds': duration, 'prediction_shape': list(outputs[0].shape),
              'parameters': model_params, 'repeat_max_abs_difference': float(np.max(np.abs(outputs[0] - outputs[1]))),
              'native_source_modified': False, 'packages': {name: importlib.metadata.version(name)
              for name in ('torch', 'numpy', 'pandas', 'transformers', 'accelerate', 'huggingface_hub', 'safetensors', 'einops')}}
    (SOURCES / ('synthetic_smoke_' + args.model + '.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
