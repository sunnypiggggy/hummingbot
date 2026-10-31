"""Synthetic-only checks through the actual study inference wrapper."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_crypto_foundation_models import synthetic_frames
from scripts.foundation_model_native import NativeForecaster


def frame_digest(frames):
    return hashlib.sha256(b''.join(frame.to_numpy().tobytes() + frame.index.asi8.tobytes() for frame in frames)).hexdigest()


def main():
    sources = ROOT / 'results/backtests/crypto_foundation_models_20261024/sources'
    source = json.loads((sources / 'source_manifest.json').read_text(encoding='utf-8'))
    weights = json.loads((sources / 'weights_manifest.json').read_text(encoding='utf-8'))
    source = {r['id']: r for r in source['sources']}
    weights = {r['name']: r for r in weights['models']}
    frames = synthetic_frames(10)
    for frame in frames:
        frame.index = pd.date_range('2024-01-01', periods=128, freq='D', tz='UTC')
    initial = frame_digest(frames)
    keys = [str(i) for i in range(1, 11)]
    date = frames[0].index[-1] + pd.Timedelta(days=1)
    results = []
    for name in ('kronos_small', 'kronos_base', 'chronos2'):
        config = {'device': 'cpu', 'weights_path': weights[name]['local_directory']}
        if name.startswith('kronos'):
            config['vendor_path'] = source['kronos']['vendor_path']
            config['tokenizer_path'] = weights['kronos_tokenizer']['local_directory']
        else:
            config['vendor_path'] = str(Path(source['chronos2']['vendor_path']) / 'src')
        model = NativeForecaster(name, config, threads=2)
        started = time.monotonic()
        prediction, seed = model.predict(frames, date, seed=19, asset_keys=keys)
        elapsed = time.monotonic() - started
        repeat, repeat_seed = model.predict(frames, date, seed=19, asset_keys=keys)
        assert prediction == repeat and seed == repeat_seed
        assert frame_digest(frames) == initial, 'A native predictor mutated its historical input data'
        isolated = None
        if name == 'chronos2':
            modified = [f.copy(deep=True) for f in frames]
            modified[0].loc[:, ['open', 'high', 'low', 'close']] += 1000
            changed, _ = model.predict(modified, date, seed=19, asset_keys=keys)
            isolated = changed[1:] == prediction[1:]
            assert isolated, 'Independent Chronos task groups shared cross-asset information'
        parameter_sha256 = model.verify_unchanged()
        row = {'model': name, 'status': 'passed', 'synthetic_only': True, 'native_wrapper': 'scripts/foundation_model_native.py',
               'batch': 10, 'context_days': 128, 'horizon_days': 1, 'threads': 2,
               'first_inference_seconds': elapsed, 'same_seed_bitwise_repeat': True,
               'historical_frames_unchanged': True, 'parameter_sha256': parameter_sha256,
               'chronos_independent_groups_unchanged_when_another_asset_perturbed': isolated}
        results.append(row)
        print(json.dumps(row), flush=True)
    (sources / 'native_wrapper_synthetic_smoke.json').write_text(json.dumps({'status': 'passed', 'results': results}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
