"""Export verified own v22 models to XGBoost's portable UBJ, no inference/training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import joblib
import numpy as np
import xgboost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((args.package/'shadow_lock.json').read_text())
    model = args.package/'models/xgboost_long_risk_gate_v22_weekly.joblib'
    assert hashlib.sha256(model.read_bytes()).hexdigest() == lock['model_sha256']
    bundle = joblib.load(model)
    metadata = {k: v for k, v in bundle.items() if k != 'pairs'}
    metadata.update(source_joblib_sha256=lock['model_sha256'], export_xgboost_version=xgboost.__version__, pairs={})
    with tarfile.open(args.output, 'w:gz') as archive:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
        for pair, item in bundle['pairs'].items():
            converted = {k: v for k, v in item.items() if k != 'weeks'}
            converted['weeks'] = []
            for week in item['weeks']:
                raw = bytes(week['model'].get_booster().save_raw(raw_format='ubj'))
                assert hashlib.sha256(raw).hexdigest() == week['model_sha256']
                filename = f"{pair}/{week['fold']}.ubj"
                add(filename, raw)
                # Deterministic probe permits a cross-platform numerical parity check.
                probe = np.zeros((1, len(item['features'])), dtype=np.float32)
                probability = float(week['model'].predict_proba(probe, validate_features=False)[0, 1])
                converted['weeks'].append({**{k: v for k, v in week.items() if k != 'model'},
                                            'ubj_file': filename, 'zero_probe_probability': probability})
            metadata['pairs'][pair] = converted
        add('portable_manifest.json', json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    print(json.dumps({'exported_pairs': list(metadata['pairs']), 'weeks_per_pair': len(converted['weeks']),
                      'source_model_sha256': lock['model_sha256'], 'archive_bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
