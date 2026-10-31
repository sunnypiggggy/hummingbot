"""Freeze public checkpoint/source provenance without generating market forecasts."""
from __future__ import annotations
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from prepare_crypto_foundation_sources import OUT, ROOT, get_json, digest, write_json

ORIGINAL = {
    'kronos_small': 'ac5c409a313c4eadd1ca78201322f5cadb9c34ab',
    'kronos_base': '31afc02ed0a3cfaa1a97238c8b948d663e45b597',
    'kronos_tokenizer': '9ef143b98ee3c2488eebd85404e0c215c112b46a',
}


def main():
    model_manifest = json.loads((OUT / 'weights_manifest.json').read_text(encoding='utf-8'))
    source_manifest = json.loads((OUT / 'source_manifest.json').read_text(encoding='utf-8'))
    models = {x['name']: x for x in model_manifest['models']}
    sources = {x['id']: x for x in source_manifest['sources']}
    comparisons = []
    for name, original in ORIGINAL.items():
        model = models[name]
        old_info = get_json(f'https://huggingface.co/api/models/{model["hf_repository"]}/revision/{original}?blobs=true')
        write_json(OUT / (name + '_original_upload_hf_revision.json'), old_info)
        old = {x['rfilename']: x for x in old_info['siblings']}
        for file in model['files']:
            if file['path'] not in ('config.json', 'model.safetensors'):
                continue
            expected = old[file['path']]
            if expected.get('lfs'):
                assert expected['lfs']['sha256'] == file['sha256']
            else:
                assert expected['blobId'] == file['hf_git_blob_sha1']
            comparisons.append({'model': name, 'path': file['path'], 'original_revision': original,
                'selected_revision': model['hf_revision'], 'unchanged_since_original_20250630_upload': True})
    write_json(OUT / 'original_upload_byte_comparisons.json', {'checks': comparisons, 'passed': True})
    availability = [
        {'id': 'kronos', 'commit': '6f586491c8dc53f2046cff0b8f5ce0e81663068e',
         'commit_time': '2025-07-11T12:18:41Z', 'filename': 'kronos_20250711_README.md',
         'url': 'https://github.com/shiyu-coder/Kronos/blob/6f586491c8dc53f2046cff0b8f5ce0e81663068e/README.md',
         'first_usable_decision_utc': '2025-07-12T00:00:00Z'},
        {'id': 'chronos2', 'commit': '5e3a6a1f137bc8a5546e05a6f7da6d470cef21b5',
         'commit_time': '2025-10-20T12:56:55Z', 'filename': 'chronos2_20251020_README.md',
         'url': 'https://github.com/amazon-science/chronos-forecasting/blob/5e3a6a1f137bc8a5546e05a6f7da6d470cef21b5/README.md',
         'first_usable_decision_utc': '2025-10-21T00:00:00Z'},
    ]
    for item in availability:
        path = OUT / item['filename']
        data = base64.b64decode(path.with_suffix(path.suffix + '.b64').read_text(encoding='utf-8'))
        if path.exists():
            assert path.read_bytes() == data
        else:
            path.write_bytes(data)
        item['sha256'] = digest(path)
        item['git_blob_sha1'] = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        body = data.decode('utf-8')
        assert ('s3://autogluon/chronos-2' in body if item['id'] == 'chronos2'
                else all(name in body for name in ('NeoQuasar/Kronos-small', 'NeoQuasar/Kronos-base', 'NeoQuasar/Kronos-Tokenizer-base')))
    write_json(OUT / 'availability_evidence.json', {
        'basis': 'Pinned historical official README linking to public checkpoints; conservatively start at next full UTC decision day. Commit metadata alone is not an independent full historical archive of repository visibility.',
        'sources': availability,
        'chronos_hf_mirror_note': 'HF amazon/chronos-2 was first uploaded 2025-10-30; original official distribution was public S3. Downloaded S3 bytes exactly match that first HF LFS SHA256; S3 headers show Last-Modified 2025-10-18, no version ID.',
        'exact_pretraining_dataset_cutoff_independently_verified': False,
        'test_window_begins_after_all_verified_public_notices': True})
    zip_path = OUT / 'official_inference_sources.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in source_manifest['sources']:
            for file in sorted(source['files'], key=lambda f: f['path']):
                path = Path(file['local_path'])
                assert digest(path) == file['sha256']
                entry = zipfile.ZipInfo(source['id'] + '/' + file['path'], date_time=(2025, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    write_json(OUT / 'source_archive_manifest.json', {'path': str(zip_path), 'sha256': digest(zip_path),
               'bytes': zip_path.stat().st_size, 'members': sum(len(s['files']) for s in source_manifest['sources']),
               'includes_licenses': True, 'algorithm_source_modified': False})
    rows = []
    for name in ('kronos_small', 'kronos_base', 'chronos2'):
        model = models[name]
        source = sources['chronos2' if name == 'chronos2' else 'kronos']
        weights = next(f for f in model['files'] if f['path'] == 'model.safetensors')
        notice = next(a for a in availability if a['id'] == source['id'])
        row = {'model_key': name, 'model': {'kronos_small': 'Kronos-small', 'kronos_base': 'Kronos-base', 'chronos2': 'Chronos-2'}[name],
            'predictor_parameters': {'kronos_small': 24741376, 'kronos_base': 102310592, 'chronos2': 119477664}[name],
            'source_repository': 'https://github.com/' + source['repository'], 'source_commit': source['commit'],
            'source_commit_time': source['commit_date'], 'source_vendor_root': source['vendor_path'],
            'vendor_path': str(Path(source['vendor_path']) / 'src') if name == 'chronos2' else source['vendor_path'],
            'weights_path': model['local_directory'], 'weights_repository': 'https://huggingface.co/' + model['hf_repository'],
            'weights_revision': model['hf_revision'], 'weights_revision_time': model['hf_revision_date'],
            'weights_sha256': weights['sha256'], 'weights_bytes': weights['size_bytes'],
            'first_verified_public_notice_utc': notice['commit_time'],
            'first_usable_decision_utc': notice['first_usable_decision_utc'],
            'public_notice_source': notice['url'], 'availability_basis': notice['filename'],
            'validation_before_availability': 'USDT cash only, no retroactive checkpoint use',
            'local_fit': False, 'pretraining_cutoff_independently_verified': False,
            'currently_declared_license': 'Apache-2.0' if name == 'chronos2' else 'MIT',
            'license_source': 'https://huggingface.co/' + model['hf_repository'],
            'checkpoint_dates_note': model['provenance_note'],
            'task': 'Native one-day OHLCVA candlestick forecast' if name.startswith('kronos') else 'Native one-day close median with five past-only OHLCVA covariates; independent task groups, no future covariates'}
        if name.startswith('kronos'):
            tokenizer = models['kronos_tokenizer']
            token_weights = next(f for f in tokenizer['files'] if f['path'] == 'model.safetensors')
            row.update(tokenizer_path=tokenizer['local_directory'], tokenizer_revision=tokenizer['hf_revision'],
                       tokenizer_sha256=token_weights['sha256'], tokenizer_parameters_additional=True)
        rows.append(row)
    write_json(OUT.parent / 'model_provenance.json', {'source': {
        'label': 'Pinned official pretrained models, immutable byte checks and pre-test public availability evidence',
        'classification': 'source_review', 'timezone': 'UTC',
        'files': [str(OUT / name) for name in ('source_manifest.json', 'weights_manifest.json', 'availability_evidence.json',
                  'original_upload_byte_comparisons.json', 'source_archive_manifest.json', 'official_inference_sources.zip', 'runtime_environment.json')],
        'caveats': ['Exact proprietary/aggregate pretraining corpus cutoff is not fully independently disclosed; public artifact availability predates the 365-day test, but is not a complete audit of all training data.',
                    'Chronos HF mirror postdates test start; original October18 S3 bytes are authenticated by full SHA256 matching and October20 official README, not by treating current HF update time as original release.',
                    'No model is fit or fine-tuned locally; forecasts still need separate fixed portfolio rules and after-cost ledger validation.',
                    'Current declared weight licenses are linked to official model cards; historical availability and contemporary license metadata are distinct evidence.']},
        'rows': rows})
    print(json.dumps({'status': 'passed', 'source_archive_sha256': digest(zip_path),
          'weights': {r['model_key']: r['weights_sha256'] for r in rows}, 'rows': len(rows)}))


if __name__ == '__main__':
    main()
