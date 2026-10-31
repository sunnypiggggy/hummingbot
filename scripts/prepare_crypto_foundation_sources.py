"""Download and verify public pre-test model/source evidence, never run backtests."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/backtests/crypto_foundation_models_20261024/sources'
CACHE = Path('C:/Users/sunny/.cache/codex-research-vendors')
MODELS = {
    'kronos_small': ('NeoQuasar/Kronos-small', '901c26c1332695a2a8f243eb2f37243a37bea320'),
    'kronos_base': ('NeoQuasar/Kronos-base', '2b554741eca47781b64468546e77fef3e85130e6'),
    'kronos_tokenizer': ('NeoQuasar/Kronos-Tokenizer-base', '0e0117387f39004a9016484a186a908917e22426'),
    'chronos2': ('amazon/chronos-2', '95a9710e2596287d08352589f42634fa5abdf0a7'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def get_json(url):
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(15, 45))
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass
        time.sleep(attempt + 1)
    raise RuntimeError('Public metadata request failed after five bounded attempts')


def ranged_s3_download(url, path, expected_sha, expected_size):
    path = Path(path)
    if path.exists():
        if path.stat().st_size != expected_size or digest(path) != expected_sha:
            raise ValueError('Preserve existing unexpected weights: ' + str(path))
        return
    chunk_size = 8 * 1024 * 1024
    chunks = path.parent / (path.name + '.chunks')
    chunks.mkdir(parents=True, exist_ok=True)

    def part_task(offset):
        end = min(offset + chunk_size, expected_size) - 1
        target = chunks / (str(offset).zfill(12) + '.part')
        if target.exists() and target.stat().st_size == end - offset + 1:
            return target
        for attempt in range(5):
            try:
                with requests.get(url, headers={'Range': f'bytes={offset}-{end}'}, stream=True,
                                  timeout=(15, 45)) as response:
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{expected_size}':
                        raise RuntimeError('Public S3 range response did not match requested interval')
                    partial = target.with_suffix('.partial')
                    with partial.open('wb') as stream:
                        for data in response.iter_content(512 * 1024):
                            if data:
                                stream.write(data)
                    if partial.stat().st_size != end - offset + 1:
                        raise RuntimeError('Public S3 range size mismatch')
                    partial.rename(target)
                    print(f'RANGE {path.parent.name}: {end + 1}/{expected_size}', flush=True)
                    return target
            except (requests.RequestException, RuntimeError):
                time.sleep(attempt + 1)
        raise RuntimeError('Public S3 range retries exhausted at offset ' + str(offset))

    offsets = list(range(0, expected_size, chunk_size))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(part_task, offset) for offset in offsets]):
            future.result()
    partial = path.with_suffix(path.suffix + '.download')
    with partial.open('wb') as stream:
        for offset in offsets:
            with (chunks / (str(offset).zfill(12) + '.part')).open('rb') as part:
                for data in iter(lambda: part.read(1024 * 1024), b''):
                    stream.write(data)
    if digest(partial) != expected_sha:
        raise AssertionError('Concatenated original S3 bytes differ from official HF LFS SHA256')
    partial.rename(path)


def download(url, path, expected_sha=None, expected_size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError('Preserve an existing unexpected-size artifact: ' + str(path))
        if expected_sha is not None and digest(path) != expected_sha:
            raise ValueError('Preserve an existing mismatched artifact: ' + str(path))
        return
    partial = path.with_suffix(path.suffix + '.download')
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        if response.status_code != 200:
            # Do not print the redirect's temporary signed URL.
            raise RuntimeError(f'Public artifact download failed status={response.status_code}, target={path.name}')
        received, next_notice = 0, 64 * 1024 * 1024
        with partial.open('wb') as stream:
            for part in response.iter_content(8 * 1024 * 1024):
                if part:
                    stream.write(part)
                    received += len(part)
                    if received >= next_notice:
                        print(f'DOWNLOAD {path.parent.name}/{path.name}: {received} bytes', flush=True)
                        next_notice += 64 * 1024 * 1024
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise AssertionError('Downloaded size does not match official metadata: ' + str(path))
    if expected_sha is not None and digest(partial) != expected_sha:
        raise AssertionError('Downloaded bytes do not match official LFS SHA256: ' + str(path))
    partial.rename(path)


def source_task(spec):
    vendor = CACHE / (spec['id'] + '-' + spec['commit'][:12])
    records = []
    for row in spec['files']:
        relative = row['path']
        path = vendor / relative
        url = f'https://raw.githubusercontent.com/{spec["repository"]}/{spec["commit"]}/{relative}'
        connector_blob = OUT / 'connector_blobs' / spec['id'] / (relative + '.b64')
        if connector_blob.exists():
            exact = base64.b64decode(connector_blob.read_text(encoding='utf-8'))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != exact:
                raise ValueError('Preserve unexpected existing source bytes: ' + str(path))
            if not path.exists():
                path.write_bytes(exact)
        else:
            download(url, path, expected_size=row['size'])
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != row['git_blob_sha1']:
            raise AssertionError('Official GitHub blob mismatch: ' + relative)
        records.append({**row, 'sha256': digest(path), 'local_path': str(path), 'official_url': url})
    return {**spec, 'vendor_path': str(vendor), 'files': records, 'algorithm_source_modified': False}


def model_task(item):
    name, (repo, revision) = item
    commits_path, info_path = OUT / (name + '_hf_commits.json'), OUT / (name + '_hf_revision.json')
    commits = json.loads(commits_path.read_text(encoding='utf-8')) if commits_path.exists() else get_json(f'https://huggingface.co/api/models/{repo}/commits/main')
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else get_json(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true')
    write_json(OUT / (name + '_hf_commits.json'), commits)
    write_json(OUT / (name + '_hf_revision.json'), info)
    selected = next(row for row in commits if row['id'] == revision)
    folder = CACHE / 'foundation-model-weights' / (name + '-' + revision[:12])
    records = []
    for row in info['siblings']:
        relative = row['rfilename']
        if relative not in ('model.safetensors', 'config.json', 'README.md', 'LICENSE'):
            continue
        path = folder / relative
        # Original Chronos-2 public distribution was S3, preceding its HF mirror.
        if name == 'chronos2' and relative in ('model.safetensors', 'config.json'):
            url = 'https://autogluon.s3.amazonaws.com/chronos-2/' + relative
            head = requests.head(url, timeout=(20, 60))
            if head.status_code != 200:
                raise RuntimeError('Original public S3 object no longer readable')
            header = {key: head.headers.get(key) for key in ('Last-Modified', 'Content-Length', 'ETag', 'x-amz-version-id')}
            write_json(OUT / ('chronos2_s3_' + relative.replace('.', '_') + '_head.json'), {'url': url, 'headers': header, 'status': head.status_code})
        else:
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{relative}'
            header = None
        expected = row.get('lfs', {}).get('sha256')
        if name == 'chronos2' and relative == 'model.safetensors':
            ranged_s3_download(url, path, expected, row['size'])
        else:
            download(url, path, expected_sha=expected, expected_size=row['size'])
        if not row.get('lfs'):
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != row['blobId']:
                raise AssertionError('Official HF git blob mismatch: ' + relative)
        records.append({'path': relative, 'local_path': str(path), 'size_bytes': path.stat().st_size,
                        'sha256': digest(path), 'expected_lfs_sha256': expected,
                        'hf_git_blob_sha1': row['blobId'], 'source_url': url, 's3_head': header})
    record = {'name': name, 'hf_repository': repo, 'hf_revision': revision,
            'hf_revision_date': selected['date'], 'local_directory': str(folder), 'files': records,
            'pretest_hf_revision': selected['date'] < '2025-10-23',
            'provenance_note': 'Original S3 object Last-Modified 2025-10-18; exact bytes matched to first official HF weights via LFS SHA256. Test-before availability is supported by the pre-test source README linking S3. No S3 version ID, and exact pretraining data cutoff not fully independently verified.' if name == 'chronos2' else 'Public HF model weights uploaded 2025-06-30; pinned revision existed before the test. Exact corpus cutoff not independently verified.'}
    write_json(OUT / (name + '_weights_manifest.json'), record)
    print('VERIFIED MODEL ' + name, flush=True)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = json.loads((OUT / 'github_source_trees.json').read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        source_records = list(pool.map(source_task, specs))
    write_json(OUT / 'source_manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'sources': source_records})
    models = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(model_task, item) for item in MODELS.items()]):
            record = future.result()
            models.append(record)
            write_json(OUT / (record['name'] + '_weights_manifest.json'), record)
            print('VERIFIED MODEL ' + record['name'], flush=True)
    write_json(OUT / 'weights_manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(),
               'models': sorted(models, key=lambda x: x['name']), 'weights_updated_or_trained_locally': False})
    print('SOURCE AND WEIGHTS READY', flush=True)


if __name__ == '__main__':
    main()
