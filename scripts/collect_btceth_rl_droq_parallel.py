"""SHA-only full snapshot transport for the explicit parallel DroQ study.

This never unpickles data, imports an agent, patches the original collector or
turns the original collision failure into a successful terminal. All work is
in new export/staging/output directories. External evidence paths are mapped
explicitly; basename guessing is forbidden.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import types
import uuid

ROOT = Path(__file__).resolve().parents[1] if globals().get('__file__') not in (None, '<stdin>') else Path.cwd()
TRANSPORT_SHA = '54ef7a4add8a3ca363ef8855989d797047e3a8c4e3f8cf2e6282fb8abc03a9db'
PROTOCOL_SHA = 'e29c93d37571dd917f95b7f517a49ff8dc52cece712f14720ce1ab496ed226da'
AUDIT_FILE = 'independent_parallel_audit.json'
FLAGS = ('designed_after_first_annual', 'test_year_already_seen', 'not_fresh_holdout')
CASES = ('equity_1h.csv.gz', 'fills.csv', 'weights.csv', 'boundary_equity.csv', 'metadata.json', 'fingerprints.json', 'member_targets.csv', 'replay.json')
GROUPS = ('summary.csv', 'daily_equity.csv', 'monthly_returns.csv', 'completion.json')


def sha(path):
    p = Path(path)
    if any(q.is_symlink() or hasattr(q, 'is_junction') and q.is_junction() for q in (p, *p.parents)):
        raise ValueError('Evidence cannot cross links or junctions')
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_transport(path):
    p = Path(path)
    if sha(p) != TRANSPORT_SHA:
        raise ValueError('Use the exact reviewed54ef safe transport')
    module = types.ModuleType('_parallel_droq_frozen_transport'); module.__file__ = str(p.resolve())
    exec(compile(p.read_bytes(), module.__file__, 'exec'), module.__dict__)
    return module


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def full_scope(audit):
    expected = dict(passed=True, status='passed', phase='test', test_complete=True,
        validation_complete=True, training_model_count=5, validation_run_count=24,
        test_run_count=12, baseline_run_count=14, run_count=26, repeat_count=12,
        failure_count=0, failures=[], fresh_holdout_success=False,
        training_budget_multiple=1, accounting_passage_is_not_goal_success=True,
        **{flag: True for flag in FLAGS})
    if any(audit.get(k) != v or isinstance(v, int) and not isinstance(v, bool) and type(audit.get(k)) is not int for k, v in expected.items()):
        raise ValueError('Complete independent5/24/12+14=26 financial audit required')
    if type(audit.get('checks_total')) is not int or audit['checks_total'] <= 0:
        raise ValueError('A positive real financial assertion count is required')
    if not isinstance(audit.get('runs'), dict) or len(audit['runs']) != 12 or not isinstance(audit.get('baseline_runs'), dict) or len(audit['baseline_runs']) != 14:
        raise ValueError('The audited twelve policies and fourteen references must be explicit')


def required_files(family):
    if family not in ('raw', 'cap075'):
        raise ValueError('Only a sealed original budget family is valid')
    names = {'protocol.json', 'protocol_receipt.json', 'selection.json', 'selection_receipt.json',
             'training_completion.json', 'accelerated_fit_process.json', 'accelerated_fit_completion.json',
             'scheduling_annex.json', 'scheduling_audit.fit.json', 'parallel_validation_process.json',
             'validation_completion.json', 'annual_authorization.json', 'test_started.json',
             'parallel_annual_resource_annex.json', 'parallel_test_process.json', 'parallel_test_runner.log',
             'test_completion.json', 'independent_parallel_audit.validation.json'}
    for seed in range(5):
        names.update(f'training/droq__balanced__seed{seed}/{leaf}' for leaf in ('started.json', 'record.json', 'model_tensors.pt', 'model.json', 'training_timeout_transitions.jsonl'))
    for split, families in (('validation', ('raw', 'cap075')), ('test', (family,))):
        if split == 'validation':
            names.update(f'{split}/{leaf}' for leaf in GROUPS[:3])
        for f in families:
            folder = f'{split}/{f}/balanced'
            names.update(f'{folder}/{leaf}' for leaf in GROUPS)
            names.update(f'{folder}/{cost}/{key}/{leaf}' for cost in ('base', 'stress') for key in ('0', '1', '2', '3', '4', 'ensemble') for leaf in CASES)
    return names


def pins(t, root, values, *, required=(), label='Evidence'):
    if not isinstance(values, dict) or not values or not set(required).issubset(values):
        raise ValueError(label + ': incomplete SHA map')
    for name, digest in values.items():
        if sha(t.safe_target(root, name)) != digest:
            raise ValueError(label + ': changed bytes: ' + name)


def guard(root, main, source, *, transport, audit_file=AUDIT_FILE, audit_source=None,
          audit_source_sha=None, external_map=None, extension_root=None, local=False):
    t = transport; root, main, source = map(Path, (root, main, source))
    p = read(t.safe_target(root, 'protocol.json'))
    if sha(root / 'protocol.json') != PROTOCOL_SHA or read(root / 'protocol_receipt.json').get('protocol_sha256') != PROTOCOL_SHA:
        raise ValueError('Original e29 protocol must remain unchanged')
    a_path = t.safe_target(root, audit_file); audit = read(a_path); full_scope(audit)
    selection = read(root / 'selection.json'); chosen = selection['candidate']['family']
    pins(t, root, audit.get('input_sha256'), required=required_files(chosen), label='Complete parallel study')
    pins(t, main, audit.get('main_input_sha256'), required={'frozen_spec.json', 'runtime.json', 'independent_audit.json'}, label='Original references')
    if audit.get('source_sha256') != p.get('source_sha256') or len(p.get('source_sha256', {})) != 9:
        raise ValueError('Original nine financial/learning sources required')
    pins(t, source, audit['source_sha256'], label='Original source')
    external = dict(audit.get('external_input_sha256') or {})
    scheduling = read(root / 'scheduling_audit.fit.json')
    for name, digest in scheduling.get('external_input_sha256', {}).items():
        if name in external and external[name] != digest:
            raise ValueError('Conflicting explicit external SHA evidence')
        external[name] = digest
    extensions = audit.get('extension_source_sha256') or {}
    if not external and not extensions:
        raise ValueError('Explicit new adapter/source SHA map required')
    if audit_source is None or audit_source_sha is None or sha(audit_source) != audit_source_sha:
        raise ValueError('Root must pin the reviewed independent parallel auditor source')
    if local:
        mapping = external_map or {}
        if not (set(external) | set(extensions)).issubset(mapping):
            raise ValueError('Every external path requires an exact explicit local mapping')
        for remote, digest in {**external, **extensions}.items():
            if sha(mapping[remote]) != digest:
                raise ValueError('Mapped external evidence differs: ' + remote)
        if not any(Path(mapping[k]).resolve() == Path(audit_source).resolve() and v == audit_source_sha for k, v in {**external, **extensions}.items()):
            raise ValueError('Mapped auditor source is absent from actual audit pins')
    else:
        for absolute, digest in external.items():
            if not PurePosixPath(absolute).is_absolute() or sha(absolute) != digest:
                raise ValueError('Actual external evidence changed: ' + absolute)
        if extensions:
            if extension_root is None:
                raise ValueError('A declared remote extension root is required for its relative SHA map')
            pins(t, extension_root, extensions, label='Explicit execution-source root')
        absolute_bound = external.get(str(Path(audit_source).resolve())) == audit_source_sha
        relative_bound = extension_root is not None and Path(audit_source).resolve() == t.safe_target(extension_root, Path(audit_source).name).resolve() and extensions.get(Path(audit_source).name) == audit_source_sha
        if not (absolute_bound or relative_bound):
            raise ValueError('Auditor source is not bound by its actual audit')
    for name in ('parallel_test_process.json', 'parallel_validation_process.json'):
        process = read(root / name)
        if process.get('status') != 'completed':
            raise ValueError('Real parallel phase completion required: ' + name)
    return dict(mode='parallel_full', protocol_sha256=PROTOCOL_SHA, audit_file=audit_file,
        audit_sha256=sha(a_path), audit_source_sha256=audit_source_sha, chosen_family=chosen,
        run_count=26, training_models=5, validation_runs=24,
        study_input_sha256=audit['input_sha256'], main_input_sha256=audit['main_input_sha256'],
        source_sha256=audit['source_sha256'], external_input_sha256=external, extension_source_sha256=extensions)


def export(root, main, source, *, transport, audit_file, audit_source, audit_source_sha, export_parent, extension_root=None):
    t = transport; root = Path(root).resolve()
    before_guard = guard(root, main, source, transport=t, audit_file=audit_file, audit_source=audit_source, audit_source_sha=audit_source_sha, extension_root=extension_root)
    before = t.inventory(root); parent = Path(export_parent).resolve()
    if parent == root or parent.is_relative_to(root):
        raise ValueError('Archive must be outside the study')
    destination = t.safe_target(parent, 'parallel-full/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex)
    destination.mkdir(parents=True, exist_ok=False); archive = destination / 'snapshot.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with tarfile.open(mode='w', fileobj=compressed, format=tarfile.PAX_FORMAT) as tar:
            for item in before:
                p = t.safe_target(root, item['path'])
                with os.fdopen(os.open(p, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)), 'rb') as f:
                    info_stat = os.fstat(f.fileno())
                    if not stat.S_ISREG(info_stat.st_mode) or info_stat.st_size != item['size']:
                        raise ValueError('Source file changed during export')
                    info = tarfile.TarInfo(item['path']); info.size = item['size']; info.mode = 0o600; info.mtime = 0
                    tar.addfile(info, f)
    if t.inventory(root) != before or guard(root, main, source, transport=t, audit_file=audit_file, audit_source=audit_source, audit_source_sha=audit_source_sha, extension_root=extension_root) != before_guard:
        raise ValueError('Study changed during export; preserve incomplete export')
    manifest = dict(schema_version=1, created_at=stamp(), remote_root=str(root), mode='parallel_full',
        files=before, file_count=len(before), total_bytes=sum(x['size'] for x in before),
        terminal_guard=before_guard, archive_name=archive.name, archive_sha256=sha(archive), archive_size=archive.stat().st_size)
    t.write_new_json(destination / 'snapshot_manifest.json', manifest)
    return dict(status='complete', export_directory=str(destination), remote_root=str(root), mode='parallel_full',
        archive_path=str(archive), archive_sha256=sha(archive), archive_size=archive.stat().st_size,
        manifest_path=str(destination / 'snapshot_manifest.json'), manifest_sha256=sha(destination / 'snapshot_manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-source', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export', action='store_true'); mode.add_argument('--stream', type=Path); mode.add_argument('--collect', action='store_true')
    for name in ('root', 'main', 'source', 'audit-source', 'output', 'external-map'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--export-parent', required=True)
    parser.add_argument('--audit-file', default=AUDIT_FILE); parser.add_argument('--audit-source-sha')
    parser.add_argument('--alias', default='v100-192-168-8-120'); parser.add_argument('--remote-python', default='/home/sunny/research-envs/btceth-rl-20261029/bin/python')
    parser.add_argument('--remote-transport', default='/home/sunny/research/btceth_rl_20261029/scripts/collect_btceth_rl_results.py')
    parser.add_argument('--remote-root'); parser.add_argument('--remote-main'); parser.add_argument('--remote-source'); parser.add_argument('--remote-audit-source'); parser.add_argument('--remote-extension-root')
    args = parser.parse_args(); t = load_transport(args.transport_source)
    if args.stream:
        p = args.stream.resolve(); parent = Path(args.export_parent).resolve()
        p.relative_to(parent)
        if p.name not in ('snapshot.tar.gz', 'snapshot_manifest.json'):
            raise ValueError('Only the generated archive or manifest can be streamed')
        sha(p)
        with p.open('rb') as f:
            import shutil
            shutil.copyfileobj(f, sys.stdout.buffer, 1024 * 1024)
    elif args.export:
        print(json.dumps(export(args.root, args.main, args.source, transport=t, audit_file=args.audit_file,
            audit_source=args.audit_source, audit_source_sha=args.audit_source_sha, export_parent=args.export_parent, extension_root=args.remote_extension_root)))
    else:
        collect(args, transport=t)


def collect(args, *, transport):
    t = transport; output = Path(args.output)
    if any(q.is_symlink() or hasattr(q, 'is_junction') and q.is_junction() for q in (output, *output.parents)):
        raise ValueError('Snapshot destination cannot cross links or junctions')
    output = output.resolve()
    main = Path(args.main).resolve()
    if output == main or output.is_relative_to(main) or main.is_relative_to(output):
        raise ValueError('Snapshot must be independent of the preserved original reference study')
    if output.exists() or output.is_symlink():
        raise FileExistsError('A complete snapshot needs a new output; never merge into prepared/old results')
    mapping = read(args.external_map)
    payload = Path(__file__).read_bytes()
    shared = ['--transport-source', args.remote_transport, '--root', args.remote_root, '--main', args.remote_main,
              '--source', args.remote_source, '--audit-file', args.audit_file, '--audit-source', args.remote_audit_source,
              '--audit-source-sha', args.audit_source_sha, '--export-parent', str(args.export_parent)]
    if args.remote_extension_root:
        shared.extend(['--remote-extension-root', args.remote_extension_root])
    response = t._remote_call(args.alias, args.remote_python, ['--export', *shared], payload)
    parent = PurePosixPath(str(args.export_parent)); directory = PurePosixPath(response['export_directory'])
    if response.get('status') != 'complete' or response.get('mode') != 'parallel_full' or response.get('remote_root') != args.remote_root or not directory.is_relative_to(parent):
        raise ValueError('Remote export is outside the exact requested root and generated parent')
    for key, leaf in (('archive_path', 'snapshot.tar.gz'), ('manifest_path', 'snapshot_manifest.json')):
        if PurePosixPath(response[key]) != directory / leaf:
            raise ValueError('Remote generated artifact path differs')
    stage = output.parent / (output.name + '.transport-' + uuid.uuid4().hex); stage.mkdir(parents=True, exist_ok=False)
    for key, leaf in (('archive_path', 'snapshot.tar.gz'), ('manifest_path', 'snapshot_manifest.json')):
        t._remote_call(args.alias, args.remote_python, ['--stream', response[key], '--export-parent', str(args.export_parent), '--transport-source', args.remote_transport], payload, destination=stage / leaf)
    manifest = read(stage / 'snapshot_manifest.json')
    if (sha(stage / 'snapshot_manifest.json') != response['manifest_sha256'] or manifest.get('archive_sha256') != response['archive_sha256']
            or manifest.get('archive_size') != response['archive_size'] or manifest.get('remote_root') != args.remote_root):
        raise ValueError('Downloaded remote manifest identity differs')
    extracted = stage / 'verified'; t.verify_extract(stage / 'snapshot.tar.gz', manifest, extracted)
    checked = guard(extracted, args.main, args.source, transport=t, audit_file=args.audit_file,
        audit_source=args.audit_source, audit_source_sha=args.audit_source_sha, external_map=mapping, local=True)
    if checked != manifest['terminal_guard']:
        raise ValueError('Local full evidence guard differs from the retained remote guard')
    if output.exists():
        raise FileExistsError('Output appeared during transport; preserve both trees')
    os.replace(extracted, output)
    t.write_new_json(output / 'parallel_transfer_receipt.json', dict(at=stamp(), remote_root=args.remote_root,
        collector_source_sha256=hashlib.sha256(payload).hexdigest(), reused_transport_sha256=TRANSPORT_SHA,
        manifest_sha256=response['manifest_sha256'], archive_sha256=response['archive_sha256'], terminal_guard=checked,
        snapshot_files=manifest['files'], external_path_map=mapping))
    print(json.dumps(dict(status='collected', output=str(output), files=manifest['file_count'], audit_sha256=checked['audit_sha256'])))


if __name__ == '__main__':
    main()
