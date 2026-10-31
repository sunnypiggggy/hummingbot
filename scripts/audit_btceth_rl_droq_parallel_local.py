"""Native local rerun with one explicit stored-POSIX-path interpretation.

The frozen Linux scheduling auditor uses Path for a path recorded in JSON.
Windows formats that one stored path with backslashes. During that original
validator only, interpret precisely that recorded string as PurePosixPath.
Actual filesystem paths, source bytes and every financial check are unchanged.
"""
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
import sys

ANNUAL_SHA = '9340ca8e95b74c0501eba8a75180603a5edc69197a8c56a31461b773571b51e7'
BASE_SHA = '52aace871ee5d04409536b3e1b8d2f13468bb37894099e1a664dd722634732e6'
SCHEDULE_SHA = '2e7f61ff1015b0449c17b19e37836cbba3a6606e3cec40f9eb5b726d65572be7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def stored_posix_performance_path(helper):
    original = helper.verify_performance
    original_path = helper.Path
    def verify(inputs, annex, worker, performance_root):
        recorded = annex['performance_adapter']['path']
        if not isinstance(recorded, str) or not PurePosixPath(recorded).is_absolute():
            raise ValueError('An absolute recorded POSIX path is required')
        def path(value, *args, **kwargs):
            if type(value) is str and value == recorded and not args and not kwargs:
                return PurePosixPath(recorded)
            return original_path(value, *args, **kwargs)
        helper.Path = path
        try:
            return original(inputs, annex, worker, performance_root)
        finally:
            helper.Path = original_path
    helper.verify_performance = verify
    try:
        yield
    finally:
        helper.verify_performance = original
        helper.Path = original_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output-dir', 'reference-main', 'dataset', 'source-root', 'extra-source-root', 'receipt-root', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Retain every prior local audit result')
    source = args.extra_source_root / 'btceth_rl_droq_parallel_annual_audit.py'
    if sha(source) != ANNUAL_SHA:
        raise ValueError('Frozen independent annual auditor required')
    spec = importlib.util.spec_from_file_location('_native_portable_annual_audit', source)
    annual = importlib.util.module_from_spec(spec); sys.modules[spec.name] = annual
    spec.loader.exec_module(annual)
    base = annual.load(args.extra_source_root, 'btceth_rl_droq_parallel_audit.py', BASE_SHA)
    helper = base.load_extra(args.extra_source_root, 'btceth_rl_droq_scheduling_audit.py')
    if sha(helper.__file__) != SCHEDULE_SHA:
        raise ValueError('Frozen original scheduling validator required')
    with stored_posix_performance_path(helper):
        result = annual.audit(args.output_dir, args.reference_main, args.dataset,
            source_root=args.source_root, extra_source_root=args.extra_source_root, receipt_root=args.receipt_root)
    result['local_metadata_portability'] = dict(adapter_file=str(Path(__file__).resolve()),
        adapter_sha256=sha(__file__), scope='stored POSIX performance path only',
        frozen_schedule_sha256=SCHEDULE_SHA, frozen_annual_sha256=ANNUAL_SHA,
        financial_functions_changed=False, filesystem_path_rules_changed=False)
    with args.report.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'checks_total', 'failure_count', 'failures', 'run_count')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
