"""Deploy only the minimal DCA Guard overlay, preserving live configuration."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/ubuntu/extra_drive/hummingbot')
ROLLOUT = ROOT / '.reentry-rollouts/20261021-recovery-v1'


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def environment(value):
    return dict(item.split('=', 1) for item in value['Config']['Env'])


def main():
    current = inspect('dca-live-guard')
    baseline = json.loads((ROLLOUT / 'backup/containers.json').read_text())
    assert current['Id'] == baseline['dca-live-guard']['id'], 'Guard changed since preview'
    env = dict(os.environ)
    env['DCA_PRESERVED_PAPER_TELEGRAM'] = environment(current)['STOCKS_PAPER_TELEGRAM_TRADING_ENABLED']
    env['GRID_PRESERVED_PAPER_TELEGRAM'] = environment(inspect('grid-live-guard'))['STOCKS_PAPER_TELEGRAM_TRADING_ENABLED']
    report = inspect('dca-live-report')
    env['CARDPUTER_REPORT_IMAGE'] = report['Config']['Image']
    destinations = {m['Destination']: m['Source'] for m in report['Mounts']}
    env['CARDPUTER_REPORT_CREDENTIALS_PATH'] = destinations['/run/secrets/cardputer_report_mqtt']
    env['CARDPUTER_REPORT_CA_PATH'] = destinations['/etc/cardputer/isrg-root-x1.pem']
    files = current['Config']['Labels']['com.docker.compose.project.config_files'].split(',')
    command = ['docker', 'compose', '-p', 'hummingbot']
    for path in files + [str(ROLLOUT / 'context/compose.reentry.yml')]:
        assert Path(path).is_file(), path
        command += ['-f', path]
    result = subprocess.run(command + ['config', '--format', 'json'], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=True)
    configured = json.loads(result.stdout)['services']['dca-live-guard']
    actual = environment(current)
    differences = [key for key, value in configured['environment'].items()
                   if str(value or '') != actual.get(key, '')]
    assert not differences, f'Unexpected environment changes: {differences}'
    expected_mounts = {(m['Source'], m['Destination'], bool(m['RW'])) for m in current['Mounts']}
    shutil.copy2(ROLLOUT / 'context/live_guard/dca_reentry.py', ROOT / 'live_guard/dca_reentry.py')
    result = subprocess.run(command + ['up', '-d', '--no-deps', '--no-build', 'dca-live-guard'],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    print(result.stdout[-1000:], result.stderr[-1500:])
    result.check_returncode()
    after = inspect('dca-live-guard')
    assert expected_mounts == {(m['Source'], m['Destination'], bool(m['RW'])) for m in after['Mounts']}
    assert environment(after) == actual, 'Guard environment changed'
    print('DCA Guard deployed; environment and mounts unchanged.')
    for name, old in baseline.items():
        if name != 'dca-live-guard':
            live = inspect(name)
            assert live['Id'] == old['id'] and live['State']['StartedAt'] == old['started_at'], name
    print('Non-target containers unchanged.')


if __name__ == '__main__':
    main()
