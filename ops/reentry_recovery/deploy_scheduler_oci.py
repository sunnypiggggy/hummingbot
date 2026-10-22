"""Update the scheduler's published Grid artifact, preserving its runtime flags."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/home/ubuntu/extra_drive/hummingbot')
ROLLOUT = ROOT / '.reentry-rollouts/20261021-recovery-v1'
INCOMING = Path('/home/ubuntu/.cache/hb-reentry-20261021')


def main():
    name = 'grid-live-fdusd-scheduler'
    current = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    actual = dict(e.split('=', 1) for e in current['Config']['Env'])
    files = current['Config']['Labels']['com.docker.compose.project.config_files'].split(',')
    context = ROLLOUT / 'context'
    (context / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(INCOMING / 'scripts/walk_forward_portfolio_grid_live.py', context / 'scripts/walk_forward_portfolio_grid_live.py')
    shutil.copy2(INCOMING / 'Dockerfile.scheduler', context / 'Dockerfile.scheduler')
    shutil.copy2(INCOMING / 'compose.reentry.yml', context / 'compose.reentry.yml')
    base = 'hummingbot/grid-live-fdusd-scheduler:reentry-base-20261021-v1'
    subprocess.run(['docker', 'tag', current['Image'], base], check=True)
    subprocess.run(['docker', 'build', '-f', str(context / 'Dockerfile.scheduler'),
                    '--build-arg', f'BASE_IMAGE={base}', '-t',
                    'hummingbot/grid-live-fdusd-scheduler:reentry-refresh-20261021-v1', str(context)], check=True)
    env = dict(os.environ)
    env['SCHEDULER_PRESERVED_PAPER_TELEGRAM'] = actual['STOCKS_PAPER_TELEGRAM_TRADING_ENABLED']
    # The shared env_file has changed since this scheduler was created. Keep
    # its exact running environment in a private rollout file, not in Git or
    # command output; unrelated Stock flags must not change during recovery.
    preserved = context / 'compose.scheduler-preserved.yml'
    preserved.write_text('services:\n  grid-live-fdusd-scheduler:\n'
                         '    env_file: !reset []\n'
                         '    environment: !override ' + json.dumps(actual) + '\n')
    preserved.chmod(0o600)
    command = ['docker', 'compose', '-p', 'hummingbot', '--profile', 'live-admin']
    for file in files + [str(context / 'compose.reentry.yml'), str(preserved)]:
        command += ['-f', file]
    result = subprocess.run(command + ['config', '--format', 'json'], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=True)
    desired = json.loads(result.stdout)['services'][name]
    differences = [k for k, v in desired['environment'].items() if str(v or '') != actual.get(k, '')]
    assert not differences, f'Scheduler environment drift: {differences}'
    state = ROOT / 'grid-live-fdusd-data/v22_weekly_automation_state.json'
    if state.exists():
        shutil.copy2(state, ROLLOUT / 'backup/scheduler_automation_before_reload.json')
    result = subprocess.run(command + ['up', '-d', '--no-deps', '--no-build', name],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    print(result.stdout[-1000:], result.stderr[-1500:])
    result.check_returncode()
    after = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
    assert dict(e.split('=', 1) for e in after['Config']['Env']) == actual
    assert {(m['Source'], m['Destination'], m['RW']) for m in after['Mounts']} == {
        (m['Source'], m['Destination'], m['RW']) for m in current['Mounts']}
    print('Scheduler artifact updated; configuration and mounts unchanged.')


if __name__ == '__main__':
    main()
