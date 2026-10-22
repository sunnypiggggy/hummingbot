from pathlib import Path, PurePosixPath

from ops.oci_home_cleanup import CACHE, HOME_ROOT, STAGING_FILES, digest, referenced


def test_active_mount_descendants_and_parents_are_protected():
    assert referenced(PurePosixPath('/home/ubuntu/serverdocker'), [], ['/home/ubuntu/serverdocker/data'], [])
    assert referenced(PurePosixPath('/home/ubuntu/serverdocker/Caddyfile'), [], ['/home/ubuntu/serverdocker'], [])
    assert not referenced(PurePosixPath('/home/ubuntu/serverdocker_old'), [], ['/home/ubuntu/serverdocker'], [])


def test_running_process_and_service_paths_are_protected():
    path=PurePosixPath('/home/ubuntu/simple_audit.py')
    assert referenced(path, ['python3 /home/ubuntu/simple_audit.py'], [], [])
    assert referenced(path, [], [], ['ExecStart=python3 /home/ubuntu/simple_audit.py'])
    assert not referenced(path, [], [], [])


def test_allowlist_excludes_secrets_databases_and_live_workspace():
    assert HOME_ROOT==Path('/home/ubuntu')
    assert CACHE==Path('/home/ubuntu/anaconda3/pkgs')
    assert not any(name.endswith(('.sqlite','.db','.key','.pem')) for name in STAGING_FILES)
    assert 'secrets' not in STAGING_FILES and 'extra_drive' not in STAGING_FILES


def test_copy_fingerprint_changes_with_content(tmp_path):
    source=tmp_path/'source'
    copied=tmp_path/'copied'
    source.write_bytes(b'archive')
    copied.write_bytes(source.read_bytes())
    assert digest(source)==digest(copied)
    copied.write_bytes(b'incomplete')
    assert digest(source)!=digest(copied)
