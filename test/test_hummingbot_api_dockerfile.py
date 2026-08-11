from pathlib import Path


def test_plain_v2_script_recent_activity_is_running():
    dockerfile = (
        Path(__file__).resolve().parents[1] / "Dockerfile.hummingbot-api"
    ).read_text(encoding="utf-8")
    assert 'replacement = \'\'\'            if recently_active:' in dockerfile
    assert 'elif len(performance) > 0:' in dockerfile
    assert "Recent MQTT activity is the runtime signal" in dockerfile


def test_non_tradable_edo_dust_is_excluded_from_account_valuation():
    compose = (
        Path(__file__).resolve().parents[1] / "docker-compose.yml"
    ).read_text(encoding="utf-8")
    assert "BANNED_TOKENS:" in compose
    assert '"EDO"' in compose


def test_controller_yaml_updates_are_atomic():
    dockerfile = (
        Path(__file__).resolve().parents[1] / "Dockerfile.hummingbot-api"
    ).read_text(encoding="utf-8")
    assert 'temporary_path = f"{file_path}.tmp"' in dockerfile
    assert "os.fsync(file.fileno())" in dockerfile
    assert "os.replace(temporary_path, file_path)" in dockerfile
