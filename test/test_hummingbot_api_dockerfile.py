from pathlib import Path


def test_plain_v2_script_recent_activity_is_running():
    dockerfile = (
        Path(__file__).resolve().parents[1] / "Dockerfile.hummingbot-api"
    ).read_text(encoding="utf-8")
    assert 'replacement = \'\'\'            if recently_active:' in dockerfile
    assert 'elif len(performance) > 0:' in dockerfile
    assert "Recent MQTT activity is the runtime signal" in dockerfile
