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


def test_bot_run_tracking_is_idempotent_and_latest_queries_are_bounded():
    dockerfile = (
        Path(__file__).resolve().parents[1] / "Dockerfile.hummingbot-api"
    ).read_text(encoding="utf-8")
    assert "pg_advisory_xact_lock" in dockerfile
    assert "hummingbot-api:bot-run:{instance_name}" in dockerfile
    assert "BotRun.instance_name == instance_name" in dockerfile
    assert "if existing:" in dockerfile
    assert "source.count(latest_needle) != 2" in dockerfile
    assert "source.count(latest_return_needle) != 1" in dockerfile
    assert ").order_by(desc(BotRun.deployed_at)).limit(1)" in dockerfile


def test_bot_run_reconciliation_closes_duplicates_before_unique_index():
    sql = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "reconcile_hummingbot_api_bot_runs.sql"
    ).read_text(encoding="utf-8")
    assert "row_number() OVER" in sql
    assert "active_rank > 1" in sql
    assert "duplicate_active_run_reconciled" in sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_bot_runs_one_active_instance" in sql
    assert "run_status IN ('RUNNING', 'CREATED')" in sql
