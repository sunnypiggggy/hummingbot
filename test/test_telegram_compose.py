from pathlib import Path

import yaml


def test_grid_scheduler_packages_telegram_endpoint_policy():
    dockerfile = (
        Path(__file__).resolve().parents[1]
        / "Dockerfile.grid-live-fdusd-scheduler"
    ).read_text(encoding="utf-8")
    assert "COPY live_guard/runtime_endpoints.py /app/runtime_endpoints.py" in dockerfile
    scheduler = (
        Path(__file__).resolve().parents[1]
        / "scheduler"
        / "fdusd_live_grid_scheduler.py"
    ).read_text(encoding="utf-8")
    assert 'scripts / "runtime_endpoints.py"' in scheduler
    notifications = (
        Path(__file__).resolve().parents[1]
        / "live_guard"
        / "telegram_notifications.py"
    ).read_text(encoding="utf-8")
    assert "from scripts.runtime_endpoints import telegram_api_base" in notifications


ROOT = Path(__file__).resolve().parents[1]


def test_existing_report_service_is_the_only_sender_and_no_service_was_added():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert "telegram-notifier" not in services
    report = services["dca-live-report"]
    assert report["command"] == ["python", "/app/dca_live_report.py"]
    assert report["env_file"] == ["./telegram-notify.env"]
    assert ".env.control" not in report["env_file"]
    assert "telegram_notify_bot_token" in report["secrets"]
    assert any(volume.endswith(":/workspace/grid:ro") for volume in report["volumes"])
    assert any(volume.endswith(":/workspace/releases:ro") for volume in report["volumes"])
    for name in ("grid-live-fdusd-scheduler", "grid-live-guard", "dca-live-guard"):
        environment = services[name].get("environment", {})
        assert "TELEGRAM_NOTIFY_BOT_TOKEN_FILE" not in environment
        assert environment["TELEGRAM_TOKEN"] == ""
        assert environment["ADMIN_USER_ID"] == ""


def test_report_notification_settings_are_persistent_and_secret_free():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["dca-live-report"]["environment"]
    assert "TELEGRAM_NOTIFY_ENABLED" not in environment
    assert "TELEGRAM_PROFIT_REPORT_ENABLED" not in environment
    assert "TELEGRAM_NOTIFY_CHANNEL_ID" not in environment
    values = {}
    for line in (ROOT / "telegram-notify.env").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    assert values["TELEGRAM_NOTIFY_ENABLED"] == "true"
    assert values["TELEGRAM_PROFIT_REPORT_ENABLED"] == "true"
    assert values["TELEGRAM_NOTIFY_CHANNEL_ID"].startswith("-100")
    assert all("TOKEN" not in key for key in values)


def test_all_seven_notification_switches_default_enabled():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["dca-live-report"]["environment"]
    mechanisms = (
        "V22_WEEKLY_BUY_GATE", "FOMC_GATE", "STRATEGY_LOSS_BREAKER",
        "STRATEGY_DRAWDOWN_BREAKER", "PORTFOLIO_LOSS_BREAKER",
        "PORTFOLIO_DRAWDOWN_BREAKER", "POSITION_PROTECTION",
    )
    for mechanism in mechanisms:
        assert environment[f"TELEGRAM_ALERT_{mechanism}_ENABLED"].endswith(":-true}")


def test_notification_sender_source_has_no_get_updates_consumer():
    source = (ROOT / "live_guard/telegram_notifications.py").read_text(encoding="utf-8")
    assert "get" + "Updates" not in source
    assert "sendMessage" in source and "sendPhoto" in source and "sendDocument" in source


def test_grid_scheduler_image_includes_technical_gate_dependency():
    dockerfile = (ROOT / "Dockerfile.grid-live-fdusd-scheduler").read_text(
        encoding="utf-8"
    )
    assert "COPY scripts/grid_technical_gate.py /app/grid_technical_gate.py" in dockerfile
