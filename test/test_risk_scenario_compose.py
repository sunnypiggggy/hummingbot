import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_scenario_compose_is_network_and_secret_isolated():
    document = yaml.safe_load(
        (ROOT / "docker-compose.risk-scenarios.yml").read_text(encoding="utf-8")
    )
    assert document["networks"]["risk-scenario"]["internal"] is True
    text = json.dumps(document, sort_keys=True)
    assert "/var/run/docker.sock" not in text
    assert "api.binance.com" not in text
    assert "api.telegram.org" not in text
    assert "GUARD_SCENARIO_MODE" in text
    assert set(document["services"]) == {
        "scenario-init", "binance-sim", "grid-guard-scenario",
        "dca-guard-scenario", "sqlite-lock-scenario", "report-scenario",
    }


def test_production_compose_does_not_gain_scenario_services():
    production = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert not set(production["services"]).intersection({
        "scenario-init", "binance-sim", "grid-guard-scenario",
        "dca-guard-scenario", "sqlite-lock-scenario", "report-scenario",
    })


def test_scenario_credentials_are_fixed_nonproduction_markers():
    credential = json.loads((
        ROOT / "test" / "fixtures" / "risk_scenarios" / "scenario_credentials.json"
    ).read_text(encoding="utf-8"))
    assert credential["api_key"].startswith("scenario-")
    assert credential["base_url"] == "http://binance-sim:8080"


def test_runtime_clients_do_not_bypass_guarded_exchange_endpoint():
    runtime_files = (
        "dca_live_guard.py", "grid_live_guard.py", "dca_live_report.py",
        "telegram_parameter_report.py",
    )
    for name in runtime_files:
        source = (ROOT / "live_guard" / name).read_text(encoding="utf-8")
        assert "https://api.binance.com" not in source, name


def test_scenario_restarts_do_not_start_compose_dependencies():
    source = (ROOT / "scripts" / "run_risk_scenario_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert 'compose("start"' not in source
    assert 'run(["docker", "start", container_id])' in source
