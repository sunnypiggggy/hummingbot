from pathlib import Path


PACKAGE = Path("release_packages/ethbtc-forced-exit")
DOCUMENTATION = PACKAGE / "documentation"
MECHANISMS = (
    "v22_weekly_buy_gate",
    "fomc_gate",
    "strategy_loss_breaker",
    "strategy_drawdown_breaker",
    "portfolio_loss_breaker",
    "portfolio_drawdown_breaker",
    "position_protection",
)


def test_release_family_contains_complete_utf8_mechanism_documentation() -> None:
    expected = {
        "README.md",
        "ONLINE_MODELS.md",
        "RISK_MECHANISMS.md",
        "V22_WEEKLY_MODEL.md",
        "FORCED_EXIT_AND_RECOVERY.md",
        "CONFIGURATION_AND_OPERATIONS.md",
        "CONTAINERS_AND_SIGNAL_FLOW.md",
        "CONTRACTS_AND_RUNTIME_FLOW.md",
        "TELEGRAM_NOTIFICATIONS.md",
        "ACCOUNT_INVENTORY.md",
        "REAL_SCENARIO_TESTING.md",
        "train_xgboost_codex_historty.md",
    }
    assert {path.name for path in DOCUMENTATION.glob("*.md")} == expected
    content = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(DOCUMENTATION.glob("*.md"))
    )
    for mechanism in MECHANISMS:
        assert mechanism in content
    assert "\ufffd" not in content
    assert "# 机制事件数" not in content


def test_contract_document_covers_runtime_schemas_and_observation_boundary() -> None:
    content = (DOCUMENTATION / "CONTRACTS_AND_RUNTIME_FLOW.md").read_text(
        encoding="utf-8"
    )
    for schema in (
        "ethbtc-forced-exit-live-contract-v1",
        "ethbtc-forced-exit-authorization-v1",
        "ethbtc-forced-exit-observer-status-v1",
        "grid-fomc-gate-v1",
        "ethbtc-telegram-event-v1",
    ):
        assert schema in content
    assert "execution_authorized=false" in content
    assert "ledger.halted=true" in content
    assert "TEST_ONLY" in content


def test_container_flow_document_describes_current_live_execution() -> None:
    content = (DOCUMENTATION / "CONTAINERS_AND_SIGNAL_FLOW.md").read_text(
        encoding="utf-8"
    )
    for container in (
        "grid-live-fdusd-400",
        "dca-live-btcusdt-200",
        "dca-live-ethusdt-200",
        "grid-live-guard",
        "dca-live-guard",
        "grid-live-fdusd-scheduler",
        "dca-macro-gateway",
        "dca-live-report",
    ):
        assert container in content
    assert "OCI 已切换到 v22 live" in content
    assert "唯一 v22 producer" in content
    assert "execution_authorized=false" in content
    assert "Fail-Closed" in content


def test_online_model_document_binds_current_release_and_retired_fallbacks() -> None:
    content = (DOCUMENTATION / "ONLINE_MODELS.md").read_text(encoding="utf-8")
    assert "73f59befa431946889a8d5885d04a05adb43c8e81eeab604f1aa89e31f0e9d60" in content
    assert "xgboost-grid-long-risk-gate-v22-weekly-250d" in content
    assert "v21 producer 已关闭" in content
    assert "禁止" in content and "SQZMOM" in content


def test_stager_binds_and_copies_documentation() -> None:
    source = Path("scripts/stage_ethbtc_forced_exit_release.py").read_text(encoding="utf-8")
    assert 'documentation_sha256 = canonical_hash(documentation_hashes)' in source
    assert 'shutil.copytree(documentation, staging / "documentation")' in source


def test_operations_document_lists_every_independent_switch() -> None:
    content = (DOCUMENTATION / "CONFIGURATION_AND_OPERATIONS.md").read_text(encoding="utf-8")
    for name in (
        "V22_WEEKLY_GATE",
        "FOMC_GATE",
        "STRATEGY_LOSS_BREAKER",
        "STRATEGY_DRAWDOWN_BREAKER",
        "PORTFOLIO_LOSS_BREAKER",
        "PORTFOLIO_DRAWDOWN_BREAKER",
        "POSITION_PROTECTION",
    ):
        assert name in content


def test_telegram_document_covers_runtime_error_lifecycle_and_safety() -> None:
    content = (DOCUMENTATION / "TELEGRAM_NOTIFICATIONS.md").read_text(encoding="utf-8")
    for value in (
        "runtime_error",
        "ERROR_OCCURRED",
        "ERROR_RECOVERED",
        "RuntimeErrorChannel",
        "[REDACTED]",
        "日志采集失败只降低日志可见性",
        "dca-live-report` 仍是唯一发送器",
    ):
        assert value in content
