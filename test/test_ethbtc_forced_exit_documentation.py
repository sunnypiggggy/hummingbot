import re
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
        "TELEGRAM_MODEL_PARAMETERS.md",
        "ACCOUNT_INVENTORY.md",
        "GRID_PAIR_PARAMETER_CUTOVER.md",
        "NO_BNB_FEE_POLICY.md",
        "REAL_SCENARIO_TESTING.md",
        "RESILIENCE_POLICY.md",
        "V22_ZERO_DOWNTIME_CUTOVER.md",
        "WEEKLY_APPROVAL_NOTIFICATIONS.md",
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


def test_every_markdown_internal_link_resolves() -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+\.md)(?:#[^)]+)?\)")
    missing: list[str] = []
    for source in DOCUMENTATION.glob("*.md"):
        if source.name == "train_xgboost_codex_historty.md":
            # Immutable conversation history contains workspace-root links, not
            # documentation-relative links. Maintained mechanism docs are checked below.
            continue
        content = source.read_text(encoding="utf-8")
        for target in link_pattern.findall(content):
            resolved = (source.parent / target).resolve()
            if not resolved.is_file():
                missing.append(f"{source.name} -> {target}")
    assert missing == []


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
    assert "schema_version=13" in content
    assert "EXPECTED_EMPTY" in content
    assert "gate_aggregate.capital.mode=alert_only" in content


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
    assert "OCI 现网容器、功能依赖与信号链路" in content
    assert "唯一 v22 producer" in content
    assert "当前实际运行容器" in content
    assert "Fail-Closed" in content


def test_online_model_document_binds_current_release_and_retired_fallbacks() -> None:
    content = (DOCUMENTATION / "ONLINE_MODELS.md").read_text(encoding="utf-8")
    assert "bc3ef0d97bad6fbfaa6e24db1d695defd69ffffaf514a800f280e166bf7e017c" in content
    assert "xgboost-grid-long-risk-gate-v22-weekly-250d" in content
    assert "fold 41" in content
    assert "medium_sideways" in content
    assert "long_volatility" in content
    assert "v21 producer 已关闭" in content
    assert "禁止" in content and "SQZMOM" in content


def test_mechanism_authority_covers_all_execution_interlocks() -> None:
    content = (DOCUMENTATION / "RISK_MECHANISMS.md").read_text(encoding="utf-8")
    for value in (
        "infrastructure_integrity_breaker",
        "EXITING/COOLDOWN/REENTRY/LATCHED",
        "统一库存归属",
        "DCA资金观察",
        "Controller/订单落地",
        "Grid订单构建与Maker保护",
        "禁止BNB抵扣",
        "通知与审计",
        "EXPECTED_EMPTY",
    ):
        assert value in content
    assert "只告警，不参与普通BUY/SELL聚合" in content
    assert "mode = alert_only" in content
    assert "enforced = false" in content
    assert "GRID_RISK_<MECHANISM>_ENABLED" in content
    assert "DCA_RISK_<MECHANISM>_ENABLED" in content


def test_grid_document_distinguishes_theoretical_and_actual_orders() -> None:
    content = (DOCUMENTATION / "GRID_PAIR_PARAMETER_CUTOVER.md").read_text(
        encoding="utf-8"
    )
    for value in (
        "schema 13",
        "18格只定义候选价格拓扑",
        "同一执行价格",
        "0张BUY",
        "额外库存额度",
        "EXPECTED_EMPTY",
        "2026-09-02",
    ):
        assert value in content


def test_current_docs_do_not_describe_old_grid_runtime_as_current() -> None:
    current_docs = (
        "CONTRACTS_AND_RUNTIME_FLOW.md",
        "GRID_PAIR_PARAMETER_CUTOVER.md",
        "ONLINE_MODELS.md",
        "RISK_MECHANISMS.md",
    )
    content = "\n".join(
        (DOCUMENTATION / name).read_text(encoding="utf-8") for name in current_docs
    )
    assert "当前 Grid runtime 已迁移到 schema 8" not in content
    assert "当前 Runtime State 为 schema v9" not in content
    assert "当前签名周 | fold 37" not in content


def test_dca_capital_observation_is_documented_as_non_blocking() -> None:
    for name in ("RISK_MECHANISMS.md", "RESILIENCE_POLICY.md", "ONLINE_MODELS.md"):
        content = (DOCUMENTATION / name).read_text(encoding="utf-8")
        assert "alert_only" in content
        assert "enforced=false" in content or "enforced = false" in content
    resilience = (DOCUMENTATION / "RESILIENCE_POLICY.md").read_text(encoding="utf-8")
    assert "不会关闭 BUY/SELL" in resilience


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


def test_weekly_release_retention_keeps_three_old_models_and_all_png_evidence() -> None:
    content = (DOCUMENTATION / "WEEKLY_APPROVAL_NOTIFICATIONS.md").read_text(
        encoding="utf-8"
    )
    for value in (
        "当前 release",
        "最近 3 个旧 release",
        "V22_WEEKLY_RETAIN_OLD_RELEASES=3",
        "12 张证据 PNG",
        "MODEL_RETENTION_PRUNED",
        "MODEL_RETENTION_FAILED",
    ):
        assert value in content
