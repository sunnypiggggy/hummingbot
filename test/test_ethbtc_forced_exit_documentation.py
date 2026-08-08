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
        "RISK_MECHANISMS.md",
        "V22_WEEKLY_MODEL.md",
        "FORCED_EXIT_AND_RECOVERY.md",
        "CONFIGURATION_AND_OPERATIONS.md",
    }
    assert {path.name for path in DOCUMENTATION.glob("*.md")} == expected
    content = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(DOCUMENTATION.glob("*.md"))
    )
    for mechanism in MECHANISMS:
        assert mechanism in content
    assert "\ufffd" not in content
    assert "# 机制事件数" not in content


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
