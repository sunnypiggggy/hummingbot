from scripts.risk_recovery import (
    ACTIVE, COOLDOWN, EXITING, LATCHED, REENTRY,
    PORTFOLIO_COOLDOWN_SECONDS, REQUIRED_HEALTHY_CYCLES,
    TECHNICAL_COOLDOWN_SECONDS,
    advance_recovery, mark_exit_complete, mark_reentry_complete, trigger_state,
)


def test_transaction_breaker_exits_cools_reenters_and_resets_baseline() -> None:
    state = trigger_state(
        mechanism="strategy_loss_breaker", scope="strategy", now=100,
        trigger_value=-16, signal_price=100, reason="loss",
    )
    assert state["phase"] == EXITING
    state = mark_exit_complete(
        state, now=103, remaining_base={"BTC-USDT": "0.00001"},
        execution={"attempts": 1},
    )
    assert state["phase"] == COOLDOWN
    state["cooldown_until"] = 200
    for now in range(200, 200 + REQUIRED_HEALTHY_CYCLES):
        state = advance_recovery(
            state, now=now, healthy=True, gates_allow_reentry=True,
        )
    assert state["phase"] == REENTRY and state["reentry_allowed"]
    state = mark_reentry_complete(state, now=204, baseline={"equity": 190})
    assert state["phase"] == ACTIVE
    assert state["episode_baseline"] == {"equity": "190"}


def test_unhealthy_cycle_resets_confirmation_and_latched_never_recovers() -> None:
    state = trigger_state(
        mechanism="portfolio_loss_breaker", scope="portfolio", now=0,
        trigger_value=-32, signal_price=100, reason="loss",
    )
    state = mark_exit_complete(state, now=1, remaining_base={}, execution={})
    assert state["cooldown_until"] == 1 + PORTFOLIO_COOLDOWN_SECONDS
    state["cooldown_until"] = 2
    state = advance_recovery(state, now=2, healthy=True, gates_allow_reentry=True)
    state = advance_recovery(state, now=3, healthy=False, gates_allow_reentry=True)
    assert state["healthy_cycles"] == 0 and state["phase"] == COOLDOWN

    locked = trigger_state(
        mechanism="infrastructure_integrity_breaker", scope="infrastructure",
        now=0, trigger_value="hash", signal_price="", reason="hash mismatch",
        latched=True,
    )
    assert advance_recovery(
        locked, now=999999, healthy=True, gates_allow_reentry=True,
    )["phase"] == LATCHED


def test_technical_exit_has_zero_fixed_cooldown_but_requires_health_confirmation() -> None:
    state = trigger_state(
        mechanism="v22_weekly_buy_gate", scope="technical", now=100,
        trigger_value=.9, signal_price=80_000, reason="risk_off",
    )
    state = mark_exit_complete(
        state, now=101, remaining_base={"BTC-FDUSD": "0"}, execution={"attempts": 1},
    )
    assert state["phase"] == COOLDOWN
    assert state["cooldown_until"] == 101 + TECHNICAL_COOLDOWN_SECONDS
    for now in range(102, 102 + REQUIRED_HEALTHY_CYCLES):
        state = advance_recovery(state, now=now, healthy=True, gates_allow_reentry=True)
    assert state["phase"] == REENTRY
    assert state["reentry_allowed"] is True
