from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_fdusd_ytd_risk_mechanisms import (  # noqa: E402
    BAR_SECONDS,
    COOLDOWN_DAYS,
    RESET_MODES,
    ROC_RECOVERY_THRESHOLDS,
    ROC_TRIGGER_THRESHOLDS,
    SQZ_RECOVERY_THRESHOLDS,
    SQZ_TRIGGER_THRESHOLDS,
    add_percentile_score,
    data_quality,
)
from validate_grid_live import Candidate, PairBreakerPolicy, simulate  # noqa: E402


def candles(price: float, count: int, *, drop_at: int | None = None,
            drop_fraction: float = 0.0) -> pd.DataFrame:
    rows = []
    timestamp = 1_767_225_600
    for index in range(count):
        close = price * (1 - drop_fraction if drop_at is not None and index >= drop_at else 1)
        rows.append({
            "timestamp": timestamp + index * BAR_SECONDS,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1,
        })
    return pd.DataFrame(rows)


def test_search_spaces_are_unique_and_complete():
    trigger = {(roc, sqz) for roc in ROC_TRIGGER_THRESHOLDS for sqz in SQZ_TRIGGER_THRESHOLDS}
    recovery = {(roc, sqz) for roc in ROC_RECOVERY_THRESHOLDS for sqz in SQZ_RECOVERY_THRESHOLDS}
    cooldown = {(days, reset) for days in COOLDOWN_DAYS for reset in RESET_MODES}
    assert len(trigger) == 56
    assert len(recovery) == 104
    assert len(cooldown) == 14


def test_pair_specific_technical_gate_does_not_pause_other_pair():
    frames = {
        "BTC-FDUSD": candles(100, 60),
        "ETH-FDUSD": candles(10, 60),
    }
    btc_gate = {int(timestamp): False for timestamp in frames["BTC-FDUSD"].timestamp}
    result, _, stats = simulate(
        frames, Candidate(0.03, 0.006, 0.006, 0.015), 0.0,
        technical_buy_gate={"BTC-FDUSD": btc_gate},
        risk_breakers_enabled=False, cost_floor_enabled=False,
    )
    assert result["technical_buy_gate_enabled"]
    assert stats["BTC-FDUSD"]["technical_risk_off_seconds"] == 59 * BAR_SECONDS
    assert stats["ETH-FDUSD"]["technical_risk_off_seconds"] == 0


def test_reset_mode_recovers_after_one_day_but_no_reset_waits_for_safe_zone():
    frame = candles(100, 650, drop_at=10, drop_fraction=0.20)
    reset_events: list[dict] = []
    no_reset_events: list[dict] = []
    common = dict(
        candles={"BTC-FDUSD": frame},
        candidate=Candidate(0.03, 0.006, 0.006, 0.015),
        maker_fee=0.0,
        taker_fee=0.001,
        risk_breakers_enabled=False,
        cost_floor_enabled=False,
    )
    reset_result, _, reset_stats = simulate(
        **common,
        pair_breaker_policy=PairBreakerPolicy("loss", 86_400, True, ("BTC-FDUSD",)),
        trade_log=reset_events,
    )
    no_reset_result, _, no_reset_stats = simulate(
        **common,
        pair_breaker_policy=PairBreakerPolicy("loss", 86_400, False, ("BTC-FDUSD",)),
        trade_log=no_reset_events,
    )
    assert reset_result["pair_breaker_policy"]["reset_baseline"] is True
    assert any(event["reason"] == "pair_breaker_recovered" for event in reset_events)
    assert not any(event["reason"] == "pair_breaker_recovered" for event in no_reset_events)
    assert reset_stats["BTC-FDUSD"]["halted_seconds"] < no_reset_stats["BTC-FDUSD"]["halted_seconds"]


def test_no_reset_safe_zone_is_rechecked_daily_not_every_bar():
    frame = candles(100, 750, drop_at=10, drop_fraction=0.20)
    # Price becomes safe twelve hours after the first one-day cooldown expires.
    recovery_index = 10 + 288 + 144
    frame.loc[recovery_index:, ["open", "close"]] = 120
    frame.loc[recovery_index:, "high"] = 120.1
    frame.loc[recovery_index:, "low"] = 119.9
    events: list[dict] = []
    simulate(
        {"BTC-FDUSD": frame}, Candidate(0.03, 0.006, 0.006, 0.015), 0.0,
        taker_fee=0.001, risk_breakers_enabled=False, cost_floor_enabled=False,
        pair_breaker_policy=PairBreakerPolicy("loss", 86_400, False, ("BTC-FDUSD",)),
        trade_log=events,
    )
    trigger = next(event for event in events if event["reason"] == "pair_breaker_flatten")
    recovered = next(event for event in events if event["reason"] == "pair_breaker_recovered")
    assert recovered["timestamp"] >= trigger["timestamp"] + 2 * 86_400


def test_pair_breaker_policy_round_trips_as_json_and_validates():
    policy = PairBreakerPolicy("drawdown", 3 * 86_400, False, ("ETH-FDUSD",))
    assert json.loads(json.dumps(asdict(policy)))["cooldown_seconds"] == 259_200
    with pytest.raises(ValueError, match="trigger"):
        PairBreakerPolicy("portfolio", 86_400, False, ("ETH-FDUSD",))
    with pytest.raises(ValueError, match="cooldown"):
        PairBreakerPolicy("loss", 0, False, ("ETH-FDUSD",))


def test_fast_mode_matches_full_mode_final_metrics():
    frame = candles(100, 180)
    kwargs = dict(
        candles={"BTC-FDUSD": frame},
        candidate=Candidate(0.03, 0.006, 0.006, 0.015),
        maker_fee=0.0,
        risk_breakers_enabled=False,
        cost_floor_enabled=False,
    )
    fast, fast_curve, fast_pairs = simulate(**kwargs, record_curve=False)
    full, full_curve, full_pairs = simulate(**kwargs, record_curve=True)
    assert fast_curve.empty
    assert not full_curve.empty
    assert fast["final_equity"] == pytest.approx(full["final_equity"])
    assert fast_pairs["BTC-FDUSD"]["net_pnl_quote"] == pytest.approx(
        full_pairs["BTC-FDUSD"]["net_pnl_quote"],
    )


def test_data_quality_and_percentile_score_contract():
    frame = candles(100, 20)
    start = int(frame.timestamp.iloc[4])
    split = int(frame.timestamp.iloc[12])
    end = int(frame.timestamp.iloc[-1]) + BAR_SECONDS
    quality = data_quality(frame, "BTC-FDUSD", start, split, end, int(frame.timestamp.iloc[0]))
    assert quality["actual_rows"] == 16
    assert quality["missing_rows"] == 0
    assert quality["duplicate_rows"] == 0

    ranked = add_percentile_score(pd.DataFrame([
        {"return_pct": 1.0, "max_drawdown_pct": -2.0, "pause_hours": 10},
        {"return_pct": 0.0, "max_drawdown_pct": -4.0, "pause_hours": 20},
    ]))
    assert ranked.loc[0, "score"] > ranked.loc[1, "score"]
