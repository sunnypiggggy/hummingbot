import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "live_guard"))

from account_inventory import ownership_from_documents  # noqa: E402
from grid_live_common import (  # noqa: E402
    BINANCE_AI_GRID_PROFILES,
    FDUSD_LEGACY_BUDGET,
    FDUSD_SOL_BUDGET,
    FDUSD_SOL_PAIRS,
    PORTFOLIOS,
    active_portfolio_pairs,
    budget_for_live_pairs,
    build_live_config,
    validate_active_selection,
    validate_live_config,
)
from sol_grid_weekly_risk import (  # noqa: E402
    CONTRACT_SCHEMA,
    MODEL_VERSION,
    feature_schema_sha256,
    load_runtime_sol_gate,
)
from scheduler.fdusd_live_grid_scheduler import Scheduler  # noqa: E402
from live_guard.telegram_notifications import canonical_sha256  # noqa: E402
from live_guard.telegram_parameter_report import build_parameter_attachments  # noqa: E402


def profile(name: str) -> dict:
    return {
        "profile": name,
        **{
            key: float(value) if isinstance(value, Decimal) else value
            for key, value in BINANCE_AI_GRID_PROFILES[name].items()
        },
    }


def selection(sol_profile: str = "short_sideways") -> dict:
    return {
        "schema_version": 3,
        "parameter_version": f"sol-{sol_profile}-v1",
        "trading_pairs": list(FDUSD_SOL_PAIRS),
        "sol_execution_enabled": True,
        "pair_parameters": {
            "BTC-FDUSD": profile("medium_sideways"),
            "ETH-FDUSD": profile("long_volatility"),
            "SOL-FDUSD": profile(sol_profile),
        },
    }


def contract(now: datetime) -> dict:
    return {
        "schema": CONTRACT_SCHEMA,
        "model_version": MODEL_VERSION,
        "release_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "feature_schema_sha256": feature_schema_sha256(),
        "strategy_sha256": "3" * 64,
        "data_sha256": "4" * 64,
        "state_lineage_sha256": "5" * 64,
        "generated_at": now.isoformat(),
        "valid_until": (now + timedelta(days=7)).isoformat(),
        "source_healthy": True,
        "deployment_allowed": True,
        "model_week": "2026-W35",
        "pairs": {
            "SOL-FDUSD": {
                "model_week": "2026-W35",
                "model_signal": "RISK_ON",
                "buy_enabled": True,
                "risk_off_active": False,
            }
        },
    }


def test_sol_budget_and_pair_set_are_inert_until_explicit_latch():
    with patch.dict(os.environ, {"GRID_SOL_FDUSD_LIVE_ENABLED": "false"}):
        assert active_portfolio_pairs(PORTFOLIOS["FDUSD"]) == ("BTC-FDUSD", "ETH-FDUSD")
        assert budget_for_live_pairs("FDUSD", active_portfolio_pairs(PORTFOLIOS["FDUSD"])) == FDUSD_LEGACY_BUDGET
    with patch.dict(os.environ, {"GRID_SOL_FDUSD_LIVE_ENABLED": "true"}):
        assert active_portfolio_pairs(PORTFOLIOS["FDUSD"]) == FDUSD_SOL_PAIRS
        assert budget_for_live_pairs("FDUSD", FDUSD_SOL_PAIRS) == FDUSD_SOL_BUDGET


def test_schema_v3_accepts_only_two_explicit_sol_profiles():
    assert validate_active_selection(selection("short_sideways"))["sol_execution_enabled"] is True
    assert validate_active_selection(selection("medium_sideways"))["pair_parameters"]["SOL-FDUSD"]["profile"] == "medium_sideways"
    invalid = selection()
    invalid["pair_parameters"]["SOL-FDUSD"] = profile("long_volatility")
    try:
        validate_active_selection(invalid)
    except ValueError as exc:
        assert "short_sideways or medium_sideways" in str(exc)
    else:
        raise AssertionError("unapproved SOL profile was accepted")


def test_three_pair_live_config_requires_isolated_gate_hashes():
    prices = {
        "BTC-FDUSD": Decimal("70000"),
        "ETH-FDUSD": Decimal("2500"),
        "SOL-FDUSD": Decimal("190"),
    }
    with patch.dict(os.environ, {
        "GRID_SOL_FDUSD_LIVE_ENABLED": "true",
        "GRID_RISK_SOL_WEEKLY_GATE_ENABLED": "true",
    }):
        config = build_live_config(PORTFOLIOS["FDUSD"], prices, Decimal("0"))
    assert config["capital_limit_quote"] == 620.0
    assert config["trading_pairs"] == list(FDUSD_SOL_PAIRS)
    config.update({
        "trading_enabled": True,
        "technical_model_sha256": "a" * 64,
        "technical_feature_sha256": "b" * 64,
    })
    try:
        validate_live_config(config)
    except ValueError as exc:
        assert "locked SOL model and feature hashes" in str(exc)
    else:
        raise AssertionError("enabled SOL config did not require isolated hashes")
    config["sol_technical_model_sha256"] = "c" * 64
    config["sol_technical_feature_sha256"] = "d" * 64
    validate_live_config(config)


def test_sol_contract_is_strict_and_stale_contract_fails_closed_only_sol():
    now = datetime(2026, 8, 27, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sol_grid_weekly_risk.json"
        path.write_text(json.dumps(contract(now)), encoding="utf-8")
        healthy = load_runtime_sol_gate(path, now=now + timedelta(seconds=30))
        assert healthy["runtime_gate_healthy"] is True
        stale = load_runtime_sol_gate(path, now=now + timedelta(seconds=151))
        assert stale["runtime_gate_healthy"] is False
        assert set(stale["pairs"]) == {"SOL-FDUSD"}
        assert stale["pairs"]["SOL-FDUSD"]["model_signal"] == "UNAVAILABLE"


def test_unified_ownership_tracks_sol_for_grid_and_never_invents_sol_dca():
    ownership = ownership_from_documents(
        reservations={"reservations": {"FDUSD": {"base": {
            "BTC": "0", "ETH": "0", "SOL": "0.50",
        }}}},
        grid_state={"bots": {"grid": {"latest": {"pairs": {
            "BTC-FDUSD": {"net_base": "0"},
            "ETH-FDUSD": {"net_base": "0"},
            "SOL-FDUSD": {"net_base": "0.02"},
        }}}}},
        managed_inventory={"pairs": {}},
        dca_state={"bots": {}},
    )
    assert ownership["SOL"]["grid:grid-live-fdusd-400"] == Decimal("0.52")
    assert all(not owner.startswith("dca:") for owner in ownership["SOL"])


def test_sol_mobile_evidence_requires_three_hash_bound_images():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        images = []
        for window in ("360d", "2026_jan_feb", "2026_may_june"):
            path = root / f"{window}.png"
            Image.new("RGB", (1440, 2400), "white").save(path)
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            images.append({"path": path.name, "window": window,
                           "pair": "SOL-FDUSD", "sha256": digest})
        identity = "7" * 64
        (root / "sol_grid_evidence_manifest.json").write_text(json.dumps({
            "schema": "sol-grid-mobile-evidence-v1",
            "identity_sha256": identity,
            "model_sha256": "8" * 64,
            "evidence_complete": True,
            "images": images,
        }), encoding="utf-8")
        attachments = build_parameter_attachments({
            "event_id": "event", "release_sha256": identity,
            "details": {"report_request": "sol_grid_360d", "evidence_root": str(root)},
        }, release_root=root, output_root=root / "out")
        assert len(attachments) == 3
        assert all(item["kind"] == "photo" for item in attachments)


def test_sol_recovery_evidence_survives_non_four_hour_rows():
    import pandas as pd
    from sol_grid_weekly_risk_v22 import GateState, advance_gate, weekly_folds

    start = int(datetime(2026, 8, 24, tzinfo=timezone.utc).timestamp())
    state = GateState(
        active=True, since=start, previous_structure=(-2, -1, -5, -1, .8),
        last_complete_4h_ts=start - 4 * 3600,
    )
    states = []
    for hour in range(53):
        relief_step = hour // 4 + 1
        state, _ = advance_gate(
            state, probability=.01, threshold=.5, signal_ts=start + hour * 3600,
            last_complete_4h_ts=start + (hour // 4) * 4 * 3600,
            structure=(relief_step, relief_step, 5, 1, .2),
        )
        states.append(state.active)
    # Three complete 4h relief observations are preserved through intervening
    # hourly rows, but the 48-hour minimum hold still applies.
    assert states[47] is True
    assert states[48] is False

    folds = weekly_folds(
        pd.Timestamp("2025-08-31T00:00:00Z"),
        pd.Timestamp("2026-08-26T00:00:00Z"),
    )
    assert len(folds) == 53
    assert folds[0] == (
        pd.Timestamp("2025-08-31T00:00:00Z"),
        pd.Timestamp("2025-09-01T00:00:00Z"),
    )
    assert folds[-1][1] == pd.Timestamp("2026-08-26T00:00:00Z")
    assert all(left[1] == right[0] for left, right in zip(folds, folds[1:]))


def test_sol_v22_feature_and_bundle_contract_is_independent():
    from sol_grid_weekly_risk_v22 import (
        FEATURES, MODEL_BUNDLE_SCHEMA, MODEL_VERSION, public_bundle_metadata,
    )

    assert all("btc" not in feature.lower() for feature in FEATURES)
    assert all("eth" not in feature.lower() for feature in FEATURES)
    marker = object()
    bundle = {
        "schema": MODEL_BUNDLE_SCHEMA, "model_version": MODEL_VERSION,
        "weeks": [{"fold": 1, "model": marker, "model_sha256": "a" * 64}],
    }
    public = public_bundle_metadata(bundle)
    assert public["weeks"] == [{"fold": 1, "model_sha256": "a" * 64}]
    assert bundle["weeks"][0]["model"] is marker


def test_scheduler_keeps_two_pair_selection_until_evidence_and_manual_approval():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        state = root / "state"
        evidence = root / "evidence"
        receipts = root / "receipts"
        bots = root / "bots"
        for path in (state, evidence, receipts, bots):
            path.mkdir(parents=True, exist_ok=True)
        current = {
            "schema_version": 2, "parameter_version": "existing",
            "trading_pairs": ["BTC-FDUSD", "ETH-FDUSD"],
            "pair_parameters": {
                "BTC-FDUSD": profile("medium_sideways"),
                "ETH-FDUSD": profile("long_volatility"),
            },
        }
        (state / "active_selection.json").write_text(json.dumps(current), encoding="utf-8")
        identity = "9" * 64
        model_sha = "a" * 64
        (evidence / "sol_grid_evidence_manifest.json").write_text(json.dumps({
            "schema": "sol-grid-mobile-evidence-v1", "identity_sha256": identity,
            "model_sha256": model_sha, "evidence_complete": True,
            "activation_eligible": False,
            "hard_gates": {"sol_account_fee_verified": False},
            "images": [{}, {}, {}],
        }), encoding="utf-8")
        env = {
            "GRID_LIVE_FDUSD_STATE_PATH": str(state), "BOTS_PATH": str(bots),
            "PARAMETER_EVIDENCE_RECEIPT_ROOT": str(receipts),
            "SOL_GRID_EVIDENCE_ROOT": str(evidence),
            "GRID_SOL_INITIAL_APPROVAL_PATH": str(state / "approval.json"),
            "GRID_SOL_FDUSD_LIVE_ENABLED": "true",
            "GRID_SOL_FDUSD_PROFILE": "medium_sideways",
            "V22_WEEKLY_AUTO_UPDATE_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            scheduler = Scheduler()
            assert scheduler.ensure_fixed_selection() == current
            event = json.loads((state / "telegram_events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()[-1])
            parameter_sha = event["parameter_sha256"]
            receipt = {
                "schema": "telegram-evidence-delivery-receipt-v1",
                "identity_sha256": identity,
                "release_sha256": identity,
                "model_sha256": model_sha,
                "parameter_sha256": parameter_sha,
                "report_request": "sol_grid_360d",
                "expected_photo_count": 3,
                "photo_sha256": ["1" * 64, "2" * 64, "3" * 64],
            }
            receipt["delivery_receipt_sha256"] = canonical_sha256(receipt)
            (receipts / f"{identity}.json").write_text(json.dumps(receipt), encoding="utf-8")
            (state / "approval.json").write_text(json.dumps({
                "schema": "sol-grid-initial-approval-v1", "approved": True,
                "consumed": False, "profile": "medium_sideways",
                "candidate_identity_sha256": identity,
                "selection_sha256": parameter_sha,
            }), encoding="utf-8")
            assert scheduler.ensure_fixed_selection() == current
            manifest = json.loads(
                (evidence / "sol_grid_evidence_manifest.json").read_text(encoding="utf-8")
            )
            manifest.update({
                "activation_eligible": True,
                "hard_gates": {"sol_account_fee_verified": True},
            })
            (evidence / "sol_grid_evidence_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            activated = scheduler.ensure_fixed_selection()
        assert activated["schema_version"] == 3
        assert activated["pair_parameters"]["SOL-FDUSD"]["profile"] == "medium_sideways"
        assert json.loads((state / "approval.json").read_text(encoding="utf-8"))["consumed"] is True
