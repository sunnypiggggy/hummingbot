from decimal import Decimal
import json

import pytest

from live_guard.account_inventory import UnifiedInventoryLedger
from scripts.migrate_grid_btc_reentry_ledger_v1 import (
    _apply_missing_buy,
    _apply_sell,
    _finalize_runtime_from_completed_job,
    _fill_metrics,
)


def ledger():
    return {
        "quote": "213.0206733700000000",
        "base": "0.00000590",
        "base_cost_quote": "0.4660970737151373153805089270",
        "fees_quote": "0.6785252446449415598354563833",
        "buys": 151, "sells": 87,
    }


def test_exact_missed_reentry_fill_reconstructs_grid_owned_btc():
    value = ledger()
    _apply_missing_buy(value, {
        "price": "79035.01", "amount": "0.00125",
        "base_fee": "0.00000125", "fee_quote": "0.0987937625",
    })
    assert Decimal(value["base"]) == Decimal("0.00125465")
    assert Decimal(value["quote"]) == Decimal("114.2269108700000000")
    assert value["buys"] == 152


def test_confirmed_sell_leaves_only_grid_dust_and_preserves_other_owners():
    value = ledger()
    _apply_missing_buy(value, {
        "price": "79035.01", "amount": "0.00125",
        "base_fee": "0.00000125", "fee_quote": "0.0987937625",
    })
    dca_owner = Decimal("0.000004482327138578")
    unattributed = Decimal("0.0000011976728614218")
    _apply_sell(value, {
        "executed": Decimal("0.00125"), "base_fee": Decimal("0"),
        "quote": Decimal("97.5"), "quote_fee": Decimal("0.0975"),
        "fee_quote": Decimal("0.0975"),
    })
    assert Decimal(value["base"]) == Decimal("0.00000465")
    assert dca_owner == Decimal("0.000004482327138578")
    assert unattributed == Decimal("0.0000011976728614218")
    assert value["sells"] == 88


def test_migration_rejects_bnb_commission():
    with pytest.raises(RuntimeError, match="BNB commission is forbidden"):
        _fill_metrics({
            "executedQty": "0.00125", "cummulativeQuoteQty": "97.5",
            "fills": [{
                "price": "78000", "commission": "0.00001",
                "commissionAsset": "BNB",
            }],
        })


def test_completed_job_repairs_runtime_after_process_crash(tmp_path):
    value = ledger()
    _apply_missing_buy(value, {
        "price": "79035.01", "amount": "0.00125",
        "base_fee": "0.00000125", "fee_quote": "0.0987937625",
    })
    runtime_path = tmp_path / "runtime.json"
    runtime = {
        "schema_version": 13,
        "ledgers": {"BTC-FDUSD": value},
        "accounting_migrations": [{
            "migration_id": "grid-btc-missed-reentry-fill-v1",
            "stage": "LEDGER_CORRECTED",
        }],
    }
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    shared = UnifiedInventoryLedger(tmp_path / "inventory")
    job = shared.start_job(
        job_id="crash-job", asset="BTC", scope="grid:grid-live-fdusd-400",
        pair="BTC-FDUSD", requested_quantity=Decimal("0.00125"),
        client_order_id="inv-crash-job",
    )
    shared.start_attempt(
        job_id="crash-job", sequence=1, client_order_id="inv-crash-job-1",
        requested_quantity="0.00125",
    )
    shared.finish_attempt(
        job_id="crash-job", sequence=1, status="FILLED", response={
            "orderId": "123", "executedQty": "0.00125",
            "cummulativeQuoteQty": "97.5",
            "fills": [{
                "price": "78000", "commission": "0.0975",
                "commissionAsset": "FDUSD",
            }],
        },
    )
    verification = {
        "order_verified": True, "balance_verified": True,
        "no_active_orders": True, "requested_quantity_verified": True,
    }
    shared.finish_job(
        "crash-job", status="COMPLETED", exchange_order_id="123",
        executed_quantity="0.00125", quote_quantity="97.5",
        fee_quote="0.0975", verification=verification,
    )
    job = shared.get_job("crash-job")
    assert job is not None
    _finalize_runtime_from_completed_job(runtime_path, runtime, job, shared)
    recovered = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert Decimal(recovered["ledgers"]["BTC-FDUSD"]["base"]) == Decimal("0.00000465")
    assert recovered["accounting_migrations"][0]["stage"] == "COMPLETED"

    # A second recovery pass is read-only and cannot account the SELL twice.
    _finalize_runtime_from_completed_job(runtime_path, recovered, job, shared)
    again = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert Decimal(again["ledgers"]["BTC-FDUSD"]["base"]) == Decimal("0.00000465")
