import json
import sys
import tempfile
from unittest.mock import patch
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_guard"))
sys.path.insert(0, str(ROOT / "scripts"))

from account_inventory import (  # noqa: E402
    UnifiedInventoryLedger, ownership_from_documents,
)
from dca_live_guard import Guard  # noqa: E402
import dca_live_guard  # noqa: E402
from risk_recovery import LATCHED  # noqa: E402


def balances(btc="0.003", eth="0.052"):
    return {
        "BTC": {"free": Decimal(btc), "locked": Decimal("0"), "total": Decimal(btc)},
        "ETH": {"free": Decimal(eth), "locked": Decimal("0"), "total": Decimal(eth)},
    }


def seed_ownership(ledger, *, btc="0", eth="0", btc_owner="dca:bot", eth_owner="dca:bot"):
    ledger.reconcile(
        account_fingerprint="test-account", balances=balances("1", "1"),
        ownership={
            "BTC": {btc_owner: btc}, "ETH": {eth_owner: eth},
        },
        evidence_sha256="seed-ownership", open_order_counts={},
        sources_healthy=True, now=1,
    )


def test_ownership_combines_grid_and_dca_evidence_without_account_balance():
    value = ownership_from_documents(
        reservations={"reservations": {"FDUSD": {"base": {
            "BTC": "0.00156", "ETH": "0.052",
        }}}},
        grid_state={"bots": {"grid": {"latest": {"pairs": {
            "BTC-FDUSD": {"net_base": "-0.00156"},
            "ETH-FDUSD": {"net_base": "-0.052"},
        }}}}},
        managed_inventory={"pairs": {
            "BTC-USDT": {"managed_base": "0.0015"},
            "ETH-USDT": {"managed_base": "0.0505"},
        }},
        dca_state={"bots": {
            "dca-live-btcusdt-200": {"latest": {"net_base": "0"}},
            "dca-live-ethusdt-200": {"latest": {"net_base": "0"}},
        }},
    )
    assert value["BTC"]["grid:grid-live-fdusd-400"] == 0
    assert value["ETH"]["grid:grid-live-fdusd-400"] == 0
    assert value["BTC"]["dca:dca-live-btcusdt-200"] == Decimal("0.0015")
    assert value["ETH"]["dca:dca-live-ethusdt-200"] == Decimal("0.0505")


def test_ownership_does_not_double_count_adjustments_already_in_latest_net_base():
    value = ownership_from_documents(
        reservations={"reservations": {"FDUSD": {"base": {
            "BTC": "0", "ETH": "0",
        }}}},
        grid_state={"bots": {"grid": {"latest": {"pairs": {
            "BTC-FDUSD": {"net_base": "0"},
            "ETH-FDUSD": {"net_base": "0"},
        }}}}},
        managed_inventory={"pairs": {
            "BTC-USDT": {"managed_base": "0.001499762327138578"},
            "ETH-USDT": {"managed_base": "0.050528420907064937"},
        }},
        dca_state={"bots": {
            "dca-live-btcusdt-200": {
                "latest": {"net_base": "-0.00149000"},
                "emergency_adjustments": [{"pair": "BTC-USDT", "base_delta": "-0.00192000"}],
            },
            "dca-live-ethusdt-200": {
                "latest": {"net_base": "-0.05095080"},
                "emergency_adjustments": [{"pair": "ETH-USDT", "base_delta": "-0.05050000"}],
            },
        }},
    )
    assert value["BTC"]["dca:dca-live-btcusdt-200"] == Decimal(
        "0.000009762327138578"
    )
    assert value["ETH"]["dca:dca-live-ethusdt-200"] == 0


def test_dca_approval_preflight_uses_current_unified_ownership():
    status = {
        "schema": "account-inventory-status-v3",
        "generated_at": 95,
        "sources_healthy": True,
        # Normal activity can make overall healthy false without invalidating
        # ownership evidence used by the weekly model preflight.
        "healthy": False,
        "account_fingerprint": "f" * 64,
        "evidence_sha256": "evidence",
        "assets": {
            "BTC": {
                "exchange": {"total": "0.00001462"},
                "owners": {"dca:dca-live-btcusdt-200": "0.000009762327138578"},
                "ownership_deficit": "0",
            },
            "ETH": {
                "exchange": {"total": "0.00189080"},
                "owners": {"dca:dca-live-ethusdt-200": "0"},
                "ownership_deficit": "0",
            },
        },
    }
    managed = {"pairs": {
        "BTC-USDT": {"managed_base": "0.001499762327138578"},
        "ETH-USDT": {"managed_base": "0.050528420907064937"},
    }}
    value = Guard._ownership_preflight_from_status(
        status, managed, observed_at=100,
    )
    assert value["BTC-USDT"]["covered"] is True
    assert value["BTC-USDT"]["managed_base"] == "0.000009762327138578"
    assert value["BTC-USDT"]["managed_base_target"] == "0.001499762327138578"
    assert value["ETH-USDT"]["covered"] is True
    assert value["ETH-USDT"]["managed_base"] == "0"

    stale = Guard._ownership_preflight_from_status(
        status, managed, observed_at=126,
    )
    assert all(not row["covered"] for row in stale.values())


def test_unattributed_requires_three_cycles_and_thirty_seconds():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        ownership = {
            "BTC": {"dca:btc": Decimal("0.0015")},
            "ETH": {"dca:eth": Decimal("0.0505")},
        }
        first = ledger.reconcile(
            account_fingerprint="account", balances=balances(), ownership=ownership,
            evidence_sha256="evidence", open_order_counts={}, sources_healthy=True,
            now=100,
        )
        second = ledger.reconcile(
            account_fingerprint="account", balances=balances(), ownership=ownership,
            evidence_sha256="evidence", open_order_counts={}, sources_healthy=True,
            now=115,
        )
        third = ledger.reconcile(
            account_fingerprint="account", balances=balances(), ownership=ownership,
            evidence_sha256="evidence", open_order_counts={}, sources_healthy=True,
            now=131,
        )
        assert first["assets"]["BTC"]["confirmation"]["cycles"] == 1
        assert not second["assets"]["BTC"]["confirmation"]["confirmed"]
        assert third["assets"]["BTC"]["confirmation"]["confirmed"]


def test_dynamic_evidence_does_not_reset_confirmation_but_deficit_is_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        owners = {"BTC": {"dca": "0.001"}, "ETH": {"dca": "0.05"}}
        ledger.reconcile(
            account_fingerprint="account", balances=balances(), ownership=owners,
            evidence_sha256="one", open_order_counts={}, sources_healthy=True, now=100,
        )
        reset = ledger.reconcile(
            account_fingerprint="account", balances=balances(), ownership=owners,
            evidence_sha256="two", open_order_counts={}, sources_healthy=True, now=140,
        )
        assert reset["assets"]["BTC"]["confirmation"]["cycles"] == 2
        deficit = ledger.reconcile(
            account_fingerprint="account", balances=balances("0.0005", "0.01"),
            ownership=owners, evidence_sha256="three", open_order_counts={},
            sources_healthy=True, now=180,
        )
        assert not deficit["healthy"]
        assert Decimal(deficit["assets"]["BTC"]["ownership_deficit"]) > 0
        assert deficit["assets"]["BTC"]["confirmation"]["cycles"] == 0


def test_deficit_is_immediately_fail_closed_but_alert_confirmation_is_delayed():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        owners = {"BTC": {"dca": "0.002"}, "ETH": {"dca": "0.05"}}
        values = []
        for now, btc_balance, evidence in (
            (100, "0.0010", "fill-one"),
            (115, "0.0012", "fill-two"),
            (131, "0.0015", "fill-three"),
        ):
            values.append(ledger.reconcile(
                account_fingerprint="account",
                balances=balances(btc_balance, "0.052"), ownership=owners,
                evidence_sha256=evidence, open_order_counts={"BTC-USDT": 1},
                sources_healthy=True, now=now,
            ))
        assert all(not value["healthy"] for value in values)
        confirmations = [
            value["assets"]["BTC"]["deficit_confirmation"] for value in values
        ]
        assert [row["cycles"] for row in confirmations] == [1, 2, 3]
        assert not confirmations[0]["confirmed"]
        assert confirmations[-1]["confirmed"]
        assert confirmations[-1]["peak_deficit"] == "0.0010"


def test_deficit_silent_recovery_clears_confirmation_episode():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        owners = {"BTC": {"dca": "0.002"}, "ETH": {"dca": "0.05"}}
        ledger.reconcile(
            account_fingerprint="account", balances=balances("0.001", "0.052"),
            ownership=owners, evidence_sha256="one", open_order_counts={},
            sources_healthy=True, now=100,
        )
        recovered = ledger.reconcile(
            account_fingerprint="account", balances=balances("0.003", "0.052"),
            ownership=owners, evidence_sha256="two", open_order_counts={},
            sources_healthy=True, now=110,
        )
        row = recovered["assets"]["BTC"]["deficit_confirmation"]
        assert row["cycles"] == 0
        assert row["confirmed"] is False


def test_deficit_alert_requires_confirmation_and_is_sent_once():
    row = {
        "ownership_deficit": "0.0048",
        "deficit_confirmation": {"cycles": 1, "confirmed": False, "notified": False},
    }
    assert not Guard._confirmed_deficit_alert(row)
    row["deficit_confirmation"].update({"cycles": 3, "confirmed": True})
    assert Guard._confirmed_deficit_alert(row)
    row["deficit_confirmation"]["notified"] = True
    assert not Guard._confirmed_deficit_alert(row)


def test_active_orders_do_not_reset_confirmation_but_keep_contract_unhealthy():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        owners = {"BTC": {"dca": "0.001"}, "ETH": {"dca": "0.05"}}
        values = []
        for now, evidence in ((100, "one"), (115, "two"), (131, "three")):
            values.append(ledger.reconcile(
                account_fingerprint="account", balances=balances(), ownership=owners,
                evidence_sha256=evidence,
                open_order_counts={"BTC-USDT": 8, "ETH-USDT": 8},
                sources_healthy=True, now=now,
            ))
        btc = values[-1]["assets"]["BTC"]
        assert not values[-1]["healthy"]
        assert values[-1]["active_order_count"] == 16
        assert btc["confirmation_eligible"] is True
        assert btc["confirmation"]["cycles"] == 3
        assert btc["confirmation"]["confirmed"] is True


def test_confirmation_episode_survives_ledger_restart():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        owners = {"BTC": {"dca": "0.001"}, "ETH": {"dca": "0.05"}}
        first = UnifiedInventoryLedger(root).reconcile(
            account_fingerprint="account", balances=balances(), ownership=owners,
            evidence_sha256="one", open_order_counts={"BTC-USDT": 4},
            sources_healthy=True, now=100,
        )
        restarted = UnifiedInventoryLedger(root).reconcile(
            account_fingerprint="account", balances=balances(), ownership=owners,
            evidence_sha256="new-dynamic-evidence", open_order_counts={"BTC-USDT": 4},
            sources_healthy=True, now=115,
        )
        assert restarted["assets"]["BTC"]["episode_id"] == first["assets"]["BTC"]["episode_id"]
        assert restarted["assets"]["BTC"]["confirmation"]["cycles"] == 2


def test_existing_dust_job_migrates_without_reconfirmation():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        ledger.start_job(
            job_id="dust", asset="ETH", scope="unattributed_dust",
            pair="ETH-USDT", requested_quantity=Decimal("0.002271579092935063"),
            client_order_id="inv-dust", now=1,
        )
        ledger.finish_job("dust", status="DUST", error="below minimum", now=2)
        value = ledger.reconcile(
            account_fingerprint="account",
            balances=balances("0", "0.0528000000000000000000000001"),
            ownership={"BTC": {}, "ETH": {"dca": "0.050528420907064937"}},
            evidence_sha256="dynamic", open_order_counts={"ETH-USDT": 8},
            sources_healthy=True, now=100,
        )
        eth = value["assets"]["ETH"]
        assert eth["inventory_phase"] == "DUST"
        assert eth["last_notified_transition"] == "INVENTORY_DUST_CLASSIFIED"
        assert eth["confirmation"]["cycles"] == 0
        assert eth["confirmation_block_reason"] == "already_classified_dust"


def test_dust_recheck_is_persistent_and_short_tradable_blip_keeps_episode():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ledger = UnifiedInventoryLedger(root)
        ledger.start_job(
            job_id="dust", asset="ETH", scope="unattributed_dust",
            pair="ETH-USDT", requested_quantity=Decimal("0.0022"),
            client_order_id="inv-dust", now=1,
        )
        ledger.finish_job("dust", status="DUST", error="below minimum", now=2)
        value = ledger.reconcile(
            account_fingerprint="account", balances=balances("0", "0.0527"),
            ownership={"BTC": {}, "ETH": {"dca": "0.0505"}},
            evidence_sha256="stable", open_order_counts={},
            sources_healthy=True, now=100,
        )
        episode_id = value["assets"]["ETH"]["episode_id"]
        observations = [ledger.observe_dust_recheck(
            "ETH", episode_id=episode_id, evidence_sha256="tradable-evidence",
            tradable_quantity=Decimal("0.0022"), eligible=True,
            now=101 + index * 0.05,
        ) for index in range(500)]
        restarted = UnifiedInventoryLedger(root).observe_dust_recheck(
            "ETH", episode_id=episode_id, evidence_sha256="tradable-evidence",
            tradable_quantity=Decimal("0.0022"), eligible=True, now=126,
        )
        cancelled = UnifiedInventoryLedger(root).observe_dust_recheck(
            "ETH", episode_id=episode_id, evidence_sha256="dust-again",
            tradable_quantity=Decimal("0.0022"), eligible=False, now=117,
        )
        after = UnifiedInventoryLedger(root).reconcile(
            account_fingerprint="account", balances=balances("0", "0.0527"),
            ownership={"BTC": {}, "ETH": {"dca": "0.0505"}},
            evidence_sha256="dynamic", open_order_counts={},
            sources_healthy=True, now=118,
        )["assets"]["ETH"]
        assert observations[0]["cycles"] == 1
        assert observations[-1]["cycles"] == 500
        assert not any(row["confirmed"] for row in observations)
        assert restarted["cycles"] == 501
        assert not cancelled["active"]
        assert after["episode_id"] == episode_id
        assert after["inventory_phase"] == "DUST"
        assert after["last_notified_transition"] == "INVENTORY_DUST_CLASSIFIED"


def test_dust_recheck_requires_three_cycles_and_thirty_seconds_before_reopen():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ledger = UnifiedInventoryLedger(root)
        ledger.start_job(
            job_id="dust", asset="ETH", scope="unattributed_dust",
            pair="ETH-USDT", requested_quantity=Decimal("0.0022"),
            client_order_id="inv-dust", now=1,
        )
        ledger.finish_job("dust", status="DUST", error="below minimum", now=2)
        old_episode = ledger.reconcile(
            account_fingerprint="account", balances=balances("0", "0.0527"),
            ownership={"BTC": {}, "ETH": {"dca": "0.0505"}},
            evidence_sha256="stable", open_order_counts={},
            sources_healthy=True, now=100,
        )["assets"]["ETH"]["episode_id"]
        results = [ledger.observe_dust_recheck(
            "ETH", episode_id=old_episode, evidence_sha256="tradable",
            tradable_quantity=Decimal("0.0022"), eligible=True, now=now,
        ) for now in (101, 116, 132)]
        assert [row["cycles"] for row in results] == [1, 2, 3]
        assert [row["confirmed"] for row in results] == [False, False, True]
        new_episode = ledger.reopen_dust_episode(
            "ETH", expected_episode_id=old_episode,
            evidence_sha256="new-stability", quantity=Decimal("0.0022"), now=132,
        )
        assert new_episode != old_episode
        with ledger._connection() as connection:
            episode = connection.execute(
                "SELECT phase,last_notified_transition FROM inventory_episodes WHERE asset='ETH'"
            ).fetchone()
            assert connection.execute(
                "SELECT COUNT(*) FROM dust_rechecks WHERE asset='ETH'"
            ).fetchone()[0] == 0
        assert episode["phase"] == "DETECTED"
        assert episode["last_notified_transition"] == ""


def test_dca_v22_current_error_is_cleared_but_recovery_history_is_preserved():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        contract_path = root / "contract.json"
        contract_path.write_text("{}", encoding="utf-8")
        guard = Guard.__new__(Guard)
        guard.v22_observation_gate_path = contract_path
        guard.v21_max_age_seconds = 150
        guard.state = {"v22_observation": {
            "release_sha256": "a" * 64,
            "last_error": "Connection reset by peer",
            "current_error_since": 90,
            # Legacy state only had per-release counters. The first healthy
            # cycle must promote them into cumulative audit fields.
            "source_errors": 14,
            "integrity_errors": 0,
        }}
        healthy = {
            "release_sha256": "a" * 64,
            "runtime_gate_healthy": True,
            "pairs": {
                "BTC-FDUSD": {"event_id": "btc"},
                "ETH-FDUSD": {"event_id": "eth"},
            },
        }
        with patch("dca_live_guard.load_runtime_v22_contract", return_value=healthy):
            guard._observe_v22_contract(100)
        observation = guard.state["v22_observation"]
        assert "last_error" not in observation
        assert "current_error_since" not in observation
        assert observation["last_recovered_error"] == "Connection reset by peer"
        assert observation["last_recovered_at"] == 100
        assert observation["source_error_total"] == 14


def test_dust_that_cleared_then_reappeared_starts_new_confirmation_episode():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        ledger.start_job(
            job_id="old-dust", asset="ETH", scope="unattributed_dust",
            pair="ETH-USDT", requested_quantity=Decimal("0.0022"),
            client_order_id="inv-old-dust", now=1,
        )
        ledger.finish_job("old-dust", status="DUST", error="below minimum", now=2)
        first = ledger.reconcile(
            account_fingerprint="account", balances=balances("0", "0.0527"),
            ownership={"BTC": {}, "ETH": {"dca": "0.0505"}},
            evidence_sha256="first", open_order_counts={}, sources_healthy=True, now=100,
        )
        original_episode = first["assets"]["ETH"]["episode_id"]
        ledger.reconcile(
            account_fingerprint="account", balances=balances("0", "0.0505"),
            ownership={"BTC": {}, "ETH": {"dca": "0.0505"}},
            evidence_sha256="clear", open_order_counts={}, sources_healthy=True, now=110,
        )
        appeared = ledger.reconcile(
            account_fingerprint="account", balances=balances("0", "0.0527"),
            ownership={"BTC": {}, "ETH": {"dca": "0.0505"}},
            evidence_sha256="again", open_order_counts={}, sources_healthy=True, now=120,
        )
        eth = appeared["assets"]["ETH"]
        assert eth["episode_id"] != original_episode
        assert eth["inventory_phase"] == "DETECTED"
        assert eth["confirmation"]["cycles"] == 1


def test_asset_lease_and_bootstrap_cap_are_persistent():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        ledger.set_bootstrap_caps({"BTC": "0.00155"}, now=1)
        assert ledger.bootstrap_cap("BTC") == Decimal("0.00155")
        assert ledger.acquire_lease("BTC", "grid", now=10)
        assert not ledger.acquire_lease("BTC", "dca", now=11)
        ledger.release_lease("BTC", "grid")
        assert ledger.acquire_lease("BTC", "dca", now=12)
        ledger.consume_bootstrap_cap("BTC", now=13)
        assert ledger.bootstrap_cap("BTC") is None


def test_completed_job_requires_persisted_four_part_verification():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        job = ledger.start_job(
            job_id="job", asset="BTC", scope="test", pair="BTC-USDT",
            requested_quantity=Decimal("0.001"), client_order_id="inv-job",
        )
        assert not ledger.completed_job_verified(job)
        try:
            ledger.finish_job("job", status="COMPLETED", executed_quantity="0.001")
        except ValueError as exc:
            assert "four-part verification" in str(exc)
        else:
            raise AssertionError("unverified liquidation must not complete")
        verification = {
            "order_verified": True, "balance_verified": True,
            "no_active_orders": True, "requested_quantity_verified": True,
        }
        ledger.finish_job(
            "job", status="COMPLETED", executed_quantity="0.001",
            verification=verification,
        )
        with ledger._connection() as connection:
            completed = dict(connection.execute(
                "SELECT * FROM liquidation_jobs WHERE job_id='job'"
            ).fetchone())
        assert ledger.completed_job_verified(completed)


def test_account_fingerprint_cannot_change_between_guards():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        ledger.bind_account("same-account")
        ledger.bind_account("same-account")
        try:
            ledger.bind_account("different-account")
        except RuntimeError as exc:
            assert "fingerprint changed" in str(exc)
        else:
            raise AssertionError("different account must fail closed")


def test_exit_preflight_rejects_shared_deficit_and_owner_overreach():
    with tempfile.TemporaryDirectory() as directory:
        ledger = UnifiedInventoryLedger(Path(directory))
        ledger.reconcile(
            account_fingerprint="account", balances=balances("0.0015", "1"),
            ownership={
                "BTC": {"grid:grid": "0.001", "dca:dca": "0.001"},
                "ETH": {"dca:dca": "0"},
            },
            evidence_sha256="deficit", open_order_counts={},
            sources_healthy=True,
        )
        try:
            ledger.assert_exit_allowed(
                asset="BTC", exchange_total="0.0015",
                owner_key="grid:grid", requested_quantity="0.001",
            )
        except RuntimeError as exc:
            assert "ownership_deficit" in str(exc)
        else:
            raise AssertionError("shared deficit must block every owner exit")

        ledger.reconcile(
            account_fingerprint="account", balances=balances("0.003", "1"),
            ownership={
                "BTC": {"grid:grid": "0.001", "dca:dca": "0.001"},
                "ETH": {"dca:dca": "0"},
            },
            evidence_sha256="healthy", open_order_counts={},
            sources_healthy=True,
        )
        try:
            ledger.assert_exit_allowed(
                asset="BTC", exchange_total="0.003",
                owner_key="grid:grid", requested_quantity="0.0011",
            )
        except RuntimeError as exc:
            assert "exit exceeds" in str(exc)
        else:
            raise AssertionError("a strategy must not sell another owner's inventory")


def test_market_filter_zero_market_step_falls_back_to_lot_size(monkeypatch):
    class ReadClient:
        def request(self, *_args, **_kwargs):
            return {"symbols": [{"filters": [
                {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.00000000"},
                {"filterType": "LOT_SIZE", "stepSize": "0.00001000"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
            ]}]}

    guard = Guard.__new__(Guard)
    guard.binance_reads = ReadClient()
    step, minimum = guard._lot_filter("BTC-USDT")
    assert step == Decimal("0.00001000")
    assert minimum == Decimal("5")


def test_third_asset_commission_is_preserved_in_audit_details():
    metrics = Guard._emergency_fill_metrics("BTC-USDT", "SELL", {
        "executedQty": "0.00155", "cummulativeQuoteQty": "101.13",
        "fills": [{
            "commission": "0.00012469", "commissionAsset": "BNB",
            "price": "65247.1",
        }],
    })
    assert metrics["fee_quote"] == "0"
    assert metrics["fee_details"] == [{
        "asset": "BNB", "commission": "0.00012469",
        "fill_price": "65247.1",
    }]


class Exchange:
    def __init__(self, btc="0.003", eth="0"):
        self.orders = []
        self.btc = Decimal(btc)
        self.eth = Decimal(eth)
        self.responses = {}

    def account_balances(self):
        return balances(str(self.btc), str(self.eth))

    def open_orders(self, pair):
        return []

    def cancel_all_orders(self, pair):
        raise AssertionError("no order should need cancellation")

    def market_order(self, pair, side, amount, client_order_id=""):
        self.orders.append((pair, side, amount, client_order_id))
        if side == "SELL":
            if pair.startswith("BTC-"):
                self.btc -= amount
            elif pair.startswith("ETH-"):
                self.eth -= amount
        response = {
            "orderId": "one", "status": "FILLED", "executedQty": str(amount),
            "cummulativeQuoteQty": str(amount * Decimal("65000")),
            "fills": [{
                "price": "65000", "qty": str(amount),
                "commission": "0", "commissionAsset": "BNB",
            }],
        }
        self.responses[client_order_id] = response
        return response

    def order_by_client_id(self, pair, client_order_id):
        return self.responses.get(client_order_id)


def test_dca_integrity_flatten_includes_managed_inventory_when_net_base_is_zero():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        managed = root / "managed_inventory.json"
        managed.write_text(json.dumps({
            "pairs": {"BTC-USDT": {"managed_base": "0.0015"}},
        }), encoding="utf-8")
        guard = Guard.__new__(Guard)
        guard.managed_inventory_path = managed
        guard.state_path = root / "guard_state.json"
        guard.state = {"bots": {"bot": {
            "managed_base_target": "0.0015", "tripped_at": 100,
            "emergency_adjustments": [],
        }}}
        guard.emergency_exchange = Exchange()
        guard.inventory_ledger = UnifiedInventoryLedger(root / "shared")
        seed_ownership(guard.inventory_ledger, btc="0.0015")
        guard._lot_filter = lambda pair: (Decimal("0.00001"), Decimal("5"))
        guard._save = lambda: None
        result = guard._flatten({
            "pair": "BTC-USDT", "net_base": "0", "mark_price": "65000",
        }, "bot")
        assert result["exit_complete"]
        assert result["amount"] == "0.0015"
        assert guard.emergency_exchange.orders[0][2] == Decimal("0.0015")


def test_existing_latch_is_marked_pending_manual_without_order():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        managed = root / "managed_inventory.json"
        managed.write_text(json.dumps({
            "pairs": {"BTC-USDT": {"managed_base": "0.0015"}},
        }), encoding="utf-8")
        guard = Guard.__new__(Guard)
        guard.managed_inventory_path = managed
        guard.state = {"bots": {"dca-live-btcusdt-200": {
            "managed_base_target": "0.0015", "latest": {"net_base": "0"},
            "action_complete": True, "flatten_response": {"status": "not_required"},
            "recovery": {"phase": LATCHED, "exit_completed_at": None},
        }}}
        guard._migrate_incomplete_latched_inventory()
        bot = guard.state["bots"]["dca-live-btcusdt-200"]
        assert not bot["action_complete"]
        assert bot["manual_exit_required"]
        assert bot["exit_status"] == "pending_manual_existing_dca_inventory"


def test_first_unattributed_liquidation_honors_approved_cap_and_settles_usdt():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        guard = Guard.__new__(Guard)
        guard.inventory_ledger = UnifiedInventoryLedger(root / "shared")
        seed_ownership(guard.inventory_ledger, btc="0.0015")
        guard.inventory_ledger.set_bootstrap_caps({"BTC": "0.00155"}, now=1)
        guard.emergency_exchange = Exchange("0.004")
        guard.notification_path = root / "events.jsonl"
        guard._lot_filter = lambda pair: (Decimal("0.00001"), Decimal("5"))
        guard._price = lambda pair: Decimal("65000")
        row = {
            "owned_total": "0.0015",
            "unattributed": "0.0025",
            "confirmation": {"cycles": 3, "confirmed": True},
        }
        guard._liquidate_unattributed("BTC", row, "evidence")
        assert len(guard.emergency_exchange.orders) == 1
        pair, side, amount, client_id = guard.emergency_exchange.orders[0]
        assert (pair, side, amount) == ("BTC-USDT", "SELL", Decimal("0.00155"))
        assert client_id.startswith("inv-")
        assert guard.inventory_ledger.bootstrap_cap("BTC") is None


def test_audit_failure_cannot_rewrite_completed_fill_or_reuse_cap():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        guard = Guard.__new__(Guard)
        guard.inventory_ledger = UnifiedInventoryLedger(root / "shared")
        seed_ownership(guard.inventory_ledger, btc="0.0015")
        guard.inventory_ledger.set_bootstrap_caps({"BTC": "0.00155"}, now=1)
        guard.emergency_exchange = Exchange("0.004")
        guard.notification_path = root / "events.jsonl"
        guard._lot_filter = lambda pair: (Decimal("0.00001"), Decimal("5"))
        guard._price = lambda pair: Decimal("65000")
        guard._audit = lambda event, **details: (
            (_ for _ in ()).throw(OSError("simulated audit disk failure"))
            if event == "inventory_liquidation_completed" else None
        )
        with patch.object(guard, "_audit", side_effect=guard._audit):
            try:
                guard._liquidate_unattributed("BTC", {
                    "owned_total": "0.0015", "unattributed": "0.0025",
                    "confirmation": {"cycles": 3, "confirmed": True},
                }, "evidence")
            except OSError as exc:
                assert "audit" in str(exc)
            else:
                raise AssertionError("audit failure should remain observable")
        with guard.inventory_ledger._connection() as connection:
            job = dict(connection.execute(
                "SELECT * FROM liquidation_jobs WHERE scope='unattributed'"
            ).fetchone())
        assert job["status"] == "COMPLETED"
        assert guard.inventory_ledger.completed_job_verified(job)
        assert guard.inventory_ledger.bootstrap_cap("BTC") is None
        assert len(guard.emergency_exchange.orders) == 1
        assert guard.inventory_ledger.pending_events() == []


def test_unattributed_below_minimum_is_persisted_as_dust_without_order():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        guard = Guard.__new__(Guard)
        guard.inventory_ledger = UnifiedInventoryLedger(root / "shared")
        seed_ownership(guard.inventory_ledger, eth="0.0505285")
        guard.inventory_ledger.set_bootstrap_caps({"ETH": "0.0022715"}, now=1)
        guard.emergency_exchange = Exchange("0")
        guard.emergency_exchange.account_balances = lambda: balances("0", "0.0528")
        guard.emergency_exchange.open_orders = lambda pair: [{"orderId": "active"}]
        guard.notification_path = root / "events.jsonl"
        guard._lot_filter = lambda pair: (Decimal("0.0001"), Decimal("5"))
        guard._price = lambda pair: Decimal("1926")
        row = {
            "owned_total": "0.0505285", "unattributed": "0.0022715",
            "confirmation": {"cycles": 3, "confirmed": True},
        }
        guard._liquidate_unattributed("ETH", row, "evidence")
        assert guard.emergency_exchange.orders == []
        with guard.inventory_ledger._connection() as connection:
            job = connection.execute(
                "SELECT status,error FROM liquidation_jobs"
            ).fetchone()
        assert job["status"] == "DUST"
        assert "minimum_notional=5" in job["error"]
        assert not guard.notification_path.exists()


def test_repeated_dust_reconciliation_writes_classification_audit_once():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        guard = Guard.__new__(Guard)
        guard.inventory_ledger = UnifiedInventoryLedger(root / "shared")
        seed_ownership(guard.inventory_ledger, eth="0.0505")
        guard.emergency_exchange = Exchange("0")
        guard.emergency_exchange.account_balances = lambda: balances("0", "0.0527")
        guard._lot_filter = lambda pair: (Decimal("0.0001"), Decimal("5"))
        guard._price = lambda pair: Decimal("1900")
        audits = []
        guard._audit = lambda event, **details: audits.append((event, details))
        row = {
            "owned_total": "0.0505", "unattributed": "0.0022",
            "last_notified_transition": "INVENTORY_UNATTRIBUTED_CONFIRMED",
            "confirmation": {"cycles": 3, "confirmed": True},
        }
        guard._liquidate_unattributed("ETH", row, "same-episode")
        guard._liquidate_unattributed("ETH", row, "same-episode")
        assert [event for event, _ in audits] == ["inventory_dust_classified"]
        assert guard.emergency_exchange.orders == []


def test_unattributed_alert_waits_for_persisted_confirmation():
    base = {
        "unattributed": "0.0075",
        "inventory_phase": "DETECTED",
        "confirmation": {"cycles": 0, "confirmed": False},
    }
    assert not Guard._confirmed_unattributed_alert(base)
    base["confirmation"] = {"cycles": 3, "confirmed": True}
    assert not Guard._confirmed_unattributed_alert(base)
    base["inventory_phase"] = "CONFIRMED"
    assert Guard._confirmed_unattributed_alert(base)
    base["unattributed"] = "0"
    assert not Guard._confirmed_unattributed_alert(base)
