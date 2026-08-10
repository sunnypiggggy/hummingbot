import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from decimal import Decimal
from pathlib import Path

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_guard"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "test" / "support"))

from account_inventory import UnifiedInventoryLedger  # noqa: E402
from dca_live_guard import ApiClient, BinanceEmergencyClient, Guard  # noqa: E402
from grid_live_guard import Guard as GridGuard  # noqa: E402
from emergency_execution import (  # noqa: E402
    execute_market_liquidation, verify_market_liquidation,
)
from risk_scenario_server import RiskScenarioServer  # noqa: E402
from runtime_endpoints import binance_api_base  # noqa: E402
from telegram_notifications import TelegramChannelClient, TelegramOutbox  # noqa: E402


FIXTURE = ROOT / "test" / "fixtures" / "risk_scenarios" / "aug10_inventory.json"


def scenario_document():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def enable_scenario(monkeypatch, server, scenario_id="test"):
    monkeypatch.setenv("GUARD_SCENARIO_MODE", "true")
    monkeypatch.setenv("GUARD_SCENARIO_ID", scenario_id)
    monkeypatch.setenv("BINANCE_API_BASE_URL", server.base_url)
    monkeypatch.setenv("TELEGRAM_API_BASE_URL", server.base_url)


def client(server):
    return BinanceEmergencyClient("scenario-key", "scenario-secret", server.base_url)


def test_production_mode_rejects_endpoint_redirect(monkeypatch):
    monkeypatch.delenv("GUARD_SCENARIO_MODE", raising=False)
    monkeypatch.setenv("BINANCE_API_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="GUARD_SCENARIO_MODE"):
        binance_api_base()


def test_scenario_mode_cannot_fall_back_to_official_binance(monkeypatch):
    monkeypatch.setenv("GUARD_SCENARIO_MODE", "true")
    monkeypatch.setenv("GUARD_SCENARIO_ID", "no-mainnet-fallback")
    monkeypatch.delenv("BINANCE_API_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="explicit isolated endpoint"):
        binance_api_base()


def test_scenario_mode_refuses_non_scenario_credentials(monkeypatch):
    monkeypatch.setenv("GUARD_SCENARIO_MODE", "true")
    monkeypatch.setenv("GUARD_SCENARIO_ID", "credential-isolation")
    with pytest.raises(RuntimeError, match="non-scenario Binance credentials"):
        BinanceEmergencyClient("production-looking-key", "secret", "http://127.0.0.1:1")


def test_inventory_health_is_false_while_any_open_order_exists(tmp_path):
    ledger = UnifiedInventoryLedger(tmp_path)
    status = ledger.reconcile(
        account_fingerprint="scenario", balances={
            "BTC": {"free": "1", "locked": "0", "total": "1"},
            "ETH": {"free": "1", "locked": "0", "total": "1"},
        }, ownership={"BTC": {"dca": "1"}, "ETH": {"dca": "1"}},
        evidence_sha256="e", open_order_counts={"BTC-USDT": 1},
        sources_healthy=True, now=1,
    )
    assert status["healthy"] is False
    assert all(not row["confirmation"]["confirmed"] for row in status["assets"].values())


@pytest.mark.parametrize("blocking_mechanism", [
    "v22_weekly_buy_gate", "fomc_gate", "strategy_loss_breaker",
    "strategy_drawdown_breaker", "portfolio_loss_breaker",
    "portfolio_drawdown_breaker", "position_protection",
])
def test_each_enabled_mechanism_independently_blocks_the_aggregate_gate(blocking_mechanism):
    guard = Guard.__new__(Guard)
    recovery_mechanisms = {
        "strategy_loss_breaker", "strategy_drawdown_breaker",
        "portfolio_loss_breaker", "portfolio_drawdown_breaker",
        "position_protection",
    }
    recovery = {
        "phase": "EXITING" if blocking_mechanism in recovery_mechanisms else "ACTIVE",
        "mechanism": blocking_mechanism if blocking_mechanism in recovery_mechanisms else "",
    }
    guard.state = {"bots": {"bot": {"recovery": recovery}}}
    guard._macro_gate = lambda: {
        "healthy": True,
        "buy_enabled": blocking_mechanism != "fomc_gate",
        "sell_enabled": blocking_mechanism != "fomc_gate",
        "reason": "scenario-fomc", "active_lease_ids": [],
    }
    guard._v21_gate = lambda: {
        "healthy": True, "reason": "scenario-v22", "pairs": {
            "BTC-USDT": {
                "buy_enabled": blocking_mechanism != "v22_weekly_buy_gate",
                "reason": "scenario-v22", "source_pair": "BTC-FDUSD",
                "event_id": "scenario-event",
            }
        },
    }
    applied = []
    guard._set_effective_gates = lambda bot, snapshot, **values: applied.append(values) or {
        "status": "applied"
    }
    guard._audit = lambda *_args, **_kwargs: None
    guard._apply_aggregate_gates(
        {"bot": {"pair": "BTC-USDT", "database": "unused"}},
        risk_actions_enabled=True,
    )
    assert applied and applied[0]["buy_enabled"] is False
    if blocking_mechanism == "v22_weekly_buy_gate":
        assert applied[0]["sell_enabled"] is True
    else:
        assert applied[0]["sell_enabled"] is False


def test_all_clear_mechanisms_are_required_before_aggregate_gate_recovers():
    guard = Guard.__new__(Guard)
    guard.state = {"bots": {"bot": {"recovery": {"phase": "ACTIVE"}}}}
    guard._macro_gate = lambda: {
        "healthy": True, "buy_enabled": True, "sell_enabled": True,
        "reason": "clear", "active_lease_ids": [],
    }
    guard._v21_gate = lambda: {
        "healthy": True, "reason": "clear", "pairs": {
            "BTC-USDT": {
                "buy_enabled": True, "reason": "clear",
                "source_pair": "BTC-FDUSD", "event_id": "clear",
            }
        },
    }
    applied = []
    guard._set_effective_gates = lambda bot, snapshot, **values: applied.append(values) or {
        "status": "applied"
    }
    guard._audit = lambda *_args, **_kwargs: None
    guard._apply_aggregate_gates(
        {"bot": {"pair": "BTC-USDT", "database": "unused"}},
        risk_actions_enabled=True,
    )
    assert applied[0]["buy_enabled"] is True
    assert applied[0]["sell_enabled"] is True


def test_aggregate_gate_crosses_real_http_control_boundary(monkeypatch, tmp_path):
    with RiskScenarioServer(scenario_document()) as server:
        enable_scenario(monkeypatch, server, "aggregate-http")
        monkeypatch.setenv("USERNAME", "scenario")
        monkeypatch.setenv("PASSWORD", "scenario")
        database = tmp_path / "bot.sqlite"
        connection = sqlite3.connect(database)
        connection.execute("CREATE TABLE Controllers (id TEXT, timestamp FLOAT, config JSON)")
        connection.execute(
            "INSERT INTO Controllers VALUES (?,?,?)",
            ("dca_btcusdt_live_200", 1, json.dumps({
                "macro_buy_enabled": True, "macro_sell_enabled": True,
            })),
        )
        connection.commit()
        connection.close()
        guard = Guard.__new__(Guard)
        guard.api = ApiClient(server.base_url)
        guard.notification_path = tmp_path / "events.jsonl"
        guard.state = {"bots": {"bot": {"recovery": {"phase": "ACTIVE"}}}}
        guard._macro_gate = lambda: {
            "healthy": True, "buy_enabled": False, "sell_enabled": True,
            "reason": "fomc-buy-block", "active_lease_ids": ["fomc"],
        }
        guard._v21_gate = lambda: {
            "healthy": True, "reason": "v22-clear", "pairs": {
                "BTC-USDT": {
                    "buy_enabled": True, "reason": "v22-clear",
                    "source_pair": "BTC-FDUSD", "event_id": "v22-clear",
                }
            },
        }
        guard._apply_aggregate_gates({"bot": {
            "pair": "BTC-USDT", "database": str(database),
        }}, risk_actions_enabled=True)
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        assert len(remote["controller_updates"]) == 1
        payload = remote["controller_updates"][0]["payload"]
        assert payload["macro_buy_enabled"] is False
        assert payload["macro_sell_enabled"] is True
        assert payload["macro_decision_id"].startswith("risk-aggregate:")


def test_aug10_redacted_incident_crosses_http_and_waits_real_confirmation(monkeypatch, tmp_path):
    with RiskScenarioServer(scenario_document()) as server:
        enable_scenario(monkeypatch, server, "aug10")
        exchange = client(server)
        ledger = UnifiedInventoryLedger(tmp_path / "ledger")
        ledger.bind_account("scenario-account")
        ledger.set_bootstrap_caps({
            "BTC": "0.0015548576728614218",
            "ETH": "0.002271579092935063",
        }, now=1)
        ownership = {
            "BTC": {"grid:grid-live-fdusd-400": "0", "dca:dca-live-btcusdt-200": "0.001499762327138578"},
            "ETH": {"grid:grid-live-fdusd-400": "0", "dca:dca-live-ethusdt-200": "0.050528420907064937"},
        }
        started = time.time()
        status = None
        for cycle in range(3):
            balances = exchange.account_balances()
            status = ledger.reconcile(
                account_fingerprint="scenario-account", balances=balances,
                ownership=ownership, evidence_sha256="stable-evidence",
                open_order_counts={}, sources_healthy=True,
                confirmation_cycles=3, confirmation_seconds=30,
            )
            if cycle < 2:
                time.sleep(15.05)
        assert time.time() - started >= 30
        assert status["assets"]["BTC"]["confirmation"]["confirmed"]
        assert status["assets"]["ETH"]["confirmation"]["confirmed"]

        guard = Guard.__new__(Guard)
        guard.inventory_ledger = ledger
        guard.emergency_exchange = exchange
        guard.notification_path = tmp_path / "telegram_events.jsonl"
        guard._liquidate_unattributed(
            "BTC", status["assets"]["BTC"], "stable-evidence"
        )
        # Reconcile after the BTC fill before evaluating independent ETH dust.
        after_btc = ledger.reconcile(
            account_fingerprint="scenario-account",
            balances=exchange.account_balances(), ownership=ownership,
            evidence_sha256="stable-evidence", open_order_counts={},
            sources_healthy=True, confirmation_cycles=1,
            confirmation_seconds=0, now=time.time() + 1,
        )
        guard._liquidate_unattributed(
            "ETH", status["assets"]["ETH"], "stable-evidence"
        )
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        btc_orders = [row for row in remote["orders"] if row["symbol"] == "BTCUSDT"]
        assert len(btc_orders) == 1
        assert btc_orders[0]["executedQty"] == "0.00155"
        assert btc_orders[0]["cummulativeQuoteQty"] == "101.1330050"
        assert Decimal(remote["balances"]["BTC"]["free"]) == Decimal("0.00150462")
        assert Decimal(remote["balances"]["ETH"]["free"]) == Decimal("0.0528")
        assert Decimal(remote["balances"]["BNB"]["free"]) == Decimal("0.09987531")
        assert Decimal(after_btc["assets"]["BTC"]["unattributed"]) == Decimal(
            "0.000004857672861422"
        )
        with ledger._connection() as connection:
            jobs = [dict(row) for row in connection.execute(
                "SELECT status,verification_json,fee_details FROM liquidation_jobs ORDER BY created_at"
            )]
        assert [row["status"] for row in jobs] == ["COMPLETED", "DUST"]
        verification = json.loads(jobs[0]["verification_json"])
        assert all(verification[key] for key in (
            "order_verified", "balance_verified", "no_active_orders",
            "requested_quantity_verified",
        ))
        assert "0.00012469" in jobs[0]["fee_details"]


def test_grid_forced_exit_crosses_http_and_persists_same_verification(monkeypatch, tmp_path):
    document = scenario_document()
    document["faults"]["POST /api/v3/order"] = [{}]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "grid-exit")
        guard = GridGuard.__new__(GridGuard)
        guard.emergency_exchange = client(server)
        guard.inventory_ledger = UnifiedInventoryLedger(tmp_path / "ledger")
        guard.inventory_ledger.bind_account("scenario")
        guard.inventory_ledger.reconcile(
            account_fingerprint="scenario",
            balances=client(server).account_balances(),
            ownership={
                "BTC": {"grid:grid-live-fdusd-400": "0.0005"},
                "ETH": {"grid:grid-live-fdusd-400": "0"},
            },
            evidence_sha256="grid-owned", open_order_counts={},
            sources_healthy=True,
        )
        guard.manifest = {
            "reservations": {"FDUSD": {"base": {"BTC": "0.0005"}}},
        }
        guard.state = {"bots": {"grid-live-fdusd-400": {}}}
        guard.save = lambda: None
        bot = guard.state["bots"]["grid-live-fdusd-400"]
        result = guard.flatten_deltas("FDUSD", {"pairs": {
            "BTC-FDUSD": {"net_base": "0", "mark": "65247.10"},
        }}, bot)
        assert result["BTC-FDUSD"]["status"] == "filled"
        assert result["BTC-FDUSD"]["executed_qty"] == "0.0005"
        assert all(result["BTC-FDUSD"]["verification"][key] for key in (
            "order_verified", "balance_verified", "no_active_orders",
            "requested_quantity_verified",
        ))
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        assert [(row["symbol"], row["executedQty"]) for row in remote["orders"]] == [
            ("BTCFDUSD", "0.0005")
        ]
        with guard.inventory_ledger._connection() as connection:
            job = dict(connection.execute("SELECT * FROM liquidation_jobs").fetchone())
        assert guard.inventory_ledger.completed_job_verified(job)


def _execute(monkeypatch, tmp_path, document, target=Decimal("0.001")):
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "fault")
        exchange = client(server)
        ledger = UnifiedInventoryLedger(tmp_path / f"ledger-{time.time_ns()}")
        ledger.bind_account("scenario")
        ledger.start_job(
            job_id="job", asset="BTC", scope="test", pair="BTC-USDT",
            requested_quantity=target, client_order_id="inv-fault",
        )
        before = exchange.account_balances()["BTC"]["total"]
        with ledger.lease("BTC", "test-holder", ttl_seconds=45):
            response = execute_market_liquidation(
                exchange=exchange, ledger=ledger, job_id="job", pair="BTC-USDT",
                side="SELL", target_quantity=target, client_order_id="inv-fault",
                step_size=Decimal("0.00001"), minimum_notional=Decimal("5"),
                mark_price=Decimal("65247.10"), lease_asset="BTC",
                lease_holder="test-holder", poll_seconds=0,
            )
            verification = verify_market_liquidation(
                exchange=exchange, pair="BTC-USDT", response=response,
                requested_quantity=target, before_total=before,
                step_size=Decimal("0.00001"), minimum_notional=Decimal("5"),
                mark_price=Decimal("65247.10"),
            )
            ledger.finish_job(
                "job", status="COMPLETED",
                executed_quantity=response["executedQty"],
                quote_quantity=response["cummulativeQuoteQty"],
                verification=verification,
            )
        return response, ledger, requests.get(
            f"{server.base_url}/scenario/state", timeout=5
        ).json()


def test_timeout_after_fill_is_recovered_by_client_id_without_duplicate(monkeypatch, tmp_path):
    document = scenario_document()
    document["faults"]["POST /api/v3/order"] = [{"drop_response": True}]
    response, ledger, remote = _execute(monkeypatch, tmp_path, document)
    assert response["executedQty"] == "0.001"
    assert len(remote["orders"]) == 1
    assert len(ledger.attempts("job")) == 1


def test_timeout_with_temporarily_invisible_order_recovers_next_cycle(monkeypatch, tmp_path):
    document = scenario_document()
    document["faults"]["POST /api/v3/order"] = [{
        "drop_response": True, "visibility_misses": 1,
    }]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "visibility")
        exchange = client(server)
        ledger = UnifiedInventoryLedger(tmp_path / "ledger")
        ledger.bind_account("scenario")
        ledger.start_job(
            job_id="job", asset="BTC", scope="test", pair="BTC-USDT",
            requested_quantity="0.001", client_order_id="inv-visibility",
        )
        with ledger.lease("BTC", "holder", ttl_seconds=45):
            with pytest.raises(requests.RequestException):
                execute_market_liquidation(
                    exchange=exchange, ledger=ledger, job_id="job",
                    pair="BTC-USDT", side="SELL",
                    target_quantity=Decimal("0.001"),
                    client_order_id="inv-visibility",
                    step_size=Decimal("0.00001"), minimum_notional=Decimal("5"),
                    mark_price=Decimal("65247.10"), lease_asset="BTC",
                    lease_holder="holder", poll_seconds=0,
                )
        # The next cycle queries the same client id. It must recover the fill
        # instead of submitting a second order after eventual consistency.
        with ledger.lease("BTC", "holder-2", ttl_seconds=45):
            response = execute_market_liquidation(
                exchange=exchange, ledger=ledger, job_id="job",
                pair="BTC-USDT", side="SELL",
                target_quantity=Decimal("0.001"),
                client_order_id="inv-visibility",
                step_size=Decimal("0.00001"), minimum_notional=Decimal("5"),
                mark_price=Decimal("65247.10"), lease_asset="BTC",
                lease_holder="holder-2", poll_seconds=0,
            )
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        assert response["executedQty"] == "0.001"
        assert len(remote["orders"]) == 1


def test_terminal_partial_fill_uses_one_deterministic_residual_child(monkeypatch, tmp_path):
    document = scenario_document()
    document["faults"]["POST /api/v3/order"] = [
        {"fill_fraction": "0.4", "status": "EXPIRED"},
        {"fill_fraction": "1", "status": "FILLED"},
    ]
    response, ledger, remote = _execute(monkeypatch, tmp_path, document)
    assert Decimal(response["executedQty"]) == Decimal("0.001")
    assert len(remote["orders"]) == 2
    assert [row["clientOrderId"] for row in remote["orders"]] == [
        "inv-fault", "inv-fault-r1",
    ]
    assert len(ledger.attempts("job")) == 2


def test_terminal_rejection_retries_only_with_deterministic_child(monkeypatch, tmp_path):
    document = scenario_document()
    document["faults"]["POST /api/v3/order"] = [
        {"fill_fraction": "0", "status": "REJECTED"},
        {"fill_fraction": "1", "status": "FILLED"},
    ]
    response, ledger, remote = _execute(monkeypatch, tmp_path, document)
    assert response["executedQty"] == "0.001"
    assert [row["clientOrderId"] for row in remote["orders"]] == [
        "inv-fault", "inv-fault-r1",
    ]
    assert [row["status"] for row in ledger.attempts("job")] == [
        "REJECTED", "FILLED",
    ]


def test_timestamp_rejection_syncs_clock_and_retries_signed_account(monkeypatch):
    document = scenario_document()
    document["faults"]["GET /api/v3/account"] = [{
        "status": 400, "code": -1021, "message": "Timestamp outside recvWindow",
    }]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "clock")
        balances = client(server).account_balances()
        assert balances["BTC"]["total"] == Decimal("0.00305462")
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        assert remote["faults_remaining"]["GET /api/v3/account"] == 0


def test_public_exchange_filter_change_is_consumed_over_real_http(monkeypatch):
    document = scenario_document()
    document["faults"]["GET /api/v3/exchangeInfo"] = [{
        "filters": {
            "lot_step": "0.0001", "market_step": "0.0002",
            "min_notional": "12",
        },
    }]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "filter-change")
        step, minimum = Guard._lot_filter("BTC-USDT")
        assert step == Decimal("0.0002")
        assert minimum == Decimal("12")


@pytest.mark.parametrize("quantity", ["0.001001", "0.00001"])
def test_exchange_simulator_independently_rejects_step_or_notional_violation(
    monkeypatch, quantity,
):
    document = scenario_document()
    document["faults"]["POST /api/v3/order"] = [{}]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "exchange-filter-enforcement")
        with pytest.raises(RuntimeError, match="Filter failure"):
            client(server).market_order(
                "BTC-USDT", "SELL", Decimal(quantity), "invalid-filter-order"
            )
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        assert remote["orders"] == []


def test_delayed_cancel_keeps_inventory_unhealthy_until_exchange_confirms(monkeypatch, tmp_path):
    document = scenario_document()
    document["open_orders"] = [{"symbol": "BTCUSDT", "orderId": 77}]
    document["faults"]["DELETE /api/v3/openOrders"] = [{"delay_queries": 2}]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "cancel-delay")
        exchange = client(server)
        assert exchange.open_orders("BTC-USDT")
        exchange.cancel_all_orders("BTC-USDT")
        pending = exchange.open_orders("BTC-USDT")
        assert pending
        ledger = UnifiedInventoryLedger(tmp_path)
        ownership = {
            "BTC": {"dca": "0.001499762327138578"},
            "ETH": {"dca": "0.050528420907064937"},
        }
        blocked = ledger.reconcile(
            account_fingerprint="scenario", balances=exchange.account_balances(),
            ownership=ownership, evidence_sha256="cancel-pending",
            open_order_counts={"BTC-USDT": len(pending)}, sources_healthy=True,
        )
        assert blocked["healthy"] is False
        assert exchange.open_orders("BTC-USDT")
        assert exchange.open_orders("BTC-USDT") == []

        status = ledger.reconcile(
            account_fingerprint="scenario", balances=exchange.account_balances(),
            ownership=ownership,
            evidence_sha256="cancel-confirmed",
            open_order_counts={"BTC-USDT": len(exchange.open_orders("BTC-USDT"))},
            sources_healthy=True,
        )
        assert status["healthy"] is True


def test_stale_balance_after_fill_prevents_completed_job(monkeypatch, tmp_path):
    document = scenario_document()
    stale = json.loads(json.dumps(document["balances"]))
    document["faults"]["GET /api/v3/account"] = [
        {}, {"balances": stale},
    ]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "stale-balance")
        exchange = client(server)
        ledger = UnifiedInventoryLedger(tmp_path)
        ledger.start_job(
            job_id="stale-job", asset="BTC", scope="test", pair="BTC-USDT",
            requested_quantity="0.001", client_order_id="stale-client",
        )
        before = exchange.account_balances()["BTC"]["total"]
        with ledger.lease("BTC", "stale-holder", ttl_seconds=45):
            response = execute_market_liquidation(
                exchange=exchange, ledger=ledger, job_id="stale-job",
                pair="BTC-USDT", side="SELL",
                target_quantity=Decimal("0.001"), client_order_id="stale-client",
                step_size=Decimal("0.00001"), minimum_notional=Decimal("5"),
                mark_price=Decimal("65247.10"), lease_asset="BTC",
                lease_holder="stale-holder", poll_seconds=0,
            )
            verification = verify_market_liquidation(
                exchange=exchange, pair="BTC-USDT", response=response,
                requested_quantity=Decimal("0.001"), before_total=before,
                step_size=Decimal("0.00001"), minimum_notional=Decimal("5"),
                mark_price=Decimal("65247.10"), ledger=ledger,
                lease_asset="BTC", lease_holder="stale-holder",
            )
            assert verification["order_verified"] is True
            assert verification["balance_verified"] is False
            with pytest.raises(ValueError, match="verification did not pass"):
                ledger.finish_job(
                    "stale-job", status="COMPLETED",
                    executed_quantity=response["executedQty"],
                    quote_quantity=response["cummulativeQuoteQty"],
                    verification=verification,
                )


@pytest.mark.parametrize("commission_asset,commission", [
    ("BTC", "0.000001"), ("USDT", "0.1"), ("BNB", "0.0001"),
])
def test_base_quote_and_third_asset_commissions_mutate_correct_balance(
    monkeypatch, tmp_path, commission_asset, commission,
):
    document = scenario_document()
    document["faults"]["POST /api/v3/order"] = [{
        "commission": commission, "commission_asset": commission_asset,
    }]
    initial = Decimal(document["balances"][commission_asset]["free"])
    _response, _ledger, remote = _execute(monkeypatch, tmp_path, document)
    final = Decimal(remote["balances"][commission_asset]["free"])
    if commission_asset == "BTC":
        assert final == initial - Decimal("0.001") - Decimal(commission)
    elif commission_asset == "USDT":
        assert final == initial + Decimal("65.24710") - Decimal(commission)
    else:
        assert final == initial - Decimal(commission)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_exchange_transport_failures_create_zero_economic_fills(monkeypatch, tmp_path, status):
    document = scenario_document()
    document["faults"]["POST /api/v3/order"] = [{
        "mode": "http_error", "status": status,
    }]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, f"http-{status}")
        exchange = client(server)
        ledger = UnifiedInventoryLedger(tmp_path / f"ledger-{status}")
        ledger.bind_account("scenario")
        ledger.start_job(
            job_id="job", asset="BTC", scope="test", pair="BTC-USDT",
            requested_quantity="0.001", client_order_id="inv-http",
        )
        with ledger.lease("BTC", "holder", ttl_seconds=45):
            with pytest.raises(RuntimeError):
                execute_market_liquidation(
                    exchange=exchange, ledger=ledger, job_id="job",
                    pair="BTC-USDT", side="SELL",
                    target_quantity=Decimal("0.001"),
                    client_order_id="inv-http", step_size=Decimal("0.00001"),
                    minimum_notional=Decimal("5"), mark_price=Decimal("65247.10"),
                    lease_asset="BTC", lease_holder="holder", poll_seconds=0,
                )
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        assert remote["orders"] == []
        assert ledger.attempts("job")[0]["status"] == "UNKNOWN"


def test_shared_asset_lease_allows_only_one_concurrent_exit(tmp_path):
    first = UnifiedInventoryLedger(tmp_path)
    second = UnifiedInventoryLedger(tmp_path)
    barrier = threading.Barrier(2)
    results = []

    def acquire(ledger, holder):
        barrier.wait()
        results.append(ledger.acquire_lease("BTC", holder, ttl_seconds=30, now=10))

    threads = [
        threading.Thread(target=acquire, args=(first, "grid")),
        threading.Thread(target=acquire, args=(second, "dca")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [False, True]


def test_blocking_exchange_call_renews_lease_until_response_returns(tmp_path):
    ledger = UnifiedInventoryLedger(tmp_path)
    assert ledger.acquire_lease("BTC", "slow-executor", ttl_seconds=1)

    class SlowExchange:
        def order_by_client_id(self, _pair, _client_order_id):
            return None

        def market_order(self, _pair, _side, quantity, _client_order_id):
            time.sleep(1.6)
            return {
                "status": "FILLED", "orderId": "slow-order",
                "executedQty": str(quantity), "cummulativeQuoteQty": "65",
                "fills": [{"price": "65000", "qty": str(quantity)}],
            }

    result = {}
    failure = []

    def execute():
        try:
            result.update(execute_market_liquidation(
                exchange=SlowExchange(), ledger=ledger, job_id="slow-job",
                pair="BTC-USDT", side="SELL",
                target_quantity=Decimal("0.001"),
                client_order_id="slow-client", step_size=Decimal("0.00001"),
                minimum_notional=Decimal("5"), mark_price=Decimal("65000"),
                lease_asset="BTC", lease_holder="slow-executor",
                lease_ttl_seconds=1, poll_attempts=1, poll_seconds=0,
            ))
        except Exception as exc:  # pragma: no cover - assertion reports details
            failure.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    time.sleep(1.2)
    contender = UnifiedInventoryLedger(tmp_path)
    assert contender.acquire_lease("BTC", "competing-executor", ttl_seconds=1) is False
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert not failure
    assert result["executedQty"] == "0.001"


def test_sqlite_writer_waits_for_real_lock_and_recovers(tmp_path):
    ledger = UnifiedInventoryLedger(tmp_path)
    blocker = sqlite3.connect(ledger.database_path, timeout=5)
    blocker.execute("BEGIN IMMEDIATE")
    outcome = []

    def writer():
        try:
            outcome.append(ledger.acquire_lease("BTC", "guard", ttl_seconds=30, now=1))
        except Exception as exc:  # pragma: no cover - asserted below
            outcome.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.2)
    blocker.rollback()
    blocker.close()
    thread.join(timeout=5)
    assert outcome == [True]


def test_process_death_inside_sqlite_transaction_rolls_back_wal(tmp_path):
    ledger = UnifiedInventoryLedger(tmp_path)
    code = (
        "import os,sqlite3,sys;"
        "c=sqlite3.connect(sys.argv[1]);c.execute('BEGIN IMMEDIATE');"
        "c.execute(\"INSERT OR REPLACE INTO meta(key,value) VALUES('crash','uncommitted')\");"
        "os._exit(137)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(ledger.database_path)], check=False
    )
    assert result.returncode != 0
    reopened = UnifiedInventoryLedger(tmp_path)
    with reopened._connection() as connection:
        assert connection.execute(
            "SELECT value FROM meta WHERE key='crash'"
        ).fetchone() is None
    assert reopened.acquire_lease("BTC", "after-crash", now=2)


def test_partial_status_json_from_killed_writer_never_replaces_last_contract(tmp_path):
    ledger = UnifiedInventoryLedger(tmp_path)
    arguments = dict(
        account_fingerprint="scenario",
        balances={
            "BTC": {"free": "1", "locked": "0", "total": "1"},
            "ETH": {"free": "1", "locked": "0", "total": "1"},
        },
        ownership={"BTC": {"dca": "1"}, "ETH": {"dca": "1"}},
        open_order_counts={}, sources_healthy=True,
    )
    expected = ledger.reconcile(
        **arguments, evidence_sha256="complete", now=1,
    )
    temporary = ledger.status_path.with_suffix(ledger.status_path.suffix + ".tmp")
    script = (
        "import os,pathlib;"
        f"pathlib.Path({str(temporary)!r}).write_text('{{\"schema\":',encoding='utf-8');"
        "os._exit(137)"
    )
    process = subprocess.run([sys.executable, "-c", script], check=False)
    assert process.returncode != 0
    assert temporary.read_text(encoding="utf-8") == '{"schema":'
    assert json.loads(ledger.status_path.read_text(encoding="utf-8")) == expected

    ledger.reconcile(**arguments, evidence_sha256="next", now=2)
    assert not temporary.exists()
    assert json.loads(
        ledger.status_path.read_text(encoding="utf-8")
    )["evidence_sha256"] == "next"


def test_locked_unattributed_inventory_fails_closed_without_consuming_cap(monkeypatch, tmp_path):
    document = scenario_document()
    document["balances"]["BTC"] = {"free": "0.0015", "locked": "0.001554857672861422"}
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "locked")
        guard = Guard.__new__(Guard)
        guard.inventory_ledger = UnifiedInventoryLedger(tmp_path / "ledger")
        guard.inventory_ledger.set_bootstrap_caps({"BTC": "0.0015548576728614218"}, now=1)
        guard.inventory_ledger.reconcile(
            account_fingerprint="scenario-account",
            balances=client(server).account_balances(),
            ownership={"BTC": {"dca": "0.0015"}, "ETH": {"dca": "0"}},
            evidence_sha256="locked-evidence", open_order_counts={},
            sources_healthy=True,
        )
        guard.emergency_exchange = client(server)
        guard.notification_path = tmp_path / "events.jsonl"
        with pytest.raises(RuntimeError, match="locked or unavailable"):
            guard._liquidate_unattributed("BTC", {
                "owned_total": "0.0015", "unattributed": "0.001554857672861422",
                "confirmation": {"cycles": 3, "confirmed": True},
            }, "locked-evidence")
        assert guard.inventory_ledger.bootstrap_cap("BTC") == Decimal(
            "0.0015548576728614218"
        )
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        assert remote["orders"] == []


def test_telegram_http_retry_is_persistent_and_deduplicated(monkeypatch, tmp_path):
    document = scenario_document()
    document["faults"]["POST telegram"] = [{"status": 429}]
    with RiskScenarioServer(document) as server:
        enable_scenario(monkeypatch, server, "telegram")
        token = tmp_path / "token"
        token.write_text("scenario-token", encoding="utf-8")
        client = TelegramChannelClient(token, "scenario-channel")
        outbox = TelegramOutbox(tmp_path / "outbox.sqlite", channel_id="scenario-channel")
        assert outbox.enqueue(event_id="event-1", kind="message", text="risk event")
        assert not outbox.enqueue(event_id="event-1", kind="message", text="risk event")
        assert outbox.drain(client, now=time.time()) == 0
        assert outbox.drain(client, now=time.time() + 10) == 1
        remote = requests.get(f"{server.base_url}/scenario/state", timeout=5).json()
        assert len(remote["telegram"]) == 1
        outbox.close()


@pytest.mark.parametrize("seed", range(5))
def test_seeded_random_inventory_sequences_preserve_fail_closed_invariants(seed, tmp_path):
    rng = random.Random(seed)
    ledger = UnifiedInventoryLedger(tmp_path / f"seed-{seed}")
    account = "scenario-random"
    for step_index in range(500):
        btc_total = Decimal(rng.randrange(0, 5000)) / Decimal("1000000")
        eth_total = Decimal(rng.randrange(0, 10000)) / Decimal("100000")
        # Deliberately create deficits in roughly one fifth of observations.
        deficit = step_index % 5 == 0
        btc_owned = btc_total + Decimal("0.00001") if deficit else btc_total * Decimal("0.8")
        eth_owned = eth_total + Decimal("0.0001") if deficit else eth_total * Decimal("0.8")
        sources_healthy = step_index % 7 != 0
        open_count = 1 if step_index % 11 == 0 else 0
        status = ledger.reconcile(
            account_fingerprint=account,
            balances={
                "BTC": {"free": btc_total, "locked": 0, "total": btc_total},
                "ETH": {"free": eth_total, "locked": 0, "total": eth_total},
            },
            ownership={"BTC": {"owner": btc_owned}, "ETH": {"owner": eth_owned}},
            evidence_sha256=f"evidence-{step_index // 3}",
            open_order_counts={"BTC-USDT": open_count},
            sources_healthy=sources_healthy, now=1000 + step_index,
        )
        expected_healthy = sources_healthy and open_count == 0 and not deficit
        assert status["healthy"] is expected_healthy
        for asset in ("BTC", "ETH"):
            row = status["assets"][asset]
            assert Decimal(row["owned_total"]) >= 0
            assert Decimal(row["unattributed"]) >= 0
            assert Decimal(row["ownership_deficit"]) >= 0
            assert not (
                Decimal(row["unattributed"]) > 0
                and Decimal(row["ownership_deficit"]) > 0
            )
            if not expected_healthy:
                assert row["confirmation"]["confirmed"] is False
