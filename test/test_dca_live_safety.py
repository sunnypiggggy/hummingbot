import sys
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "live_guard"))

from dca_live_common import (  # noqa: E402
    LIVE_PAIRS,
    extract_balances,
    layer_quote_amounts,
    live_controller_config,
    required_balances,
    trade_pnl_from_rows,
    validate_config,
    validate_exchange_filters,
)
from dca_live_guard import (  # noqa: E402
    BinanceEmergencyClient, Guard, _read_retry_session,
)
from macro_control.telemetry import build_sanitized_snapshot  # noqa: E402
from deploy_dca_live import (  # noqa: E402
    ApiClient,
    active_container_exists,
    can_start_eth,
    stage_configs,
)
from risk_recovery import COOLDOWN, EXITING, trigger_state  # noqa: E402


class FakeApi:
    def __init__(self):
        self.stopped = []
        self.orders = []
        self.containers = []
        self.active_order_values = []
        self.cancelled = []
        self.docker_stopped = []
        self.controller_updates = []

    def stop_bot(self, bot_name):
        self.stopped.append(bot_name)
        return {"status": "success", "response": {"success": True}}

    def status(self):
        return {"data": {}}

    def market_order(self, pair, side, amount):
        self.orders.append((pair, side, amount))
        return {"status": "submitted"}

    def active_containers(self, name_filter):
        return {"data": [
            {"name": name} for name in self.containers if name_filter in name
        ]}

    def active_orders(self, pair):
        return {"data": [
            value for value in self.active_order_values
            if value.get("trading_pair") == pair
        ]}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        self.active_order_values = [
            value for value in self.active_order_values
            if value.get("client_order_id") != order_id
        ]
        return {"message": f"cancelled {order_id}"}

    def stop_container(self, name):
        self.docker_stopped.append(name)
        self.containers = [value for value in self.containers if value != name]
        return {"status": "stopped"}

    def update_controller(self, bot_name, controller_name, profile):
        self.controller_updates.append((bot_name, controller_name, profile))
        return {"success": True}


class FakeEmergencyExchange:
    def __init__(self):
        self.orders = []
        self.open_order_values = []
        self.cancel_all_calls = []
        self.free_usdt = Decimal("1000")
        self.account_balance_calls = 0

    def open_orders(self, pair):
        return list(self.open_order_values)

    def cancel_all_orders(self, pair):
        self.cancel_all_calls.append(pair)
        self.open_order_values = []
        return []

    def account_balances(self):
        self.account_balance_calls += 1
        return {
            "BTC": {"free": Decimal("10"), "locked": Decimal("0"), "total": Decimal("10")},
            "ETH": {"free": Decimal("10"), "locked": Decimal("0"), "total": Decimal("10")},
            "USDT": {"free": self.free_usdt, "locked": Decimal("0"), "total": self.free_usdt},
        }

    def market_order(self, pair, side, amount):
        self.orders.append((pair, side, amount))
        return {
            "orderId": 123,
            "status": "FILLED",
            "executedQty": str(amount),
            "cummulativeQuoteQty": str(amount * Decimal("65000")),
            "fills": [],
        }


class FakeEmergencyDocker:
    def __init__(self, api):
        self.api = api

    def matching_containers(self, bot_name):
        return [
            name for name in self.api.containers
            if name == bot_name or name.startswith(f"{bot_name}-")
        ]

    def stop(self, name):
        self.api.docker_stopped.append(name)
        self.api.containers = [value for value in self.api.containers if value != name]
        return {}


class FakeResponse:
    status_code = 400
    text = '{"code":-2011,"msg":"Unknown order sent."}'
    content = text.encode()

    @staticmethod
    def json():
        return {"code": -2011, "msg": "Unknown order sent."}


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, params, timeout):
        self.calls.append((method, url, params, timeout))
        return FakeResponse()


class CanaryTests(unittest.TestCase):
    def test_eth_requires_24_hours_and_fresh_btc_guard_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard_state.json"
            now = 200_000.0
            bot_name = LIVE_PAIRS["BTC-USDT"].bot_name
            state.write_text(
                json.dumps(
                    {
                        "bots": {
                            bot_name: {
                                "started_at": now - 86_401,
                                "tripped": False,
                                "latest": {"updated_at": now - 10},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(can_start_eth(state, now)[0])

            value = json.loads(state.read_text(encoding="utf-8"))
            value["bots"][bot_name]["latest"]["updated_at"] = now - 61
            state.write_text(json.dumps(value), encoding="utf-8")
            allowed, reason = can_start_eth(state, now)
            self.assertFalse(allowed)
            self.assertIn("fresh guard monitoring", reason)


class DcaLiveSafetyTest(unittest.TestCase):
    def test_transient_contract_failure_uses_grace_without_latch(self):
        guard = Guard.__new__(Guard)
        guard.state = {}
        guard.fail_closed_seconds = 60
        guard.runtime_errors = Mock()

        self.assertFalse(guard._integrity_failure_requires_latch(
            "v22_contract", "fail_closed:ConnectionResetError", 100,
        ))
        self.assertFalse(guard._integrity_failure_requires_latch(
            "v22_contract", "fail_closed:ReadTimeout", 159.9,
        ))
        self.assertTrue(guard._integrity_failure_requires_latch(
            "v22_contract", "fail_closed:ReadTimeout", 160,
        ))
        self.assertEqual(
            3, guard.state["integrity_failure_grace"]["v22_contract"]["attempts"]
        )

    def test_binance_500_server_error_uses_grace_without_latch(self):
        guard = Guard.__new__(Guard)
        guard.state = {}
        guard.fail_closed_seconds = 60
        guard.runtime_errors = Mock()
        reason = (
            "fail_closed:500 Server Error: Internal Server Error for url: "
            "https://api.binance.com/api/v3/time"
        )

        self.assertFalse(guard._integrity_failure_requires_latch(
            "v22_contract", reason, 100,
        ))
        decision = guard.state["integrity_failure_grace"]["v22_contract"]
        self.assertEqual("transient_transport", decision["classification"])
        self.assertFalse(decision["expired"])

    def test_deterministic_contract_failure_bypasses_grace(self):
        guard = Guard.__new__(Guard)
        guard.state = {}
        guard.fail_closed_seconds = 60
        guard.runtime_errors = Mock()

        self.assertTrue(guard._integrity_failure_requires_latch(
            "v22_contract", "fail_closed:model hash mismatch", 100,
        ))

    def test_http_retry_policy_never_replays_trading_writes(self):
        session = _read_retry_session()
        retry = session.get_adapter("https://").max_retries
        self.assertEqual(2, retry.total)
        self.assertEqual(frozenset({"GET"}), retry.allowed_methods)
        self.assertNotIn("POST", retry.allowed_methods)
        self.assertNotIn("DELETE", retry.allowed_methods)

    def test_cancel_all_orders_treats_binance_already_clear_as_idempotent_success(self):
        client = BinanceEmergencyClient("key", "secret")
        client.session = FakeSession()

        result = client.cancel_all_orders("BTC-USDT")

        self.assertEqual("already_clear", result["status"])
        self.assertEqual(-2011, result["code"])
        self.assertEqual("DELETE", client.session.calls[0][0])

    def test_cooldown_cancels_orders_restored_by_restarted_bot(self):
        guard = Guard.__new__(Guard)
        guard.state = {"bots": {"bot": {"recovery": {
            "phase": COOLDOWN,
            "mechanism": "infrastructure_integrity_breaker",
            "scope": "infrastructure",
            "cooldown_until": 100,
        }}}}
        guard.emergency_exchange = FakeEmergencyExchange()
        guard.emergency_exchange.open_order_values = [
            {"symbol": "BTCUSDT", "orderId": 1},
        ]
        guard._lot_filter = lambda pair: (Decimal("0.00001"), Decimal("5"))
        audits = []
        guard._audit = lambda event, **details: audits.append((event, details))

        guard._process_recoverable(
            "bot",
            {"pair": "BTC-USDT", "mark_price": "64000", "database": "unused"},
            macro={"healthy": True, "buy_enabled": True, "sell_enabled": True},
            technical={"buy_enabled": True, "execution_authorized": True},
            now=200,
        )

        self.assertEqual(["BTC-USDT"], guard.emergency_exchange.cancel_all_calls)
        self.assertEqual([], guard.emergency_exchange.open_order_values)
        self.assertEqual(COOLDOWN, guard.state["bots"]["bot"]["recovery"]["phase"])
        self.assertEqual("recoverable_residual_orders_cancelled", audits[0][0])

    def test_quote_shortage_is_buy_only_and_self_recovering(self):
        guard = Guard.__new__(Guard)
        guard.state = {}
        guard.emergency_exchange = FakeEmergencyExchange()
        guard.quote_budget_buffer_pct = Decimal("0.002")
        guard.quote_balance_cache_seconds = 30
        guard.emergency_exchange.free_usdt = Decimal("89.52")

        blocked = guard._quote_budget_status(Decimal("95"), now=100)
        self.assertFalse(blocked["buy_ready"])
        self.assertEqual("insufficient_quote_budget", blocked["reason"])

        guard.emergency_exchange.free_usdt = Decimal("100")
        recovered = guard._quote_budget_status(
            Decimal("95"), now=110, force_refresh=True,
        )
        self.assertTrue(recovered["buy_ready"])
        self.assertEqual("quote_budget_available", recovered["reason"])

    def test_quote_gate_reuses_fresh_reconciliation_balance(self):
        guard = Guard.__new__(Guard)
        exchange = FakeEmergencyExchange()
        guard.state = {"quote_balance_source": {
            "free_quote": "100", "observed_at": 100,
            "source": "account_reconciliation",
        }}
        guard.emergency_exchange = exchange
        guard.quote_budget_buffer_pct = Decimal("0.002")
        guard.quote_balance_cache_seconds = 30

        result = guard._quote_budget_status(Decimal("95"), now=110)

        self.assertTrue(result["buy_ready"])
        self.assertTrue(result["cached"])
        self.assertEqual(0, exchange.account_balance_calls)

    def test_aggregate_capital_shortage_is_alert_only_and_keeps_both_sides_enabled(self):
        bot_name = LIVE_PAIRS["ETH-USDT"].bot_name
        guard = Guard.__new__(Guard)
        guard.state = {"bots": {bot_name: {"recovery": {"phase": "ACTIVE"}}}}
        guard.emergency_exchange = FakeEmergencyExchange()
        guard.emergency_exchange.free_usdt = Decimal("89.52")
        guard.quote_budget_buffer_pct = Decimal("0.002")
        guard.quote_balance_cache_seconds = 30
        guard._macro_gate = lambda: {
            "healthy": True, "buy_enabled": True, "sell_enabled": True,
            "reason": "macro_state_healthy", "active_lease_ids": [],
        }
        guard._v21_gate = lambda: {
            "healthy": True, "reason": "healthy", "pairs": {
                "ETH-USDT": {
                    "buy_enabled": True, "source_pair": "ETH-FDUSD",
                    "reason": "risk_on", "event_id": "eth-on",
                },
            },
        }
        applied = []
        guard._set_effective_gates = lambda bot, snapshot, **gates: applied.append(gates) or {
            "status": "applied",
        }
        guard._audit = lambda *args, **kwargs: None

        guard._apply_aggregate_gates(
            {bot_name: {"pair": "ETH-USDT", "database": "unused"}},
            risk_actions_enabled=True,
        )

        self.assertTrue(applied[0]["buy_enabled"])
        self.assertTrue(applied[0]["sell_enabled"])
        aggregate = guard.state["gate_aggregate"]
        self.assertFalse(aggregate["capital"]["buy_ready"])
        self.assertFalse(aggregate["capital"]["enforced"])
        self.assertEqual("alert_only", aggregate["capital"]["mode"])
        self.assertEqual("insufficient_quote_budget", aggregate["capital"]["reason"])
        self.assertFalse(aggregate["bots"][bot_name]["capital_gate_enforced"])

    def test_capital_alert_notification_explicitly_has_no_trade_impact(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.notification_path = Path(directory) / "events.jsonl"

            guard._emit_notification("capital_budget_gate_transition", {
                "buy_enabled": False,
                "reason": "insufficient_quote_budget",
                "free_quote": "18.68",
                "required_quote": "190",
                "action": "capital_alert_only_no_trade_block",
            })

            event = json.loads(guard.notification_path.read_text(encoding="utf-8"))
            self.assertEqual("capital_budget_gate", event["mechanism"])
            self.assertEqual("TRIGGERED", event["transition"])
            self.assertFalse(event["details"]["trading_permissions_changed"])
            self.assertEqual("alert_only", event["details"]["enforcement_mode"])

    def test_reentry_waits_for_base_rebuild_and_buy_side_cash_without_liquidation(self):
        bot_name = LIVE_PAIRS["BTC-USDT"].bot_name
        guard = Guard.__new__(Guard)
        guard.state = {
            "bots": {bot_name: {"recovery": {
                "phase": "REENTRY", "mechanism": "v22_weekly_buy_gate",
                "scope": "technical", "healthy_cycles": 3,
                "cooldown_until": 0,
            }}},
            "gate_aggregate": {"v22": {"pairs": {
                "BTC-USDT": {"buy_enabled": True},
                "ETH-USDT": {"buy_enabled": False},
            }}},
        }
        guard.auto_reentry_enabled = True
        guard.quote_budget_buffer_pct = Decimal("0.002")
        guard.quote_balance_cache_seconds = 30
        guard.emergency_exchange = FakeEmergencyExchange()
        # Enough to rebuild ~95 USDT of BTC, but not enough to preserve the
        # robot's separate ~95 USDT BUY-side budget afterwards.
        guard.emergency_exchange.free_usdt = Decimal("180")
        guard._lot_filter = lambda pair: (Decimal("0.00001"), Decimal("5"))
        guard._executor_counts = lambda database: {
            "active_buy_executors": 0, "trading_buy_executors": 0,
            "active_sell_executors": 0, "trading_sell_executors": 0,
            "open_orders": 0,
        }
        audits = []
        guard._audit = lambda event, **details: audits.append((event, details))

        guard._process_recoverable(
            bot_name,
            {"pair": "BTC-USDT", "mark_price": "65000", "database": "unused"},
            macro={"healthy": True, "buy_enabled": True, "sell_enabled": True},
            technical={"buy_enabled": True, "execution_authorized": True},
            now=200,
        )

        recovery = guard.state["bots"][bot_name]["recovery"]
        self.assertEqual("REENTRY", recovery["phase"])
        self.assertFalse(recovery["reentry_allowed"])
        self.assertEqual("insufficient_quote_budget", recovery["reentry_block_reason"])
        self.assertEqual([], guard.emergency_exchange.orders)
        self.assertEqual("recoverable_reentry_capital_wait", audits[-1][0])

    def test_recoverable_exit_is_capped_by_deployment_owned_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "managed_inventory.json").write_text(json.dumps({
                "pairs": {"BTC-USDT": {"managed_base": "0.002"}},
            }), encoding="utf-8")
            guard = Guard.__new__(Guard)
            guard.managed_inventory_path = root / "managed_inventory.json"
            guard.state_path = root / "guard_state.json"
            guard.audit_path = root / "risk_audit.jsonl"
            guard.emergency_exchange = FakeEmergencyExchange()
            guard.state = {"bots": {"bot": {"recovery": trigger_state(
                mechanism="strategy_loss_breaker", scope="strategy", now=100,
                trigger_value=-16, signal_price=65000, reason="loss",
            )}}}
            guard._notify = lambda message: None
            snapshot = {"pair": "BTC-USDT", "mark_price": "65000", "net_base": "0.001"}
            with patch.object(guard, "_lot_filter", return_value=(Decimal("0.000001"), Decimal("5"))):
                guard._process_recoverable(
                    "bot", snapshot,
                    macro={"healthy": True, "buy_enabled": True, "sell_enabled": True},
                    technical={"buy_enabled": True}, now=104,
                )
            self.assertEqual(Decimal("0.003"), guard.emergency_exchange.orders[0][2])
            self.assertEqual("COOLDOWN", guard.state["bots"]["bot"]["recovery"]["phase"])

    def test_v21_maps_fdusd_signals_to_usdt_without_recomputing_features(self):
        guard = Guard.__new__(Guard)
        guard.mechanisms = {"v21_buy_gate": True}
        guard.v21_gate_path = Path("ignored.json")
        guard.v21_max_age_seconds = 150
        contract = {
            "runtime_gate_healthy": True,
            "reason": "xgboost_gate_healthy",
            "model_version": "xgboost-grid-long-risk-gate-v21-250d",
            "generated_at": "2026-08-07T00:00:00Z",
            "pairs": {
                "BTC-FDUSD": {"buy_enabled": False, "risk_off_active": True,
                               "transition": "long:hold", "reason": "long_risk_off"},
                "ETH-FDUSD": {"buy_enabled": True, "risk_off_active": False,
                               "transition": "long:clear", "reason": "long_channel_clear"},
            },
        }
        with patch("dca_live_guard.load_runtime_xgboost_gate", return_value=contract):
            result = guard._v21_gate()
        self.assertFalse(result["pairs"]["BTC-USDT"]["buy_enabled"])
        self.assertTrue(result["pairs"]["ETH-USDT"]["buy_enabled"])
        self.assertEqual("BTC-FDUSD", result["pairs"]["BTC-USDT"]["source_pair"])

    def test_v21_unhealthy_contract_fails_closed_for_both_dca_pairs(self):
        guard = Guard.__new__(Guard)
        guard.mechanisms = {"v21_buy_gate": True}
        guard.v21_gate_path = Path("ignored.json")
        guard.v21_max_age_seconds = 150
        failed = {
            "runtime_gate_healthy": False,
            "reason": "fail_closed:model hash mismatch",
            "pairs": {"BTC-FDUSD": {}, "ETH-FDUSD": {}},
        }
        with patch("dca_live_guard.load_runtime_xgboost_gate", return_value=failed):
            result = guard._v21_gate()
        self.assertFalse(result["healthy"])
        self.assertTrue(all(
            not item["buy_enabled"] for item in result["pairs"].values()
        ))

    def test_aggregate_gate_disables_buy_and_preserves_macro_sell_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bot.sqlite"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE Controllers (id TEXT, timestamp FLOAT, config JSON)"
            )
            connection.execute(
                "INSERT INTO Controllers VALUES (?, ?, ?)",
                (
                    "dca_btcusdt_live_200",
                    1,
                    json.dumps({
                        "macro_buy_enabled": True,
                        "macro_sell_enabled": False,
                        "macro_decision_id": "external-sell-lease",
                    }),
                ),
            )
            connection.commit()
            connection.close()
            guard = Guard.__new__(Guard)
            guard.api = FakeApi()

            result = guard._set_effective_gates(
                "dca-live-btcusdt-200",
                {"database": str(database)},
                buy_enabled=False,
                sell_enabled=False,
                reasons={"v21": "risk_off", "fomc": "positive_event"},
            )

            self.assertEqual("applied", result["status"])
            profile = guard.api.controller_updates[0][2]
            self.assertFalse(profile["macro_buy_enabled"])
            self.assertFalse(profile["macro_sell_enabled"])

    def test_controller_profile_prefers_live_yaml_over_stale_database(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = Path(directory) / "instances" / "bot"
            database = instance / "data" / "bot.sqlite"
            database.parent.mkdir(parents=True)
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE Controllers (id TEXT, timestamp FLOAT, config JSON)"
            )
            connection.execute(
                "INSERT INTO Controllers VALUES (?, ?, ?)",
                (
                    "dca_btcusdt_live_200",
                    1,
                    json.dumps({
                        "macro_buy_enabled": False,
                        "macro_sell_enabled": False,
                        "macro_decision_id": "stale-database",
                    }),
                ),
            )
            connection.commit()
            connection.close()
            config = instance / "conf" / "controllers" / "dca_btcusdt_live_200.yml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "macro_buy_enabled: true\n"
                "macro_sell_enabled: true\n"
                "macro_decision_id: live-yaml\n",
                encoding="utf-8",
            )

            controller, profile = Guard._controller_profile(database)

            self.assertEqual("dca_btcusdt_live_200", controller)
            self.assertTrue(profile["macro_buy_enabled"])
            self.assertTrue(profile["macro_sell_enabled"])
            self.assertEqual("live-yaml", profile["macro_decision_id"])

    def test_aggregate_gate_corrects_live_yaml_even_when_database_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = Path(directory) / "instances" / "bot"
            database = instance / "data" / "bot.sqlite"
            database.parent.mkdir(parents=True)
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE Controllers (id TEXT, timestamp FLOAT, config JSON)"
            )
            connection.execute(
                "INSERT INTO Controllers VALUES (?, ?, ?)",
                (
                    "dca_btcusdt_live_200",
                    1,
                    json.dumps({
                        "macro_buy_enabled": False,
                        "macro_sell_enabled": False,
                    }),
                ),
            )
            connection.commit()
            connection.close()
            config = instance / "conf" / "controllers" / "dca_btcusdt_live_200.yml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "macro_buy_enabled: true\nmacro_sell_enabled: true\n",
                encoding="utf-8",
            )
            guard = Guard.__new__(Guard)
            guard.api = FakeApi()

            result = guard._set_effective_gates(
                "dca-live-btcusdt-200",
                {"database": str(database)},
                buy_enabled=False,
                sell_enabled=False,
                reasons={"v22": "risk_off", "fomc": "clear"},
            )

            self.assertEqual("applied", result["status"])
            self.assertEqual(1, len(guard.api.controller_updates))
            profile = guard.api.controller_updates[0][2]
            self.assertFalse(profile["macro_buy_enabled"])
            self.assertFalse(profile["macro_sell_enabled"])

    def test_v21_recovery_does_not_override_fomc_buy_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bot.sqlite"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE Controllers (id TEXT, timestamp FLOAT, config JSON)"
            )
            connection.execute(
                "INSERT INTO Controllers VALUES (?, ?, ?)",
                (
                    "dca_btcusdt_live_200",
                    1,
                    json.dumps({
                        "macro_buy_enabled": False,
                        "macro_sell_enabled": True,
                        "macro_decision_id": "external-negative-lease",
                    }),
                ),
            )
            connection.commit()
            connection.close()
            guard = Guard.__new__(Guard)
            guard.api = FakeApi()

            guard.state = {"bots": {}}
            guard._macro_gate = lambda: {
                "healthy": True, "buy_enabled": False, "sell_enabled": True,
                "reason": "active_negative_fomc", "active_lease_ids": ["fomc"],
            }
            guard._v21_gate = lambda: {
                "healthy": True, "reason": "healthy", "pairs": {
                    "BTC-USDT": {"buy_enabled": True, "source_pair": "BTC-FDUSD",
                                 "reason": "long_channel_clear"},
                },
            }
            guard._apply_aggregate_gates(
                {"dca-live-btcusdt-200": {
                    "database": str(database), "pair": "BTC-USDT"
                }},
                risk_actions_enabled=True,
            )
            self.assertFalse(guard.api.controller_updates)

    def test_database_inactivity_does_not_make_fresh_observation_unhealthy(self):
        snapshot = build_sanitized_snapshot(
            [
                {
                    "bot_name": "dca-live-btcusdt-200",
                    "controller_name": "btc",
                    "trading_pair": "BTC-USDT",
                    "strategy_equity": 190,
                    "peak_strategy_equity": 190,
                    "data_age_seconds": 0,
                    "observation_age_seconds": 0,
                    "database_event_age_seconds": 7200,
                    "healthy": True,
                }
            ],
            {
                "BTC-USDT": {
                    "mid_price": 65000,
                    "spread_bps": 1,
                    "volatility_ratio_30m": 0.01,
                    "data_age_seconds": 0,
                }
            },
        )
        self.assertTrue(snapshot["telemetry_healthy"])
        self.assertEqual(7200, snapshot["bots"][0]["database_event_age_seconds"])

    def test_fixed_name_duplicate_check_is_exact(self):
        payload = [
            {"name": "dca-live-btcusdt-200-20260727-063530", "status": "running"}
        ]
        self.assertFalse(active_container_exists(payload, "dca-live-btcusdt-200"))
        payload.append({"name": "dca-live-btcusdt-200", "status": "running"})
        self.assertTrue(active_container_exists(payload, "dca-live-btcusdt-200"))

    def test_deploy_uses_fixed_singleton_name(self):
        client = ApiClient.__new__(ApiClient)
        calls = []
        client.request = lambda method, path, payload: calls.append(
            (method, path, payload)
        ) or {"success": True}
        client.deploy("dca-live-btcusdt-200", "dca_btcusdt_live_200", "account")
        method, path, payload = calls[0]
        self.assertEqual("POST", method)
        self.assertEqual(
            "/bot-orchestration/deploy-v2-script?use_timestamp=false", path
        )
        self.assertEqual("dca-live-btcusdt-200", payload["instance_name"])
        self.assertEqual("dca-live-btcusdt-200.yml", payload["script_config"])

    def test_stage_configs_writes_fixed_bot_script_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            written = stage_configs(Path(directory))
            expected = Path(directory) / "conf" / "scripts" / "dca-live-btcusdt-200.yml"
            self.assertIn(expected, written)
            payload = __import__("yaml").safe_load(expected.read_text(encoding="utf-8"))
            self.assertEqual("v2_with_controllers.py", payload["script_file_name"])
            self.assertEqual(
                ["dca_btcusdt_live_200.yml"], payload["controllers_config"]
            )

    def test_configs_are_spot_only_and_budgeted(self):
        for pair in LIVE_PAIRS:
            config = live_controller_config(pair)
            validate_config(config)
            self.assertEqual("binance", config["connector_name"])
            self.assertEqual(1, config["leverage"])
            self.assertEqual(190.0, config["total_amount_quote"])
            self.assertEqual(18000, config["time_limit"])
            self.assertEqual(18000, config["executor_refresh_time"])
            self.assertTrue(config["skip_rebalance"])

    def test_each_side_has_expected_layers(self):
        self.assertEqual(
            [Decimal("9.50"), Decimal("19.00"), Decimal("28.50"), Decimal("38.00")],
            layer_quote_amounts(),
        )

    def test_min_notional_validation(self):
        symbol_info = {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "LOT_SIZE", "minQty": "0.00001000", "stepSize": "0.00001000"},
                {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
            ],
        }
        validate_exchange_filters(symbol_info, Decimal("65000"))
        symbol_info["filters"][1]["minNotional"] = "10"
        with self.assertRaises(ValueError):
            validate_exchange_filters(symbol_info, Decimal("65000"))

    def test_shared_balance_requirements_reserve_ten_per_bot(self):
        requirements = required_balances({
            "BTC-USDT": Decimal("65000"),
            "ETH-USDT": Decimal("2000"),
        })
        self.assertEqual(Decimal("210"), requirements["USDT"])
        self.assertEqual(Decimal("95") / Decimal("65000"), requirements["BTC"])
        self.assertEqual(Decimal("95") / Decimal("2000"), requirements["ETH"])

    def test_portfolio_response_is_normalized(self):
        payload = {
            "binance_live_dca_200": {
                "binance": [
                    {"token": "USDT", "units": 210},
                    {"token": "BTC", "units": "0.002"},
                ]
            }
        }
        self.assertEqual(
            {"USDT": Decimal("210"), "BTC": Decimal("0.002")},
            extract_balances(payload),
        )

    def test_trade_pnl_ignores_starting_account_inventory(self):
        # Buy 0.01 BTC at 60k, sell 0.005 at 64k, then mark remaining at 65k.
        rows = [
            ("BUY", 60_000_000_000, 10_000, 120_000, 1),
            ("SELL", 64_000_000_000, 5_000, 64_000, 2),
        ]
        metrics = trade_pnl_from_rows(rows, Decimal("65000"))
        self.assertEqual(Decimal("0.005"), metrics["net_base"])
        self.assertEqual(Decimal("44.816"), metrics["pnl_quote"])
        self.assertEqual(Decimal("0.184"), metrics["fees_quote"])

    def test_flatten_restores_only_bot_inventory_delta(self):
        guard = Guard.__new__(Guard)
        guard.api = FakeApi()
        guard.emergency_exchange = FakeEmergencyExchange()
        guard._lot_filter = lambda pair: (Decimal("0.00001"), Decimal("5"))
        result = guard._flatten({
            "pair": "BTC-USDT",
            "net_base": "0.00123456",
            "mark_price": "65000",
        })
        self.assertEqual("SELL", result["side"])
        self.assertEqual("0.00123", result["amount"])
        self.assertEqual(
            [("BTC-USDT", "SELL", Decimal("0.00123"))],
            guard.emergency_exchange.orders,
        )

    def test_trip_is_persistent_and_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.api = FakeApi()
            guard.emergency_exchange = FakeEmergencyExchange()
            guard.emergency_docker = FakeEmergencyDocker(guard.api)
            guard.state_dir = Path(directory)
            guard.state_path = Path(directory) / "guard_state.json"
            guard.audit_path = Path(directory) / "audit.jsonl"
            guard.state = {"bots": {}}
            guard._notify = lambda message: None
            guard._flatten = lambda snapshot, bot_name="": {
                "status": "dust", "remaining_base": "0", "exit_complete": True,
            }
            guard._snapshot = lambda bot_name, pair: {
                "pair": pair, "net_base": "0", "mark_price": "65000"
            }
            guard._wait_for_instance_terminal = lambda name, database, requested_at: {
                "instance": name,
                "mqtt_running": False,
                "database_terminal": True,
            }
            guard._trip("dca-live-btcusdt-200", "test loss", {
                "pair": "BTC-USDT",
                "net_base": "0",
                "mark_price": "65000",
            })
            guard._trip("dca-live-btcusdt-200", "test loss", None)
            state = guard.state["bots"]["dca-live-btcusdt-200"]
            self.assertTrue(state["tripped"])
            self.assertTrue(state["action_complete"])
            self.assertEqual(["dca-live-btcusdt-200"], guard.api.stopped)

    def test_trip_flattens_only_post_stop_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.api = FakeApi()
            guard.emergency_exchange = FakeEmergencyExchange()
            guard.emergency_docker = FakeEmergencyDocker(guard.api)
            guard.state_dir = Path(directory)
            guard.state_path = Path(directory) / "guard_state.json"
            guard.audit_path = Path(directory) / "audit.jsonl"
            guard.state = {"bots": {}}
            guard._notify = lambda message: None
            guard._secure_stop = lambda bot_name, pair: {"status": "stopped"}
            snapshots = [
                {"pair": "BTC-USDT", "net_base": "0", "mark_price": "65000"},
                {"pair": "BTC-USDT", "net_base": "0", "mark_price": "65000"},
            ]
            guard._snapshot = lambda bot_name, pair: snapshots.pop(0)
            flattened = []
            guard._flatten = lambda snapshot, bot_name="": flattened.append(snapshot) or {
                "status": "not_required"
            }

            guard._trip(
                "dca-live-btcusdt-200",
                "test stale snapshot",
                {"pair": "BTC-USDT", "net_base": "0.01", "mark_price": "65000"},
            )

            self.assertEqual("0", flattened[0]["net_base"])

    def test_mqtt_failure_uses_independent_exchange_and_docker_channels(self):
        guard = Guard.__new__(Guard)
        guard.api = FakeApi()
        guard.api.stop_bot = lambda name: (_ for _ in ()).throw(RuntimeError("mqtt down"))
        guard.api.containers = ["dca-live-btcusdt-200"]
        guard.emergency_exchange = FakeEmergencyExchange()
        guard.emergency_exchange.open_order_values = [
            {"clientOrderId": "live-order", "symbol": "BTCUSDT"}
        ]
        guard.emergency_docker = FakeEmergencyDocker(guard.api)
        guard.bots_path = Path("not-present")

        result = guard._secure_stop("dca-live-btcusdt-200", "BTC-USDT")

        self.assertTrue(result["emergency_path_used"])
        self.assertTrue(result["verified_no_active_orders"])
        self.assertTrue(result["verified_no_live_instances"])
        self.assertEqual(["BTC-USDT"], guard.emergency_exchange.cancel_all_calls)

    def test_trip_resolves_timestamp_instance_cancels_and_stops_it(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Guard.__new__(Guard)
            guard.api = FakeApi()
            guard.emergency_exchange = FakeEmergencyExchange()
            guard.emergency_docker = FakeEmergencyDocker(guard.api)
            actual = "dca-live-btcusdt-200-20260727-120000"
            guard.api.containers = [actual, "unrelated"]
            guard.bots_path = Path(directory) / "bots"
            guard.api.active_order_values = [
                {"client_order_id": "buy-1", "trading_pair": "BTC-USDT"},
                {"client_order_id": "eth-1", "trading_pair": "ETH-USDT"},
            ]
            guard.emergency_exchange.open_order_values = [
                {"clientOrderId": "buy-1", "symbol": "BTCUSDT"}
            ]
            guard.state_dir = Path(directory)
            guard.state_path = Path(directory) / "guard_state.json"
            guard.audit_path = Path(directory) / "audit.jsonl"
            guard.state = {"bots": {}}
            guard._notify = lambda message: None
            guard._flatten = lambda snapshot, bot_name="": {
                "status": "dust", "remaining_base": "0", "exit_complete": True,
            }
            guard._snapshot = lambda bot_name, pair: {
                "pair": pair, "net_base": "0", "mark_price": "65000"
            }
            guard._wait_for_instance_terminal = lambda name, database, requested_at: {
                "instance": name,
                "mqtt_running": False,
                "database_terminal": True,
            }

            guard._trip("dca-live-btcusdt-200", "test mapping", {
                "pair": "BTC-USDT",
                "net_base": "0",
                "mark_price": "65000",
            })

            state = guard.state["bots"]["dca-live-btcusdt-200"]
            self.assertTrue(state["action_complete"])
            self.assertEqual(
                ["buy-1"], state["stop_response"]["cancelled_order_ids"]
            )
            self.assertEqual(
                ["BTC-USDT"], guard.emergency_exchange.cancel_all_calls
            )
            self.assertEqual([actual], guard.api.docker_stopped)
            self.assertTrue(state["stop_response"]["verified_no_active_orders"])
            self.assertTrue(
                state["stop_response"]["verified_mqtt_and_database_terminal"]
            )
            self.assertEqual(
                ["dca-live-btcusdt-200", actual],
                guard.api.stopped,
            )

    def test_mqtt_stop_confirmation_uses_post_request_ack(self):
        payload = {"data": {"dca-live-btcusdt-200": {
            "status": "running",
            "general_logs": [
                {"timestamp": 99, "msg": "Hummingbot stopped."},
                {"timestamp": 101, "msg": "Hummingbot stopped."},
            ],
        }}}
        self.assertTrue(Guard._mqtt_stop_confirmed(
            payload, "dca-live-btcusdt-200", 100
        ))
        self.assertFalse(Guard._mqtt_stop_confirmed(
            payload, "dca-live-btcusdt-200", 102.5
        ))

    def test_macro_telemetry_reads_side_counts_and_controller_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bot.sqlite"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE Executors "
                "(is_active BOOLEAN, is_trading BOOLEAN, config JSON)"
            )
            connection.execute(
                "CREATE TABLE Controllers "
                "(id TEXT, timestamp FLOAT, config JSON)"
            )
            connection.executemany(
                "INSERT INTO Executors VALUES (?, ?, ?)",
                [
                    (1, 1, json.dumps({"side": 1})),
                    (1, 0, json.dumps({"side": 2})),
                    (0, 0, json.dumps({"side": 2})),
                ],
            )
            connection.execute(
                "INSERT INTO Controllers VALUES (?, ?, ?)",
                (
                    "dca_btc",
                    1,
                    json.dumps(
                        {
                            "macro_buy_enabled": False,
                            "macro_sell_enabled": True,
                            "macro_decision_id": "decision",
                        }
                    ),
                ),
            )
            connection.commit()
            connection.close()

            self.assertEqual(
                {
                    "active_buy_executors": 1,
                    "trading_buy_executors": 1,
                    "active_sell_executors": 1,
                    "trading_sell_executors": 0,
                    "open_orders": 0,
                },
                Guard._executor_counts(database),
            )
            gates = Guard._controller_gates(database)
            self.assertFalse(gates["macro_buy_enabled"])
            self.assertTrue(gates["macro_sell_enabled"])
            self.assertEqual("decision", gates["macro_decision_id"])

    def test_macro_telemetry_falls_back_to_live_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bot.sqlite"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE Executors "
                "(is_active BOOLEAN, is_trading BOOLEAN, config JSON)"
            )
            connection.execute(
                'CREATE TABLE "Order" '
                "(id TEXT, amount INTEGER, last_status TEXT)"
            )
            connection.execute(
                "CREATE TABLE OrderStatus "
                "(order_id TEXT, timestamp INTEGER, status TEXT)"
            )
            connection.execute(
                "CREATE TABLE TradeFill (order_id TEXT, amount INTEGER)"
            )
            connection.executemany(
                'INSERT INTO "Order" VALUES (?, ?, ?)',
                [
                    ("buy", 10, "BuyOrderCreated"),
                    ("sell", 10, "SellOrderCreated"),
                    ("cancelled", 10, "OrderCancelled"),
                ],
            )
            connection.executemany(
                "INSERT INTO OrderStatus VALUES (?, ?, ?)",
                [
                    ("buy", 1, "BuyOrderCreated"),
                    ("sell", 1, "SellOrderCreated"),
                    ("cancelled", 1, "BuyOrderCreated"),
                ],
            )
            connection.commit()
            connection.close()

            self.assertEqual(
                {
                    "active_buy_executors": 1,
                    "trading_buy_executors": 0,
                    "active_sell_executors": 1,
                    "trading_sell_executors": 0,
                    "open_orders": 2,
                },
                Guard._executor_counts(database),
            )


if __name__ == "__main__":
    unittest.main()
