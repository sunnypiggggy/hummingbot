import json
import hashlib
import importlib
import inspect
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from macro_control.approval import proposal_sha256
from macro_control.app import create_app
from macro_control.executor import DecisionRejected, MacroExecutor
from macro_control.file_telemetry import JsonFileTelemetryProvider
from macro_control.policy import POLICY_VERSION, RiskWindowDecision, RiskWindowPolicy
from macro_control.profiles import event_config
from macro_control.security import NonceCache, sign_request, verify_request
from macro_control.storage import StateStore
from macro_control.trading_report import JsonTradingReportProvider
from hummingbot.strategy_v2.controllers.controller_base import ControllerConfigBase
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.controllers.market_making_controller_base import (
    MarketMakingControllerConfigBase,
)


UTC = timezone.utc
NOW = datetime(2026, 7, 28, 18, 0, tzinfo=UTC)
EVENT_AT = datetime(2026, 7, 29, 18, 0, tzinfo=UTC)


def decision(
    decision_id: str,
    impact: str,
    *,
    effective_at: datetime | None = None,
    resume_at: datetime | None = None,
    approved: bool = True,
) -> dict:
    value = {
        "decision_id": decision_id,
        "event_id": "fomc-2026-07",
        "event_kind": "fomc",
        "event_at": EVENT_AT.isoformat(),
        "event_source_url": (
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        ),
        "decision_at": NOW.isoformat(),
        "market_impact": impact,
        "confidence": 0.9,
        "effective_at": (effective_at or EVENT_AT - timedelta(hours=2)).isoformat(),
        "resume_at": (resume_at or EVENT_AT + timedelta(minutes=30)).isoformat(),
        "snapshot_id": "snapshot-v3",
        "reason": "Hermes-authored qualitative judgment.",
        "evidence": [
            {
                "source_id": "one",
                "source_url": "https://example.com/one",
                "observed_at": (NOW - timedelta(minutes=20)).isoformat(),
                "summary": "First independent source.",
                "kind": "decision",
            },
            {
                "source_id": "two",
                "source_url": "https://example.org/two",
                "observed_at": (NOW - timedelta(minutes=10)).isoformat(),
                "summary": "Second independent source.",
                "kind": "decision",
            },
        ],
        "hermes": {
            "agent_id": "hermes-macro",
            "model": "test",
            "prompt_version": "v1",
            "run_id": decision_id,
        },
        "policy_version": POLICY_VERSION,
    }
    if impact != "neutral" and approved:
        value["approval"] = {
            "status": "approved",
            "channel": "telegram",
            "action": "approve",
            "telegram_user_id": "owner",
            "telegram_chat_id": "chat",
            "telegram_update_id": f"update-{decision_id}",
            "telegram_callback_query_id": f"callback-{decision_id}",
            "telegram_message_id": f"message-{decision_id}",
            "approved_at": NOW.isoformat(),
            "proposal_sha256": proposal_sha256(value),
        }
    return value


class MutableTelemetry:
    def __init__(self):
        self.hard_breaker = False
        self.buy_breaker = False
        self.gates = {"buy": True, "sell": True}
        self.executors = {"buy": 1, "sell": 1}

    def snapshot(self):
        return {
            "snapshot_id": "snapshot-v3",
            "observed_at": NOW.isoformat(),
            "bots": [
                {
                    "bot_name": "dca-btc",
                    "macro_buy_enabled": self.gates["buy"],
                    "macro_sell_enabled": self.gates["sell"],
                    "active_buy_executors": self.executors["buy"],
                    "trading_buy_executors": self.executors["buy"],
                    "active_sell_executors": self.executors["sell"],
                    "trading_sell_executors": self.executors["sell"],
                    "hard_circuit_breaker": self.hard_breaker,
                    "buy_circuit_breaker": self.buy_breaker,
                }
            ],
        }


class FakeAPI:
    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.updates = []
        self.fail = False

    def update_controller(self, bot_name, controller_name, profile):
        if self.fail:
            raise RuntimeError("simulated API failure")
        self.updates.append((bot_name, controller_name, profile))
        self.telemetry.gates = {
            "buy": profile["macro_buy_enabled"],
            "sell": profile["macro_sell_enabled"],
        }
        if not profile["macro_buy_enabled"]:
            self.telemetry.executors["buy"] = 0
        if not profile["macro_sell_enabled"]:
            self.telemetry.executors["sell"] = 0
        return {"ok": True}


class FixedExecutor(MacroExecutor):
    @staticmethod
    def _now():
        return NOW


class RiskWindowPolicyTests(unittest.TestCase):
    def test_valid_fomc_window_and_direction(self):
        value = RiskWindowDecision.from_mapping(decision("negative", "negative"))
        assessment = RiskWindowPolicy().assess(value)
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.creates_lease)

    def test_rejects_non_official_source_and_window_beyond_limit(self):
        value = decision(
            "invalid",
            "negative",
            effective_at=EVENT_AT - timedelta(hours=25),
        )
        value["event_source_url"] = "https://example.com/calendar"
        assessment = RiskWindowPolicy().assess(
            RiskWindowDecision.from_mapping(value)
        )
        self.assertFalse(assessment.accepted)
        self.assertIn("official_calendar_source_missing", assessment.reasons)
        self.assertIn("effective_at_outside_event_window", assessment.reasons)

    def test_v2_policy_is_rejected(self):
        value = decision("old-policy", "negative")
        value["policy_version"] = "dca-macro-v2"
        assessment = RiskWindowPolicy().assess(
            RiskWindowDecision.from_mapping(value)
        )
        self.assertFalse(assessment.accepted)
        self.assertIn("policy_version_mismatch", assessment.reasons)

    def test_profile_contains_only_v3_directional_gates(self):
        profile = event_config(
            "BTC-USDT",
            macro_buy_enabled=False,
            macro_sell_enabled=True,
            decision_id="v3-only",
        )
        self.assertEqual("dman_maker_v3_macro", profile["controller_name"])
        self.assertEqual("dca-macro-v3", profile["policy_version"])
        self.assertEqual(18000, profile["time_limit"])
        self.assertEqual(18000, profile["executor_refresh_time"])
        self.assertTrue(profile["time_limit_from_first_fill"])
        self.assertTrue(profile["stop_loss_on_partial_fills"])
        self.assertNotIn("macro_long_enabled", profile)

    def test_dynamic_loader_selects_v3_config_before_imported_base(self):
        sys.modules.pop("controllers.market_making.dman_maker_v3_macro", None)
        sys.modules.pop("controllers.market_making.dman_maker_v2", None)
        with mock.patch.dict(
            sys.modules, {"pandas_ta": types.ModuleType("pandas_ta")}
        ):
            module = importlib.import_module(
                "controllers.market_making.dman_maker_v3_macro"
            )

        config_class = next(
            (
                member
                for _, member in inspect.getmembers(module)
                if inspect.isclass(member)
                and member
                not in {
                    ControllerConfigBase,
                    MarketMakingControllerConfigBase,
                    DirectionalTradingControllerConfigBase,
                }
                and issubclass(member, ControllerConfigBase)
            ),
            None,
        )
        self.assertIs(module.DManMakerV3MacroConfig, config_class)


class ExecutorV3Tests(unittest.TestCase):
    def make_executor(self, directory, *, execution_enabled=True):
        telemetry = MutableTelemetry()
        api = FakeAPI(telemetry)
        executor = FixedExecutor(
            api,
            telemetry,
            StateStore(Path(directory)),
            [
                {
                    "bot_name": "dca-btc",
                    "controller_name": "btc",
                    "trading_pair": "BTC-USDT",
                    "close_confirmation_timeout": 0,
                }
            ],
            execution_enabled=execution_enabled,
            telegram_approver_user_id="owner",
            telegram_approver_chat_id="chat",
        )
        return executor, api, telemetry

    def test_negative_disables_buy_and_positive_disables_sell(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, api, _ = self.make_executor(directory)
            executor.apply(
                decision(
                    "negative",
                    "negative",
                    effective_at=NOW,
                )
            )
            self.assertFalse(api.updates[-1][2]["macro_buy_enabled"])
            self.assertTrue(api.updates[-1][2]["macro_sell_enabled"])
            executor.apply(
                decision(
                    "positive",
                    "positive",
                    effective_at=NOW,
                )
            )
            self.assertFalse(api.updates[-1][2]["macro_buy_enabled"])
            self.assertFalse(api.updates[-1][2]["macro_sell_enabled"])

    def test_reconcile_uses_live_gates_not_persisted_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, api, telemetry = self.make_executor(directory)
            state = executor.store.read()
            state["bot_gate_state"] = {
                "dca-btc": {"buy": True, "sell": True}
            }
            executor.store.write(state)
            telemetry.gates = {"buy": False, "sell": False}

            executor.reconcile(now=NOW)

            self.assertEqual(1, len(api.updates))
            self.assertTrue(api.updates[0][2]["macro_buy_enabled"])
            self.assertTrue(api.updates[0][2]["macro_sell_enabled"])

    def test_non_neutral_decision_requires_matching_owner_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, _, _ = self.make_executor(directory)
            with self.assertRaisesRegex(
                DecisionRejected, "owner approval is required"
            ):
                executor.apply(
                    decision(
                        "unapproved",
                        "negative",
                        effective_at=NOW,
                        approved=False,
                    )
                )

            mismatched = decision(
                "mismatched",
                "negative",
                effective_at=NOW,
            )
            mismatched["approval"]["proposal_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                DecisionRejected, "proposal hash does not match"
            ):
                executor.apply(mismatched)

    def test_neutral_is_audited_without_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, api, _ = self.make_executor(directory)
            result = executor.apply(
                decision("neutral", "neutral", approved=False)
            )
            self.assertEqual("recorded_no_change", result["status"])
            self.assertFalse(api.updates)

    def test_telegram_callback_cannot_approve_two_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, _, _ = self.make_executor(directory)
            first = decision("first", "negative", effective_at=NOW)
            executor.apply(first)
            second = decision("second", "positive", effective_at=NOW)
            second["approval"]["telegram_callback_query_id"] = first["approval"][
                "telegram_callback_query_id"
            ]
            second["approval"]["proposal_sha256"] = proposal_sha256(second)
            with self.assertRaisesRegex(
                DecisionRejected, "interaction was reused"
            ):
                executor.apply(second)

    def test_overlapping_lease_expiry_restores_only_expired_side(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, api, telemetry = self.make_executor(directory)
            negative = decision(
                "negative",
                "negative",
                effective_at=NOW,
                resume_at=EVENT_AT,
            )
            positive = decision(
                "positive",
                "positive",
                effective_at=NOW,
                resume_at=EVENT_AT + timedelta(hours=1),
            )
            executor.apply(negative)
            executor.apply(positive)
            telemetry.executors["buy"] = 0
            telemetry.executors["sell"] = 0
            result = executor.reconcile(
                now=EVENT_AT + timedelta(minutes=10),
                snapshot=telemetry.snapshot(),
            )
            self.assertEqual({"buy": True, "sell": False}, result["desired_gates"])
            self.assertTrue(api.updates[-1][2]["macro_buy_enabled"])
            self.assertFalse(api.updates[-1][2]["macro_sell_enabled"])

    def test_hard_breaker_blocks_automatic_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, _, telemetry = self.make_executor(directory)
            executor.apply(
                decision(
                    "negative",
                    "negative",
                    effective_at=NOW,
                    resume_at=EVENT_AT,
                )
            )
            telemetry.hard_breaker = True
            result = executor.reconcile(
                now=EVENT_AT + timedelta(minutes=1),
                snapshot=telemetry.snapshot(),
            )
            self.assertEqual("blocked_by_hard_breaker", result["status"])
            self.assertFalse(
                executor.store.read()["bot_gate_state"]["dca-btc"]["buy"]
            )

    def test_buy_breaker_blocks_buy_resume_but_allows_sell_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, api, telemetry = self.make_executor(directory)
            executor.apply(
                decision(
                    "negative",
                    "negative",
                    effective_at=NOW,
                    resume_at=EVENT_AT,
                )
            )
            telemetry.gates["sell"] = False
            telemetry.executors["sell"] = 0
            telemetry.buy_breaker = True

            result = executor.reconcile(
                now=EVENT_AT + timedelta(minutes=1),
                snapshot=telemetry.snapshot(),
            )

            self.assertEqual("partially_applied", result["status"])
            self.assertFalse(api.updates[-1][2]["macro_buy_enabled"])
            self.assertTrue(api.updates[-1][2]["macro_sell_enabled"])

    def test_approved_revocation_cannot_bypass_hard_breaker(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, _, telemetry = self.make_executor(directory)
            payload = decision(
                "revoke-me",
                "negative",
                effective_at=NOW,
                resume_at=EVENT_AT,
            )
            executor.apply(payload)
            telemetry.hard_breaker = True
            revocation = {
                **payload["approval"],
                "action": "revoke",
                "telegram_update_id": "revoke-update",
                "telegram_callback_query_id": "revoke-callback",
                "telegram_message_id": "revoke-message",
            }
            result = executor.revoke(
                "revoke-me",
                revocation,
                snapshot=telemetry.snapshot(),
            )
            self.assertEqual("revoked", result["status"])
            self.assertEqual(
                "blocked_by_hard_breaker", result["reconcile_status"]
            )
            self.assertFalse(telemetry.gates["buy"])

    def test_hard_breaker_still_allows_other_side_to_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, api, telemetry = self.make_executor(directory)
            executor.apply(
                decision(
                    "negative",
                    "negative",
                    effective_at=NOW,
                    resume_at=EVENT_AT,
                )
            )
            executor.apply(
                decision(
                    "positive",
                    "positive",
                    effective_at=EVENT_AT,
                    resume_at=EVENT_AT + timedelta(hours=1),
                )
            )
            telemetry.hard_breaker = True
            result = executor.reconcile(
                now=EVENT_AT + timedelta(minutes=1),
                snapshot=telemetry.snapshot(),
            )
            self.assertEqual("partially_applied", result["status"])
            self.assertFalse(api.updates[-1][2]["macro_buy_enabled"])
            self.assertFalse(api.updates[-1][2]["macro_sell_enabled"])

    def test_shadow_mode_records_without_api_update(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, api, _ = self.make_executor(
                directory, execution_enabled=False
            )
            result = executor.apply(
                decision("negative", "negative", effective_at=NOW)
            )
            self.assertEqual("shadow", result["status"])
            self.assertFalse(api.updates)

    def test_decision_id_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, _, _ = self.make_executor(directory)
            executor.apply(decision("same", "negative", effective_at=NOW))
            conflicting = decision("same", "positive", effective_at=NOW)
            with self.assertRaises(DecisionRejected):
                executor.apply(conflicting)

    def test_state_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, _, _ = self.make_executor(
                directory, execution_enabled=False
            )
            executor.apply(decision("persisted", "negative", effective_at=NOW))
            reloaded = StateStore(Path(directory)).read()
            self.assertIn("persisted", reloaded["leases"])
            self.assertEqual(3, reloaded["schema_version"])

    def test_api_failure_uses_bounded_exponential_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            executor, api, telemetry = self.make_executor(directory)
            api.fail = True
            result = executor.apply(
                decision("retry", "negative", effective_at=NOW)
            )
            self.assertEqual("failed", result["status"])
            retry = executor.store.read()["retry_state"]["dca-btc"]
            self.assertEqual(5.0, retry["delay_seconds"])
            waiting = executor.reconcile(
                now=NOW + timedelta(seconds=4),
                snapshot=telemetry.snapshot(),
            )
            self.assertEqual("retry_scheduled", waiting["status"])
            executor.reconcile(
                now=NOW + timedelta(seconds=5),
                snapshot=telemetry.snapshot(),
            )
            retry = executor.store.read()["retry_state"]["dca-btc"]
            self.assertEqual(10.0, retry["delay_seconds"])


class MultiKeySecurityTests(unittest.TestCase):
    def test_two_keys_and_unknown_key_id(self):
        body = json.dumps({"ok": True}).encode()
        timestamp = str(NOW.timestamp())
        signature = sign_request(
            "next-secret", "POST", "/v1/event-decisions", timestamp, "nonce", body
        )
        secrets = {"primary": "primary-secret", "next": "next-secret"}
        self.assertEqual(
            (True, "ok"),
            verify_request(
                secrets,
                "POST",
                "/v1/event-decisions",
                timestamp,
                "nonce",
                body,
                signature,
                NonceCache(),
                "next",
                now=NOW.timestamp(),
            ),
        )
        self.assertEqual(
            "unknown_key_id",
            verify_request(
                secrets,
                "POST",
                "/v1/event-decisions",
                timestamp,
                "nonce-2",
                body,
                signature,
                NonceCache(),
                "missing",
                now=NOW.timestamp(),
            )[1],
        )


class GatewayV3Tests(unittest.TestCase):
    @staticmethod
    def signed_headers(path, body, nonce, *, method="POST"):
        timestamp = str(time.time())
        return {
            "X-Client-Cert-Verified": "SUCCESS",
            "X-Client-Cert-Fingerprint": "client",
            "X-Hermes-Key-Id": "primary",
            "X-Hermes-Timestamp": timestamp,
            "X-Hermes-Nonce": nonce,
            "X-Hermes-Signature": sign_request(
                "a" * 32,
                method,
                path,
                timestamp,
                nonce,
                body,
            ),
            "Content-Type": "application/json",
        }

    def test_v3_endpoint_requires_known_key_and_is_replay_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = MutableTelemetry()
            executor = FixedExecutor(
                FakeAPI(telemetry),
                telemetry,
                StateStore(Path(directory)),
                [
                    {
                        "bot_name": "dca-btc",
                        "controller_name": "btc",
                        "trading_pair": "BTC-USDT",
                        "close_confirmation_timeout": 0,
                    }
                ],
                execution_enabled=False,
                telegram_approver_user_id="owner",
                telegram_approver_chat_id="chat",
            )
            app = create_app(
                executor,
                telemetry,
                {"primary": "a" * 32, "next": "b" * 32},
            )
            payload = decision("gateway-v3", "negative", effective_at=NOW)
            body = json.dumps(payload, separators=(",", ":")).encode()
            timestamp = str(time.time())
            headers = {
                "X-Client-Cert-Verified": "SUCCESS",
                "X-Client-Cert-Fingerprint": "client",
                "X-Hermes-Key-Id": "next",
                "X-Hermes-Timestamp": timestamp,
                "X-Hermes-Nonce": "v3-nonce",
                "X-Hermes-Signature": sign_request(
                    "b" * 32,
                    "POST",
                    "/v1/event-decisions",
                    timestamp,
                    "v3-nonce",
                    body,
                ),
                "Content-Type": "application/json",
            }
            with TestClient(app) as client:
                response = client.post(
                    "/v1/event-decisions", content=body, headers=headers
                )
                self.assertEqual(200, response.status_code)
                self.assertEqual("shadow", response.json()["status"])
                self.assertEqual(
                    401,
                    client.post(
                        "/v1/event-decisions", content=body, headers=headers
                    ).status_code,
                )

    def test_gateway_rejects_bypass_and_accepts_approved_revocation(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = MutableTelemetry()
            executor = FixedExecutor(
                FakeAPI(telemetry),
                telemetry,
                StateStore(Path(directory)),
                [
                    {
                        "bot_name": "dca-btc",
                        "controller_name": "btc",
                        "trading_pair": "BTC-USDT",
                        "close_confirmation_timeout": 0,
                    }
                ],
                execution_enabled=False,
                telegram_approver_user_id="owner",
                telegram_approver_chat_id="chat",
            )
            app = create_app(executor, telemetry, {"primary": "a" * 32})
            unapproved = decision(
                "gateway-bypass",
                "negative",
                effective_at=NOW,
                approved=False,
            )
            body = json.dumps(unapproved, separators=(",", ":")).encode()
            with TestClient(app) as client:
                response = client.post(
                    "/v1/event-decisions",
                    content=body,
                    headers=self.signed_headers(
                        "/v1/event-decisions", body, "bypass"
                    ),
                )
                self.assertEqual(409, response.status_code)

                approved = decision(
                    "gateway-revoke",
                    "negative",
                    effective_at=NOW,
                )
                approved_body = json.dumps(
                    approved, separators=(",", ":")
                ).encode()
                self.assertEqual(
                    200,
                    client.post(
                        "/v1/event-decisions",
                        content=approved_body,
                        headers=self.signed_headers(
                            "/v1/event-decisions",
                            approved_body,
                            "approved",
                        ),
                    ).status_code,
                )
                revocation = {
                    "approval": {
                        **approved["approval"],
                        "action": "revoke",
                        "telegram_update_id": "revoke-update",
                        "telegram_callback_query_id": "revoke-callback",
                        "telegram_message_id": "revoke-message",
                    }
                }
                revoke_body = json.dumps(
                    revocation, separators=(",", ":")
                ).encode()
                path = "/v1/decisions/gateway-revoke/revoke"
                revoked = client.post(
                    path,
                    content=revoke_body,
                    headers=self.signed_headers(
                        path, revoke_body, "revoke"
                    ),
                )
                self.assertEqual(200, revoked.status_code)
                self.assertEqual("revoked", revoked.json()["status"])

    def test_authenticated_trading_report_and_png_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chart = b"\x89PNG\r\n\x1a\nchart"
            chart_path = root / "chart.png"
            report_path = root / "report.json"
            chart_path.write_bytes(chart)
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "report_id": "report-v3",
                        "chart_sha256": hashlib.sha256(chart).hexdigest(),
                        "bots": [],
                    }
                ),
                encoding="utf-8",
            )
            telemetry = MutableTelemetry()
            api = FakeAPI(telemetry)
            executor = FixedExecutor(
                api,
                telemetry,
                StateStore(root / "state"),
                [],
                execution_enabled=False,
                telegram_approver_user_id="owner",
                telegram_approver_chat_id="chat",
            )
            app = create_app(
                executor,
                telemetry,
                {"primary": "a" * 32},
                trading_report=JsonTradingReportProvider(
                    report_path, chart_path
                ),
            )
            with TestClient(app) as client:
                report_path_api = "/v1/trading-report"
                report = client.get(
                    report_path_api,
                    headers=self.signed_headers(
                        report_path_api,
                        b"",
                        "report-get",
                        method="GET",
                    ),
                )
                self.assertEqual(200, report.status_code)
                self.assertEqual("report-v3", report.json()["report_id"])

                chart_path_api = "/v1/trading-chart"
                chart_response = client.get(
                    chart_path_api,
                    headers=self.signed_headers(
                        chart_path_api,
                        b"",
                        "chart-get",
                        method="GET",
                    ),
                )
                self.assertEqual(200, chart_response.status_code)
                self.assertEqual("image/png", chart_response.headers["content-type"])
                self.assertEqual(chart, chart_response.content)
                self.assertEqual([], api.updates)

                no_certificate = self.signed_headers(
                    report_path_api,
                    b"",
                    "no-certificate",
                    method="GET",
                )
                no_certificate.pop("X-Client-Cert-Verified")
                self.assertEqual(
                    401,
                    client.get(
                        report_path_api, headers=no_certificate
                    ).status_code,
                )


class FileTelemetryTests(unittest.TestCase):
    def test_snapshot_id_is_stable_and_previous_snapshot_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.json"
            payload = {
                "observed_at": NOW.isoformat(),
                "bot_statuses": [],
                "market": {
                    "BTC-USDT": {
                        "mid_price": 100,
                        "spread_bps": 1,
                        "volatility_ratio_30m": 0.01,
                        "data_age_seconds": 0,
                    }
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            provider = JsonFileTelemetryProvider(path)
            first = provider.snapshot()
            self.assertEqual(first["snapshot_id"], provider.snapshot()["snapshot_id"])

            payload["observed_at"] = (NOW + timedelta(seconds=10)).isoformat()
            path.write_text(json.dumps(payload), encoding="utf-8")
            second = provider.snapshot()
            self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
            self.assertEqual(
                first, provider.snapshot_by_id(first["snapshot_id"])
            )


if __name__ == "__main__":
    unittest.main()
