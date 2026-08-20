#!/usr/bin/env python3
"""External fail-closed guard for the isolated live Grid portfolios."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from grid_live_common import PORTFOLIO_DRAWDOWN_LIMIT_PCT, PORTFOLIOS, budget_for_quote
from grid_xgboost_risk_gate import MODEL_VERSION as XGBOOST_MODEL_VERSION
from grid_xgboost_risk_gate import SCHEMA as XGBOOST_GATE_SCHEMA
from grid_xgboost_risk_gate import atomic_json as atomic_gate_json
from grid_xgboost_risk_gate import load_runtime_xgboost_gate
from ethbtc_forced_exit_contract import (
    MODEL_VERSION as V22_MODEL_VERSION,
    PACKAGE_ID as V22_PACKAGE_ID,
    SCHEMA as V22_GATE_SCHEMA,
    atomic_json as atomic_v22_json,
    load_runtime_contract as load_runtime_v22_contract,
)
from risk_recovery import (
    EMERGENCY_ESCALATION_SECONDS,
    EXITING,
    advance_integrity_failure,
)
try:
    from account_inventory import (
        SCHEMA as INVENTORY_STATUS_SCHEMA,
        UnifiedInventoryLedger,
        api_key_fingerprint,
        canonical_sha256,
    )
    from emergency_execution import execute_market_liquidation, verify_market_liquidation
    from runtime_endpoints import OFFICIAL_BINANCE_API, binance_api_base, scenario_mode
except ModuleNotFoundError:
    from live_guard.account_inventory import (
        SCHEMA as INVENTORY_STATUS_SCHEMA,
        UnifiedInventoryLedger,
        api_key_fingerprint,
        canonical_sha256,
    )
    from live_guard.emergency_execution import (
        execute_market_liquidation, verify_market_liquidation,
    )
    from live_guard.runtime_endpoints import (
        OFFICIAL_BINANCE_API, binance_api_base, scenario_mode,
    )
try:
    from telegram_notifications import (
        RuntimeErrorChannel, append_event, build_event, runtime_error_lines,
    )
except ModuleNotFoundError:
    from live_guard.telegram_notifications import (
        RuntimeErrorChannel, append_event, build_event, runtime_error_lines,
    )

try:
    from live_guard.dca_live_guard import BinanceEmergencyClient, DockerEmergencyClient
except ImportError:  # Docker image layout
    from dca_live_guard import BinanceEmergencyClient, DockerEmergencyClient


LOG = logging.getLogger("grid-live-guard")
BINANCE_API = OFFICIAL_BINANCE_API
SCALE = Decimal("1000000")


def fill_pnl(rows: list[tuple[Any, ...]], mark_price: Decimal) -> tuple[Decimal, Decimal]:
    cashflow, net_base, fees = Decimal("0"), Decimal("0"), Decimal("0")
    for side, price_raw, amount_raw, fee_raw in rows:
        price, amount = Decimal(price_raw) / SCALE, Decimal(amount_raw) / SCALE
        fee = Decimal(fee_raw or 0) / SCALE
        if str(side).upper() == "BUY":
            cashflow -= price * amount
            net_base += amount
        elif str(side).upper() == "SELL":
            cashflow += price * amount
            net_base -= amount
        fees += fee
    return cashflow + net_base * mark_price - fees, net_base


def peak_drawdown(current_equity: Decimal, stored_peak: Decimal,
                  initial_equity: Decimal) -> tuple[Decimal, Decimal]:
    peak = max(stored_peak, initial_equity, current_equity)
    drawdown = (peak - current_equity) / peak if peak > 0 else Decimal("0")
    return peak, drawdown


class ApiClient:
    def __init__(self):
        self.base = os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000").rstrip("/")
        self.session = requests.Session()
        self.session.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])

    def request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Any:
        response = self.session.request(method, f"{self.base}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json() if response.content else {}

    def status(self) -> Any:
        return self.request("GET", "/bot-orchestration/status")

    def stop(self, bot_name: str) -> Any:
        return self.request("POST", "/bot-orchestration/stop-bot", {
            "bot_name": bot_name, "skip_order_cancellation": False, "async_backend": False,
        })

    def active_containers(self, name_filter: str) -> Any:
        response = self.session.get(
            f"{self.base}/docker/active-containers",
            params={"name_filter": name_filter},
            timeout=30,
        )
        response.raise_for_status()
        return response.json() if response.content else []

    def market(self, profile: str, pair: str, side: str, amount: Decimal) -> Any:
        return self.request("POST", "/trading/orders", {
            "account_name": profile, "connector_name": "binance", "trading_pair": pair,
            "trade_type": side, "amount": str(amount), "order_type": "MARKET", "position_action": "OPEN",
        })


class Guard:
    def __init__(self):
        self.bots_path = Path(os.getenv("BOTS_PATH", "/workspace/bots"))
        self.state_dir = Path(os.getenv("GRID_LIVE_STATE_PATH", "/workspace/grid-live-state"))
        self.manifest_path = Path(os.getenv(
            "GRID_LIVE_RESERVATIONS", "/workspace/grid-live-state/capital_reservations.json"
        ))
        self.manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists()
            else {"reservations": {}}
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "guard_state.json"
        self.audit_path = self.state_dir / "risk_audit.jsonl"
        self.notification_path = self.state_dir / "telegram_events.jsonl"
        self.runtime_errors = RuntimeErrorChannel(
            event_path=self.notification_path,
            state_path=self.state_dir / "runtime_error_state.json",
            source="grid-live-guard", strategy="grid",
            bot="grid-live-fdusd-400", pair="BTC-FDUSD,ETH-FDUSD",
        )
        self.runtime_log_cursor_path = self.state_dir / "runtime_log_cursor.json"
        try:
            self.runtime_log_cursors = json.loads(
                self.runtime_log_cursor_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            self.runtime_log_cursors = {"grid-live-fdusd-400": time.time()}
        self.inventory_ledger = UnifiedInventoryLedger(Path(os.getenv(
            "ACCOUNT_INVENTORY_LEDGER_PATH", "/workspace/account-inventory"
        )))
        self.interval = max(2, int(os.getenv("GRID_LIVE_GUARD_INTERVAL", "10")))
        self.fail_closed_seconds = max(
            20, int(os.getenv("GRID_LIVE_FAIL_CLOSED_SECONDS", "60"))
        )
        self.technical_refresh_seconds = max(
            10, int(os.getenv("GRID_XGBOOST_GATE_DISTRIBUTION_SECONDS", "30"))
        )
        self.armed = os.getenv("GRID_LIVE_TRADING_ENABLED", "false").lower() == "true"
        self.shadow = os.getenv("GRID_LIVE_GUARD_SHADOW", "false").lower() == "true"
        # FDUSD pair breakers live in the strategy so each pair can halt and
        # restore its own inventory.  An external full-bot trip would collapse
        # that isolation and is therefore intentionally disabled for FDUSD.
        self.fdusd_external_breakers_enabled = os.getenv(
            "GRID_FDUSD_EXTERNAL_BREAKERS_ENABLED", "false"
        ).lower() == "true"
        if self.armed and self.shadow:
            raise ValueError("Grid Guard cannot be armed and shadowed at the same time")
        self.technical_gate_path = self.state_dir / "xgboost_risk_gate.json"
        self.v21_in_guard_enabled = os.getenv(
            "GRID_V21_IN_GUARD_ENABLED", "true"
        ).lower() == "true"
        self.v21_live_authorized = os.getenv(
            "GRID_V21_LIVE_AUTHORIZED", "false"
        ).lower() == "true"
        self.v22_in_guard_enabled = os.getenv(
            "GRID_V22_IN_GUARD_ENABLED", "true"
        ).lower() == "true"
        self.v22_execution_mode = os.getenv(
            "GRID_V22_EXECUTION_MODE", "observe"
        ).lower()
        if self.v22_execution_mode not in {"observe", "live"}:
            raise ValueError("GRID_V22_EXECUTION_MODE must be observe or live")
        self.v22_observation_gate_path = self.state_dir / "ethbtc_forced_exit_observation.json"
        self.mechanisms = {
            "v22_weekly_buy_gate": os.getenv("GRID_RISK_V22_WEEKLY_GATE_ENABLED", "true").lower() == "true",
            "fomc_gate": os.getenv("GRID_RISK_FOMC_GATE_ENABLED", "true").lower() == "true",
            "strategy_loss_breaker": os.getenv("GRID_RISK_STRATEGY_LOSS_BREAKER_ENABLED", "true").lower() == "true",
            "strategy_drawdown_breaker": os.getenv("GRID_RISK_STRATEGY_DRAWDOWN_BREAKER_ENABLED", "true").lower() == "true",
            "portfolio_loss_breaker": os.getenv("GRID_RISK_PORTFOLIO_LOSS_BREAKER_ENABLED", "true").lower() == "true",
            "portfolio_drawdown_breaker": os.getenv("GRID_RISK_PORTFOLIO_DRAWDOWN_BREAKER_ENABLED", "true").lower() == "true",
            "position_protection": os.getenv("GRID_RISK_POSITION_PROTECTION_ENABLED", "true").lower() == "true",
        }
        if self.v21_in_guard_enabled:
            # Keep the heavyweight XGBoost/joblib dependency inside the Guard
            # container's inference path.  Importing Guard utilities for
            # preflight or unit tests must not require the ML runtime.
            from grid_v21_live_gate import V21LiveGateProducer
            self.v21_producer = V21LiveGateProducer(
                package_dir=Path(os.getenv("GRID_V21_PACKAGE_PATH", "/workspace/package")),
                cache_dir=Path(os.getenv("GRID_V21_CANDLE_PATH", "/workspace/v21-candles")),
                seed_cache_dir=Path(os.getenv(
                    "GRID_V21_SEED_CANDLE_PATH", "/workspace/research-candles"
                )),
                state_dir=self.state_dir,
                authorized=self.v21_live_authorized,
                refresh_binance=True,
            )
        else:
            self.v21_producer = None
        if self.v22_in_guard_enabled:
            from grid_v22_live_gate import V22LiveGateProducer
            self.v22_producer = V22LiveGateProducer(
                package_dir=Path(os.getenv("GRID_V22_PACKAGE_PATH", "/workspace/v22-package")),
                cache_dir=Path(os.getenv("GRID_V22_CANDLE_PATH", "/workspace/v22-candles")),
                seed_cache_dir=Path(os.getenv(
                    "GRID_V22_SEED_CANDLE_PATH", "/workspace/research-candles"
                )),
                state_dir=self.state_dir,
                authorization_path=Path(os.getenv(
                    "GRID_V22_AUTHORIZATION_PATH",
                    "/workspace/state/ethbtc_forced_exit_authorization.json",
                )),
                refresh_binance=True,
                runtime_root=Path(os.getenv(
                    "V22_RUNTIME_ROOT", "/workspace/state/v22-runtime",
                )),
            )
            self.v22_producer.output = self.v22_observation_gate_path
        else:
            self.v22_producer = None
        if self.v22_execution_mode == "live" and self.v21_in_guard_enabled:
            raise RuntimeError("live v22 mode forbids an enabled v21 producer")
        if self.armed and not self.v22_in_guard_enabled:
            raise RuntimeError("armed Grid Guard requires the in-process v22 producer")
        self.next_technical_refresh = 0.0
        self.api = ApiClient()
        secret_path = Path(os.getenv(
            "GRID_BINANCE_EMERGENCY_CREDENTIALS_FILE",
            "/run/secrets/grid_binance_emergency_credentials",
        ))
        self.emergency_exchange = (
            BinanceEmergencyClient.from_secret_file(secret_path)
            if secret_path.exists()
            else None
        )
        if self.emergency_exchange is not None:
            self.inventory_ledger.bind_account(
                api_key_fingerprint(self.emergency_exchange.api_key)
            )
        docker_socket = os.getenv("GRID_DOCKER_SOCKET", "/var/run/docker.sock")
        self.emergency_docker = (
            DockerEmergencyClient(docker_socket) if Path(docker_socket).exists() else None
        )
        manifest_keys = self.manifest.get("reservations", {}).keys()
        self.portfolio_keys = [key for key in manifest_keys if key in PORTFOLIOS]
        if not self.portfolio_keys and self.manifest.get("portfolio") in PORTFOLIOS:
            self.portfolio_keys = [str(self.manifest["portfolio"])]
        if not self.portfolio_keys:
            requested = os.getenv("GRID_LIVE_PORTFOLIOS", "FDUSD")
            self.portfolio_keys = [
                key.strip().upper() for key in requested.split(",")
                if key.strip().upper() in PORTFOLIOS
            ]
        if not self.portfolio_keys:
            raise RuntimeError("Grid Guard has no supported portfolio configured")
        if self.armed:
            if self.emergency_exchange is None or self.emergency_docker is None:
                raise RuntimeError(
                    "armed Grid Guard requires independent Binance credentials "
                    "and the Docker emergency socket"
                )
            pairs = [pair for key in self.portfolio_keys for pair in PORTFOLIOS[key].pairs]
            self.emergency_exchange.verify_ready(pairs)
            for key in self.portfolio_keys:
                self.emergency_docker.matching_containers(PORTFOLIOS[key].bot_name)
        self.state = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {"version": 3, "bots": {}, "first_failure_at": None, "last_success_at": 0}
        )
        # ``technical_buy_gate`` was the retired ROC/SQZMOM gate snapshot.
        # Keeping it after the v22 cutover makes status consumers report a
        # contradictory Risk-On state while the executable v22 contract is
        # Risk-Off.  It is telemetry only and must not survive state migration.
        self.state.pop("technical_buy_gate", None)
        self.state.update({
            "armed": self.armed,
            "shadow": self.shadow,
            "technical_gate_schema": XGBOOST_GATE_SCHEMA,
            "technical_model_version": XGBOOST_MODEL_VERSION,
            "mechanism1_runtime_fallback": False,
            "mechanisms": dict(self.mechanisms),
            "v21_in_guard": self.v21_in_guard_enabled,
            "v21_live_authorized": self.v21_live_authorized,
            "v22_in_guard": self.v22_in_guard_enabled,
            "v22_execution_mode": self.v22_execution_mode,
            "v22_package_id": V22_PACKAGE_ID,
            "fdusd_external_breakers_enabled": self.fdusd_external_breakers_enabled,
        })
        emergency_error = None
        shadow_preflight = None
        if (self.shadow or self.armed) and self.emergency_exchange is not None:
            try:
                pairs = [pair for key in self.portfolio_keys for pair in PORTFOLIOS[key].pairs]
                shadow_preflight = self.verify_shadow_exchange_ready(pairs)
            except Exception as exc:
                emergency_error = repr(exc)
        if self.emergency_docker is not None:
            for key in self.portfolio_keys:
                portfolio = PORTFOLIOS[key]
                if self.emergency_docker.matching_containers(portfolio.bot_name):
                    self.state["bots"].setdefault(portfolio.bot_name, {
                        "portfolio": key,
                        "started_at": time.time(),
                        "tripped": False,
                        "action_complete": False,
                    })
            self.state["emergency_ready"] = bool(
                self.emergency_exchange is not None
                and self.emergency_docker is not None
                and emergency_error is None
            )
            self.state["emergency_checked_at"] = time.time()
            if emergency_error:
                self.state["emergency_error"] = emergency_error
            else:
                self.state.pop("emergency_error", None)
            if shadow_preflight is not None:
                self.state["shadow_preflight"] = shadow_preflight
            self.save()

    def save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _scan_runtime_logs(self) -> None:
        if self.emergency_docker is None:
            return
        now = time.time()
        name = "grid-live-fdusd-400"
        since = float(self.runtime_log_cursors.get(name, now))
        try:
            lines = self.emergency_docker.logs_since(name, since)
        except Exception as exc:
            self.runtime_errors.failure(
                "runtime_log_monitor", exc,
                trading_impact=(
                    "仅交易机器人日志采集暂时不可用；Grid Guard 主风控循环和交易权限不受影响。"
                ),
                severity="warning", action="retry_log_collection_next_cycle",
            )
            return
        self.runtime_errors.recovered(
            "runtime_log_monitor",
            trading_status="Grid 机器人日志采集恢复；主风控循环始终未受影响",
        )
        self.runtime_log_cursors[name] = now
        temporary = self.runtime_log_cursor_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.runtime_log_cursors, indent=2), encoding="utf-8")
        temporary.replace(self.runtime_log_cursor_path)
        errors = runtime_error_lines(lines)
        component = f"container_log:{name}"
        if errors:
            self.runtime_errors.failure(
                component, errors[-1],
                trading_impact="交易机器人日志出现错误；是否限制交易仍由现有 Grid 风控状态决定。",
                severity="warning", action="inspect_and_continue_existing_safety_logic",
                details={"matched_log_lines": len(errors)},
            )
        self.runtime_errors.recover_if_quiet(
            component, quiet_seconds=300,
            trading_status="连续5分钟无新的 Grid 机器人日志错误；当前风控状态保持不变",
        )

    def audit(self, event: str, **details):
        with self.audit_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **details,
            }, default=str) + "\n")
        self._emit_notification(event, details)

    def _emit_notification(self, audit_event: str, details: Dict[str, Any]) -> None:
        if audit_event == "grid_xgboost_risk_gate_transition":
            gate = details.get("gate") if isinstance(details.get("gate"), dict) else {}
            if not details.get("runtime_healthy"):
                failure_reason = str(details.get("reason") or "v22 runtime contract unhealthy")
                append_event(self.notification_path, build_event(
                    source="grid-live-guard", strategy="grid",
                    bot=PORTFOLIOS["FDUSD"].bot_name, pair="BTC-FDUSD,ETH-FDUSD",
                    mechanism="infrastructure_integrity_breaker", transition="TRIGGERED",
                    reason=failure_reason,
                    severity="critical", phase_to="FAIL_CLOSED",
                    action="ordinary_orders_blocked",
                    release_sha256=str(gate.get("release_sha256", "")),
                    model_sha256=str(gate.get("model_sha256", "")),
                    correlation_id=f"v22-integrity:{failure_reason}",
                ))
            elif details.get("previous_failure"):
                append_event(self.notification_path, build_event(
                    source="grid-live-guard", strategy="grid",
                    bot=PORTFOLIOS["FDUSD"].bot_name, pair="BTC-FDUSD,ETH-FDUSD",
                    mechanism="infrastructure_integrity_breaker", transition="RECOVERED",
                    reason="v22 runtime contract integrity restored", severity="info",
                    phase_from="FAIL_CLOSED", phase_to="ACTIVE",
                    action="release_only_this_integrity_gate",
                    release_sha256=str(gate.get("release_sha256", "")),
                    model_sha256=str(gate.get("model_sha256", "")),
                    correlation_id=f"v22-integrity-recovered:{details.get('previous_failure')}",
                ))
            previous_states = details.get("previous_risk_off", {})
            for pair, item in gate.get("pairs", {}).items():
                active = bool(item.get("risk_off_active"))
                if pair in previous_states and bool(previous_states[pair]) == active:
                    continue
                append_event(self.notification_path, build_event(
                    source="grid-live-guard", strategy="grid",
                    bot=PORTFOLIOS["FDUSD"].bot_name, pair=str(pair),
                    mechanism="v22_weekly_buy_gate",
                    transition="TRIGGERED" if active else "RECOVERED",
                    reason=str(item.get("reason") or item.get("transition") or "v22 state changed"),
                    severity="warning" if active else "info",
                    phase_from="ACTIVE" if active else "EXITING",
                    phase_to="EXITING" if active else "ACTIVE",
                    action="cancel_and_market_exit" if active else "await_all_gates_then_reentry",
                    trigger_value=item.get("probability"), threshold=item.get("entry_threshold"),
                    release_sha256=str(gate.get("release_sha256", "")),
                    model_sha256=str(gate.get("model_sha256", "")),
                    correlation_id=str(item.get("event_id", "")),
                ))
            return
        if audit_event not in {
            "grid_circuit_breaker_complete", "grid_circuit_breaker_action_failed",
        }:
            return
        failed = audit_event.endswith("action_failed")
        append_event(self.notification_path, build_event(
            source="grid-live-guard", strategy="grid",
            bot=str(details.get("bot", "")), pair="BTC-FDUSD,ETH-FDUSD",
            mechanism="infrastructure_integrity_breaker",
            transition="ACTION_FAILED" if failed else "LATCHED",
            reason=str(details.get("reason") or details.get("error") or audit_event),
            severity="critical", phase_to="LATCHED",
            action="manual_recovery_required" if not failed else "fail_closed_retry",
            requires_manual_action=not failed,
            correlation_id=str(details.get("reason") or audit_event),
        ))

    def verify_shadow_exchange_ready(self, pairs: list[str]) -> dict:
        if self.emergency_exchange is None:
            raise RuntimeError("Grid shadow preflight requires Binance credentials")
        fee_policy = self.emergency_exchange.verify_ready(pairs)
        restrictions = self.emergency_exchange._signed(
            "GET", "/sapi/v1/account/apiRestrictions"
        )
        if bool(restrictions.get("enableWithdrawals")):
            raise RuntimeError("Grid Binance key must not permit withdrawals")
        if not bool(restrictions.get("ipRestrict")):
            raise RuntimeError("Grid Binance key must use an IP allowlist")
        if bool(restrictions.get("enableFutures")):
            raise RuntimeError("Grid Binance key must not permit futures trading")
        if bool(restrictions.get("enableMargin")):
            raise RuntimeError("Grid Binance key must not permit margin trading")
        tested = []
        open_order_counts = {}
        commissions = {}
        for pair in pairs:
            open_order_counts[pair] = len(self.emergency_exchange.open_orders(pair))
            fee_payload = self.emergency_exchange._signed(
                "GET", "/sapi/v1/asset/tradeFee",
                {"symbol": pair.replace("-", "")},
            )
            fee_row = fee_payload[0] if isinstance(fee_payload, list) and fee_payload else fee_payload
            commissions[pair] = {
                "maker_fee": str(fee_row["makerCommission"]),
                "taker_fee": str(fee_row["takerCommission"]),
            }
            self.emergency_exchange._signed(
                "POST",
                "/api/v3/order/test",
                {
                    "symbol": pair.replace("-", ""),
                    "side": "BUY",
                    "type": "MARKET",
                    "quoteOrderQty": "10",
                },
            )
            tested.append(pair)
        balances = self.emergency_exchange.account_balances()
        # The Grid and DCA guards share one Binance account.  The immutable
        # capital reservation is only the starting allocation; after a forced
        # exit its audited Grid ownership can legitimately be zero while DCA
        # still owns the account's BTC/ETH.  Reading the reservation here used
        # to double-claim DCA inventory and kept Grid emergency readiness false.
        # The DCA guard continuously publishes the transactionally reconciled
        # account-level ownership contract, so fail closed unless that contract
        # is fresh, source-healthy, and free of ownership deficits.  The v3
        # contract deliberately reports overall healthy=false while ordinary
        # strategy orders are open.  Those orders do not make ownership
        # evidence untrustworthy and must not disable the emergency channel;
        # the liquidation path still cancels orders, refreshes balances,
        # acquires the asset lease, and rechecks ownership before any sell.
        inventory_path = self.inventory_ledger.status_path
        if not inventory_path.is_file():
            raise RuntimeError("shared inventory status is missing")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if inventory.get("schema") != INVENTORY_STATUS_SCHEMA:
            raise RuntimeError(
                f"shared inventory status schema is not {INVENTORY_STATUS_SCHEMA}"
            )
        age = time.time() - float(inventory.get("generated_at") or 0)
        if age < -5 or age > 30:
            raise RuntimeError(f"shared inventory status is stale: age={age:.1f}s")
        if inventory.get("sources_healthy") is not True:
            raise RuntimeError("shared inventory sources are unhealthy")
        if not inventory.get("account_fingerprint"):
            raise RuntimeError("shared inventory account fingerprint is missing")
        if not inventory.get("evidence_sha256"):
            raise RuntimeError("shared inventory evidence hash is missing")
        try:
            active_order_count = int(inventory.get("active_order_count", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("shared inventory active order count is invalid") from exc
        if active_order_count < 0:
            raise RuntimeError("shared inventory active order count is invalid")
        ownership_coverage = {}
        for pair in pairs:
            asset = pair.split("-")[0]
            asset_status = inventory.get("assets", {}).get(asset, {})
            if Decimal(str(asset_status.get("ownership_deficit", "0"))) > 0:
                raise RuntimeError(f"shared inventory ownership deficit for {asset}")
            owner_key = f"grid:{PORTFOLIOS['FDUSD'].bot_name}"
            owners = asset_status.get("owners", {})
            if owner_key not in owners:
                raise RuntimeError(f"shared inventory has no Grid owner row for {asset}")
            managed = Decimal(str(owners[owner_key]))
            available = balances.get(asset, {}).get("total", Decimal("0"))
            ownership_coverage[pair] = {
                "managed_base": str(managed), "account_total_base": str(available),
                "covered": managed >= 0 and managed <= available,
                "source": "shared_account_inventory_v3",
            }
        if not all(item["covered"] for item in ownership_coverage.values()):
            raise RuntimeError("Grid managed inventory exceeds the emergency account balance")
        return {
            "account_read": True,
            "spot_trading": True,
            "spot_bnb_burn_disabled": bool(
                fee_policy.get("spot_bnb_burn_disabled")
            ),
            "withdrawals_disabled": True,
            "ip_restricted": True,
            "futures_disabled": True,
            "margin_disabled": True,
            "test_order_no_fill": True,
            "test_order_pairs": tested,
            "open_order_counts": open_order_counts,
            "commissions": commissions,
            "ownership_coverage": ownership_coverage,
            "inventory_contract": {
                "schema": INVENTORY_STATUS_SCHEMA,
                "sources_healthy": True,
                "overall_healthy": bool(inventory.get("healthy")),
                "active_order_count": active_order_count,
                "ordinary_orders_allowed_during_readiness": True,
                "orders_must_be_cancelled_before_liquidation": True,
            },
        }

    def _technical_gate_targets(self) -> list[Path]:
        targets = [self.technical_gate_path]
        for key in self.portfolio_keys:
            bot_name = PORTFOLIOS[key].bot_name
            for instance in (self.bots_path / "instances").glob(f"{bot_name}*"):
                targets.append(instance / "data" / "xgboost_risk_gate.json")
        return sorted(set(targets))

    @staticmethod
    def _is_distributable_technical_gate(gate: dict) -> bool:
        """Only the frozen live schema may cross into Hummingbot instances."""
        return bool(
            (gate.get("schema") == XGBOOST_GATE_SCHEMA
             and gate.get("model_version") == XGBOOST_MODEL_VERSION)
            or (gate.get("schema") == V22_GATE_SCHEMA
                and gate.get("model_version") == V22_MODEL_VERSION
                and gate.get("package_id") == V22_PACKAGE_ID)
        )

    @classmethod
    def _is_usable_cached_technical_gate(cls, gate: dict, now: float) -> bool:
        """Return true only while the prior signed contract remains valid."""
        if not cls._is_distributable_technical_gate(gate):
            return False
        if gate.get("source_healthy") is not True:
            return False
        if gate.get("schema") == V22_GATE_SCHEMA and gate.get("execution_authorized") is not True:
            return False
        if not all(pair in gate.get("pairs", {}) for pair in ("BTC-FDUSD", "ETH-FDUSD")):
            return False
        try:
            generated = datetime.fromisoformat(
                str(gate["generated_at"]).replace("Z", "+00:00")
            ).timestamp()
            valid_until = datetime.fromisoformat(
                str(gate["valid_until"]).replace("Z", "+00:00")
            ).timestamp()
        except (KeyError, TypeError, ValueError):
            return False
        return generated <= now + 10 and now <= valid_until

    @staticmethod
    def _runtime_error_is_active(runtime_errors: Any, component: str) -> bool:
        state = getattr(runtime_errors, "state", {})
        if not isinstance(state, dict):
            return False
        components = state.get("components", {})
        return bool(
            isinstance(components, dict)
            and isinstance(components.get(component), dict)
            and components[component].get("active")
        )

    def publish_technical_buy_gate(self, *, force: bool = False) -> dict:
        """Observe v22 beside v21, then permanently cut over at activation."""
        now = time.time()
        if not force and now < self.next_technical_refresh:
            gate = dict(self.state.get("xgboost_risk_gate", {}))
            if self._is_distributable_technical_gate(gate):
                for target in self._technical_gate_targets():
                    if target != self.technical_gate_path:
                        atomic_gate_json(target, gate)
            return gate
        self.next_technical_refresh = now + self.technical_refresh_seconds
        previous_gate = self.state.get("xgboost_risk_gate", {})
        if self.v22_producer is None:
            raise RuntimeError("in-guard v22 producer is disabled")
        try:
            v22_gate = self.v22_producer.produce(int(now))
            v22_runtime = load_runtime_v22_contract(self.v22_observation_gate_path)
        except Exception as exc:
            v22_gate = {}
            v22_runtime = {"runtime_gate_healthy": False, "reason": f"fail_closed:{exc!r}"}
        observation = self.state.setdefault("v22_observation", {})
        release = str(v22_gate.get("release_sha256", ""))
        if release and observation.get("release_sha256") != release:
            observation.clear()
            observation.update({"release_sha256": release, "started_at": now, "cycles": 0,
                                "source_errors": 0, "integrity_errors": 0})
        observation["last_seen_at"] = now
        observation["cycles"] = int(observation.get("cycles", 0)) + 1
        observation["event_ids"] = {
            pair: item.get("event_id") for pair, item in v22_gate.get("pairs", {}).items()
        }
        if not v22_runtime.get("runtime_gate_healthy"):
            failure = str(v22_runtime.get("reason", ""))
            category = "source_errors" if any(
                marker in failure.lower() for marker in ("timeout", "connection", "temporarily")
            ) else "integrity_errors"
            observation[category] = int(observation.get(category, 0)) + 1
            observation["last_error"] = failure
        cutover = bool(self.state.get("v22_cutover_complete"))
        if bool(v22_gate.get("execution_authorized")):
            cutover = True
            self.state["v22_cutover_complete"] = True
            self.state["v22_activated_at"] = now
        if cutover or self.v22_execution_mode == "live":
            gate = v22_gate
            runtime = v22_runtime
            healthy = bool(runtime.get("runtime_gate_healthy"))
            self.state["active_technical_producer"] = "v22"
        else:
            if self.v21_producer is None:
                raise RuntimeError("v22 observation requires the existing v21 live producer")
            self.v21_producer.produce(int(now))
            gate = json.loads(self.technical_gate_path.read_text(encoding="utf-8"))
            runtime = load_runtime_xgboost_gate(self.technical_gate_path)
            healthy = bool(runtime.get("runtime_gate_healthy"))
            self.state["active_technical_producer"] = "v21_observation_bridge"

        raw_failure = None if healthy else str(
            runtime.get("reason") or "technical runtime unhealthy"
        )
        held_cached_contract = False
        if raw_failure is not None:
            failure = advance_integrity_failure(
                self.state.get("technical_refresh_failure"),
                reason=raw_failure,
                now=now,
                grace_seconds=self.fail_closed_seconds,
            )
            self.state["technical_refresh_failure"] = failure
            if (
                failure["classification"] == "transient_transport"
                and not failure["expired"]
                and self._is_usable_cached_technical_gate(previous_gate, now)
            ):
                # Never extend or rewrite timestamps/hashes.  The last verified
                # contract is reused only inside its original signed validity.
                gate = dict(previous_gate)
                healthy = True
                held_cached_contract = True
                runtime = {
                    "runtime_gate_healthy": True,
                    "reason": "transient_transport_grace_using_cached_contract",
                }
                if failure["attempts"] == 1:
                    self.audit(
                        "grid_v22_transport_grace_started",
                        reason=raw_failure,
                        grace_seconds=self.fail_closed_seconds,
                    )
                if hasattr(self, "runtime_errors"):
                    self.runtime_errors.failure(
                        "v22_contract_refresh",
                        raw_failure,
                        trading_impact=(
                            "Transient source failure: keep the last still-valid signed contract; "
                            "retry without forced exit or integrity latch."
                        ),
                    )
            else:
                if (
                    failure["classification"] == "transient_transport"
                    and failure["expired"]
                    and isinstance(gate, dict)
                ):
                    gate = dict(gate)
                    gate["reason"] = (
                        "fail_closed:transient_grace_expired:"
                        f"{self.fail_closed_seconds}s:{raw_failure}"
                    )
                    runtime = {
                        **dict(runtime),
                        "runtime_gate_healthy": False,
                        "reason": gate["reason"],
                    }
                if hasattr(self, "runtime_errors"):
                    self.runtime_errors.failure(
                        "v22_contract_refresh",
                        raw_failure,
                        trading_impact=(
                            "Integrity failure or transport grace expired; publish Fail-Closed "
                            "and let the existing exit/latch path take control."
                        ),
                    )
        else:
            previous_refresh_failure = self.state.pop("technical_refresh_failure", None)
            runtime_failure_active = bool(
                hasattr(self, "runtime_errors")
                and self._runtime_error_is_active(
                    self.runtime_errors, "v22_contract_refresh"
                )
            )
            if previous_refresh_failure or runtime_failure_active:
                self.audit(
                    "grid_v22_transport_grace_recovered",
                    previous_failure=previous_refresh_failure or {},
                )
            if hasattr(self, "runtime_errors") and runtime_failure_active:
                self.runtime_errors.recovered(
                    "v22_contract_refresh",
                    trading_status="v22 contract refresh healthy; normal signed signal applies",
                )
        atomic_v22_json(self.technical_gate_path, gate)
        distributable = self._is_distributable_technical_gate(gate)
        for target in self._technical_gate_targets():
            if target != self.technical_gate_path and distributable:
                atomic_gate_json(target, gate)
        previous_states = {
            pair: bool(value.get("risk_off_active"))
            for pair, value in previous_gate.get("pairs", {}).items()
        }
        current_states = {
            pair: bool(value.get("risk_off_active"))
            for pair, value in gate.get("pairs", {}).items()
        }
        self.state["xgboost_risk_gate"] = gate
        current_failure = None if healthy else str(runtime.get("reason") or "v22 runtime unhealthy")
        previous_failure = self.state.get("v22_notification_failure")
        self.state["v22_notification_failure"] = current_failure
        if previous_states != current_states or current_failure != previous_failure:
            self.audit(
                "grid_xgboost_risk_gate_transition",
                previous_risk_off=previous_states,
                current_risk_off=current_states,
                runtime_healthy=healthy,
                reason=runtime.get("reason"), gate=gate,
                previous_failure=previous_failure,
                source_failure=raw_failure,
                held_cached_contract=held_cached_contract,
            )
        return gate

    def database(self, bot_name: str) -> Path | None:
        candidates = sorted((self.bots_path / "instances").glob(f"{bot_name}*/data/*.sqlite"),
                            key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def price(pair: str) -> Decimal:
        response = requests.get(f"{binance_api_base()}/api/v3/ticker/price",
                                params={"symbol": pair.replace("-", "")}, timeout=15)
        response.raise_for_status()
        return Decimal(str(response.json()["price"]))

    @staticmethod
    def market_filter(pair: str) -> tuple[Decimal, Decimal]:
        response = requests.get(
            f"{binance_api_base()}/api/v3/exchangeInfo",
            params={"symbol": pair.replace("-", "")},
            timeout=15,
        )
        response.raise_for_status()
        filters = {
            item["filterType"]: item
            for item in response.json()["symbols"][0]["filters"]
        }
        lot = filters.get("MARKET_LOT_SIZE") or filters["LOT_SIZE"]
        if Decimal(str(lot.get("stepSize", "0"))) <= 0:
            lot = filters["LOT_SIZE"]
        notional = filters.get("NOTIONAL") or filters["MIN_NOTIONAL"]
        return Decimal(str(lot["stepSize"])), Decimal(str(notional["minNotional"]))

    @staticmethod
    def quantity_step(pair: str) -> Decimal:
        return Guard.market_filter(pair)[0]

    @staticmethod
    def rows(database: Path, pair: str) -> list[tuple[Any, ...]]:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10)
        try:
            return connection.execute(
                "SELECT trade_type, price, amount, trade_fee_in_quote FROM TradeFill WHERE symbol=?",
                (pair,),
            ).fetchall()
        finally:
            connection.close()

    def snapshot(self, key: str) -> dict | None:
        portfolio = PORTFOLIOS[key]
        database = self.database(portfolio.bot_name)
        if database is None:
            return None
        pnl, pairs = Decimal("0"), {}
        reservations = self.manifest["reservations"][key]["base"]
        bot_state = self.state.get("bots", {}).get(portfolio.bot_name, {})
        for pair in portfolio.pairs:
            mark = self.price(pair)
            fill_value, net_base = fill_pnl(self.rows(database, pair), mark)
            adjustment_pnl = Decimal("0")
            for adjustment in bot_state.get("emergency_adjustments", []):
                if adjustment.get("pair") != pair:
                    continue
                base_delta = Decimal(str(adjustment["base_delta"]))
                quote_cashflow = Decimal(str(adjustment["quote_cashflow"]))
                fee_quote = Decimal(str(adjustment.get("fee_quote", 0)))
                net_base += base_delta
                adjustment_pnl += quote_cashflow - fee_quote + base_delta * mark
            fill_value += adjustment_pnl
            base = pair.split("-")[0]
            initial_base = Decimal(reservations[base])
            start_price = Decimal(self.manifest["prices"][pair])
            holding_pnl = initial_base * (mark - start_price)
            pair_pnl = fill_value + holding_pnl
            pnl += pair_pnl
            pairs[pair] = {"pnl": str(pair_pnl), "net_base": str(net_base), "mark": str(mark)}
        observed_at = time.time()
        return {
            "pnl": str(pnl),
            "pairs": pairs,
            "database": str(database),
            "updated_at": observed_at,
            "observed_at": observed_at,
            "database_event_at": database.stat().st_mtime,
        }

    @staticmethod
    def _emergency_fill_metrics(pair: str, side: str, response: dict) -> Dict[str, Any]:
        executed = Decimal(str(response.get("executedQty", "0")))
        quote = Decimal(str(response.get("cummulativeQuoteQty", "0")))
        if executed <= 0:
            raise RuntimeError(f"emergency order reported no executed quantity: {response}")
        base_asset, quote_asset = pair.split("-", 1)
        base_delta = executed if side == "BUY" else -executed
        quote_cashflow = -quote if side == "BUY" else quote
        fee_quote = Decimal("0")
        fee_details = []
        for fill in response.get("fills", []):
            commission = Decimal(str(fill.get("commission", "0")))
            asset = str(fill.get("commissionAsset", ""))
            fill_price = Decimal(str(fill.get("price", "0")))
            fee_details.append({
                "asset": asset, "commission": str(commission),
                "fill_price": str(fill_price),
            })
            if asset == quote_asset:
                fee_quote += commission
            elif asset == base_asset:
                base_delta -= commission
                fee_quote += commission * fill_price
        return {
            "base_delta": str(base_delta),
            "quote_cashflow": str(quote_cashflow),
            "fee_quote": str(fee_quote),
            "fee_details": fee_details,
        }

    def _idempotent_market_order(
        self, pair: str, side: str, amount: Decimal, client_order_id: str,
    ) -> dict:
        existing = None
        if hasattr(self.emergency_exchange, "order_by_client_id"):
            existing = self.emergency_exchange.order_by_client_id(pair, client_order_id)
        if isinstance(existing, dict) and existing.get("status") == "FILLED":
            return existing
        try:
            return self.emergency_exchange.market_order(
                pair, side, amount, client_order_id
            )
        except TypeError:
            return self.emergency_exchange.market_order(pair, side, amount)
        except Exception:
            if hasattr(self.emergency_exchange, "order_by_client_id"):
                existing = self.emergency_exchange.order_by_client_id(pair, client_order_id)
                if isinstance(existing, dict) and existing.get("status") == "FILLED":
                    return existing
            raise

    def flatten_deltas(self, key: str, snapshot: dict, bot: dict):
        portfolio = PORTFOLIOS[key]
        results = bot.setdefault("flatten", {})
        reservations = getattr(self, "manifest", {}).get("reservations", {}).get(key, {}).get("base", {})
        for pair, values in snapshot["pairs"].items():
            if pair in results:
                continue
            delta, mark = Decimal(values["net_base"]), Decimal(values["mark"])
            base_asset = pair.split("-", 1)[0]
            if base_asset not in reservations:
                # No ownership proof means no emergency market order.
                results[pair] = "ownership_unavailable_no_action"
                self.save()
                continue
            # Ownership is bounded by the signed capital reservation plus this
            # bot's audited fills; never derive an amount from account balance.
            owned = max(Decimal(str(reservations[base_asset])) + delta, Decimal("0"))
            step, minimum_notional = self.market_filter(pair)
            if owned * mark < minimum_notional:
                results[pair] = "dust"
                self.save()
                continue
            amount = (owned / step).to_integral_value(rounding=ROUND_DOWN) * step
            if amount <= 0 or amount * mark < minimum_notional:
                results[pair] = "dust_after_exchange_rounding"
                self.save()
                continue
            side = "SELL"
            if self.emergency_exchange is None:
                raise RuntimeError("independent Binance emergency client is unavailable")
            digest = canonical_sha256({
                "bot": portfolio.bot_name, "pair": pair, "amount": str(amount),
                "tripped_at": bot.get("tripped_at"),
            })
            client_order_id = f"inv-{digest[:24]}"
            ledger = getattr(self, "inventory_ledger", None)
            lease_holder = f"grid-live-guard:{portfolio.bot_name}"
            lease = ledger.lease(
                base_asset, lease_holder, ttl_seconds=45
            ) if ledger else nullcontext()
            with lease:
                if ledger:
                    job = ledger.start_job(
                        job_id=digest, asset=base_asset,
                        scope=f"grid:{portfolio.bot_name}", pair=pair,
                        requested_quantity=amount,
                        client_order_id=client_order_id,
                    )
                    if job.get("status") == "COMPLETED":
                        if not ledger.completed_job_verified(job):
                            raise RuntimeError(
                                f"completed Grid liquidation job {digest} lacks verification"
                            )
                        executed = str(job.get("executed_quantity") or "0")
                        adjustments = bot.setdefault("emergency_adjustments", [])
                        if not any(
                            str(row.get("client_order_id", "")) == client_order_id
                            for row in adjustments
                        ):
                            adjustments.append({
                                "recorded_at": datetime.now(timezone.utc).isoformat(),
                                "pair": pair, "side": "SELL",
                                "client_order_id": client_order_id,
                                "order_id": str(job.get("exchange_order_id") or ""),
                                "executed_qty": executed,
                                "cummulative_quote_qty": str(job.get("quote_quantity") or "0"),
                                "base_delta": str(-Decimal(executed)),
                                "quote_cashflow": str(job.get("quote_quantity") or "0"),
                                "fee_quote": str(job.get("fee_quote") or "0"),
                                "recovered_from_shared_ledger": True,
                            })
                            self.save()
                        results[pair] = {
                            "status": "filled", "order_id": job.get("exchange_order_id"),
                            "executed_qty": job.get("executed_quantity"),
                            "client_order_id": client_order_id,
                        }
                        continue
                try:
                    if ledger:
                        balances = self.emergency_exchange.account_balances()
                        before_total = balances.get(base_asset, {}).get(
                            "total", Decimal("0")
                        )
                        if amount > before_total + step:
                            raise RuntimeError(
                                f"Grid owned {base_asset} {amount} exceeds exchange total {before_total}"
                            )
                        ledger.assert_exit_allowed(
                            asset=base_asset, exchange_total=before_total,
                            owner_key=f"grid:{portfolio.bot_name}",
                            requested_quantity=amount, tolerance=step,
                        )
                        response = execute_market_liquidation(
                            exchange=self.emergency_exchange, ledger=ledger,
                            job_id=digest, pair=pair, side=side,
                            target_quantity=amount,
                            client_order_id=client_order_id,
                            step_size=step, minimum_notional=minimum_notional,
                            mark_price=mark, lease_asset=base_asset,
                            lease_holder=lease_holder,
                        )
                        verification = verify_market_liquidation(
                            exchange=self.emergency_exchange, pair=pair,
                            response=response, requested_quantity=amount,
                            before_total=before_total, step_size=step,
                            minimum_notional=minimum_notional,
                            mark_price=mark,
                            ledger=ledger, lease_asset=base_asset,
                            lease_holder=lease_holder,
                        )
                    else:
                        response = self._idempotent_market_order(
                            pair, side, amount, client_order_id
                        )
                        verification = None
                    metrics = self._emergency_fill_metrics(pair, side, response)
                    if ledger:
                        ledger.finish_job(
                            digest, status="COMPLETED",
                            exchange_order_id=str(response.get("orderId", "")),
                            executed_quantity=response.get("executedQty", "0"),
                            quote_quantity=response.get("cummulativeQuoteQty", "0"),
                            fee_quote=metrics.get("fee_quote", "0"),
                            fee_details=metrics.get("fee_details", []),
                            verification=verification,
                        )
                except Exception as exc:
                    if ledger:
                        ledger.finish_job(digest, status="FAILED", error=repr(exc))
                    raise
            adjustment = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "pair": pair,
                "side": side,
                "client_order_id": client_order_id,
                "order_id": str(response.get("orderId", "")),
                "executed_qty": str(response.get("executedQty", "0")),
                "cummulative_quote_qty": str(response.get("cummulativeQuoteQty", "0")),
                **metrics,
            }
            adjustments = bot.setdefault("emergency_adjustments", [])
            if not any(
                str(row.get("client_order_id", "")) == client_order_id
                for row in adjustments
            ):
                adjustments.append(adjustment)
            results[pair] = {"status": "filled", **adjustment}
            results[pair]["verification"] = verification
            self.save()
        return results

    def _stuck_recoverable_exit(self, snapshot: dict) -> tuple[bool, str]:
        if not snapshot.get("database"):
            return False, "runtime database absent"
        runtime_state_path = Path(snapshot["database"]).parent / "live_grid_runtime_state.json"
        if not runtime_state_path.exists():
            return False, "runtime state absent"
        runtime = json.loads(runtime_state_path.read_text(encoding="utf-8"))
        now = time.time()
        states = list(runtime.get("pair_recovery", {}).items())
        states.append(("PORTFOLIO", runtime.get("portfolio_recovery", {})))
        for pair, state in states:
            if state.get("phase") != EXITING:
                continue
            age = now - float(state.get("triggered_at") or now)
            if age >= EMERGENCY_ESCALATION_SECONDS:
                return True, f"recoverable {pair} exit remained incomplete for {age:.1f}s"
        return False, "no stuck recoverable exit"

    @staticmethod
    def _items(payload: Any) -> list[dict]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for name in ("data", "items", "containers", "result", "response"):
                nested = payload.get(name)
                values = Guard._items(nested)
                if values:
                    return values
        return []

    def _actual_instances(self, bot_name: str) -> list[str]:
        names = []
        for item in self._items(self.api.active_containers(bot_name)):
            name = str(item.get("name", "")).lstrip("/")
            if name == bot_name or name.startswith(f"{bot_name}-"):
                names.append(name)
        return sorted(set(names))

    def _secure_stop(self, key: str) -> dict:
        if self.emergency_exchange is None or self.emergency_docker is None:
            raise RuntimeError("independent emergency clients are unavailable")
        portfolio = PORTFOLIOS[key]
        instances = set(self.emergency_docker.matching_containers(portfolio.bot_name))
        discovery_errors = []
        try:
            instances.update(self._actual_instances(portfolio.bot_name))
        except Exception as exc:
            discovery_errors.append(repr(exc))
        result = {
            "logical_bot_name": portfolio.bot_name,
            "resolved_instances": sorted(instances),
            "discovery_errors": discovery_errors,
            "graceful_stops": [],
            "exchange_cancellations": {},
            "container_stops": [],
        }
        for candidate in dict.fromkeys([portfolio.bot_name, *sorted(instances)]):
            try:
                result["graceful_stops"].append({
                    "name": candidate,
                    "response": self.api.stop(candidate),
                })
            except Exception as exc:
                result["graceful_stops"].append({"name": candidate, "error": repr(exc)})
        for pair in portfolio.pairs:
            orders = self.emergency_exchange.open_orders(pair)
            result["exchange_cancellations"][pair] = [
                str(item.get("clientOrderId") or item.get("orderId")) for item in orders
            ]
            if orders:
                self.emergency_exchange.cancel_all_orders(pair)
        for instance in sorted(instances):
            result["container_stops"].append({
                "name": instance,
                "response": self.emergency_docker.stop(instance),
            })
        for pair in portfolio.pairs:
            remaining = self.emergency_exchange.open_orders(pair)
            for _ in range(5):
                if not remaining:
                    break
                time.sleep(1)
                self.emergency_exchange.cancel_all_orders(pair)
                remaining = self.emergency_exchange.open_orders(pair)
            if remaining:
                raise RuntimeError(f"active exchange orders remain for {pair}: {remaining}")
        live_instances = self.emergency_docker.matching_containers(portfolio.bot_name)
        for _ in range(5):
            if not live_instances:
                break
            time.sleep(1)
            live_instances = self.emergency_docker.matching_containers(portfolio.bot_name)
        if live_instances:
            raise RuntimeError(f"Grid containers remain live: {live_instances}")
        result["verified_no_active_orders"] = True
        result["verified_no_live_instances"] = True
        return result

    def trip(self, key: str, reason: str, snapshot: Optional[dict]):
        portfolio = PORTFOLIOS[key]
        bot = self.state["bots"].setdefault(portfolio.bot_name, {})
        if getattr(self, "shadow", False):
            observation = {
                "reason": reason,
                "observed_at": time.time(),
                "snapshot": snapshot,
            }
            bot["would_trip"] = observation
            self.audit(
                "grid_circuit_breaker_would_trip",
                bot=portfolio.bot_name,
                **observation,
            )
            self.save()
            return
        if bot.get("action_complete"):
            return
        bot.update({"tripped": True, "reason": reason, "tripped_at": time.time()})
        self.save()
        try:
            stop = self._secure_stop(key)
            # The stopped strategy may have filled or flattened while stopping.
            # Re-read the database before any independent compensation so a
            # stale pre-stop delta can never create the opposite exposure.
            post_stop = None
            for _ in range(5):
                post_stop = self.snapshot(key)
                if post_stop is not None:
                    break
                time.sleep(1)
            if post_stop is None:
                raise RuntimeError("post-stop Grid inventory could not be read")
            bot["flatten"] = {}
            flatten = self.flatten_deltas(key, post_stop, bot)
            reconciled = self.snapshot(key)
            bot.update({
                "stop": stop,
                "stop_complete": True,
                "action_complete": True,
                "post_stop_snapshot": post_stop,
                "reconciled_snapshot": reconciled,
            })
            self.audit("grid_circuit_breaker_complete", bot=portfolio.bot_name,
                       reason=reason, snapshot=snapshot, post_stop=post_stop,
                       reconciled=reconciled, stop=stop, flatten=flatten)
        except Exception as exc:
            bot["action_complete"] = False
            bot["last_action_error"] = repr(exc)
            self.audit("grid_circuit_breaker_action_failed", bot=portfolio.bot_name,
                       reason=reason, error=repr(exc))
        finally:
            self.save()

    def cycle(self):
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        status = json.dumps(self.api.status(), ensure_ascii=True)
        self.publish_technical_buy_gate()
        snapshots = {}
        for key in self.portfolio_keys:
            portfolio = PORTFOLIOS[key]
            docker_running = bool(
                self.emergency_docker
                and self.emergency_docker.matching_containers(portfolio.bot_name)
            )
            if portfolio.bot_name not in status and not docker_running:
                continue
            snapshot = self.snapshot(key)
            if snapshot is None:
                raise RuntimeError(f"Missing SQLite for running bot {portfolio.bot_name}")
            snapshots[key] = snapshot
            bot = self.state["bots"].setdefault(portfolio.bot_name, {})
            budget = budget_for_quote(key)
            equity = budget.capital_limit + Decimal(snapshot["pnl"])
            peak, drawdown = peak_drawdown(
                equity,
                Decimal(str(bot.get("peak_equity", budget.capital_limit))),
                budget.capital_limit,
            )
            snapshot.update({
                "equity": str(equity),
                "peak_equity": str(peak),
                "drawdown_pct": str(drawdown),
            })
            bot["peak_equity"] = str(peak)
            bot["latest"] = snapshot
            stuck, stuck_reason = self._stuck_recoverable_exit(snapshot) if key == "FDUSD" else (False, "")
            if stuck:
                self.trip(key, stuck_reason, snapshot)
                continue
            pair_limit = budget.pair_loss_limit
            breached_pair = next(
                (
                    pair for pair, values in snapshot["pairs"].items()
                    if Decimal(values["pnl"]) <= -pair_limit
                ),
                None,
            )
            pnl = Decimal(snapshot["pnl"])
            if bot.get("tripped") and not bot.get("action_complete"):
                self.trip(key, bot.get("reason", "retry incomplete breaker"), snapshot)
            elif breached_pair is not None and (
                key != "FDUSD" or getattr(self, "fdusd_external_breakers_enabled", False)
            ):
                self.trip(
                    key,
                    f"pair PnL {breached_pair} {snapshot['pairs'][breached_pair]['pnl']} "
                    f"<= -{pair_limit} {key}",
                    snapshot,
                )
            elif pnl <= -budget.portfolio_loss_limit and (
                key != "FDUSD" or getattr(self, "fdusd_external_breakers_enabled", False)
            ):
                self.trip(
                    key,
                    f"portfolio PnL {pnl} <= -{budget.portfolio_loss_limit} {key}",
                    snapshot,
                )
            elif drawdown >= PORTFOLIO_DRAWDOWN_LIMIT_PCT and (
                key != "FDUSD" or getattr(self, "fdusd_external_breakers_enabled", False)
            ):
                self.trip(
                    key,
                    f"portfolio peak drawdown {drawdown:.2%} >= {PORTFOLIO_DRAWDOWN_LIMIT_PCT:.2%}",
                    snapshot,
                )
        combined_pnl = sum((Decimal(item["pnl"]) for item in snapshots.values()), Decimal("0"))
        self.state["combined_pnl"] = str(combined_pnl)
        if len(snapshots) == len(self.portfolio_keys):
            initial = sum(
                (budget_for_quote(key).capital_limit for key in snapshots), Decimal("0")
            )
            combined_equity = initial + combined_pnl
            combined_peak, combined_drawdown = peak_drawdown(
                combined_equity,
                Decimal(str(self.state.get("combined_peak_equity", initial))),
                initial,
            )
            self.state["combined_peak_equity"] = str(combined_peak)
            self.state["combined_drawdown_pct"] = str(combined_drawdown)
        else:
            combined_drawdown = Decimal("0")
        if (
            getattr(self, "fdusd_external_breakers_enabled", False)
            and
            len(snapshots) == len(self.portfolio_keys)
            and combined_drawdown >= PORTFOLIO_DRAWDOWN_LIMIT_PCT
        ):
            for key, snapshot in snapshots.items():
                self.trip(
                    key,
                    f"combined peak drawdown {combined_drawdown:.2%} >= "
                    f"{PORTFOLIO_DRAWDOWN_LIMIT_PCT:.2%}",
                    snapshot,
                )
        self.state["first_failure_at"] = None
        self.state["last_success_at"] = time.time()
        self.save()

    def run(self):
        while True:
            try:
                self.cycle()
                self._scan_runtime_logs()
                self.runtime_errors.recovered(
                    "guard_cycle",
                    trading_status="Guard healthy；existing risk gates unchanged",
                )
            except Exception as exc:
                LOG.exception("Grid guard cycle failed")
                first = self.state.get("first_failure_at") or time.time()
                self.runtime_errors.failure(
                    "guard_cycle", exc,
                    trading_impact=(
                        "本周期监控失败，Guard 正在自动重试；达到连续失败阈值前不改变交易权限，"
                        "达到阈值后由既有 Fail-Closed 风控事件接管。"
                    ),
                    severity="warning", action="retry_then_fail_closed_on_threshold",
                    details={"fail_closed_after_seconds": self.fail_closed_seconds},
                    now=first,
                )
                self.state["first_failure_at"] = self.state.get("first_failure_at") or time.time()
                self.save()
                if time.time() - self.state["first_failure_at"] >= self.fail_closed_seconds:
                    for key in self.portfolio_keys:
                        portfolio = PORTFOLIOS[key]
                        if portfolio.bot_name in self.state["bots"]:
                            self.trip(
                                key,
                                f"monitor unavailable for {self.fail_closed_seconds} seconds",
                                self.state["bots"][portfolio.bot_name].get("latest"),
                            )
            time.sleep(self.interval)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    if "--scenario-smoke" in sys.argv[1:]:
        if not scenario_mode():
            raise RuntimeError("Grid scenario smoke requires GUARD_SCENARIO_MODE=true")
        guard = Guard()
        result = {
            "scenario_id": os.environ["GUARD_SCENARIO_ID"],
            "prices": {pair: str(guard.price(pair)) for pair in ("BTC-FDUSD", "ETH-FDUSD")},
            "filters": {
                pair: [str(value) for value in guard.market_filter(pair)]
                for pair in ("BTC-FDUSD", "ETH-FDUSD")
            },
            "balances": guard.emergency_exchange.account_balances(),
        }
        print(json.dumps(result, default=str), flush=True)
        return 0
    armed = os.getenv("GRID_LIVE_TRADING_ENABLED", "false").lower() == "true"
    shadow = os.getenv("GRID_LIVE_GUARD_SHADOW", "false").lower() == "true"
    if not armed and not shadow:
        LOG.warning("Grid live guard is disabled; no API action will be taken.")
        return 0
    Guard().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
