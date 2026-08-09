#!/usr/bin/env python3
"""Fail-closed risk guard for the isolated 200 USD live DCA bots."""

from __future__ import annotations

import json
import hashlib
import hmac
import http.client
import logging
import os
import socket
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlencode

import requests

from dca_live_common import (
    ACCOUNT_NAME,
    COMBINED_LOSS_LIMIT,
    CONNECTOR,
    LIVE_PAIRS,
    SINGLE_BOT_LOSS_LIMIT,
    STRATEGY_BUDGET_QUOTE,
    side_budget,
    trade_pnl_from_rows,
)
from grid_xgboost_risk_gate import load_runtime_xgboost_gate
from ethbtc_forced_exit_contract import (
    SCHEMA as V22_CONTRACT_SCHEMA,
    load_runtime_contract as load_runtime_v22_contract,
)
from risk_recovery import (
    ACTIVE, COOLDOWN, EXITING, LATCHED, REENTRY,
    EMERGENCY_ESCALATION_SECONDS, EXIT_CRITICAL_SECONDS,
    advance_recovery, active_state,
    mark_exit_complete, mark_reentry_complete, normalize_state, trigger_state,
)
try:
    from telegram_notifications import append_event, build_event
except ModuleNotFoundError:
    from live_guard.telegram_notifications import append_event, build_event


LOG = logging.getLogger("dca-live-guard")
BINANCE_API = "https://api.binance.com"
V21_PAIR_MAP = {"BTC-USDT": "BTC-FDUSD", "ETH-USDT": "ETH-FDUSD"}
V22_PAIR_MAP = V21_PAIR_MAP


def _env_enabled(name: str, default: bool = True) -> bool:
    return os.getenv(name, "true" if default else "false").lower() == "true"


class BinanceEmergencyClient:
    """Minimal signed Binance client independent of Hummingbot and MQTT."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = BINANCE_API):
        if not api_key or not api_secret:
            raise ValueError("Binance emergency credentials are incomplete")
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
        self.time_offset_ms = 0

    @classmethod
    def from_secret_file(cls, path: Path) -> "BinanceEmergencyClient":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Binance emergency secret must be a JSON object")
        return cls(
            str(value.get("api_key", "")).strip(),
            str(value.get("api_secret", "")).strip(),
            str(value.get("base_url", BINANCE_API)).strip(),
        )

    @staticmethod
    def symbol(pair: str) -> str:
        return pair.replace("-", "").upper()

    def sync_time(self) -> None:
        response = self.session.get(f"{self.base_url}/api/v3/time", timeout=10)
        response.raise_for_status()
        self.time_offset_ms = int(response.json()["serverTime"]) - int(time.time() * 1000)

    def _signed(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000) + self.time_offset_ms
        params["recvWindow"] = 5000
        query = urlencode(params)
        params["signature"] = hmac.new(
            self.api_secret, query.encode(), hashlib.sha256
        ).hexdigest()
        response = self.session.request(
            method, f"{self.base_url}{path}", params=params, timeout=15
        )
        if response.status_code == 400 and "-1021" in response.text:
            self.sync_time()
            return self._signed(method, path, {k: v for k, v in params.items() if k not in {"timestamp", "recvWindow", "signature"}})
        if response.status_code >= 400:
            raise RuntimeError(
                f"Binance emergency {method} {path} failed "
                f"({response.status_code}): {response.text[:300]}"
            )
        return response.json() if response.content else {}

    def open_orders(self, pair: str) -> list[dict]:
        value = self._signed("GET", "/api/v3/openOrders", {"symbol": self.symbol(pair)})
        return value if isinstance(value, list) else []

    def cancel_all_orders(self, pair: str) -> Any:
        return self._signed("DELETE", "/api/v3/openOrders", {"symbol": self.symbol(pair)})

    def verify_ready(self, pairs: list[str]) -> None:
        self.sync_time()
        account = self._signed("GET", "/api/v3/account")
        if not isinstance(account, dict) or account.get("canTrade") is not True:
            raise RuntimeError("Binance emergency key does not have trading permission")
        # Binance's account-level canWithdraw flag does not expose the API
        # key's withdrawal permission, so it must not be used to reject an
        # otherwise valid key. Withdrawal permission and IP restrictions are
        # enforced in Binance API Management and audited during deployment.
        for pair in pairs:
            self.open_orders(pair)

    def account_balances(self) -> Dict[str, Dict[str, Decimal]]:
        account = self._signed("GET", "/api/v3/account")
        result: Dict[str, Dict[str, Decimal]] = {}
        for row in account.get("balances", []):
            free = Decimal(str(row.get("free", "0")))
            locked = Decimal(str(row.get("locked", "0")))
            result[str(row.get("asset", ""))] = {
                "free": free, "locked": locked, "total": free + locked,
            }
        return result

    def market_order(self, pair: str, side: str, amount: Decimal) -> dict:
        value = self._signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": self.symbol(pair),
                "side": side,
                "type": "MARKET",
                "quantity": format(amount, "f"),
                "newOrderRespType": "FULL",
            },
        )
        if not isinstance(value, dict) or value.get("status") != "FILLED":
            raise RuntimeError(f"Binance emergency market order was not FILLED: {value}")
        return value

class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class DockerEmergencyClient:
    """Small Docker Engine client used when orchestration/MQTT is unavailable."""

    def __init__(self, socket_path: str = "/var/run/docker.sock"):
        self.socket_path = socket_path

    def _request(self, method: str, path: str) -> Any:
        connection = _UnixHTTPConnection(self.socket_path)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            payload = response.read()
            if response.status >= 400 and response.status != 304:
                raise RuntimeError(
                    f"Docker emergency {method} {path} failed ({response.status}): "
                    f"{payload[:300].decode(errors='replace')}"
                )
            return json.loads(payload) if payload else {}
        finally:
            connection.close()

    def matching_containers(self, bot_name: str) -> list[str]:
        filters = quote(json.dumps({"name": [bot_name]}), safe="")
        values = self._request("GET", f"/containers/json?filters={filters}")
        names = []
        for item in values if isinstance(values, list) else []:
            for raw_name in item.get("Names", []):
                name = str(raw_name).lstrip("/")
                if name == bot_name or name.startswith(f"{bot_name}-"):
                    names.append(name)
        return sorted(set(names))

    def stop(self, container_name: str) -> Any:
        return self._request("POST", f"/containers/{quote(container_name, safe='')}/stop?t=10")


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (os.environ["USERNAME"], os.environ["PASSWORD"])

    def request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        response = self.session.request(method, f"{self.base_url}{path}", json=payload, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} failed ({response.status_code}): {response.text[:500]}")
        return response.json() if response.content else {}

    def status(self) -> Any:
        return self.request("GET", "/bot-orchestration/status")

    def stop_bot(self, bot_name: str) -> Any:
        return self.request("POST", "/bot-orchestration/stop-bot", {
            "bot_name": bot_name,
            "skip_order_cancellation": False,
            "async_backend": False,
        })

    def active_containers(self, name_filter: str) -> Any:
        response = self.session.get(
            f"{self.base_url}/docker/active-containers",
            params={"name_filter": name_filter},
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"GET /docker/active-containers failed ({response.status_code}): "
                f"{response.text[:500]}"
            )
        return response.json() if response.content else []

    def stop_container(self, container_name: str) -> Any:
        encoded = quote(container_name, safe="")
        return self.request("POST", f"/docker/stop-container/{encoded}")

    def active_orders(self, pair: str) -> Any:
        return self.request("POST", "/trading/orders/active", {
            "limit": 1000,
            "account_names": [ACCOUNT_NAME],
            "connector_names": [CONNECTOR],
            "trading_pairs": [pair],
        })

    def cancel_order(self, client_order_id: str) -> Any:
        encoded = quote(client_order_id, safe="")
        return self.request(
            "POST",
            f"/trading/{ACCOUNT_NAME}/{CONNECTOR}/orders/{encoded}/cancel",
        )

    def market_order(self, pair: str, side: str, amount: Decimal) -> Any:
        return self.request("POST", "/trading/orders", {
            "account_name": ACCOUNT_NAME,
            "connector_name": CONNECTOR,
            "trading_pair": pair,
            "trade_type": side,
            "amount": str(amount),
            "order_type": "MARKET",
            "position_action": "OPEN",
        })

    def update_controller(
        self, bot_name: str, controller_name: str, profile: Dict[str, Any]
    ) -> Any:
        return self.request(
            "POST",
            f"/controllers/bots/{quote(bot_name, safe='')}/"
            f"{quote(controller_name, safe='')}/config",
            profile,
        )


class Guard:
    def __init__(self):
        self.bots_path = Path(os.getenv("BOTS_PATH", "/workspace/bots"))
        self.state_dir = Path(os.getenv("DCA_LIVE_STATE_PATH", "/workspace/state"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "guard_state.json"
        self.audit_path = self.state_dir / "risk_audit.jsonl"
        self.notification_path = self.state_dir / "telegram_events.jsonl"
        self.telemetry_path = self.state_dir / "dca_macro_telemetry.json"
        self.managed_inventory_path = self.state_dir / "managed_inventory.json"
        self.interval = max(2, int(os.getenv("DCA_LIVE_GUARD_INTERVAL", "10")))
        self.fail_closed_seconds = max(20, int(os.getenv("DCA_LIVE_FAIL_CLOSED_SECONDS", "60")))
        self.v21_gate_path = Path(os.getenv(
            "DCA_V22_GATE_PATH",
            os.getenv("DCA_V21_GATE_PATH", "/workspace/technical/xgboost_risk_gate.json"),
        ))
        self.v22_observation_gate_path = Path(os.getenv(
            "DCA_V22_OBSERVATION_GATE_PATH",
            "/workspace/technical/ethbtc_forced_exit_observation.json",
        ))
        self.macro_state_path = Path(os.getenv(
            "DCA_MACRO_STATE_PATH", "/workspace/macro/state.json"
        ))
        self.macro_max_age_seconds = max(
            10, int(os.getenv("DCA_MACRO_STATE_MAX_AGE_SECONDS", "30"))
        )
        self.v21_max_age_seconds = max(
            30, int(os.getenv("DCA_V22_MAX_AGE_SECONDS", os.getenv("DCA_V21_MAX_AGE_SECONDS", "150")))
        )
        self.mechanisms = {
            "v22_weekly_buy_gate": _env_enabled("DCA_RISK_V22_WEEKLY_GATE_ENABLED"),
            "fomc_gate": _env_enabled("DCA_RISK_FOMC_GATE_ENABLED"),
            "strategy_loss_breaker": _env_enabled("DCA_RISK_STRATEGY_LOSS_BREAKER_ENABLED"),
            "strategy_drawdown_breaker": _env_enabled("DCA_RISK_STRATEGY_DRAWDOWN_BREAKER_ENABLED"),
            "portfolio_loss_breaker": _env_enabled("DCA_RISK_PORTFOLIO_LOSS_BREAKER_ENABLED"),
            "portfolio_drawdown_breaker": _env_enabled("DCA_RISK_PORTFOLIO_DRAWDOWN_BREAKER_ENABLED"),
            "position_protection": _env_enabled("DCA_RISK_POSITION_PROTECTION_ENABLED"),
        }
        self.strategy_drawdown_limit = Decimal(os.getenv(
            "DCA_STRATEGY_DRAWDOWN_LIMIT_PCT", "0.08"
        ))
        self.portfolio_drawdown_limit = Decimal(os.getenv(
            "DCA_PORTFOLIO_DRAWDOWN_LIMIT_PCT", "0.08"
        ))
        self.auto_reentry_enabled = _env_enabled("DCA_RISK_AUTO_REENTRY_ENABLED", False)
        if not Decimal("0") < self.strategy_drawdown_limit < Decimal("1"):
            raise ValueError("DCA strategy drawdown limit must be between zero and one")
        if not Decimal("0") < self.portfolio_drawdown_limit < Decimal("1"):
            raise ValueError("DCA portfolio drawdown limit must be between zero and one")
        self.api = ApiClient(os.getenv("HUMMINGBOT_API_URL", "http://hummingbot-api:8000"))
        secret_path = Path(
            os.getenv(
                "DCA_BINANCE_EMERGENCY_CREDENTIALS_FILE",
                "/run/secrets/dca_binance_emergency_credentials",
            )
        )
        self.emergency_exchange = (
            BinanceEmergencyClient.from_secret_file(secret_path)
            if secret_path.exists()
            else None
        )
        docker_socket = os.getenv("DCA_DOCKER_SOCKET", "/var/run/docker.sock")
        self.emergency_docker = (
            DockerEmergencyClient(docker_socket)
            if Path(docker_socket).exists()
            else None
        )
        if os.getenv("DCA_LIVE_TRADING_ENABLED", "false").lower() == "true":
            if self.emergency_exchange is None or self.emergency_docker is None:
                raise RuntimeError(
                    "armed DCA Guard requires independent Binance credentials "
                    "and the Docker emergency socket"
                )
            self.emergency_exchange.verify_ready(list(LIVE_PAIRS))
            for spec in LIVE_PAIRS.values():
                self.emergency_docker.matching_containers(spec.bot_name)
        self.state = self._load_state()
        self.state["emergency_ready"] = bool(
            self.emergency_exchange is not None and self.emergency_docker is not None
        )
        if self.state["emergency_ready"] and self.managed_inventory_path.exists():
            ownership = json.loads(self.managed_inventory_path.read_text(encoding="utf-8"))
            balances = self.emergency_exchange.account_balances()
            coverage = {}
            for pair, spec in LIVE_PAIRS.items():
                managed = Decimal(str(
                    ownership.get("pairs", {}).get(pair, {}).get("managed_base", "0")
                ))
                available = balances.get(spec.base_asset, {}).get("total", Decimal("0"))
                coverage[pair] = {
                    "managed_base": str(managed), "account_total_base": str(available),
                    "covered": managed > 0 and managed <= available,
                }
            self.state["ownership_preflight"] = coverage
        self._save()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            state = {"version": 2, "armed": True, "bots": {}, "last_success_at": 0,
                     "created_at": time.time()}
        legacy = state.get("roc_buy_guard")
        if legacy is not None:
            state["roc_buy_guard"] = {
                "retired": True,
                "retired_reason": "replaced_by_ethbtc_forced_exit_v22",
                "previous_active": bool(legacy.get("active", False)),
            }
        state["version"] = 2
        state["mechanisms"] = dict(self.mechanisms)
        for bot in state.get("bots", {}).values():
            bot["recovery"] = normalize_state(bot.get("recovery"))
        return state

    def _save(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def _audit(self, event: str, **details: Any) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        audit_path = getattr(self, "audit_path", None)
        if audit_path is not None:
            with audit_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        LOG.warning("%s %s", event, json.dumps(details, ensure_ascii=False, default=str))
        self._emit_notification(event, details)

    def _emit_notification(self, audit_event: str, details: Dict[str, Any]) -> None:
        path = getattr(self, "notification_path", None)
        if path is None:
            return
        if audit_event in {"fomc_gate_transition", "v22_gate_transition"}:
            mechanism = "fomc_gate" if audit_event.startswith("fomc") else "v22_weekly_buy_gate"
            blocked = not bool(details.get("buy_enabled")) or (
                mechanism == "fomc_gate" and not bool(details.get("sell_enabled"))
            )
            transition = "TRIGGERED" if blocked else "RECOVERED"
            if mechanism == "v22_weekly_buy_gate":
                phase_from = "RISK_ON" if blocked else "RISK_OFF"
                phase_to = "RISK_OFF" if blocked else "RISK_ON"
            else:
                phase_from = "ACTIVE" if blocked else "RESTRICTED"
                phase_to = "RESTRICTED" if blocked else "ACTIVE"
            append_event(path, build_event(
                source="dca-live-guard", strategy="dca",
                bot=str(details.get("bot", "")), pair=str(details.get("pair", "")),
                mechanism=mechanism, transition=transition,
                reason=str(details.get("reason") or audit_event),
                severity="warning" if blocked else "info",
                phase_from=phase_from, phase_to=phase_to,
                action=("aggregate_gate_blocks_orders" if blocked
                        else "await_remaining_gates"),
                trigger_value=details.get("probability"),
                release_sha256=str(details.get("release_sha256", "")),
                model_sha256=str(details.get("model_sha256", "")),
                correlation_id=str(details.get("correlation_id") or (
                    f"{audit_event}:{details.get('bot')}:{blocked}:{details.get('reason')}"
                )),
                details={
                    "buy_enabled": details.get("buy_enabled"),
                    "sell_enabled": details.get("sell_enabled"),
                    "source_pair": details.get("source_pair"),
                    "effective_buy_enabled": details.get("effective_buy_enabled"),
                    "effective_sell_enabled": details.get("effective_sell_enabled"),
                    "recovery_phase": details.get("recovery_phase"),
                    "execution_applied": details.get("execution_applied"),
                    "controller_update_status": details.get("controller_update_status"),
                    "controller_update_error": details.get("controller_update_error"),
                },
            ))
            return
        mapping = {
            "recoverable_breaker_triggered": (["TRIGGERED", "EXITING"], "warning"),
            "integrity_failure_exit_then_latch": (["TRIGGERED", "EXITING"], "critical"),
            "recoverable_exit_complete": (["EXIT_COMPLETE"], "info"),
            "recoverable_exit_critical_delay": ("EXIT_DELAY", "critical"),
            "recoverable_reentry_ready": ("REENTRY", "info"),
            "recoverable_reentry_complete": ("RECOVERED", "info"),
            "portfolio_reentry_committed": ("RECOVERED", "info"),
            "circuit_breaker_complete": ("LATCHED", "critical"),
            "circuit_breaker_action_failed": ("ACTION_FAILED", "critical"),
        }
        if audit_event not in mapping:
            return
        transitions, severity = mapping[audit_event]
        if isinstance(transitions, str):
            transitions = [transitions]
        recovery = details.get("recovery") if isinstance(details.get("recovery"), dict) else {}
        bot = str(details.get("bot") or ",".join(details.get("bots", [])))
        pair = str(details.get("pair") or "")
        mechanism = str(recovery.get("mechanism") or recovery.get("previous_mechanism") or (
            "infrastructure_integrity_breaker" if "circuit" in audit_event or "integrity" in audit_event
            else details.get("mechanism") or "infrastructure_integrity_breaker"
        ))
        final_phase = str(recovery.get("phase") or "")
        if audit_event == "recoverable_reentry_complete" and final_phase == "REENTRY":
            transitions = ["REENTRY"]
        if audit_event == "recoverable_exit_complete" and final_phase in {"COOLDOWN", "LATCHED"}:
            transitions.append(final_phase)
        reason = str(details.get("reason") or recovery.get("reason") or audit_event)
        contract = getattr(self, "state", {}).get("v22_observation", {})
        correlation = str(recovery.get("triggered_at") or details.get("reason") or audit_event)
        for transition in transitions:
            phase_to = final_phase if transition == "EXIT_COMPLETE" and final_phase else transition
            manual_action = transition == "LATCHED" or (
                transition == "REENTRY" and not bool(getattr(self, "auto_reentry_enabled", False))
            )
            event = build_event(
                source="dca-live-guard", strategy="dca", bot=bot, pair=pair,
                mechanism=mechanism, transition=transition, reason=reason,
                severity=severity, phase_from="ACTIVE" if transition == "TRIGGERED" else "",
                phase_to=phase_to,
                action="cancel_orders_and_flatten" if transition in {"TRIGGERED", "EXITING", "LATCHED"} else audit_event,
                trigger_value=recovery.get("trigger_value"),
                release_sha256=str(contract.get("release_sha256", "")),
                model_sha256=str(contract.get("model_sha256", "")),
                requires_manual_action=manual_action,
                correlation_id=correlation,
            )
            append_event(path, event)

    def _notify(self, message: str) -> None:
        # Telegram delivery is intentionally centralized in dca-live-report.
        # Guards only persist auditable events and must never hold the Hermes or
        # channel bot token.
        LOG.warning("guard notification queued through audit event: %s", message)

    def _database(self, bot_name: str) -> Optional[Path]:
        exact = self.bots_path / "instances" / bot_name / "data" / f"{bot_name}.sqlite"
        if exact.exists():
            return exact
        candidates = sorted(
            (self.bots_path / "instances").glob(f"{bot_name}*/data/*.sqlite"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _price(pair: str) -> Decimal:
        response = requests.get(
            f"{BINANCE_API}/api/v3/ticker/price",
            params={"symbol": pair.replace("-", "")},
            timeout=15,
        )
        response.raise_for_status()
        return Decimal(str(response.json()["price"]))

    @staticmethod
    def _lot_filter(pair: str) -> tuple[Decimal, Decimal]:
        response = requests.get(
            f"{BINANCE_API}/api/v3/exchangeInfo",
            params={"symbol": pair.replace("-", "")},
            timeout=15,
        )
        response.raise_for_status()
        filters = {item["filterType"]: item for item in response.json()["symbols"][0]["filters"]}
        # MARKET_LOT_SIZE is authoritative for the forced-exit/re-entry market
        # orders. Some symbols expose a different market step than LOT_SIZE.
        lot = filters.get("MARKET_LOT_SIZE") or filters["LOT_SIZE"]
        notional = filters.get("NOTIONAL") or filters["MIN_NOTIONAL"]
        return Decimal(str(lot["stepSize"])), Decimal(str(notional["minNotional"]))

    @staticmethod
    def _rows(database: Path, pair: str) -> list[tuple[Any, ...]]:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10)
        try:
            return connection.execute(
                "SELECT trade_type, price, amount, trade_fee_in_quote, timestamp "
                "FROM TradeFill WHERE symbol = ? ORDER BY timestamp, rowid",
                (pair,),
            ).fetchall()
        finally:
            connection.close()

    def _snapshot(self, bot_name: str, pair: str) -> Optional[Dict[str, Any]]:
        database = self._database(bot_name)
        if database is None:
            return None
        price = self._price(pair)
        rows = self._rows(database, pair)
        metrics = trade_pnl_from_rows(rows, price)
        for adjustment in self.state.get("bots", {}).get(bot_name, {}).get(
            "emergency_adjustments", []
        ):
            metrics["net_base"] += Decimal(str(adjustment["base_delta"]))
            metrics["quote_cashflow"] += Decimal(
                str(adjustment["quote_cashflow"])
            )
            metrics["fees_quote"] += Decimal(str(adjustment.get("fee_quote", 0)))
            metrics["trades"] += Decimal("1")
        metrics["pnl_quote"] = (
            metrics["quote_cashflow"]
            - metrics["fees_quote"]
            + metrics["net_base"] * price
        )
        observed_at = time.time()
        database_event_at = database.stat().st_mtime
        if rows:
            last_fill_at = float(rows[-1][4])
            while last_fill_at > 10_000_000_000:
                last_fill_at /= 1000
            database_event_at = max(database_event_at, last_fill_at)
        return {
            "pair": pair,
            "database": str(database),
            "mark_price": str(price),
            **{key: str(value) for key, value in metrics.items()},
            "updated_at": observed_at,
            "observed_at": observed_at,
            "database_event_at": database_event_at,
            "database_event_age_seconds": max(0.0, observed_at - database_event_at),
            "latest_stop_loss_at": self._latest_stop_loss_at(database),
        }

    @staticmethod
    def _latest_stop_loss_at(database: Path) -> float:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10)
        try:
            row = connection.execute(
                "SELECT MAX(close_timestamp) FROM Executors WHERE close_type = 2"
            ).fetchone()
        finally:
            connection.close()
        value = float(row[0] or 0)
        while value > 10_000_000_000:
            value /= 1000
        return value

    @staticmethod
    def _side(value: Any) -> str:
        text = str(value).upper()
        if text in {"1", "BUY", "TRADETYPE.BUY"}:
            return "buy"
        if text in {"2", "SELL", "TRADETYPE.SELL"}:
            return "sell"
        return "unknown"

    @staticmethod
    def _executor_counts(database: Path) -> Dict[str, int]:
        connection = sqlite3.connect(
            f"file:{database}?mode=ro", uri=True, timeout=10
        )
        try:
            rows = connection.execute(
                "SELECT is_active, is_trading, config FROM Executors "
                "WHERE is_active = 1 OR is_trading = 1"
            ).fetchall()
            try:
                order_rows = connection.execute(
                    'SELECT orders.id, orders.amount, orders.last_status, '
                    'COALESCE((SELECT SUM(fills.amount) FROM TradeFill fills '
                    'WHERE fills.order_id = orders.id), 0), '
                    'COALESCE((SELECT statuses.status FROM OrderStatus statuses '
                    'WHERE statuses.order_id = orders.id '
                    "AND statuses.status IN ('BuyOrderCreated', 'SellOrderCreated') "
                    'ORDER BY statuses.timestamp LIMIT 1), orders.last_status) '
                    'FROM "Order" orders'
                ).fetchall()
            except sqlite3.OperationalError:
                order_rows = []
        finally:
            connection.close()
        counts = {
            "active_buy_executors": 0,
            "trading_buy_executors": 0,
            "active_sell_executors": 0,
            "trading_sell_executors": 0,
            "open_orders": 0,
        }
        for is_active, is_trading, raw_config in rows:
            try:
                side = Guard._side(json.loads(raw_config).get("side"))
            except (TypeError, ValueError):
                continue
            if side not in {"buy", "sell"}:
                continue
            counts[f"active_{side}_executors"] += int(bool(is_active))
            counts[f"trading_{side}_executors"] += int(bool(is_trading))
        open_sides: set[str] = set()
        trading_sides: set[str] = set()
        for _, raw_amount, status, raw_filled, creation_status in order_rows:
            normalized_status = str(status).lower()
            terminal = any(
                value in normalized_status
                for value in ("cancel", "failure", "expired", "completed")
            )
            amount = Decimal(str(raw_amount or 0))
            filled = Decimal(str(raw_filled or 0))
            if terminal or (amount > 0 and filled >= amount):
                continue
            side = (
                "buy"
                if str(creation_status) == "BuyOrderCreated"
                else "sell"
                if str(creation_status) == "SellOrderCreated"
                else "unknown"
            )
            if side == "unknown":
                continue
            counts["open_orders"] += 1
            open_sides.add(side)
            if filled > 0:
                trading_sides.add(side)
        # Active executors are not persisted by some runtime releases until
        # completion. In that case, live orders are the authoritative fallback.
        if not rows:
            for side in open_sides:
                counts[f"active_{side}_executors"] = 1
            for side in trading_sides:
                counts[f"trading_{side}_executors"] = 1
        return counts

    @staticmethod
    def _controller_profile(database: Path) -> tuple[str, Dict[str, Any]]:
        connection = sqlite3.connect(
            f"file:{database}?mode=ro", uri=True, timeout=10
        )
        try:
            row = connection.execute(
                "SELECT id, config FROM Controllers "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return "", {}
        controller_id, raw_config = row
        try:
            config = json.loads(raw_config)
        except (TypeError, ValueError):
            config = {}
        return str(controller_id), config

    @staticmethod
    def _controller_gates(database: Path) -> Dict[str, Any]:
        controller_id, config = Guard._controller_profile(database)
        if not controller_id:
            return {
                "controller_name": "",
                "macro_buy_enabled": True,
                "macro_sell_enabled": True,
                "macro_decision_id": "",
            }
        buy_enabled = bool(config.get("macro_buy_enabled", True))
        return {
            "controller_name": str(controller_id),
            "macro_buy_enabled": buy_enabled,
            "macro_sell_enabled": bool(config.get("macro_sell_enabled", True)),
            "macro_decision_id": str(config.get("macro_decision_id", "")),
        }

    def _macro_gate(self, *, now: float | None = None) -> Dict[str, Any]:
        if not self.mechanisms["fomc_gate"]:
            return {"healthy": True, "buy_enabled": True, "sell_enabled": True,
                    "reason": "fomc_gate_disabled", "active_lease_ids": []}
        observed = now if now is not None else time.time()
        try:
            payload = json.loads(self.macro_state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 3:
                raise ValueError("unsupported macro state schema")
            reconciled = datetime.fromisoformat(
                str(payload["last_reconcile"]).replace("Z", "+00:00")
            ).astimezone(timezone.utc).timestamp()
            age = observed - reconciled
            if age < -10 or age > self.macro_max_age_seconds:
                raise ValueError(f"macro state age is {age:.0f}s")
            desired = payload["desired_gates"]
            if not isinstance(desired.get("buy"), bool) or not isinstance(desired.get("sell"), bool):
                raise ValueError("macro desired gates are invalid")
            active = sorted(
                key for key, lease in payload.get("leases", {}).items()
                if lease.get("status") == "active"
            )
            return {"healthy": True, "buy_enabled": desired["buy"],
                    "sell_enabled": desired["sell"], "reason": "macro_state_healthy",
                    "active_lease_ids": active, "age_seconds": max(0, age)}
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return {"healthy": False, "buy_enabled": False, "sell_enabled": False,
                    "reason": f"fail_closed:{exc}", "active_lease_ids": []}

    def _v21_gate(self) -> Dict[str, Any]:
        # Keep the legacy key readable only for pre-cutover state/test
        # compatibility. It does not reinstate ROC/SQZMOM or a model fallback.
        enabled = self.mechanisms.get(
            "v22_weekly_buy_gate", self.mechanisms.get("v21_buy_gate", False)
        )
        if not enabled:
            return {"healthy": True, "reason": "v22_gate_disabled", "pairs": {
                pair: {"buy_enabled": True, "source_pair": source}
                for pair, source in V22_PAIR_MAP.items()
            }}
        contract = load_runtime_xgboost_gate(
            self.v21_gate_path, max_age_seconds=self.v21_max_age_seconds
        )
        healthy = bool(contract.get("runtime_gate_healthy"))
        mapped = {}
        for dca_pair, source_pair in V22_PAIR_MAP.items():
            source = contract.get("pairs", {}).get(source_pair, {})
            mapped[dca_pair] = {
                "source_pair": source_pair,
                "buy_enabled": bool(source.get("buy_enabled")) if healthy else False,
                "risk_off_active": bool(source.get("risk_off_active", True)),
                "transition": source.get("transition", "fail_closed"),
                "reason": source.get("reason", contract.get("reason", "v21_unhealthy")),
                "event_id": source.get("event_id"),
                "force_exit": bool(source.get("force_exit", False)),
                "probability": source.get("probability"),
                "execution_authorized": bool(contract.get("execution_authorized", False)),
            }
        return {"healthy": healthy, "reason": contract.get("reason"),
                "schema": contract.get("schema"),
                "release_sha256": contract.get("release_sha256"),
                "model_sha256": contract.get("model_sha256"),
                "execution_authorized": bool(contract.get("execution_authorized", False)),
                "model_version": contract.get("model_version"),
                "generated_at": contract.get("generated_at"), "pairs": mapped}

    def _observe_v22_contract(self, now: float) -> None:
        if not self.v22_observation_gate_path.exists():
            return
        contract = load_runtime_v22_contract(
            self.v22_observation_gate_path,
            now=datetime.fromtimestamp(now, timezone.utc),
            max_age_seconds=self.v21_max_age_seconds,
        )
        observation = self.state.setdefault("v22_observation", {})
        release = str(contract.get("release_sha256", ""))
        if release and observation.get("release_sha256") != release:
            observation.clear()
            observation.update({"release_sha256": release, "started_at": now,
                                "cycles": 0, "source_errors": 0, "integrity_errors": 0})
        observation["last_seen_at"] = now
        observation["cycles"] = int(observation.get("cycles", 0)) + 1
        observation["event_ids"] = {
            pair: contract.get("pairs", {}).get(source, {}).get("event_id")
            for pair, source in V22_PAIR_MAP.items()
        }
        if not contract.get("runtime_gate_healthy"):
            failure = str(contract.get("reason", ""))
            category = "source_errors" if any(
                marker in failure.lower() for marker in ("timeout", "connection", "temporarily")
            ) else "integrity_errors"
            observation[category] = int(observation.get(category, 0)) + 1
            observation["last_error"] = failure

    def _set_effective_gates(
        self, bot_name: str, snapshot: Dict[str, Any], *, buy_enabled: bool,
        sell_enabled: bool, reasons: Dict[str, Any],
    ) -> Dict[str, Any]:
        database = Path(snapshot["database"])
        controller_name, profile = self._controller_profile(database)
        if not controller_name or not profile:
            raise RuntimeError(f"controller config is unavailable for {bot_name}")
        actual_buy = bool(profile.get("macro_buy_enabled", True))
        actual_sell = bool(profile.get("macro_sell_enabled", True))
        if actual_buy == buy_enabled and actual_sell == sell_enabled:
            return {
                "status": "unchanged",
                "macro_buy_enabled": actual_buy,
                "macro_sell_enabled": actual_sell,
                "macro_decision_id": str(profile.get("macro_decision_id", "")),
            }
        digest = hashlib.sha256(json.dumps(
            {"buy": buy_enabled, "sell": sell_enabled, "reasons": reasons},
            sort_keys=True, default=str, separators=(",", ":"),
        ).encode()).hexdigest()[:16]
        profile["macro_buy_enabled"] = buy_enabled
        profile["macro_sell_enabled"] = sell_enabled
        profile["macro_decision_id"] = f"risk-aggregate:{digest}"
        response = self.api.update_controller(bot_name, controller_name, profile)
        return {
            "status": "applied",
            "macro_buy_enabled": buy_enabled,
            "macro_sell_enabled": sell_enabled,
            "macro_decision_id": profile["macro_decision_id"],
            "response": response,
        }

    def _apply_aggregate_gates(
        self, snapshots: Dict[str, Dict[str, Any]], *, risk_actions_enabled: bool,
    ) -> None:
        macro = self._macro_gate()
        v21 = self._v21_gate()
        previous_aggregate = self.state.get("gate_aggregate", {})
        previous_macro = previous_aggregate.get("macro", {})
        previous_bots = previous_aggregate.get("bots", {})
        aggregate = {"macro": macro, "v22": v21, "bots": {}}
        for bot_name, snapshot in snapshots.items():
            bot_state = self.state.get("bots", {}).get(bot_name, {})
            if bot_state.get("tripped"):
                continue
            pair = str(snapshot["pair"])
            technical = v21["pairs"][pair]
            recovery = normalize_state(bot_state.get("recovery"))
            recoverable_blocked = recovery["phase"] != ACTIVE
            buy_enabled = bool(
                macro["buy_enabled"] and technical["buy_enabled"]
                and not recoverable_blocked
            )
            sell_enabled = bool(macro["sell_enabled"] and not recoverable_blocked)
            reasons = {
                "fomc": macro["reason"],
                "v22": technical["reason"],
                "v22_source_pair": technical["source_pair"],
                "recovery_phase": recovery["phase"],
                "recovery_mechanism": recovery.get("mechanism", ""),
            }
            aggregate["bots"][bot_name] = {
                "pair": pair, "buy_enabled": buy_enabled,
                "sell_enabled": sell_enabled, "reasons": reasons,
                "fomc_buy_enabled": bool(macro["buy_enabled"]),
                "fomc_sell_enabled": bool(macro["sell_enabled"]),
                "v22_buy_enabled": bool(technical["buy_enabled"]),
                "v22_event_id": technical.get("event_id"),
            }
            previous_bot = previous_bots.get(bot_name, {})
            macro_changed = (
                previous_macro.get("buy_enabled") != macro.get("buy_enabled")
                or previous_macro.get("sell_enabled") != macro.get("sell_enabled")
                or previous_macro.get("healthy") != macro.get("healthy")
            )
            v22_changed = (
                previous_bot.get("v22_buy_enabled") != technical.get("buy_enabled")
            )
            controller_result: Dict[str, Any] = {"status": "observation_only"}
            controller_error = ""
            if risk_actions_enabled:
                try:
                    controller_result = self._set_effective_gates(
                        bot_name, snapshot, buy_enabled=buy_enabled,
                        sell_enabled=sell_enabled, reasons=reasons,
                    )
                    if controller_result["status"] != "unchanged":
                        self._audit("aggregate_gate_update", bot=bot_name,
                                    desired=aggregate["bots"][bot_name], result=controller_result)
                except Exception as exc:
                    controller_error = repr(exc)
                    controller_result = {"status": "failed"}
                    self._audit("aggregate_gate_update_failed", bot=bot_name,
                                desired=aggregate["bots"][bot_name], error=controller_error)
            common_transition_details = {
                "effective_buy_enabled": buy_enabled,
                "effective_sell_enabled": sell_enabled,
                "recovery_phase": recovery["phase"],
                "execution_applied": bool(risk_actions_enabled and not controller_error),
                "controller_update_status": controller_result.get("status", "unknown"),
                "controller_update_error": controller_error,
            }
            if macro_changed:
                self._audit(
                    "fomc_gate_transition", bot=bot_name, pair=pair,
                    buy_enabled=macro["buy_enabled"], sell_enabled=macro["sell_enabled"],
                    reason=macro["reason"], **common_transition_details,
                    correlation_id="|".join(macro.get("active_lease_ids", []))
                    or f"fomc:{macro['buy_enabled']}:{macro['sell_enabled']}:{macro['reason']}",
                )
            if v22_changed:
                self._audit(
                    "v22_gate_transition", bot=bot_name, pair=pair,
                    buy_enabled=technical["buy_enabled"], sell_enabled=True,
                    reason=technical["reason"], probability=technical.get("probability"),
                    source_pair=technical.get("source_pair"),
                    **common_transition_details,
                    release_sha256=v21.get("release_sha256", ""),
                    model_sha256=v21.get("model_sha256", ""),
                    correlation_id=technical.get("event_id") or "",
                )
        self.state["gate_aggregate"] = aggregate

    @staticmethod
    def _market_telemetry(pair: str) -> Dict[str, float]:
        symbol = pair.replace("-", "")
        book = requests.get(
            f"{BINANCE_API}/api/v3/ticker/bookTicker",
            params={"symbol": symbol},
            timeout=15,
        )
        book.raise_for_status()
        book_value = book.json()
        bid = float(book_value["bidPrice"])
        ask = float(book_value["askPrice"])
        mid = (bid + ask) / 2
        klines = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={"symbol": symbol, "interval": "1m", "limit": 31},
            timeout=15,
        )
        klines.raise_for_status()
        closes = [float(item[4]) for item in klines.json()]
        returns = [
            closes[index] / closes[index - 1] - 1
            for index in range(1, len(closes))
            if closes[index - 1] > 0
        ]
        return {
            "mid_price": mid,
            "spread_bps": 0.0 if mid <= 0 else (ask - bid) / mid * 10_000,
            "volatility_ratio_30m": (
                statistics.pstdev(returns) if len(returns) > 1 else 0.0
            ),
            "data_age_seconds": 0.0,
        }

    def _write_macro_telemetry(
        self, snapshots: Dict[str, Dict[str, Any]]
    ) -> None:
        bot_statuses = []
        for pair, spec in LIVE_PAIRS.items():
            snapshot = snapshots.get(spec.bot_name)
            if snapshot is None:
                continue
            database = Path(snapshot["database"])
            bot_state = self.state["bots"].get(spec.bot_name, {})
            equity = float(Decimal("190") + Decimal(snapshot["pnl_quote"]))
            peak = max(float(bot_state.get("peak_strategy_equity", equity)), equity)
            bot_state["peak_strategy_equity"] = peak
            counts = self._executor_counts(database)
            gates = self._controller_gates(database)
            observation_age = max(
                0.0, time.time() - float(snapshot["observed_at"])
            )
            database_event_age = max(
                0.0, time.time() - float(snapshot["database_event_at"])
            )
            bot_statuses.append(
                {
                    "bot_name": spec.bot_name,
                    "trading_pair": pair,
                    "strategy_equity": equity,
                    "peak_strategy_equity": peak,
                    "strategy_owned_long_base": max(
                        0.0, float(snapshot["net_base"])
                    ),
                    "active_executors": (
                        counts["active_buy_executors"]
                        + counts["active_sell_executors"]
                    ),
                    "open_orders": counts["open_orders"],
                    "fills": int(snapshot["trades"]),
                    # Compatibility field now means collector observation
                    # freshness, not database activity. A quiet five-hour
                    # ladder must not look unhealthy merely because it has no
                    # new fills.
                    "data_age_seconds": observation_age,
                    "observation_age_seconds": observation_age,
                    "database_event_age_seconds": database_event_age,
                    "database_event_at": datetime.fromtimestamp(
                        float(snapshot["database_event_at"]), timezone.utc
                    ).isoformat(),
                    "healthy": observation_age <= 60,
                    "hard_circuit_breaker": bool(bot_state.get("tripped")),
                    "buy_circuit_breaker": bool(
                        self.state.get("roc_buy_guard", {}).get("active", False)
                    ),
                    **counts,
                    **gates,
                }
            )
        payload = {
            "schema_version": 3,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "bot_statuses": bot_statuses,
            "market": {
                pair: self._market_telemetry(pair) for pair in LIVE_PAIRS
            },
        }
        temporary = self.telemetry_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.telemetry_path)

    @staticmethod
    def _emergency_fill_metrics(pair: str, side: str, response: dict) -> Dict[str, str]:
        executed = Decimal(str(response.get("executedQty", "0")))
        quote = Decimal(str(response.get("cummulativeQuoteQty", "0")))
        if executed <= 0:
            raise RuntimeError(f"emergency order reported no executed quantity: {response}")
        base_asset, quote_asset = pair.split("-", 1)
        base_delta = executed if side == "BUY" else -executed
        quote_cashflow = -quote if side == "BUY" else quote
        fee_quote = Decimal("0")
        for fill in response.get("fills", []):
            commission = Decimal(str(fill.get("commission", "0")))
            asset = str(fill.get("commissionAsset", ""))
            fill_price = Decimal(str(fill.get("price", "0")))
            if asset == quote_asset:
                fee_quote += commission
            elif asset == base_asset:
                base_delta -= commission
                fee_quote += commission * fill_price
            # Third-asset fees remain explicit in the audit response and are
            # intentionally not guessed without an independent conversion.
        return {
            "base_delta": str(base_delta),
            "quote_cashflow": str(quote_cashflow),
            "fee_quote": str(fee_quote),
        }

    def _flatten(self, snapshot: Dict[str, Any], bot_name: str = "") -> Dict[str, Any]:
        pair = snapshot["pair"]
        net_base = Decimal(snapshot["net_base"])
        if net_base == 0:
            return {"status": "not_required"}
        side = "SELL" if net_base > 0 else "BUY"
        step_size, minimum_notional = self._lot_filter(pair)
        amount = (abs(net_base) / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
        mark_price = Decimal(snapshot["mark_price"])
        if amount <= 0 or amount * mark_price < minimum_notional:
            return {"status": "dust", "side": side, "amount": str(amount),
                    "notional": str(amount * mark_price)}
        if self.emergency_exchange is None:
            raise RuntimeError("independent Binance emergency client is unavailable")
        response = self.emergency_exchange.market_order(pair, side, amount)
        metrics = self._emergency_fill_metrics(pair, side, response)
        if bot_name:
            adjustment = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "pair": pair,
                "side": side,
                "order_id": str(response.get("orderId", "")),
                "executed_qty": str(response.get("executedQty", "0")),
                "cummulative_quote_qty": str(
                    response.get("cummulativeQuoteQty", "0")
                ),
                **metrics,
            }
            self.state["bots"].setdefault(bot_name, {}).setdefault(
                "emergency_adjustments", []
            ).append(adjustment)
            self._save()
        return {
            "status": "filled",
            "side": side,
            "amount": str(amount),
            "order_id": str(response.get("orderId", "")),
            "executed_qty": str(response.get("executedQty", "0")),
            **metrics,
        }

    @staticmethod
    def _items(payload: Any) -> list[Dict[str, Any]]:
        """Normalize the list wrappers used by different API releases."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("data", "items", "containers", "result", "response"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = Guard._items(value)
                if nested:
                    return nested
        return []

    def _actual_instances(self, bot_name: str) -> list[str]:
        instances = []
        for item in self._items(self.api.active_containers(bot_name)):
            name = str(item.get("name", "")).lstrip("/")
            if name == bot_name or name.startswith(f"{bot_name}-"):
                instances.append(name)
        return sorted(set(instances))

    @staticmethod
    def _mqtt_bot_running(payload: Any, instance_name: str) -> bool:
        node = payload.get("data", payload) if isinstance(payload, dict) else {}
        if not isinstance(node, dict):
            return False
        value = node.get(instance_name)
        if not isinstance(value, dict):
            return False
        return str(value.get("status", "")).lower() in {
            "running", "started", "starting"
        }

    @staticmethod
    def _mqtt_stop_confirmed(
        payload: Any, instance_name: str, requested_at: float
    ) -> bool:
        node = payload.get("data", payload) if isinstance(payload, dict) else {}
        value = node.get(instance_name) if isinstance(node, dict) else None
        if not isinstance(value, dict):
            return False
        if str(value.get("status", "")).lower() == "stopped":
            return True
        # The process intentionally keeps MQTT alive after stopping the
        # strategy. Its command acknowledgement is published in general_logs.
        return any(
            float(record.get("timestamp", 0)) >= requested_at - 1
            and "Hummingbot stopped." in str(record.get("msg", ""))
            for record in value.get("general_logs", [])
            if isinstance(record, dict)
        )

    def _instance_database(self, instance_name: str) -> Optional[Path]:
        exact = (
            self.bots_path / "instances" / instance_name / "data"
            / f"{instance_name}.sqlite"
        )
        return exact if exact.exists() else None

    def _wait_for_instance_terminal(
        self, instance_name: str, database: Optional[Path], requested_at: float
    ) -> Dict[str, Any]:
        timeout = max(10, int(os.getenv("DCA_LIVE_STOP_CONFIRM_SECONDS", "30")))
        deadline = time.monotonic() + timeout
        last: Dict[str, Any] = {}
        while time.monotonic() <= deadline:
            mqtt_stop_confirmed = self._mqtt_stop_confirmed(
                self.api.status(), instance_name, requested_at
            )
            counts = self._executor_counts(database) if database is not None else None
            database_terminal = counts is not None and all(
                counts[key] == 0
                for key in (
                    "active_buy_executors",
                    "trading_buy_executors",
                    "active_sell_executors",
                    "trading_sell_executors",
                    "open_orders",
                )
            )
            last = {
                "instance": instance_name,
                "mqtt_stop_confirmed": mqtt_stop_confirmed,
                "database": str(database) if database is not None else None,
                "database_terminal": database_terminal,
                "executor_counts": counts,
            }
            if mqtt_stop_confirmed and database_terminal:
                return last
            time.sleep(1)
        raise RuntimeError(
            f"bot terminal confirmation timed out after {timeout}s: {last}"
        )

    def _active_order_ids(self, pair: str) -> list[str]:
        identifiers = []
        for item in self._items(self.api.active_orders(pair)):
            order_pair = str(item.get("trading_pair", pair)).upper()
            if order_pair and order_pair != pair.upper():
                continue
            identifier = item.get("client_order_id") or item.get("order_id") or item.get("id")
            if identifier:
                identifiers.append(str(identifier))
        return sorted(set(identifiers))

    @staticmethod
    def _stop_succeeded(response: Any) -> bool:
        if not isinstance(response, dict):
            return False
        if response.get("success") is True:
            return True
        return Guard._stop_succeeded(response.get("response"))

    def _secure_stop(self, bot_name: str, pair: str) -> Dict[str, Any]:
        """Stop a bot with independent exchange and Docker fallbacks."""
        if self.emergency_exchange is None or self.emergency_docker is None:
            raise RuntimeError("independent emergency clients are unavailable")
        discovery_errors = []
        instances = set(self.emergency_docker.matching_containers(bot_name))
        try:
            instances.update(self._actual_instances(bot_name))
        except Exception as exc:
            discovery_errors.append(repr(exc))
        result: Dict[str, Any] = {
            "logical_bot_name": bot_name,
            "resolved_instances": sorted(instances),
            "discovery_errors": discovery_errors,
            "graceful_stops": [],
            "terminal_confirmations": [],
            "cancelled_order_ids": [],
            "container_stops": [],
            "emergency_path_used": False,
        }
        candidates = [bot_name, *result["resolved_instances"]]
        successful_instances = set()
        stop_requested_at: Dict[str, float] = {}
        for candidate in dict.fromkeys(candidates):
            try:
                stop_requested_at[candidate] = time.time()
                response = self.api.stop_bot(candidate)
                succeeded = self._stop_succeeded(response)
                result["graceful_stops"].append({
                    "name": candidate,
                    "succeeded": succeeded,
                    "response": response,
                })
                if succeeded and candidate in result["resolved_instances"]:
                    successful_instances.add(candidate)
            except Exception as exc:
                result["graceful_stops"].append({
                    "name": candidate,
                    "succeeded": False,
                    "error": repr(exc),
                })

        for instance in result["resolved_instances"]:
            if instance not in successful_instances:
                continue
            try:
                result["terminal_confirmations"].append(
                    self._wait_for_instance_terminal(
                        instance,
                        self._instance_database(instance),
                        stop_requested_at[instance],
                    )
                )
            except Exception as exc:
                result["terminal_confirmations"].append(
                    {"instance": instance, "confirmed": False, "error": repr(exc)}
                )

        unresolved = sorted(set(result["resolved_instances"]) - successful_instances)
        terminal_failed = any(
            item.get("confirmed") is False or item.get("database_terminal") is False
            for item in result["terminal_confirmations"]
        )
        result["emergency_path_used"] = bool(unresolved or terminal_failed)

        # Binance, not the orchestration API, is authoritative. Cancel all
        # orders for this isolated pair before terminating the process. This
        # remains available when MQTT and Hummingbot API are unavailable.
        exchange_orders = self.emergency_exchange.open_orders(pair)
        result["cancelled_order_ids"] = [
            str(item.get("clientOrderId") or item.get("orderId"))
            for item in exchange_orders
        ]
        if exchange_orders:
            self.emergency_exchange.cancel_all_orders(pair)

        # Stop through the Docker Engine socket, independently of MQTT and the
        # Hummingbot orchestration service. Stopping after exchange cancellation
        # prevents the bot from recreating orders during the final check.
        for instance in result["resolved_instances"]:
            response = self.emergency_docker.stop(instance)
            result["container_stops"].append(
                {"name": instance, "channel": "docker_socket", "response": response}
            )

        remaining_orders = self.emergency_exchange.open_orders(pair)
        for _ in range(5):
            if not remaining_orders:
                break
            time.sleep(1)
            self.emergency_exchange.cancel_all_orders(pair)
            remaining_orders = self.emergency_exchange.open_orders(pair)
        if remaining_orders:
            raise RuntimeError(
                f"active exchange orders remain for {bot_name}: {remaining_orders}"
            )

        live_instances = self.emergency_docker.matching_containers(bot_name)
        for _ in range(5):
            if not live_instances:
                break
            time.sleep(1)
            live_instances = self.emergency_docker.matching_containers(bot_name)
        if live_instances:
            raise RuntimeError(
                f"Docker instances remain for {bot_name}: {live_instances}"
            )
        result["verified_no_active_orders"] = True
        result["verified_mqtt_and_database_terminal"] = not unresolved and not terminal_failed
        result["verified_no_live_instances"] = True
        return result

    def _managed_base_target(self, pair: str) -> Decimal:
        """Return the deployment-audited base allocation; never use account-wide balance."""
        payload = json.loads(self.managed_inventory_path.read_text(encoding="utf-8"))
        value = payload.get("pairs", {}).get(pair, {}).get("managed_base")
        if value is None:
            raise RuntimeError(f"managed base ownership is not recorded for {pair}")
        amount = Decimal(str(value))
        if amount <= 0:
            raise RuntimeError(f"managed base ownership is invalid for {pair}")
        return amount

    def _owned_base(self, bot_name: str, snapshot: Dict[str, Any]) -> Decimal:
        bot = self.state["bots"].setdefault(bot_name, {})
        target = Decimal(str(bot.get("managed_base_target", "0")))
        if target <= 0:
            target = self._managed_base_target(str(snapshot["pair"]))
            bot["managed_base_target"] = str(target)
        return max(target + Decimal(str(snapshot["net_base"])), Decimal("0"))

    def _trigger_recoverable(
        self, bot_name: str, snapshot: Dict[str, Any], *, mechanism: str,
        scope: str, trigger_value: Any, reason: str,
    ) -> None:
        bot = self.state["bots"].setdefault(bot_name, {})
        current = normalize_state(bot.get("recovery"))
        if current["phase"] != ACTIVE:
            return
        bot["recovery"] = trigger_state(
            mechanism=mechanism, scope=scope, now=time.time(),
            trigger_value=trigger_value, signal_price=snapshot["mark_price"], reason=reason,
        )
        self._audit("recoverable_breaker_triggered", bot=bot_name,
                    pair=snapshot["pair"], recovery=bot["recovery"])
        self._save()

    def _latch_integrity_failure(self, bot_name: str, snapshot: Dict[str, Any], reason: str) -> None:
        bot = self.state["bots"].setdefault(bot_name, {})
        if normalize_state(bot.get("recovery"))["phase"] in {EXITING, LATCHED}:
            return
        bot["recovery"] = trigger_state(
            mechanism="infrastructure_integrity_breaker", scope="infrastructure",
            now=time.time(), trigger_value=reason, signal_price=snapshot["mark_price"],
            reason=reason, latch_after_exit=True,
        )
        self._audit("integrity_failure_exit_then_latch", bot=bot_name,
                    pair=snapshot["pair"], reason=reason)
        # The integrity latch is a safety boundary.  Persist it immediately so
        # a process crash cannot reopen trading on restart.
        self._save()

    def _record_emergency_fill(self, bot_name: str, pair: str, side: str,
                               response: Dict[str, Any]) -> Dict[str, str]:
        metrics = self._emergency_fill_metrics(pair, side, response)
        self.state["bots"].setdefault(bot_name, {}).setdefault(
            "emergency_adjustments", []
        ).append({
            "recorded_at": datetime.now(timezone.utc).isoformat(), "pair": pair,
            "side": side, "order_id": str(response.get("orderId", "")),
            "executed_qty": str(response.get("executedQty", "0")),
            "cummulative_quote_qty": str(response.get("cummulativeQuoteQty", "0")),
            **metrics,
        })
        return metrics

    def _process_recoverable(
        self, bot_name: str, snapshot: Dict[str, Any], *,
        macro: Dict[str, Any], technical: Dict[str, Any], now: float,
        portfolio_all_gates: bool = True,
    ) -> None:
        bot = self.state["bots"][bot_name]
        state = normalize_state(bot.get("recovery"))
        if state["phase"] in {ACTIVE, LATCHED}:
            return
        pair = str(snapshot["pair"])
        step, minimum_notional = self._lot_filter(pair)
        mark = Decimal(snapshot["mark_price"])
        if state["phase"] == EXITING:
            # Give the in-process executor one short window to complete its
            # market close. The independent channel takes over only after the
            # persisted deadline, preventing a stale double-close race.
            if now - float(state.get("triggered_at") or now) < EMERGENCY_ESCALATION_SECONDS:
                bot["recovery"] = state
                return
            self.emergency_exchange.cancel_all_orders(pair)
            try:
                refreshed = self._snapshot(bot_name, pair)
                if refreshed is not None:
                    snapshot = refreshed
                    mark = Decimal(snapshot["mark_price"])
            except Exception as exc:
                self._audit("exit_snapshot_refresh_failed", bot=bot_name,
                            pair=pair, error=repr(exc))
            owned = self._owned_base(bot_name, snapshot)
            amount = (owned / step).to_integral_value(rounding=ROUND_DOWN) * step
            state["remaining_base"] = {pair: str(owned)}
            state["exit_attempts"] = int(state.get("exit_attempts", 0)) + 1
            if amount > 0 and amount * mark >= minimum_notional:
                state["first_exit_order_at"] = state.get("first_exit_order_at") or now
                response = self.emergency_exchange.market_order(pair, "SELL", amount)
                metrics = self._record_emergency_fill(bot_name, pair, "SELL", response)
                state["last_exit_fill"] = {"response": response, "metrics": metrics}
                # A FILLED response is authoritative; subtract only that fill
                # from the ownership-capped amount rather than reading account balance.
                owned = max(owned - Decimal(str(response["executedQty"])), Decimal("0"))
                amount = (owned / step).to_integral_value(rounding=ROUND_DOWN) * step
            if amount <= 0 or amount * mark < minimum_notional:
                state = mark_exit_complete(
                    state, now=now, remaining_base={pair: owned},
                    execution={"target": "quote_only", "attempts": state["exit_attempts"],
                               "last_fill": state.get("last_exit_fill", {})},
                )
                self._audit("recoverable_exit_complete", bot=bot_name, pair=pair, recovery=state)
            elif now - float(state.get("triggered_at") or now) >= EXIT_CRITICAL_SECONDS:
                if not state.get("critical_alerted"):
                    state["critical_alerted"] = True
                    self._notify(f"DCA CRITICAL EXIT DELAY: {bot_name} owns {owned} base")
                    self._audit("recoverable_exit_critical_delay", bot=bot_name,
                                pair=pair, remaining_base=str(owned))
            bot["recovery"] = state
            return

        counts = self._executor_counts(Path(snapshot["database"]))
        no_runtime_risk = not self.emergency_exchange.open_orders(pair) and all(
            counts[key] == 0 for key in counts
        )
        underlying_healthy = bool(macro["healthy"] and technical.get("buy_enabled"))
        gates_allow = bool(
            self.auto_reentry_enabled and macro["buy_enabled"] and macro["sell_enabled"]
            and technical.get("buy_enabled") and technical.get("execution_authorized")
        )
        if state.get("scope") == "portfolio":
            gates_allow = gates_allow and portfolio_all_gates
        previous_phase = state["phase"]
        state = advance_recovery(
            state, now=now, healthy=underlying_healthy and no_runtime_risk,
            gates_allow_reentry=gates_allow,
        )
        if state["phase"] == REENTRY and previous_phase != REENTRY:
            self._audit("recoverable_reentry_ready", bot=bot_name, pair=pair,
                        recovery=state)
        if state.get("reentry_allowed"):
            target_quote = side_budget()
            amount = ((target_quote / mark) / step).to_integral_value(rounding=ROUND_DOWN) * step
            if amount <= 0 or amount * mark < minimum_notional:
                raise RuntimeError(f"DCA reentry amount is below exchange minimum for {pair}")
            response = self.emergency_exchange.market_order(pair, "BUY", amount)
            metrics = self._record_emergency_fill(bot_name, pair, "BUY", response)
            baseline = {"base": response["executedQty"], "target_quote": target_quote,
                        "mark_price": mark}
            if state.get("scope") == "portfolio":
                state["reentry_filled"] = True
                state["reentry_baseline"] = {key: str(value) for key, value in baseline.items()}
                state["reentry_allowed"] = False
            else:
                state = mark_reentry_complete(state, now=now, baseline=baseline)
            bot["peak_equity"] = str(STRATEGY_BUDGET_QUOTE)
            bot["pnl_offset_pending"] = True
            self._audit("recoverable_reentry_complete", bot=bot_name, pair=pair,
                        response=response, metrics=metrics, recovery=state)
        bot["recovery"] = state

    def _trip(self, bot_name: str, reason: str, snapshot: Optional[Dict[str, Any]]) -> None:
        bot_state = self.state["bots"].setdefault(bot_name, {})
        if bot_state.get("action_complete"):
            return
        observed = time.time()
        signal_price = snapshot.get("mark_price", "") if snapshot else ""
        bot_state.update({
            "tripped": True, "trip_reason": reason, "tripped_at": observed,
            "recovery": trigger_state(
                mechanism="infrastructure_integrity_breaker", scope="infrastructure",
                now=observed, trigger_value=reason, signal_price=signal_price,
                reason=reason, latched=True,
            ),
        })
        self._save()
        try:
            pair = snapshot["pair"] if snapshot is not None else next(
                pair for pair, spec in LIVE_PAIRS.items() if spec.bot_name == bot_name
            )
            stop_response = self._secure_stop(bot_name, pair)
            # Reconcile after bot-side closes, exchange cancellations and
            # process termination. Never flatten from the pre-stop snapshot:
            # doing so can double-close and create the opposite exposure.
            post_stop_snapshot = None
            for _ in range(5):
                post_stop_snapshot = self._snapshot(bot_name, pair)
                if post_stop_snapshot is not None:
                    break
                time.sleep(1)
            if post_stop_snapshot is None:
                raise RuntimeError("post-stop strategy inventory could not be read")
            flatten_response = self._flatten(post_stop_snapshot, bot_name)
            reconciled_snapshot = self._snapshot(bot_name, pair)
            bot_state.update({
                "action_complete": True,
                "stop_response": stop_response,
                "flatten_response": flatten_response,
                "post_stop_snapshot": post_stop_snapshot,
                "reconciled_snapshot": reconciled_snapshot,
            })
            self._audit("circuit_breaker_complete", bot=bot_name, reason=reason,
                        snapshot=snapshot, post_stop_snapshot=post_stop_snapshot,
                        reconciled_snapshot=reconciled_snapshot,
                        stop=stop_response, flatten=flatten_response)
            self._notify(
                f"DCA LIVE CIRCUIT BREAKER: {bot_name}\n"
                f"Reason: {reason}\nBot stopped, orders cancelled, inventory restore submitted."
            )
        except Exception as exc:
            bot_state["action_complete"] = False
            bot_state["last_action_error"] = repr(exc)
            self._audit("circuit_breaker_action_failed", bot=bot_name, reason=reason, error=repr(exc))
            self._notify(f"DCA LIVE CIRCUIT BREAKER ACTION FAILED: {bot_name}\n{exc}")
        finally:
            self._save()

    def cycle(self, *, risk_actions_enabled: bool = True) -> None:
        status_text = json.dumps(self.api.status(), ensure_ascii=True)
        snapshots: Dict[str, Dict[str, Any]] = {}
        now = time.time()
        self._observe_v22_contract(now)
        mechanisms = getattr(self, "mechanisms", {})
        for pair, spec in LIVE_PAIRS.items():
            if spec.bot_name not in status_text:
                continue
            bot_state = self.state["bots"].setdefault(spec.bot_name, {
                "pair": pair,
                "started_at": now,
                "tripped": False,
                "action_complete": False,
                "recovery": active_state(),
            })
            bot_state["recovery"] = normalize_state(bot_state.get("recovery"))
            snapshot = self._snapshot(spec.bot_name, pair)
            if snapshot is None:
                continue
            snapshots[spec.bot_name] = snapshot
            raw_pnl = Decimal(snapshot["pnl_quote"])
            if bot_state.pop("pnl_offset_pending", False):
                bot_state["pnl_offset_quote"] = str(-raw_pnl)
            pnl = raw_pnl + Decimal(str(bot_state.get("pnl_offset_quote", "0")))
            snapshot["raw_pnl_quote"] = str(raw_pnl)
            snapshot["pnl_quote"] = str(pnl)
            equity = STRATEGY_BUDGET_QUOTE + pnl
            peak = max(
                STRATEGY_BUDGET_QUOTE,
                Decimal(str(bot_state.get("peak_equity", STRATEGY_BUDGET_QUOTE))),
                equity,
            )
            drawdown = (peak - equity) / peak if peak > 0 else Decimal("0")
            snapshot.update({
                "equity": str(equity), "peak_equity": str(peak),
                "drawdown_pct": str(drawdown),
                "executor_counts": self._executor_counts(Path(snapshot["database"])),
            })
            bot_state["peak_equity"] = str(peak)
            bot_state["latest"] = snapshot
            if risk_actions_enabled and bot_state.get("tripped"):
                if not bot_state.get("action_complete"):
                    self._trip(spec.bot_name, bot_state.get("trip_reason", "retry"), snapshot)
                continue
            latest_stop = float(snapshot.get("latest_stop_loss_at", 0))
            if "last_position_stop_seen" not in bot_state:
                bot_state["last_position_stop_seen"] = latest_stop
            last_seen_stop = float(bot_state.get("last_position_stop_seen", 0))
            if (
                risk_actions_enabled
                and mechanisms.get("position_protection", True)
                and latest_stop > last_seen_stop
                and bot_state["recovery"]["phase"] == ACTIVE
            ):
                bot_state["last_position_stop_seen"] = latest_stop
                self._trigger_recoverable(
                    spec.bot_name, snapshot, mechanism="position_protection",
                    scope="position", trigger_value="5%", reason="executor_stop_loss",
                )
            strategy_loss_enabled = mechanisms.get("strategy_loss_breaker", True)
            strategy_drawdown_enabled = mechanisms.get("strategy_drawdown_breaker", True)
            if risk_actions_enabled and strategy_loss_enabled and pnl <= -SINGLE_BOT_LOSS_LIMIT:
                self._trigger_recoverable(
                    spec.bot_name, snapshot, mechanism="strategy_loss_breaker",
                    scope="strategy", trigger_value=pnl,
                    reason=f"single bot PnL {pnl} <= -{SINGLE_BOT_LOSS_LIMIT} USDT",
                )
            elif (
                risk_actions_enabled and strategy_drawdown_enabled
                and drawdown >= getattr(self, "strategy_drawdown_limit", Decimal("0.08"))
            ):
                limit = getattr(self, "strategy_drawdown_limit", Decimal("0.08"))
                self._trigger_recoverable(
                    spec.bot_name, snapshot, mechanism="strategy_drawdown_breaker",
                    scope="strategy", trigger_value=drawdown,
                    reason=(f"single bot peak drawdown {drawdown:.2%} >= "
                            f"{limit:.2%}"),
                )

        combined_pnl = sum((Decimal(item["pnl_quote"]) for item in snapshots.values()), Decimal("0"))
        self.state["combined_pnl_quote"] = str(combined_pnl)
        initial_equity = STRATEGY_BUDGET_QUOTE * Decimal(len(LIVE_PAIRS))
        combined_equity = initial_equity + combined_pnl
        combined_peak = max(
            initial_equity,
            Decimal(str(self.state.get("combined_peak_equity", initial_equity))),
            combined_equity,
        )
        combined_drawdown = (
            (combined_peak - combined_equity) / combined_peak
            if combined_peak > 0 else Decimal("0")
        )
        self.state["combined_equity"] = str(combined_equity)
        self.state["combined_peak_equity"] = str(combined_peak)
        self.state["combined_drawdown_pct"] = str(combined_drawdown)
        if (
            risk_actions_enabled
            and len(snapshots) == len(LIVE_PAIRS)
            and mechanisms.get("portfolio_loss_breaker", True)
            and combined_pnl <= -COMBINED_LOSS_LIMIT
        ):
            for bot_name, snapshot in snapshots.items():
                self._trigger_recoverable(
                    bot_name, snapshot, mechanism="portfolio_loss_breaker",
                    scope="portfolio", trigger_value=combined_pnl,
                    reason=f"combined PnL {combined_pnl} <= -{COMBINED_LOSS_LIMIT} USDT",
                )
        elif (
            risk_actions_enabled
            and len(snapshots) == len(LIVE_PAIRS)
            and mechanisms.get("portfolio_drawdown_breaker", True)
            and combined_drawdown >= getattr(self, "portfolio_drawdown_limit", Decimal("0.08"))
        ):
            limit = getattr(self, "portfolio_drawdown_limit", Decimal("0.08"))
            for bot_name, snapshot in snapshots.items():
                self._trigger_recoverable(
                    bot_name, snapshot, mechanism="portfolio_drawdown_breaker",
                    scope="portfolio", trigger_value=combined_drawdown,
                    reason=(f"combined peak drawdown {combined_drawdown:.2%} >= "
                            f"{limit:.2%}"),
                )
        macro_contract = self._macro_gate()
        v21_contract = self._v21_gate()
        if risk_actions_enabled:
            for bot_name, snapshot in snapshots.items():
                if not macro_contract["healthy"]:
                    self._latch_integrity_failure(bot_name, snapshot, str(macro_contract["reason"]))
                elif not v21_contract["healthy"]:
                    self._latch_integrity_failure(bot_name, snapshot, str(v21_contract["reason"]))
                else:
                    technical = v21_contract["pairs"][snapshot["pair"]]
                    if (
                        v21_contract.get("schema") == V22_CONTRACT_SCHEMA
                        and technical.get("force_exit")
                    ):
                        self._trigger_recoverable(
                            bot_name, snapshot, mechanism="v22_weekly_buy_gate",
                            scope="technical", trigger_value=technical.get("probability"),
                            reason=str(technical.get("reason", "v22_risk_off")),
                        )
        self._apply_aggregate_gates(
            snapshots, risk_actions_enabled=risk_actions_enabled
        )
        if risk_actions_enabled:
            macro = macro_contract
            v21 = v21_contract
            portfolio_all_gates = bool(
                macro["healthy"] and macro["buy_enabled"] and macro["sell_enabled"]
                and all(item.get("buy_enabled") for item in v21["pairs"].values())
            )
            for bot_name, snapshot in snapshots.items():
                self._process_recoverable(
                    bot_name, snapshot, macro=macro,
                    technical=v21["pairs"][snapshot["pair"]], now=now,
                    portfolio_all_gates=portfolio_all_gates,
                )
            portfolio_states = [
                normalize_state(self.state["bots"][name].get("recovery"))
                for name in snapshots
            ]
            if (
                len(portfolio_states) == len(LIVE_PAIRS)
                and all(state.get("scope") == "portfolio" for state in portfolio_states)
                and all(state.get("reentry_filled") for state in portfolio_states)
            ):
                for bot_name, snapshot in snapshots.items():
                    state = normalize_state(self.state["bots"][bot_name]["recovery"])
                    committed = mark_reentry_complete(
                        state, now=now, baseline=state["reentry_baseline"],
                    )
                    self.state["bots"][bot_name]["recovery"] = committed
                    self._audit(
                        "portfolio_reentry_committed", bot=bot_name,
                        pair=snapshot["pair"], recovery=committed,
                    )
        self.state["last_success_at"] = now
        self.state.pop("last_monitor_error", None)
        self.state.pop("first_failure_at", None)
        self._write_macro_telemetry(snapshots)
        self._save()

    def fail_closed(self, error: Exception) -> None:
        now = time.time()
        self.state["last_monitor_error"] = repr(error)
        self.state.setdefault("first_failure_at", now)
        self._save()
        if now - float(self.state["first_failure_at"]) < self.fail_closed_seconds:
            return
        for spec in LIVE_PAIRS.values():
            bot_state = self.state["bots"].get(spec.bot_name)
            if bot_state and not bot_state.get("tripped"):
                self._trip(spec.bot_name, f"monitor unavailable for {self.fail_closed_seconds}s",
                           bot_state.get("latest"))

    def run(self, *, observe_only: bool = False) -> None:
        LOG.info(
            "DCA live guard %s for %s",
            "observing with risk actions disabled" if observe_only else "armed",
            ", ".join(spec.bot_name for spec in LIVE_PAIRS.values()),
        )
        while True:
            try:
                self.cycle(risk_actions_enabled=not observe_only)
            except Exception as exc:
                LOG.error("Guard cycle failed: %s", exc)
                if not observe_only:
                    self.fail_closed(exc)
            time.sleep(self.interval)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--emergency-stop-all" in sys.argv[1:]:
        guard = Guard()
        results = {
            pair: guard._secure_stop(spec.bot_name, pair)
            for pair, spec in LIVE_PAIRS.items()
        }
        print(json.dumps({"status": "stopped", "results": results}, default=str))
        return 0
    if "--check-emergency" in sys.argv[1:]:
        guard = Guard()
        open_order_counts = {
            pair: len(guard.emergency_exchange.open_orders(pair))
            for pair in LIVE_PAIRS
        }
        live_containers = {
            spec.bot_name: guard.emergency_docker.matching_containers(spec.bot_name)
            for spec in LIVE_PAIRS.values()
        }
        print(
            json.dumps(
                {
                    "emergency_exchange": guard.emergency_exchange is not None,
                    "emergency_docker": guard.emergency_docker is not None,
                    "pairs": sorted(LIVE_PAIRS),
                    "open_order_counts": open_order_counts,
                    "live_containers": live_containers,
                    "status": "ready",
                },
                sort_keys=True,
            )
        )
        return 0
    if os.getenv("DCA_LIVE_TRADING_ENABLED", "false").lower() != "true":
        LOG.warning(
            "DCA live trading is DISABLED. Guard emits telemetry only; "
            "no live action will be taken."
        )
        Guard().run(observe_only=True)
    Guard().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
