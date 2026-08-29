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
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlencode

import requests
import yaml

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
try:
    from account_inventory import (
        UnifiedInventoryLedger, api_key_fingerprint, canonical_sha256,
        liquidation_identity, ownership_from_documents,
    )
    from emergency_execution import execute_market_liquidation, verify_market_liquidation
    from runtime_endpoints import (
        OFFICIAL_BINANCE_API, binance_api_base, guarded_endpoint, scenario_mode,
    )
except ModuleNotFoundError:
    from live_guard.account_inventory import (
        UnifiedInventoryLedger, api_key_fingerprint, canonical_sha256,
        liquidation_identity, ownership_from_documents,
    )
    from live_guard.emergency_execution import (
        execute_market_liquidation, verify_market_liquidation,
    )
    from live_guard.runtime_endpoints import (
        OFFICIAL_BINANCE_API, binance_api_base, guarded_endpoint, scenario_mode,
    )
from risk_recovery import (
    ACTIVE, COOLDOWN, EXITING, LATCHED, REENTRY,
    EMERGENCY_ESCALATION_SECONDS, EXIT_CRITICAL_SECONDS,
    POSITION_COOLDOWN_SECONDS, PORTFOLIO_COOLDOWN_SECONDS,
    REQUIRED_HEALTHY_CYCLES, STRATEGY_COOLDOWN_SECONDS,
    advance_integrity_failure, advance_recovery, active_state,
    classify_integrity_failure,
    mark_exit_complete, mark_reentry_complete, normalize_state, trigger_state,
)
try:
    from telegram_notifications import (
        RuntimeErrorChannel, append_event, build_event, runtime_error_lines,
    )
except ModuleNotFoundError:
    from live_guard.telegram_notifications import (
        RuntimeErrorChannel, append_event, build_event, runtime_error_lines,
    )


LOG = logging.getLogger("dca-live-guard")
BINANCE_API = OFFICIAL_BINANCE_API
V21_PAIR_MAP = {"BTC-USDT": "BTC-FDUSD", "ETH-USDT": "ETH-FDUSD"}
V22_PAIR_MAP = V21_PAIR_MAP
STOP_LOSS_EVENT_CURSOR_SCHEMA = "dca-stop-loss-event-v3"


try:
    from get_only_read_client import GetOnlyReadClient, read_retry_session
except ModuleNotFoundError:
    from scripts.get_only_read_client import GetOnlyReadClient, read_retry_session


def _read_retry_session() -> requests.Session:
    return read_retry_session(retry_total=2)


def _env_enabled(name: str, default: bool = True) -> bool:
    return os.getenv(name, "true" if default else "false").lower() == "true"


class BinanceEmergencyClient:
    """Minimal signed Binance client independent of Hummingbot and MQTT."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = BINANCE_API):
        if not api_key or not api_secret:
            raise ValueError("Binance emergency credentials are incomplete")
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = guarded_endpoint(
            base_url, official=BINANCE_API, purpose="Binance signed API"
        )
        if scenario_mode() and not api_key.startswith("scenario-"):
            raise RuntimeError("scenario mode refuses non-scenario Binance credentials")
        self.session = _read_retry_session()
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
            # Binance can return -2011 when DELETE /openOrders races with the
            # final order disappearing (or when the symbol is already clear).
            # Cancellation is an idempotent safety action, so that terminal
            # state is success rather than a monitor failure.  Keep the
            # exception behavior for every other endpoint and error code.
            try:
                error_payload = response.json()
            except (TypeError, ValueError):
                error_payload = {}
            if (
                method.upper() == "DELETE"
                and path == "/api/v3/openOrders"
                and error_payload.get("code") == -2011
            ):
                return {
                    "status": "already_clear",
                    "code": -2011,
                    "message": str(error_payload.get("msg", "Unknown order sent.")),
                }
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

    def spot_bnb_burn_enabled(self) -> bool:
        """Return Binance's account-level Spot BNB fee-discount setting.

        The live Grid/DCA accounting is denominated in FDUSD/USDT.  Allowing
        Binance to deduct fees in BNB makes a completed fill depend on an
        unrelated rate oracle and can cause the quote fee to be persisted as
        zero while that oracle is warming up.  Treat a missing or malformed
        response as an integrity failure instead of guessing.
        """
        value = self._signed("GET", "/sapi/v1/bnbBurn")
        if not isinstance(value, dict) or not isinstance(value.get("spotBNBBurn"), bool):
            raise RuntimeError(f"Invalid Binance BNB burn status response: {value!r}")
        return value["spotBNBBurn"]

    def set_spot_bnb_burn(self, enabled: bool) -> bool:
        """Set and verify the account-level Spot BNB fee-discount setting."""
        value = self._signed(
            "POST", "/sapi/v1/bnbBurn",
            {"spotBNBBurn": "true" if enabled else "false"},
        )
        if not isinstance(value, dict) or value.get("spotBNBBurn") is not enabled:
            raise RuntimeError(f"Binance did not confirm the requested BNB burn setting: {value!r}")
        confirmed = self.spot_bnb_burn_enabled()
        if confirmed is not enabled:
            raise RuntimeError(
                "Binance BNB burn setting verification disagrees with the update response"
            )
        return confirmed

    def verify_ready(self, pairs: list[str]) -> Dict[str, Any]:
        self.sync_time()
        account = self._signed("GET", "/api/v3/account")
        if not isinstance(account, dict) or account.get("canTrade") is not True:
            raise RuntimeError("Binance emergency key does not have trading permission")
        if self.spot_bnb_burn_enabled():
            raise RuntimeError(
                "Binance Spot BNB fee deduction must be disabled (spotBNBBurn=true)"
            )
        # Binance's account-level canWithdraw flag does not expose the API
        # key's withdrawal permission, so it must not be used to reject an
        # otherwise valid key. Withdrawal permission and IP restrictions are
        # enforced in Binance API Management and audited during deployment.
        for pair in pairs:
            self.open_orders(pair)
        return {
            "spot_trading": True,
            "spot_bnb_burn_disabled": True,
        }

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

    def order_by_client_id(self, pair: str, client_order_id: str) -> Optional[dict]:
        try:
            value = self._signed(
                "GET", "/api/v3/order",
                {"symbol": self.symbol(pair), "origClientOrderId": client_order_id},
            )
        except RuntimeError as exc:
            if "-2013" in str(exc):
                return None
            raise
        if not isinstance(value, dict):
            return None
        if Decimal(str(value.get("executedQty", "0"))) > 0 and not value.get("fills"):
            value = dict(value)
            value["fills"] = self.order_trades(pair, str(value.get("orderId", "")))
        return value

    def order_trades(self, pair: str, order_id: str) -> list[dict]:
        if not order_id:
            return []
        value = self._signed(
            "GET", "/api/v3/myTrades",
            {"symbol": self.symbol(pair), "orderId": order_id},
        )
        if not isinstance(value, list):
            return []
        return [{
            "price": str(row.get("price", "0")),
            "qty": str(row.get("qty", "0")),
            "commission": str(row.get("commission", "0")),
            "commissionAsset": str(row.get("commissionAsset", "")),
        } for row in value if isinstance(row, dict)]

    def market_order(
        self, pair: str, side: str, amount: Decimal,
        client_order_id: str = "",
    ) -> dict:
        parameters = {
            "symbol": self.symbol(pair),
            "side": side,
            "type": "MARKET",
            "quantity": format(amount, "f"),
            "newOrderRespType": "FULL",
        }
        if client_order_id:
            parameters["newClientOrderId"] = client_order_id
        value = self._signed(
            "POST", "/api/v3/order", parameters,
        )
        if not isinstance(value, dict) or not value.get("status"):
            raise RuntimeError(f"Binance emergency market order response is invalid: {value}")
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

    def logs_since(self, container_name: str, since: float, *, tail: int = 200) -> list[str]:
        query = urlencode({
            "stdout": "1", "stderr": "1", "timestamps": "1",
            "since": str(max(0, int(float(since)))),
            "tail": str(int(tail)),
        })
        connection = _UnixHTTPConnection(self.socket_path)
        try:
            connection.request(
                "GET", f"/containers/{quote(container_name, safe='')}/logs?{query}"
            )
            response = connection.getresponse()
            payload = response.read()
            if response.status >= 400:
                raise RuntimeError(
                    f"Docker logs read failed ({response.status}): "
                    f"{payload[:200].decode(errors='replace')}"
                )
        finally:
            connection.close()
        chunks: list[bytes] = []
        offset = 0
        while offset + 8 <= len(payload) and payload[offset] in {0, 1, 2}:
            size = int.from_bytes(payload[offset + 4:offset + 8], "big")
            end = offset + 8 + size
            if end > len(payload):
                break
            chunks.append(payload[offset + 8:end])
            offset = end
        raw = b"".join(chunks) if chunks else payload
        return [line for line in raw.decode(errors="replace").splitlines() if line.strip()]

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
        self.session = _read_retry_session()
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
        self.runtime_errors = RuntimeErrorChannel(
            event_path=self.notification_path,
            state_path=self.state_dir / "runtime_error_state.json",
            source="dca-live-guard", strategy="dca",
            bot="dca-live-btcusdt-200,dca-live-ethusdt-200",
            pair="BTC-USDT,ETH-USDT",
        )
        self.runtime_log_cursor_path = self.state_dir / "runtime_log_cursor.json"
        try:
            self.runtime_log_cursors = json.loads(
                self.runtime_log_cursor_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            self.runtime_log_cursors = {
                spec.bot_name: time.time() for spec in LIVE_PAIRS.values()
            }
        self.telemetry_path = self.state_dir / "dca_macro_telemetry.json"
        self.managed_inventory_path = self.state_dir / "managed_inventory.json"
        self.grid_inventory_state_path = Path(os.getenv(
            "GRID_ACCOUNT_INVENTORY_STATE_PATH", "/workspace/grid/guard_state.json"
        ))
        self.grid_reservations_path = Path(os.getenv(
            "GRID_ACCOUNT_INVENTORY_RESERVATIONS_PATH",
            "/workspace/grid/capital_reservations.json",
        ))
        self.inventory_ledger = UnifiedInventoryLedger(Path(os.getenv(
            "ACCOUNT_INVENTORY_LEDGER_PATH", "/workspace/account-inventory"
        )))
        self.inventory_confirmation_cycles = max(
            3, int(os.getenv("ACCOUNT_INVENTORY_CONFIRMATION_CYCLES", "3"))
        )
        self.inventory_confirmation_seconds = max(
            30, int(os.getenv("ACCOUNT_INVENTORY_CONFIRMATION_SECONDS", "30"))
        )
        self.unattributed_auto_liquidate = _env_enabled(
            "ACCOUNT_INVENTORY_UNATTRIBUTED_AUTO_LIQUIDATE_ENABLED", False
        )
        self.inventory_filter_cache: Dict[str, tuple[float, Decimal, Decimal]] = {}
        self.inventory_ledger.set_bootstrap_caps({
            "BTC": os.getenv(
                "ACCOUNT_INVENTORY_BOOTSTRAP_BTC_CAP",
                "0.001554857672861421803102676728",
            ),
            "ETH": os.getenv(
                "ACCOUNT_INVENTORY_BOOTSTRAP_ETH_CAP",
                "0.00227157909293506300096269939",
            ),
        })
        self.interval = max(2, int(os.getenv("DCA_LIVE_GUARD_INTERVAL", "10")))
        self.binance_reads = GetOnlyReadClient(binance_api_base())
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
        # Quote capital is advisory for ordinary DCA trading.  It is observed
        # and reported, but it must never participate in the aggregate
        # BUY/SELL permission decision: open BUY orders legitimately move
        # funds from ``free`` to ``locked`` and would otherwise create a
        # cancel/recreate feedback loop.  Transaction-level affordability is
        # still checked before an emergency market re-entry is submitted.
        self.quote_budget_buffer_pct = Decimal(os.getenv(
            "DCA_QUOTE_BUDGET_BUFFER_PCT", "0.002"
        ))
        self.quote_balance_cache_seconds = max(
            10, int(os.getenv("DCA_QUOTE_BALANCE_CACHE_SECONDS", "30"))
        )
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
        self.inventory_account_fingerprint = (
            api_key_fingerprint(self.emergency_exchange.api_key)
            if self.emergency_exchange is not None else "unavailable"
        )
        if self.emergency_exchange is not None:
            self.inventory_ledger.bind_account(self.inventory_account_fingerprint)
        docker_socket = os.getenv("DCA_DOCKER_SOCKET", "/var/run/docker.sock")
        self.emergency_docker = (
            DockerEmergencyClient(docker_socket)
            if Path(docker_socket).exists()
            else None
        )
        fee_policy = None
        if os.getenv("DCA_LIVE_TRADING_ENABLED", "false").lower() == "true":
            if self.emergency_exchange is None or self.emergency_docker is None:
                raise RuntimeError(
                    "armed DCA Guard requires independent Binance credentials "
                    "and the Docker emergency socket"
                )
            fee_policy = self.emergency_exchange.verify_ready(list(LIVE_PAIRS))
            for spec in LIVE_PAIRS.values():
                self.emergency_docker.matching_containers(spec.bot_name)
        self.state = self._load_state()
        if fee_policy is not None:
            self.state["fee_policy"] = fee_policy
        self.state["emergency_ready"] = bool(
            self.emergency_exchange is not None and self.emergency_docker is not None
        )
        if self.state["emergency_ready"] and self.inventory_ledger.status_path.exists():
            try:
                inventory_status = self._read_json(self.inventory_ledger.status_path)
                managed = self._read_json(self.managed_inventory_path)
                self.state["ownership_preflight"] = self._ownership_preflight_from_status(
                    inventory_status, managed,
                )
            except Exception as exc:
                self.state["ownership_preflight"] = {
                    pair: {
                        "covered": False,
                        "source": "unified_account_inventory_v3",
                        "reason": f"inventory_status_unavailable:{type(exc).__name__}",
                    }
                    for pair in LIVE_PAIRS
                }
        self._migrate_incomplete_latched_inventory()
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
        state["mechanism_parameters"] = {
            "v22_weekly_buy_gate": {
                "update_cycle": "weekly", "contract_max_age_seconds": self.v21_max_age_seconds,
                "scope": "ordinary_buy_only", "threshold_source": "fold_local_signed_model",
            },
            "fomc_gate": {
                "contract_max_age_seconds": self.macro_max_age_seconds,
                "scope": "configured_direction",
            },
            "strategy_loss_breaker": {
                "loss_limit_quote": str(SINGLE_BOT_LOSS_LIMIT), "quote_asset": "USDT",
                "cooldown_seconds": STRATEGY_COOLDOWN_SECONDS,
            },
            "strategy_drawdown_breaker": {
                "drawdown_limit_pct": str(self.strategy_drawdown_limit),
                "cooldown_seconds": STRATEGY_COOLDOWN_SECONDS,
            },
            "portfolio_loss_breaker": {
                "loss_limit_quote": str(COMBINED_LOSS_LIMIT), "quote_asset": "USDT",
                "cooldown_seconds": PORTFOLIO_COOLDOWN_SECONDS,
            },
            "portfolio_drawdown_breaker": {
                "drawdown_limit_pct": str(self.portfolio_drawdown_limit),
                "cooldown_seconds": PORTFOLIO_COOLDOWN_SECONDS,
            },
            "position_protection": {
                "stop_loss_pct": "0.05", "cooldown_seconds": POSITION_COOLDOWN_SECONDS,
                "healthy_cycles_before_reentry": REQUIRED_HEALTHY_CYCLES,
            },
            "capital_budget_gate": {
                "mode": "alert_only", "strategy_budget_quote": str(STRATEGY_BUDGET_QUOTE),
                "quote_asset": "USDT",
            },
            "infrastructure_integrity_breaker": {
                "fail_closed_seconds": self.fail_closed_seconds,
                "auto_recovery": False,
            },
        }
        for bot in state.get("bots", {}).values():
            bot["recovery"] = normalize_state(bot.get("recovery"))
        return state

    def _save(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def _record_read_retry_events(self) -> None:
        client = getattr(self, "binance_reads", None)
        if client is None:
            return
        telemetry = self.state.setdefault("read_retry_telemetry", {})
        for retry in client.consume_read_retry_events():
            row = telemetry.setdefault("binance_public", {
                "retry_events": 0, "retry_attempts": 0,
                "pool_replacements": 0, "longest_recovery_seconds": 0.0,
            })
            row["retry_events"] = int(row.get("retry_events", 0)) + 1
            row["retry_attempts"] = int(row.get("retry_attempts", 0)) + int(
                retry.get("attempts", 1)
            )
            row["pool_replacements"] = int(row.get("pool_replacements", 0)) + int(
                bool(retry.get("pool_replaced"))
            )
            duration = float(retry.get("duration_seconds", 0.0))
            row["longest_recovery_seconds"] = max(
                float(row.get("longest_recovery_seconds", 0.0)), duration,
            )
            row["last_recovered_at"] = time.time()
            row["last_reason"] = str(retry.get("reason") or "transient GET retry")
            self.runtime_errors.record_transient_recovery(
                "guard_read:binance_public",
                retry.get("reason", "transient GET retry"),
                occurrences=int(retry.get("attempts", 1)),
                duration_seconds=duration,
            )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"inventory evidence is not an object: {path}")
        return value

    @staticmethod
    def _ownership_preflight_from_status(
        status: Dict[str, Any], managed_inventory: Dict[str, Any],
        *, observed_at: float | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Build approval coverage from current ownership, not startup inventory.

        ``latest.net_base`` already contains all Guard emergency adjustments.
        The immutable managed inventory remains an auditable cycle baseline, but
        it is not the amount that must still be present after confirmed exits.
        """
        now = time.time() if observed_at is None else observed_at
        generated_at = float(status.get("generated_at") or 0)
        fresh = generated_at > 0 and now - generated_at < 30
        contract_healthy = bool(
            fresh
            and status.get("schema") == "account-inventory-status-v3"
            and status.get("sources_healthy") is True
            and bool(status.get("account_fingerprint"))
            and bool(status.get("evidence_sha256"))
        )
        result: Dict[str, Dict[str, Any]] = {}
        for pair, spec in LIVE_PAIRS.items():
            row = status.get("assets", {}).get(spec.base_asset, {})
            owner_key = f"dca:{spec.bot_name}"
            owned = Decimal(str(row.get("owners", {}).get(owner_key, "0")))
            total = Decimal(str(row.get("exchange", {}).get("total", "0")))
            deficit = Decimal(str(row.get("ownership_deficit", "0")))
            target = Decimal(str(
                managed_inventory.get("pairs", {}).get(pair, {}).get("managed_base", "0")
            ))
            covered = bool(
                contract_healthy and deficit <= 0 and owned >= 0 and owned <= total
            )
            result[pair] = {
                "managed_base": str(owned),
                "managed_base_target": str(target),
                "owned_base": str(owned),
                "account_total_base": str(total),
                "ownership_deficit": str(deficit),
                "covered": covered,
                "source": "unified_account_inventory_v3",
                "evidence_sha256": status.get("evidence_sha256"),
                "generated_at": generated_at,
                "age_seconds": max(0.0, now - generated_at) if generated_at else None,
                "reason": "current_strategy_ownership_covered" if covered else (
                    "inventory_contract_untrusted_or_stale" if not contract_healthy
                    else "current_strategy_ownership_exceeds_account_balance"
                ),
            }
        return result

    def _migrate_incomplete_latched_inventory(self) -> None:
        """Do not keep claiming an integrity exit completed when startup stock remains.

        The migration intentionally does not submit an order.  Existing latches
        are marked for manual exit because this rollout was approved to clear
        only unattributed inventory.
        """
        if not self.managed_inventory_path.exists():
            return
        managed = self._read_json(self.managed_inventory_path)
        for pair, spec in LIVE_PAIRS.items():
            bot = self.state.get("bots", {}).get(spec.bot_name)
            if not isinstance(bot, dict):
                continue
            recovery = normalize_state(bot.get("recovery"))
            flatten = bot.get("flatten_response", {})
            if not (
                recovery.get("phase") == LATCHED
                and bot.get("action_complete") is True
                and isinstance(flatten, dict)
                and flatten.get("status") == "not_required"
                and recovery.get("exit_completed_at") is None
            ):
                continue
            target = Decimal(str(
                bot.get("managed_base_target")
                or managed.get("pairs", {}).get(pair, {}).get("managed_base", "0")
            ))
            remaining = max(
                target + Decimal(str(bot.get("latest", {}).get("net_base", "0"))),
                Decimal("0"),
            )
            if remaining <= 0:
                continue
            recovery["remaining_base"] = {pair: str(remaining)}
            recovery["exit_completed_at"] = None
            bot.update({
                "action_complete": False,
                "manual_exit_required": True,
                "exit_status": "pending_manual_existing_dca_inventory",
                "recovery": recovery,
            })

    def _inventory_evidence(self) -> tuple[dict[str, dict[str, Decimal]], str, bool]:
        reservations = self._read_json(self.grid_reservations_path)
        grid_state = self._read_json(self.grid_inventory_state_path)
        managed = self._read_json(self.managed_inventory_path)
        ownership = ownership_from_documents(
            reservations=reservations, grid_state=grid_state,
            managed_inventory=managed, dca_state=self.state,
        )
        running = {}
        for name in ("grid-live-fdusd-400", *[spec.bot_name for spec in LIVE_PAIRS.values()]):
            running[name] = bool(
                self.emergency_docker and self.emergency_docker.matching_containers(name)
            )
        now = time.time()
        grid_latest = next(iter(grid_state.get("bots", {}).values()), {}).get("latest", {})
        timestamps = [float(grid_latest.get("observed_at") or 0)]
        timestamps.extend(
            float(self.state.get("bots", {}).get(spec.bot_name, {}).get("latest", {}).get("observed_at") or 0)
            for spec in LIVE_PAIRS.values()
        )
        # A stopped bot has immutable trade evidence; a running bot requires a
        # fresh monitor snapshot before an account-wide reconciliation is safe.
        sources_healthy = all(
            (not running[name]) or (timestamps[index] > 0 and now - timestamps[index] < 30)
            for index, name in enumerate(running)
        )
        evidence = {
            "ownership": {
                asset: {key: str(value) for key, value in owners.items()}
                for asset, owners in ownership.items()
            },
            "reservations_generated_at": reservations.get("generated_at"),
            "managed_source_sha256": managed.get("source_preflight_sha256"),
            "grid_database_event_at": grid_latest.get("database_event_at"),
            "dca_database_event_at": {
                spec.bot_name: self.state.get("bots", {}).get(spec.bot_name, {}).get(
                    "latest", {}
                ).get("database_event_at")
                for spec in LIVE_PAIRS.values()
            },
            "running": running,
        }
        return ownership, canonical_sha256(evidence), sources_healthy

    def _inventory_runtime_status(
        self, order_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        grid_state = self._read_json(self.grid_inventory_state_path)
        grid_bot = next(iter(grid_state.get("bots", {}).values()), {})
        grid_recovery = grid_bot.get("recovery") or {}
        grid_phase = str(
            grid_recovery.get("phase")
            or ("TRIPPED" if grid_bot.get("tripped") else "ACTIVE")
        )
        robots: Dict[str, Dict[str, Any]] = {
            "grid-live-fdusd-400": {
                "phase": grid_phase,
                "running": bool(
                    self.emergency_docker
                    and self.emergency_docker.matching_containers("grid-live-fdusd-400")
                ),
            },
        }
        for spec in LIVE_PAIRS.values():
            bot = self.state.get("bots", {}).get(spec.bot_name, {})
            recovery = bot.get("recovery") or {}
            phase = str(recovery.get("phase") or "ACTIVE")
            robots[spec.bot_name] = {
                "phase": phase,
                "running": bool(
                    self.emergency_docker
                    and self.emergency_docker.matching_containers(spec.bot_name)
                ),
            }
        trading_normal = all(
            row["running"] and row["phase"] == "ACTIVE" for row in robots.values()
        )
        return {
            "robots": robots,
            "active_order_count": sum(int(value) for value in order_counts.values()),
            "open_order_counts": dict(order_counts),
            "trading_normal": trading_normal,
        }

    def _inventory_market_diagnostics(
        self, asset: str, row: Dict[str, Any],
    ) -> Dict[str, Any]:
        pair = f"{asset}-USDT"
        now = time.time()
        cache = getattr(self, "inventory_filter_cache", {})
        cached = cache.get(pair)
        if cached is None or now - cached[0] >= 300:
            step, minimum_notional = self._lot_filter(pair)
            cache[pair] = (now, step, minimum_notional)
            self.inventory_filter_cache = cache
            filter_fetched_at = now
        else:
            filter_fetched_at, step, minimum_notional = cached
        bot_name = f"dca-live-{asset.lower()}usdt-200"
        latest_mark = self.state.get("bots", {}).get(bot_name, {}).get(
            "latest", {}
        ).get("mark_price")
        mark = Decimal(str(latest_mark)) if latest_mark is not None else self._price(pair)
        unattributed = Decimal(str(row["unattributed"]))
        free = Decimal(str(row.get("exchange", {}).get("free", "0")))
        intended = min(unattributed, free)
        bootstrap_cap = self.inventory_ledger.bootstrap_cap(asset)
        if bootstrap_cap is not None:
            intended = min(intended, bootstrap_cap)
        amount = (intended / step).to_integral_value(rounding=ROUND_DOWN) * step
        estimated_notional = amount * mark
        if amount <= 0:
            dust_reason = "rounded_quantity_zero"
        elif estimated_notional < minimum_notional:
            dust_reason = "below_minimum_notional"
        else:
            dust_reason = ""
        return {
            "tradable_quantity": str(amount),
            "estimated_notional": str(estimated_notional),
            "minimum_notional": str(minimum_notional),
            "mark_price": str(mark),
            "step_size": str(step),
            "filter_fetched_at": filter_fetched_at,
            "price_source": "guard_snapshot" if latest_mark is not None else "binance_ticker",
            "dust_reason": dust_reason,
            "tradable": not dust_reason,
        }

    def _inventory_event(
        self, *, transition: str, asset: str, reason: str,
        severity: str, action: str, correlation_id: str,
        details: Dict[str, Any],
    ) -> None:
        event = build_event(
            source="dca-live-guard", strategy="account", bot="shared-binance-spot",
            pair=f"{asset}-USDT", mechanism="account_inventory",
            transition=transition, reason=reason, severity=severity,
            phase_from="", phase_to=transition, action=action,
            correlation_id=correlation_id, details=details,
        )
        if self.inventory_ledger.stage_event(event["event_id"], transition, event):
            append_event(self.notification_path, event)
            self.inventory_ledger.mark_event_delivered(event["event_id"])
            self.inventory_ledger.mark_episode_notified(asset, transition)

    def _inventory_deficit_diagnostics(
        self, asset: str, row: Dict[str, Any],
    ) -> Dict[str, Any]:
        pair = f"{asset}-USDT"
        bot_name = f"dca-live-{asset.lower()}usdt-200"
        latest_mark = self.state.get("bots", {}).get(bot_name, {}).get(
            "latest", {}
        ).get("mark_price")
        mark = Decimal(str(latest_mark)) if latest_mark is not None else self._price(pair)
        deficit = Decimal(str(row.get("ownership_deficit", "0")))
        return {
            "deficit_quantity": str(deficit),
            "deficit_estimated_notional": str(deficit * mark),
            "mark_price": str(mark),
            "price_source": "guard_snapshot" if latest_mark is not None else "binance_ticker",
        }

    @staticmethod
    def _confirmed_deficit_alert(row: Dict[str, Any]) -> bool:
        confirmation = row.get("deficit_confirmation", {})
        return bool(
            Decimal(str(row.get("ownership_deficit", "0"))) > 0
            and confirmation.get("confirmed") is True
            and confirmation.get("notified") is not True
        )

    def _flush_inventory_events(self) -> int:
        delivered = 0
        for row in self.inventory_ledger.pending_events():
            append_event(self.notification_path, row["payload"])
            self.inventory_ledger.mark_event_delivered(row["event_id"])
            delivered += 1
        return delivered

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
            # Compatibility for unit-test fakes. Production clients always
            # accept the deterministic client id.
            return self.emergency_exchange.market_order(pair, side, amount)
        except Exception:
            if hasattr(self.emergency_exchange, "order_by_client_id"):
                existing = self.emergency_exchange.order_by_client_id(pair, client_order_id)
                if isinstance(existing, dict) and existing.get("status") == "FILLED":
                    return existing
            raise

    def _liquidate_unattributed(self, asset: str, row: Dict[str, Any], evidence: str) -> None:
        pair = f"{asset}-USDT"
        holder = f"dca-live-guard:unattributed:{asset}"
        with self.inventory_ledger.lease(asset, holder, ttl_seconds=45):
            balances = self.emergency_exchange.account_balances()
            actual_free = balances.get(asset, {}).get("free", Decimal("0"))
            owned = Decimal(str(row["owned_total"]))
            self.inventory_ledger.assert_exit_allowed(
                asset=asset,
                exchange_total=balances.get(asset, {}).get("total", Decimal("0")),
            )
            live_unattributed = max(
                balances.get(asset, {}).get("total", Decimal("0")) - owned,
                Decimal("0"),
            )
            quantity = min(live_unattributed, actual_free)
            bootstrap_cap = self.inventory_ledger.bootstrap_cap(asset)
            if bootstrap_cap is not None:
                quantity = min(quantity, bootstrap_cap)
            step, minimum_notional = self._lot_filter(pair)
            mark = self._price(pair)
            intended = live_unattributed
            if bootstrap_cap is not None:
                intended = min(intended, bootstrap_cap)
            if actual_free + step < intended and intended * mark >= minimum_notional:
                raise RuntimeError(
                    f"unattributed {asset} is locked or unavailable: "
                    f"intended={intended};free={actual_free}"
                )
            amount = (quantity / step).to_integral_value(rounding=ROUND_DOWN) * step
            if amount <= 0 or amount * mark < minimum_notional:
                dust_quantity = max(quantity, Decimal("0"))
                job_id, client_order_id = liquidation_identity(
                    asset, "unattributed_dust", dust_quantity, evidence
                )
                job = self.inventory_ledger.start_job(
                    job_id=job_id, asset=asset, scope="unattributed_dust",
                    pair=pair, requested_quantity=dust_quantity,
                    client_order_id=client_order_id,
                )
                self.inventory_ledger.finish_job(
                    job_id, status="DUST", error=(
                        f"rounded_quantity={amount};notional={amount * mark};"
                        f"minimum_notional={minimum_notional}"
                    ),
                    consume_bootstrap_asset=asset if bootstrap_cap is not None else "",
                )
                self.inventory_ledger.set_episode_phase(asset, "DUST")
                # Expected exchange dust is durable audit evidence, not an
                # operator alert. A fill/ownership snapshot race commonly
                # settles here after the next monitor cycle.
                last_transition = str(row.get("last_notified_transition") or "")
                first_classification = bool(
                    str(job.get("status") or "") != "DUST"
                    and last_transition != "INVENTORY_DUST_CLASSIFIED"
                )
                if first_classification:
                    self._audit(
                        "inventory_dust_classified",
                        asset=asset, pair=pair, job_id=job_id,
                        quantity=str(dust_quantity), rounded_quantity=str(amount),
                        notional=str(amount * mark),
                        minimum_notional=str(minimum_notional),
                        dust_reason=(
                            "rounded_quantity_zero" if amount <= 0
                            else "below_minimum_notional"
                        ),
                    )
                self.inventory_ledger.mark_episode_notified(
                    asset, "INVENTORY_DUST_CLASSIFIED",
                )
                return
            related_pairs = (f"{asset}-USDT", f"{asset}-FDUSD")
            for related in related_pairs:
                orders = self.emergency_exchange.open_orders(related)
                if orders:
                    self.emergency_exchange.cancel_all_orders(related)
                    return
            job_id, client_order_id = liquidation_identity(
                asset, "unattributed", amount, evidence
            )
            job = self.inventory_ledger.start_job(
                job_id=job_id, asset=asset, scope="unattributed", pair=pair,
                requested_quantity=amount, client_order_id=client_order_id,
            )
            if job.get("status") == "COMPLETED":
                if not self.inventory_ledger.completed_job_verified(job):
                    raise RuntimeError(
                        f"completed inventory job {job_id} lacks mandatory verification"
                    )
                return
            self.inventory_ledger.set_episode_phase(asset, "LIQUIDATING")
            self._audit(
                "inventory_liquidation_started",
                asset=asset, pair=pair, job_id=job_id,
                quantity=str(amount), client_order_id=client_order_id,
            )
            try:
                before_total = balances.get(asset, {}).get("total", Decimal("0"))
                response = execute_market_liquidation(
                    exchange=self.emergency_exchange,
                    ledger=self.inventory_ledger, job_id=job_id, pair=pair,
                    side="SELL", target_quantity=amount,
                    client_order_id=client_order_id, step_size=step,
                    minimum_notional=minimum_notional, mark_price=mark,
                    lease_asset=asset, lease_holder=holder,
                )
                verification = verify_market_liquidation(
                    exchange=self.emergency_exchange, pair=pair,
                    response=response, requested_quantity=amount,
                    before_total=before_total, step_size=step,
                    minimum_notional=minimum_notional, mark_price=mark,
                    ledger=self.inventory_ledger, lease_asset=asset,
                    lease_holder=holder,
                )
                metrics = self._emergency_fill_metrics(pair, "SELL", response)
                self.inventory_ledger.finish_job(
                    job_id, status="COMPLETED",
                    exchange_order_id=str(response.get("orderId", "")),
                    executed_quantity=response.get("executedQty", "0"),
                    quote_quantity=response.get("cummulativeQuoteQty", "0"),
                    fee_quote=metrics.get("fee_quote", "0"),
                    fee_details=metrics.get("fee_details", []),
                    verification=verification,
                    consume_bootstrap_asset=asset if bootstrap_cap is not None else "",
                )
            except Exception as exc:
                self.inventory_ledger.finish_job(
                    job_id, status="FAILED", error=repr(exc)
                )
                self.inventory_ledger.set_episode_phase(asset, "CONFIRMED")
                self._inventory_event(
                    transition="INVENTORY_LIQUIDATION_FAILED", asset=asset,
                    reason=repr(exc), severity="critical",
                    action="fail_closed_retry", correlation_id=job_id,
                    details={"quantity": str(amount)},
                )
                raise
            self._audit(
                "inventory_liquidation_completed",
                asset=asset, pair=pair, job_id=job_id,
                quantity=response.get("executedQty", "0"),
                quote_quantity=response.get("cummulativeQuoteQty", "0"),
                fee_quote=metrics.get("fee_quote", "0"),
                fee_details=metrics.get("fee_details", []),
                order_id=response.get("orderId", ""),
                verification=verification,
            )

    @staticmethod
    def _confirmed_unattributed_alert(row: Dict[str, Any]) -> bool:
        """Alert only after the persisted 3-cycle/30-second confirmation."""
        return bool(
            Decimal(str(row.get("unattributed", "0"))) > 0
            and str(row.get("inventory_phase") or "CLEAR") == "CONFIRMED"
            and row.get("confirmation", {}).get("confirmed") is True
        )

    def _reconcile_account_inventory(self, *, risk_actions_enabled: bool) -> dict[str, Any]:
        if self.emergency_exchange is None:
            raise RuntimeError("shared inventory reconciliation requires emergency exchange")
        self._flush_inventory_events()
        ownership, evidence, sources_healthy = self._inventory_evidence()
        pairs = ("BTC-USDT", "ETH-USDT", "BTC-FDUSD", "ETH-FDUSD")
        order_counts = {
            pair: len(self.emergency_exchange.open_orders(pair)) for pair in pairs
        }
        balances = self.emergency_exchange.account_balances()
        self.state["quote_balance_source"] = {
            "free_quote": str(balances.get("USDT", {}).get("free", Decimal("0"))),
            "observed_at": time.time(), "source": "account_reconciliation",
        }
        status = self.inventory_ledger.reconcile(
            account_fingerprint=self.inventory_account_fingerprint,
            balances=balances, ownership=ownership,
            evidence_sha256=evidence, open_order_counts=order_counts,
            sources_healthy=sources_healthy,
            confirmation_cycles=self.inventory_confirmation_cycles,
            confirmation_seconds=self.inventory_confirmation_seconds,
        )
        runtime = self._inventory_runtime_status(order_counts)
        status["runtime"] = runtime
        managed = self._read_json(self.managed_inventory_path)
        ownership_preflight = self._ownership_preflight_from_status(
            status, managed,
        )
        self.state["ownership_preflight"] = ownership_preflight
        ownership_ready = bool(ownership_preflight) and all(
            bool(row.get("covered")) for row in ownership_preflight.values()
        )
        self.state["ownership_preflight_checked_at"] = time.time()
        self.state["ownership_preflight_healthy_cycles"] = (
            int(self.state.get("ownership_preflight_healthy_cycles", 0)) + 1
            if ownership_ready else 0
        )
        self.state["account_inventory"] = {
            "healthy": status["healthy"], "evidence_sha256": evidence,
            "status_path": str(self.inventory_ledger.status_path),
        }
        for asset, row in status["assets"].items():
            row["runtime"] = runtime
            deficit = Decimal(row["ownership_deficit"])
            unattributed = Decimal(row["unattributed"])
            if deficit > 0:
                try:
                    row.update(self._inventory_deficit_diagnostics(asset, row))
                except Exception as exc:
                    row.update({
                        "deficit_quantity": str(deficit),
                        "deficit_estimated_notional": "",
                        "market_data_error": f"{type(exc).__name__}: {exc}",
                    })
                confirmation = row.get("deficit_confirmation", {})
                # Reconciliation is unhealthy immediately; only notification waits.
                if self._confirmed_deficit_alert(row):
                    episode_id = f"deficit:{asset}:{confirmation.get('first_seen', '')}"
                    self._inventory_event(
                        transition="INVENTORY_OWNERSHIP_DEFICIT", asset=asset,
                        reason="confirmed_strategy_ownership_exceeds_exchange_balance",
                        severity="critical", action="fail_closed_no_liquidation",
                        correlation_id=episode_id, details=row,
                    )
                    self.inventory_ledger.mark_deficit_notified(asset)
                continue
            try:
                market = self._inventory_market_diagnostics(asset, row)
                market["market_data_healthy"] = True
            except Exception as exc:
                market = {
                    "tradable_quantity": "0", "estimated_notional": "0",
                    "minimum_notional": "0", "dust_reason": "",
                    "tradable": False, "market_data_healthy": False,
                    "market_data_error": f"{type(exc).__name__}: {exc}",
                }
            row.update(market)
            if row["inventory_phase"] == "DUST":
                transitional_robot = any(
                    str(value.get("phase") or "ACTIVE") in {EXITING, REENTRY}
                    for value in runtime.get("robots", {}).values()
                )
                recheck_eligible = bool(
                    status.get("sources_healthy")
                    and Decimal(row["ownership_deficit"]) == 0
                    and market.get("market_data_healthy")
                    and market.get("tradable")
                    and not transitional_robot
                )
                recheck_evidence = canonical_sha256({
                    "episode_id": row.get("episode_id"),
                    "stability_sha256": row.get("stability_sha256"),
                    "tradable_quantity": market.get("tradable_quantity"),
                    "minimum_notional": market.get("minimum_notional"),
                    "step_size": market.get("step_size"),
                })
                recheck = self.inventory_ledger.observe_dust_recheck(
                    asset, episode_id=str(row.get("episode_id") or ""),
                    evidence_sha256=recheck_evidence,
                    tradable_quantity=Decimal(str(market.get("tradable_quantity") or "0")),
                    eligible=recheck_eligible,
                    confirmation_cycles=self.inventory_confirmation_cycles,
                    confirmation_seconds=self.inventory_confirmation_seconds,
                )
                row["dust_recheck"] = recheck
                if recheck.get("confirmed"):
                    new_episode = self.inventory_ledger.reopen_dust_episode(
                        asset,
                        expected_episode_id=str(row.get("episode_id") or ""),
                        evidence_sha256=str(row.get("stability_sha256") or ""),
                        quantity=unattributed,
                    )
                    row.update({
                        "episode_id": new_episode,
                        "inventory_phase": "DETECTED",
                        "last_notified_transition": "",
                        "confirmation_eligible": True,
                        "confirmation_block_reason": "dust_became_tradable_reconfirm",
                        "confirmation": {
                            "cycles": 0, "first_seen": status["generated_at"],
                            "last_seen": status["generated_at"], "confirmed": False,
                        },
                        "dust_recheck": {
                            "active": False, "cycles": 0, "confirmed": False,
                        },
                    })
                else:
                    row["confirmation_eligible"] = False
                    row["confirmation_block_reason"] = (
                        "dust_tradable_recheck_pending"
                        if recheck_eligible else "already_classified_dust"
                    )
            phase = str(row.get("inventory_phase") or "CLEAR")
            episode_id = str(row.get("episode_id") or "")
            confirmed_unattributed = self._confirmed_unattributed_alert(row)
            already_notified = str(row.get("last_notified_transition") or "") in {
                "INVENTORY_UNATTRIBUTED_CONFIRMED", "INVENTORY_DUST_CLASSIFIED",
            }
            if confirmed_unattributed and not already_notified:
                self._audit(
                    "inventory_unattributed_confirmed",
                    asset=asset, episode_id=episode_id,
                    unattributed=str(unattributed), details=row,
                )
                self.inventory_ledger.mark_episode_notified(
                    asset, "INVENTORY_UNATTRIBUTED_CONFIRMED",
                )
                row["last_notified_transition"] = "INVENTORY_UNATTRIBUTED_CONFIRMED"
            if (
                risk_actions_enabled and self.unattributed_auto_liquidate
                and confirmed_unattributed
                and market.get("market_data_healthy")
            ):
                self._liquidate_unattributed(asset, row, episode_id)
            elif confirmed_unattributed and not (
                risk_actions_enabled and self.unattributed_auto_liquidate
            ):
                self._inventory_event(
                    transition="INVENTORY_UNATTRIBUTED_DETECTED", asset=asset,
                    reason="confirmed_unattributed_inventory_has_no_auto_handler",
                    severity="warning", action="manual_review_required",
                    correlation_id=episode_id, details=row,
                )
        self.inventory_ledger.write_status(status)
        return status

    def _audit(self, event: str, **details: Any) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        audit_path = getattr(self, "audit_path", None)
        if audit_path is not None:
            with audit_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        log = LOG.info if event == "inventory_dust_classified" else LOG.warning
        log("%s %s", event, json.dumps(details, ensure_ascii=False, default=str))
        self._emit_notification(event, details)

    def _emit_notification(self, audit_event: str, details: Dict[str, Any]) -> None:
        path = getattr(self, "notification_path", None)
        if path is None:
            return
        if audit_event == "capital_budget_gate_transition":
            ready = bool(details.get("buy_enabled"))
            append_event(path, build_event(
                source="dca-live-guard", strategy="dca",
                bot=",".join(spec.bot_name for spec in LIVE_PAIRS.values()),
                pair=",".join(LIVE_PAIRS),
                mechanism="capital_budget_gate",
                transition="RECOVERED" if ready else "TRIGGERED",
                reason=str(details.get("reason") or audit_event),
                severity="info" if ready else "warning",
                phase_from="ALERT_ONLY" if ready else "OK",
                phase_to="OK" if ready else "ALERT_ONLY",
                action=str(details.get("action") or
                           "capital_alert_only_no_trade_block"),
                trigger_value=details.get("free_quote"),
                threshold=details.get("required_quote"),
                correlation_id=(
                    f"capital-budget-alert-only:{ready}:"
                    f"{details.get('reason')}"
                ),
                details={
                    "free_quote": details.get("free_quote"),
                    "required_quote": details.get("required_quote"),
                    "enforcement_mode": "alert_only",
                    "trading_permissions_changed": False,
                },
            ))
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

    def _scan_runtime_logs(self) -> None:
        if self.emergency_docker is None:
            return
        now = time.time()
        for spec in LIVE_PAIRS.values():
            name = spec.bot_name
            since = float(self.runtime_log_cursors.get(name, now))
            monitor_component = f"runtime_log_monitor:{name}"
            try:
                lines = self.emergency_docker.logs_since(name, since)
            except Exception as exc:
                self.runtime_errors.failure(
                    monitor_component, exc,
                    trading_impact=(
                        "仅该 DCA 机器人日志采集暂时不可用；Guard 主风控循环和交易权限不受影响。"
                    ),
                    severity="warning", action="retry_log_collection_next_cycle",
                )
                continue
            self.runtime_errors.recovered(
                monitor_component,
                trading_status="DCA 机器人日志采集恢复；主风控循环始终未受影响",
            )
            self.runtime_log_cursors[name] = now
            errors = runtime_error_lines(lines)
            component = f"container_log:{name}"
            if errors:
                self.runtime_errors.failure(
                    component, errors[-1],
                    trading_impact=(
                        "交易机器人日志出现错误；是否限制交易仍由现有 DCA 聚合风控门决定。"
                    ),
                    severity="warning",
                    action="inspect_and_continue_existing_executor_safety_logic",
                    details={"matched_log_lines": len(errors)},
                )
            self.runtime_errors.recover_if_quiet(
                component, quiet_seconds=300,
                trading_status="连续5分钟无新的 DCA 机器人日志错误；当前聚合风控状态保持不变",
            )
        temporary = self.runtime_log_cursor_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.runtime_log_cursors, indent=2), encoding="utf-8")
        temporary.replace(self.runtime_log_cursor_path)

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

    def _public_reads(self) -> GetOnlyReadClient:
        client = getattr(self, "binance_reads", None)
        if client is None:
            client = GetOnlyReadClient(binance_api_base())
            self.binance_reads = client
        return client

    def _price(self, pair: str) -> Decimal:
        response = self._public_reads().request(
            "GET", "/api/v3/ticker/price",
            params={"symbol": pair.replace("-", "")},
            timeout=15,
        )
        return Decimal(str(response["price"]))

    def _lot_filter(self, pair: str) -> tuple[Decimal, Decimal]:
        response = self._public_reads().request(
            "GET", "/api/v3/exchangeInfo",
            params={"symbol": pair.replace("-", "")},
            timeout=15,
        )
        filters = {item["filterType"]: item for item in response["symbols"][0]["filters"]}
        # MARKET_LOT_SIZE is authoritative for the forced-exit/re-entry market
        # orders. Some symbols expose a different market step than LOT_SIZE.
        lot = filters.get("MARKET_LOT_SIZE") or filters["LOT_SIZE"]
        if Decimal(str(lot.get("stepSize", "0"))) <= 0:
            lot = filters["LOT_SIZE"]
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
        latest_stop_loss = self._latest_stop_loss_event(database)
        return {
            "pair": pair,
            "database": str(database),
            "mark_price": str(price),
            **{key: str(value) for key, value in metrics.items()},
            "updated_at": observed_at,
            "observed_at": observed_at,
            "database_event_at": database_event_at,
            "database_event_age_seconds": max(0.0, observed_at - database_event_at),
            "latest_stop_loss_at": float(latest_stop_loss.get("timestamp", 0)),
            "latest_stop_loss_event": latest_stop_loss,
        }

    @staticmethod
    def _latest_stop_loss_event(database: Path) -> Dict[str, Any]:
        """Return the newest durable STOP_LOSS signal from either persistence path.

        Final executor rows expose ``close_timestamp``.  The live DCA runtime also
        writes a trigger snapshot immediately, before its in-memory closed-executor
        retention buffer is flushed; that timestamp lives in ``custom_info``.
        Reading both makes the event contract resilient to either writer lagging.
        """
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10)
        try:
            rows = connection.execute(
                "SELECT id, close_timestamp, custom_info FROM Executors WHERE close_type = 2"
            ).fetchall()
        finally:
            connection.close()
        latest = {"timestamp": 0.0, "executor_id": None, "source": None, "side": None}
        for executor_id, close_timestamp, raw_custom_info in rows:
            try:
                custom_info = (
                    json.loads(raw_custom_info)
                    if isinstance(raw_custom_info, str)
                    else (raw_custom_info or {})
                )
            except (TypeError, ValueError):
                custom_info = {}
            trigger_timestamp = custom_info.get("stop_loss_trigger_timestamp")
            candidates = (
                (trigger_timestamp, "executor_trigger_snapshot"),
                (close_timestamp, "executor_terminal_row"),
            )
            for raw_timestamp, source in candidates:
                try:
                    value = float(raw_timestamp or 0)
                except (TypeError, ValueError):
                    continue
                while value > 10_000_000_000:
                    value /= 1000
                if value > float(latest["timestamp"]):
                    latest = {
                        "timestamp": value,
                        "executor_id": str(executor_id),
                        "source": source,
                        "side": Guard._side(custom_info.get("side")),
                    }
        return latest

    @staticmethod
    def _latest_stop_loss_at(database: Path) -> float:
        """Compatibility wrapper retained for existing diagnostics and tests."""
        return float(Guard._latest_stop_loss_event(database).get("timestamp", 0))

    @staticmethod
    def _consume_position_stop_event(
        bot_state: Dict[str, Any], snapshot: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        """Advance the durable STOP_LOSS cursor without replaying upgrade history.

        The v3 cursor records when monitoring began. On the first upgraded
        cycle, baseline the newest durable row. A database restored or mounted
        later may contain a row newer than the old cursor, but events predating
        this monitoring epoch are still history and must never cause an exit.
        A genuine STOP_LOSS written while Guard is temporarily unavailable is
        newer than the epoch and remains consumable after Guard recovers.
        """
        event = dict(snapshot.get("latest_stop_loss_event") or {})
        try:
            timestamp = float(
                event.get("timestamp") or snapshot.get("latest_stop_loss_at") or 0
            )
        except (TypeError, ValueError):
            timestamp = 0.0
        event_id = str(event.get("executor_id") or "")
        if bot_state.get("position_stop_cursor_schema") != STOP_LOSS_EVENT_CURSOR_SCHEMA:
            bot_state["position_stop_cursor_schema"] = STOP_LOSS_EVENT_CURSOR_SCHEMA
            bot_state["last_position_stop_seen"] = timestamp
            bot_state["last_position_stop_event_id"] = event_id
            bot_state["position_stop_monitor_started_at"] = float(
                snapshot.get("observed_at") or time.time()
            )
            return None, True

        try:
            last_timestamp = float(bot_state.get("last_position_stop_seen") or 0)
        except (TypeError, ValueError):
            last_timestamp = 0.0
        last_event_id = str(bot_state.get("last_position_stop_event_id") or "")
        is_new = timestamp > last_timestamp or (
            timestamp > 0 and timestamp == last_timestamp and event_id != last_event_id
        )
        if not is_new:
            return None, False
        bot_state["last_position_stop_seen"] = timestamp
        bot_state["last_position_stop_event_id"] = event_id
        monitor_started_at = float(bot_state.get("position_stop_monitor_started_at") or 0)
        if timestamp < monitor_started_at:
            return None, True
        return event, False

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
        # Strategy V2 reloads this YAML every clock tick, so it is the live
        # source of truth. The Controllers row is historical executor evidence
        # and can lag a controller update indefinitely.
        live_config_path = (
            database.parent.parent / "conf" / "controllers"
            / f"{controller_id}.yml"
        )
        try:
            live_config = yaml.safe_load(live_config_path.read_text(encoding="utf-8"))
            if isinstance(live_config, dict):
                return str(controller_id), live_config
        except (OSError, yaml.YAMLError):
            # Preserve the database fallback for offline snapshots and the
            # short interval before a deployed config directory is mounted.
            pass
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
                "model_signal": (
                    str(source.get("model_signal") or (
                        "RISK_OFF" if source.get("risk_off_active", True) else "RISK_ON"
                    )) if healthy else "UNAVAILABLE"
                ),
                "transition": source.get("transition", "fail_closed"),
                "reason": source.get("reason", contract.get("reason", "v21_unhealthy")),
                "event_id": source.get("event_id"),
                "force_exit": bool(source.get("force_exit", False)),
                "probability": source.get("probability"),
                "model_week": source.get("model_week"),
                "week_model_sha256": source.get("week_model_sha256"),
                "execution_authorized": bool(contract.get("execution_authorized", False)),
            }
        return {"healthy": healthy, "reason": contract.get("reason"),
                "schema": contract.get("schema"),
                "release_sha256": contract.get("release_sha256"),
                "model_sha256": contract.get("model_sha256"),
                "execution_authorized": bool(contract.get("execution_authorized", False)),
                "model_version": contract.get("model_version"),
                "runtime_generation": contract.get("runtime_generation"),
                "predecessor_release_sha256": contract.get("predecessor_release_sha256"),
                "state_lineage_sha256": contract.get("state_lineage_sha256"),
                "cutover_phase": contract.get("cutover_phase"),
                "fold_boundary": contract.get("fold_boundary"),
                "system_health": contract.get("system_health"),
                "generated_at": contract.get("generated_at"), "pairs": mapped}

    def _integrity_failure_requires_latch(
        self, source: str, reason: str, now: float,
    ) -> bool:
        """Apply grace only to transient transport failures.

        During grace the aggregate gate remains closed, so no new DCA risk is
        created, but inventory is not force-exited and the bot is not latched.
        """
        episodes = self.state.setdefault("integrity_failure_grace", {})
        decision = advance_integrity_failure(
            episodes.get(source), reason=reason, now=now,
            grace_seconds=self.fail_closed_seconds,
        )
        episodes[source] = decision
        if decision["classification"] == "transient_transport" and not decision["expired"]:
            self.runtime_errors.failure(
                f"{source}_transport",
                reason,
                trading_impact=(
                    "Aggregate order gate is temporarily closed while the Guard retries; "
                    "no forced exit or integrity latch before the grace deadline."
                ),
                details={
                    "grace_seconds": self.fail_closed_seconds,
                    "remaining_seconds": decision["remaining_seconds"],
                },
                now=now,
            )
            return False
        return True

    def _clear_integrity_failure(self, source: str, now: float) -> None:
        episode = self.state.setdefault("integrity_failure_grace", {}).pop(source, None)
        if episode:
            self.runtime_errors.recovered(
                f"{source}_transport",
                trading_status="contract healthy; aggregate gates again follow all active mechanisms",
                now=now,
            )

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
            source_error_total = int(observation.get("source_error_total", 0))
            integrity_error_total = int(observation.get("integrity_error_total", 0))
            observation.clear()
            observation.update({"release_sha256": release, "started_at": now,
                                "cycles": 0, "source_errors": 0, "integrity_errors": 0,
                                "source_error_total": source_error_total,
                                "integrity_error_total": integrity_error_total})
        observation["last_seen_at"] = now
        observation["cycles"] = int(observation.get("cycles", 0)) + 1
        observation["event_ids"] = {
            pair: contract.get("pairs", {}).get(source, {}).get("event_id")
            for pair, source in V22_PAIR_MAP.items()
        }
        if not contract.get("runtime_gate_healthy"):
            failure = str(contract.get("reason", ""))
            category = (
                "source_errors"
                if classify_integrity_failure(failure) == "transient_transport"
                else "integrity_errors"
            )
            observation[category] = int(observation.get(category, 0)) + 1
            total_key = (
                "source_error_total"
                if category == "source_errors" else "integrity_error_total"
            )
            observation[total_key] = int(observation.get(total_key, 0)) + 1
            observation["last_error"] = failure
            observation["current_error_since"] = observation.get(
                "current_error_since", now,
            )
        elif observation.get("last_error"):
            observation["last_recovered_error"] = observation.pop("last_error")
            observation["last_recovered_at"] = now
            observation.pop("current_error_since", None)

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
        long_only_enabled = bool(profile.get("long_only_enabled", False))
        sell_stop_event = dict(reasons.get("sell_stop_event") or {})
        desired_stop_at = float(sell_stop_event.get("timestamp") or 0)
        desired_stop_id = str(sell_stop_event.get("executor_id") or "")
        actual_stop_at = float(profile.get("sell_stop_event_at") or 0)
        actual_stop_id = str(profile.get("sell_stop_event_id") or "")
        if (
            actual_buy == buy_enabled and actual_sell == sell_enabled
            and actual_stop_at == desired_stop_at
            and actual_stop_id == desired_stop_id
        ):
            return {
                "status": "unchanged",
                "macro_buy_enabled": actual_buy,
                "macro_sell_enabled": actual_sell,
                "long_only_enabled": long_only_enabled,
                "macro_decision_id": str(profile.get("macro_decision_id", "")),
            }
        digest = hashlib.sha256(json.dumps(
            {"buy": buy_enabled, "sell": sell_enabled, "reasons": reasons},
            sort_keys=True, default=str, separators=(",", ":"),
        ).encode()).hexdigest()[:16]
        profile["macro_buy_enabled"] = buy_enabled
        profile["macro_sell_enabled"] = sell_enabled
        profile["sell_stop_event_at"] = desired_stop_at
        profile["sell_stop_event_id"] = desired_stop_id
        profile["macro_decision_id"] = f"risk-aggregate:{digest}"
        response = self.api.update_controller(bot_name, controller_name, profile)
        return {
            "status": "applied",
            "macro_buy_enabled": buy_enabled,
            "macro_sell_enabled": sell_enabled,
            "long_only_enabled": bool(profile.get("long_only_enabled", False)),
            "macro_decision_id": profile["macro_decision_id"],
            "response": response,
        }

    def _quote_budget_status(
        self, required_quote: Decimal, *, now: Optional[float] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Observe quote capital without changing ordinary trade permissions."""
        observed = time.time() if now is None else float(now)
        required = max(Decimal(str(required_quote)), Decimal("0"))
        buffer_pct = max(
            Decimal(str(getattr(self, "quote_budget_buffer_pct", "0.002"))),
            Decimal("0"),
        )
        required_with_buffer = required * (Decimal("1") + buffer_pct)
        previous = self.state.get("quote_budget_observation", {})
        source = self.state.get("quote_balance_source", {})
        exchange = getattr(self, "emergency_exchange", None)

        if required <= 0:
            result = {
                "healthy": True, "buy_ready": True, "free_quote": "0",
                "required_quote": "0", "required_with_buffer": "0",
                "reason": "no_buy_capital_required", "observed_at": observed,
                "cached": False,
            }
            self.state["quote_budget_observation"] = result
            return result

        # Offline/observation Guards may intentionally omit a signed exchange
        # channel. Armed production construction already rejects that setup.
        if exchange is None:
            return {
                "healthy": False, "buy_ready": True, "free_quote": None,
                "required_quote": str(required),
                "required_with_buffer": str(required_with_buffer),
                "reason": "quote_balance_channel_unavailable_observation_only",
                "observed_at": observed, "cached": False,
            }

        source_age = observed - float(source.get("observed_at") or 0)
        cache_seconds = int(getattr(self, "quote_balance_cache_seconds", 30))
        if (
            not force_refresh and source.get("free_quote") is not None
            and 0 <= source_age <= cache_seconds
        ):
            free_quote = Decimal(str(source["free_quote"]))
            ready = free_quote >= required_with_buffer
            result = {
                "healthy": True, "buy_ready": ready,
                "free_quote": str(free_quote),
                "required_quote": str(required),
                "required_with_buffer": str(required_with_buffer),
                "reason": "quote_budget_available" if ready else "insufficient_quote_budget",
                "observed_at": float(source["observed_at"]), "cached": True,
                "cache_age_seconds": source_age,
                "source": str(source.get("source", "account_reconciliation")),
            }
            self.state["quote_budget_observation"] = result
            return result

        try:
            balances = exchange.account_balances()
            free_quote = Decimal(str(balances.get("USDT", {}).get("free", "0")))
            self.state["quote_balance_source"] = {
                "free_quote": str(free_quote), "observed_at": observed,
                "source": "direct_preflight",
            }
            ready = free_quote >= required_with_buffer
            result = {
                "healthy": True, "buy_ready": ready,
                "free_quote": str(free_quote),
                "required_quote": str(required),
                "required_with_buffer": str(required_with_buffer),
                "reason": "quote_budget_available" if ready else "insufficient_quote_budget",
                "observed_at": observed, "cached": False,
            }
            self.state["quote_budget_observation"] = result
            return result
        except Exception as exc:
            age = observed - float(previous.get("observed_at") or 0)
            if (
                not force_refresh and previous.get("healthy")
                and previous.get("free_quote") is not None
                and 0 <= age <= cache_seconds
            ):
                free_quote = Decimal(str(previous["free_quote"]))
                ready = free_quote >= required_with_buffer
                return {
                    **previous, "buy_ready": ready,
                    "required_quote": str(required),
                    "required_with_buffer": str(required_with_buffer),
                    "reason": "quote_budget_cached_after_transient_read_error",
                    "cached": True, "cache_age_seconds": age,
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            result = {
                "healthy": False, "buy_ready": False, "free_quote": None,
                "required_quote": str(required),
                "required_with_buffer": str(required_with_buffer),
                "reason": "quote_balance_temporarily_unavailable",
                "observed_at": observed, "cached": False,
                "last_error": f"{type(exc).__name__}: {exc}",
            }
            self.state["quote_budget_observation"] = result
            return result

    def _reentry_quote_requirement(
        self, current_bot: str, v22_contract: Dict[str, Any]
    ) -> Decimal:
        """Cash needed to rebuild pending base while preserving BUY budgets."""
        risk_on = {
            spec.bot_name for pair, spec in LIVE_PAIRS.items()
            if v22_contract.get("pairs", {}).get(pair, {}).get("buy_enabled")
        }
        pending = set()
        for bot_name in risk_on:
            phase = normalize_state(
                self.state.get("bots", {}).get(bot_name, {}).get("recovery")
            )["phase"]
            if phase == REENTRY:
                pending.add(bot_name)
        pending.add(current_bot)
        # Every Risk-On bot retains one BUY-side budget. Every pending bot
        # additionally needs one side budget to rebuild base inventory.
        return side_budget() * Decimal(len(risk_on) + len(pending & risk_on))

    def _apply_aggregate_gates(
        self, snapshots: Dict[str, Dict[str, Any]], *, risk_actions_enabled: bool,
    ) -> None:
        macro = self._macro_gate()
        v21 = self._v21_gate()
        previous_aggregate = self.state.get("gate_aggregate", {})
        previous_macro = previous_aggregate.get("macro", {})
        previous_capital = previous_aggregate.get("capital", {})
        previous_bots = previous_aggregate.get("bots", {})
        candidate_buy_bots = []
        for bot_name, snapshot in snapshots.items():
            bot_state = self.state.get("bots", {}).get(bot_name, {})
            if bot_state.get("tripped"):
                continue
            technical = v21["pairs"][str(snapshot["pair"])]
            recovery = normalize_state(bot_state.get("recovery"))
            if (
                recovery["phase"] == ACTIVE and macro["buy_enabled"]
                and technical["buy_enabled"]
            ):
                candidate_buy_bots.append(bot_name)
        capital = self._quote_budget_status(
            side_budget() * Decimal(len(candidate_buy_bots))
        )
        # The capital observation is intentionally advisory.  Keep the raw
        # readiness result in the contract for reports and alerts, while the
        # aggregate permission remains the AND of actual risk mechanisms.
        capital["mode"] = "alert_only"
        capital["enforced"] = False
        aggregate = {"macro": macro, "v22": v21, "capital": capital, "bots": {}}
        capital_mode_changed = bool(
            previous_capital.get("mode") != "alert_only"
            or previous_capital.get("enforced") is not False
        )
        if capital_mode_changed or (
            previous_capital.get("buy_ready") is not None
            and previous_capital.get("buy_ready") != capital.get("buy_ready")
        ):
            self._audit(
                "capital_budget_gate_transition",
                buy_enabled=bool(capital["buy_ready"]),
                reason=capital["reason"],
                free_quote=capital.get("free_quote"),
                required_quote=capital.get("required_quote"),
                action=("capital_alert_cleared_no_trade_change"
                        if capital["buy_ready"] else
                        "capital_alert_only_no_trade_block"),
                enforcement_mode="alert_only",
                trading_permissions_changed=False,
            )
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
                "capital_gate": capital["reason"],
                "capital_gate_mode": "alert_only",
                "free_quote": capital.get("free_quote"),
                "required_quote": capital.get("required_quote"),
                "sell_stop_event": dict(
                    bot_state.get("sell_stop_recovery_event") or {}
                ),
            }
            aggregate["bots"][bot_name] = {
                "pair": pair, "buy_enabled": buy_enabled,
                "sell_enabled": sell_enabled, "reasons": reasons,
                "fomc_buy_enabled": bool(macro["buy_enabled"]),
                "fomc_sell_enabled": bool(macro["sell_enabled"]),
                "v22_buy_enabled": bool(technical["buy_enabled"]),
                "v22_event_id": technical.get("event_id"),
                "capital_buy_ready": bool(capital["buy_ready"]),
                "capital_gate_enforced": False,
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
            actual_buy = controller_result.get("macro_buy_enabled")
            actual_sell = controller_result.get("macro_sell_enabled")
            controller_applied = bool(
                risk_actions_enabled and not controller_error
                and isinstance(actual_buy, bool) and isinstance(actual_sell, bool)
                and actual_buy == buy_enabled and actual_sell == sell_enabled
            )
            aggregate["bots"][bot_name].update({
                "controller_update_status": controller_result.get("status", "unknown"),
                "controller_actual_buy_enabled": actual_buy,
                "controller_actual_sell_enabled": actual_sell,
                "long_only_enabled": bool(controller_result.get("long_only_enabled", False)),
                "controller_applied": controller_applied,
                "controller_update_error": controller_error,
            })
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

    def _market_telemetry(self, pair: str) -> Dict[str, float]:
        symbol = pair.replace("-", "")
        read_client = self._public_reads()
        book_value = read_client.request(
            "GET", "/api/v3/ticker/bookTicker",
            params={"symbol": symbol},
            timeout=15,
        )
        bid = float(book_value["bidPrice"])
        ask = float(book_value["askPrice"])
        mid = (bid + ask) / 2
        kline_values = read_client.request(
            "GET", "/api/v3/klines",
            params={"symbol": symbol, "interval": "1m", "limit": 31},
            timeout=15,
        )
        closes = [float(item[4]) for item in kline_values]
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
            # Third-asset fees remain explicit in the audit response and are
            # intentionally not guessed without an independent conversion.
        return {
            "base_delta": str(base_delta),
            "quote_cashflow": str(quote_cashflow),
            "fee_quote": str(fee_quote),
            "fee_details": fee_details,
        }

    def _flatten(self, snapshot: Dict[str, Any], bot_name: str = "") -> Dict[str, Any]:
        pair = snapshot["pair"]
        owned = (
            self._owned_base(bot_name, snapshot)
            if bot_name else max(Decimal(snapshot["net_base"]), Decimal("0"))
        )
        step_size, minimum_notional = self._lot_filter(pair)
        mark_price = Decimal(snapshot["mark_price"])
        amount = (owned / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
        if amount <= 0 or amount * mark_price < minimum_notional:
            return {
                "status": "dust", "side": "SELL", "amount": str(amount),
                "notional": str(amount * mark_price), "remaining_base": str(owned),
                "exit_complete": True,
            }
        if self.emergency_exchange is None:
            raise RuntimeError("independent Binance emergency client is unavailable")
        base_asset = pair.split("-", 1)[0]
        digest = canonical_sha256({
            "bot": bot_name, "pair": pair, "amount": str(amount),
            "triggered_at": getattr(self, "state", {}).get("bots", {}).get(bot_name, {}).get("tripped_at"),
        })
        client_order_id = f"inv-{digest[:24]}"
        ledger = getattr(self, "inventory_ledger", None)
        lease_holder = f"dca-live-guard:{bot_name}"
        lease = ledger.lease(base_asset, lease_holder, ttl_seconds=45) if ledger else nullcontext()
        job_id = digest
        with lease:
            if ledger:
                balances = self.emergency_exchange.account_balances()
                total = balances.get(base_asset, {}).get("total", Decimal("0"))
                free = balances.get(base_asset, {}).get("free", Decimal("0"))
                if owned > total + step_size:
                    raise RuntimeError(
                        f"DCA owned {base_asset} {owned} exceeds exchange total {total}"
                    )
                if free + step_size < amount:
                    raise RuntimeError(
                        f"DCA owned {base_asset} is locked or not currently sellable"
                    )
                ledger.assert_exit_allowed(
                    asset=base_asset, exchange_total=total,
                    owner_key=f"dca:{bot_name}", requested_quantity=amount,
                    tolerance=step_size,
                )
                job = ledger.start_job(
                    job_id=job_id, asset=base_asset, scope=f"dca:{bot_name}",
                    pair=pair, requested_quantity=amount,
                    client_order_id=client_order_id,
                )
                if job.get("status") == "COMPLETED":
                    if not ledger.completed_job_verified(job):
                        raise RuntimeError(
                            f"completed DCA liquidation job {job_id} lacks verification"
                        )
                    executed = Decimal(str(job.get("executed_quantity") or "0"))
                    remaining = max(owned - executed, Decimal("0"))
                    bot = self.state["bots"].setdefault(bot_name, {})
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
                            "executed_qty": str(executed),
                            "cummulative_quote_qty": str(job.get("quote_quantity") or "0"),
                            "base_delta": str(-executed),
                            "quote_cashflow": str(job.get("quote_quantity") or "0"),
                            "fee_quote": str(job.get("fee_quote") or "0"),
                            "recovered_from_shared_ledger": True,
                        })
                        self._save()
                    return {
                        "status": "filled", "side": "SELL", "amount": str(amount),
                        "order_id": str(job.get("exchange_order_id") or ""),
                        "executed_qty": str(executed), "remaining_base": str(remaining),
                        "exit_complete": remaining * mark_price < minimum_notional,
                        "client_order_id": client_order_id,
                    }
            try:
                if ledger:
                    response = execute_market_liquidation(
                        exchange=self.emergency_exchange, ledger=ledger,
                        job_id=job_id, pair=pair, side="SELL",
                        target_quantity=amount, client_order_id=client_order_id,
                        step_size=step_size, minimum_notional=minimum_notional,
                        mark_price=mark_price, lease_asset=base_asset,
                        lease_holder=lease_holder,
                    )
                    verification = verify_market_liquidation(
                        exchange=self.emergency_exchange, pair=pair,
                        response=response, requested_quantity=amount,
                        before_total=total, step_size=step_size,
                        minimum_notional=minimum_notional,
                        mark_price=mark_price,
                        ledger=ledger, lease_asset=base_asset,
                        lease_holder=lease_holder,
                    )
                else:
                    response = self._idempotent_market_order(
                        pair, "SELL", amount, client_order_id
                    )
                    verification = None
                metrics = self._emergency_fill_metrics(pair, "SELL", response)
                if ledger:
                    ledger.finish_job(
                        job_id, status="COMPLETED",
                        exchange_order_id=str(response.get("orderId", "")),
                        executed_quantity=response.get("executedQty", "0"),
                        quote_quantity=response.get("cummulativeQuoteQty", "0"),
                        fee_quote=metrics.get("fee_quote", "0"),
                        fee_details=metrics.get("fee_details", []),
                        verification=verification,
                    )
            except Exception as exc:
                if ledger:
                    ledger.finish_job(job_id, status="FAILED", error=repr(exc))
                raise
        executed = Decimal(str(response.get("executedQty", "0")))
        remaining = max(owned - executed, Decimal("0"))
        if bot_name:
            adjustment = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "pair": pair,
                "side": "SELL",
                "client_order_id": client_order_id,
                "order_id": str(response.get("orderId", "")),
                "executed_qty": str(response.get("executedQty", "0")),
                "cummulative_quote_qty": str(
                    response.get("cummulativeQuoteQty", "0")
                ),
                **metrics,
            }
            adjustments = self.state["bots"].setdefault(bot_name, {}).setdefault(
                "emergency_adjustments", []
            )
            if not any(
                str(row.get("client_order_id", "")) == client_order_id
                for row in adjustments
            ):
                adjustments.append(adjustment)
            self._save()
        return {
            "status": "filled",
            "side": "SELL",
            "amount": str(amount),
            "order_id": str(response.get("orderId", "")),
            "executed_qty": str(response.get("executedQty", "0")),
            "remaining_base": str(remaining),
            "exit_complete": bool(
                remaining <= 0 or remaining * mark_price < minimum_notional
            ),
            "client_order_id": client_order_id,
            "verification": verification,
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
        adjustments = Decimal("0")
        for row in bot.get("emergency_adjustments", []):
            if str(row.get("pair", "")) != str(snapshot["pair"]):
                continue
            if row.get("base_delta") is not None:
                adjustments += Decimal(str(row["base_delta"]))
            else:
                executed = Decimal(str(row.get("executed_qty", "0")))
                adjustments += executed if str(row.get("side", "")).upper() == "BUY" else -executed
        return max(
            target + Decimal(str(snapshot["net_base"])) + adjustments,
            Decimal("0"),
        )

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

        # A restarted bot can restore connector-tracked orders from its
        # database even though the persisted aggregate controller gate is
        # already closed.  COOLDOWN/REENTRY are quote-only phases: actively
        # remove such residual orders instead of waiting forever for
        # no_runtime_risk to become true.
        residual_orders = self.emergency_exchange.open_orders(pair)
        if residual_orders:
            self.emergency_exchange.cancel_all_orders(pair)
            self._audit(
                "recoverable_residual_orders_cancelled",
                bot=bot_name,
                pair=pair,
                phase=state["phase"],
                order_count=len(residual_orders),
            )
            bot["recovery"] = state
            return
        counts = self._executor_counts(Path(snapshot["database"]))
        no_runtime_risk = all(counts[key] == 0 for key in counts)
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
            v22_contract = self.state.get("gate_aggregate", {}).get("v22", {})
            required_quote = self._reentry_quote_requirement(bot_name, v22_contract)
            capital = self._quote_budget_status(
                required_quote, now=now, force_refresh=True,
            )
            if not capital["buy_ready"]:
                previous_reason = str(state.get("reentry_block_reason", ""))
                state["reentry_allowed"] = False
                state["reentry_block_reason"] = str(capital["reason"])
                state["reentry_capital"] = capital
                if previous_reason != state["reentry_block_reason"]:
                    self._audit(
                        "recoverable_reentry_capital_wait",
                        bot=bot_name, pair=pair, recovery=state,
                        free_quote=capital.get("free_quote"),
                        required_quote=capital.get("required_quote"),
                        action="wait_without_liquidation_or_latch",
                    )
                bot["recovery"] = state
                return
            state.pop("reentry_block_reason", None)
            state.pop("reentry_capital", None)
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
        if bot_state.get("manual_exit_required"):
            return
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
            if not bool(flatten_response.get("exit_complete")):
                raise RuntimeError(
                    f"DCA inventory exit remains incomplete: {flatten_response}"
                )
            reconciled_snapshot = self._snapshot(bot_name, pair)
            recovery = normalize_state(bot_state.get("recovery"))
            recovery.update({
                "exit_completed_at": time.time(),
                "remaining_base": {
                    pair: str(flatten_response.get("remaining_base", "0"))
                },
            })
            bot_state.update({
                "action_complete": True,
                "exit_status": "complete",
                "stop_response": stop_response,
                "flatten_response": flatten_response,
                "post_stop_snapshot": post_stop_snapshot,
                "reconciled_snapshot": reconciled_snapshot,
                "recovery": recovery,
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
        snapshots: Dict[str, Dict[str, Any]] = {}
        now = time.time()
        self._observe_v22_contract(now)
        self._reconcile_account_inventory(risk_actions_enabled=risk_actions_enabled)
        status_text = json.dumps(self.api.status(), ensure_ascii=True)
        mechanisms = getattr(self, "mechanisms", {})
        for pair, spec in LIVE_PAIRS.items():
            # A stopped but deployed bot still has a live controller YAML that
            # must be fail-closed before the process is restarted.  Requiring
            # it to appear in MQTT status created a startup race where stale
            # permissive gates were loaded before the Guard could correct them.
            if spec.bot_name not in status_text and self._database(spec.bot_name) is None:
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
                if (
                    not bot_state.get("action_complete")
                    and not bot_state.get("manual_exit_required")
                ):
                    self._trip(spec.bot_name, bot_state.get("trip_reason", "retry"), snapshot)
                continue
            stop_event, cursor_migrated = self._consume_position_stop_event(
                bot_state, snapshot
            )
            if cursor_migrated:
                self._audit(
                    "POSITION_STOP_CURSOR_BASELINED",
                    bot=spec.bot_name,
                    pair=pair,
                    cursor_schema=STOP_LOSS_EVENT_CURSOR_SCHEMA,
                    executor_id=bot_state.get("last_position_stop_event_id"),
                    event_at=bot_state.get("last_position_stop_seen", 0),
                )
            if stop_event is not None and stop_event.get("side") == "sell":
                bot_state["sell_stop_recovery_event"] = dict(stop_event)
            if (
                risk_actions_enabled
                and mechanisms.get("position_protection", True)
                and stop_event is not None
                and bot_state["recovery"]["phase"] == ACTIVE
            ):
                self._trigger_recoverable(
                    spec.bot_name, snapshot, mechanism="position_protection",
                    scope="position", trigger_value="5%",
                    reason=(
                        "executor_stop_loss:"
                        f"{stop_event.get('executor_id') or 'unknown'}"
                    ),
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
        if macro_contract["healthy"]:
            self._clear_integrity_failure("macro_contract", now)
        if v21_contract["healthy"]:
            self._clear_integrity_failure("v22_contract", now)
        macro_requires_latch = bool(
            not macro_contract["healthy"]
            and self._integrity_failure_requires_latch(
                "macro_contract", str(macro_contract["reason"]), now,
            )
        )
        v22_requires_latch = bool(
            macro_contract["healthy"]
            and not v21_contract["healthy"]
            and self._integrity_failure_requires_latch(
                "v22_contract", str(v21_contract["reason"]), now,
            )
        )
        if risk_actions_enabled:
            for bot_name, snapshot in snapshots.items():
                if not macro_contract["healthy"]:
                    reason = str(macro_contract["reason"])
                    if macro_requires_latch:
                        self._latch_integrity_failure(bot_name, snapshot, reason)
                elif not v21_contract["healthy"]:
                    reason = str(v21_contract["reason"])
                    if v22_requires_latch:
                        self._latch_integrity_failure(bot_name, snapshot, reason)
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
        self._record_read_retry_events()
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
                self._scan_runtime_logs()
                self.runtime_errors.recovered(
                    "guard_cycle",
                    trading_status="Guard healthy；existing aggregate gates apply normally",
                )
            except Exception as exc:
                LOG.error("Guard cycle failed: %s", exc)
                first = float(self.state.get("first_failure_at") or time.time())
                self.runtime_errors.failure(
                    "guard_cycle", exc,
                    trading_impact=(
                        "本周期监控失败，Guard 正在自动重试；达到连续失败阈值前不新增限制，"
                        "达到阈值后由既有 Fail-Closed 清仓与锁存流程接管。"
                    ),
                    severity="warning", action="retry_then_fail_closed_on_threshold",
                    details={"fail_closed_after_seconds": self.fail_closed_seconds},
                    now=first,
                )
                if not observe_only:
                    self.fail_closed(exc)
            time.sleep(self.interval)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--scenario-inventory-cycles" in sys.argv[1:]:
        if not scenario_mode():
            raise RuntimeError("scenario inventory loop requires GUARD_SCENARIO_MODE=true")
        index = sys.argv.index("--scenario-inventory-cycles")
        cycles = int(sys.argv[index + 1])
        sleep_seconds = float(os.getenv("GUARD_SCENARIO_CYCLE_SECONDS", "15.1"))
        guard = Guard()
        reports = []
        for number in range(cycles):
            reports.append(guard._reconcile_account_inventory(risk_actions_enabled=True))
            if number + 1 < cycles:
                time.sleep(sleep_seconds)
        print(json.dumps({
            "scenario_id": os.environ["GUARD_SCENARIO_ID"],
            "cycles": cycles, "final": reports[-1],
        }, default=str), flush=True)
        return 0
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
