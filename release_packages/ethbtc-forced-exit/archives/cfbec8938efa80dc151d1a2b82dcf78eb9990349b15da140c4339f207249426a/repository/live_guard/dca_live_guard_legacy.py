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
    trade_pnl_from_rows,
)


LOG = logging.getLogger("dca-live-guard")
BINANCE_API = "https://api.binance.com"


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
        self.telemetry_path = self.state_dir / "dca_macro_telemetry.json"
        self.interval = max(2, int(os.getenv("DCA_LIVE_GUARD_INTERVAL", "10")))
        self.fail_closed_seconds = max(20, int(os.getenv("DCA_LIVE_FAIL_CLOSED_SECONDS", "60")))
        self.roc_buy_guard_enabled = (
            os.getenv("DCA_ROC_BUY_GUARD_ENABLED", "true").lower() == "true"
        )
        self.roc_buy_guard_refresh_seconds = max(
            30, int(os.getenv("DCA_ROC_BUY_GUARD_REFRESH_SECONDS", "60"))
        )
        self.roc_trigger_pct = float(
            os.getenv("DCA_ROC_BUY_GUARD_TRIGGER_PCT", "-8")
        )
        self.sqz_trigger_pct = float(
            os.getenv("DCA_SQZ_BUY_GUARD_TRIGGER_PCT", "-3")
        )
        if self.roc_trigger_pct >= 0 or self.sqz_trigger_pct >= 0:
            raise ValueError("ROC and SQZMOM BUY guard trigger thresholds must be negative")
        self._roc_signal_cache: Dict[str, Any] = {}
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

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"version": 1, "armed": True, "bots": {}, "last_success_at": 0, "created_at": time.time()}

    def _save(self) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def _audit(self, event: str, **details: Any) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        with self.audit_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        LOG.warning("%s %s", event, json.dumps(details, ensure_ascii=False, default=str))

    def _notify(self, message: str) -> None:
        token = os.getenv("TELEGRAM_TOKEN", "").strip()
        chat_id = os.getenv("ADMIN_USER_ID", "").strip()
        if not token or not chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message},
                timeout=15,
            ).raise_for_status()
        except Exception as exc:
            LOG.error("Telegram alert failed: %s", exc)

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
        return {
            "pair": pair,
            "database": str(database),
            "mark_price": str(price),
            **{key: str(value) for key, value in metrics.items()},
            "updated_at": observed_at,
            "observed_at": observed_at,
            "database_event_at": database_event_at,
            "database_event_age_seconds": max(0.0, observed_at - database_event_at),
        }

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

    @staticmethod
    def _linreg_endpoint(values: list[float]) -> float:
        length = len(values)
        if length < 2:
            raise ValueError("linear regression requires at least two values")
        x_mean = (length - 1) / 2
        y_mean = sum(values) / length
        denominator = sum((index - x_mean) ** 2 for index in range(length))
        slope = sum(
            (index - x_mean) * (value - y_mean)
            for index, value in enumerate(values)
        ) / denominator
        intercept = y_mean - slope * x_mean
        return intercept + slope * (length - 1)

    @classmethod
    def _roc_sqz_signal_from_klines(
        cls,
        klines: list[list[Any]],
        *,
        roc_length: int = 12,
        sqz_length: int = 20,
    ) -> Dict[str, Any]:
        if len(klines) < sqz_length * 2:
            raise ValueError(
                f"at least {sqz_length * 2} closed klines are required"
            )
        highs = [float(item[2]) for item in klines]
        lows = [float(item[3]) for item in klines]
        closes = [float(item[4]) for item in klines]
        if any(value <= 0 for value in closes):
            raise ValueError("kline closes must be positive")
        sqz_sources = []
        for index in range(sqz_length - 1, len(klines)):
            start = index - sqz_length + 1
            highest = max(highs[start:index + 1])
            lowest = min(lows[start:index + 1])
            close_sma = sum(closes[start:index + 1]) / sqz_length
            midpoint = ((highest + lowest) / 2 + close_sma) / 2
            sqz_sources.append(closes[index] - midpoint)
        current_sqz = cls._linreg_endpoint(sqz_sources[-sqz_length:])
        previous_sqz = cls._linreg_endpoint(sqz_sources[-sqz_length - 1:-1])
        roc_pct = (closes[-1] / closes[-1 - roc_length] - 1) * 100
        sqz_pct = current_sqz / closes[-1] * 100
        return {
            "bar_open_time": int(klines[-1][0]),
            "bar_close_time": int(klines[-1][6]),
            "close": closes[-1],
            "roc_48h_pct": roc_pct,
            "sqzmom": current_sqz,
            "sqzmom_previous": previous_sqz,
            "sqzmom_pct": sqz_pct,
            "sqzmom_red": current_sqz < 0 and current_sqz < previous_sqz,
            "sqzmom_green": current_sqz > 0,
        }

    def _roc_buy_signal(self) -> Dict[str, Any]:
        now = time.time()
        cached_at = float(self._roc_signal_cache.get("cached_at", 0))
        if now - cached_at < self.roc_buy_guard_refresh_seconds:
            if "error" in self._roc_signal_cache:
                raise RuntimeError(self._roc_signal_cache["error"])
            return dict(self._roc_signal_cache["signal"])
        try:
            server_time = requests.get(f"{BINANCE_API}/api/v3/time", timeout=15)
            server_time.raise_for_status()
            server_now_ms = int(server_time.json()["serverTime"])
            response = requests.get(
                f"{BINANCE_API}/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "4h", "limit": 64},
                timeout=15,
            )
            response.raise_for_status()
            closed = [
                item for item in response.json()
                if isinstance(item, list) and len(item) > 6 and int(item[6]) < server_now_ms
            ]
            signal = self._roc_sqz_signal_from_klines(closed)
            signal["trigger"] = bool(
                signal["roc_48h_pct"] <= self.roc_trigger_pct
                and signal["sqzmom_pct"] <= self.sqz_trigger_pct
                and signal["sqzmom_red"]
            )
            signal["recover"] = bool(signal["sqzmom_green"])
            self._roc_signal_cache = {"cached_at": now, "signal": signal}
            return dict(signal)
        except Exception as exc:
            self._roc_signal_cache = {"cached_at": now, "error": repr(exc)}
            raise

    def _set_roc_buy_gate(
        self,
        bot_name: str,
        snapshot: Dict[str, Any],
        *,
        enabled: bool,
        decision_id: str,
    ) -> Dict[str, Any]:
        database = Path(snapshot["database"])
        controller_name, profile = self._controller_profile(database)
        if not controller_name or not profile:
            raise RuntimeError(f"controller config is unavailable for {bot_name}")
        actual = bool(profile.get("macro_buy_enabled", True))
        current_decision = str(profile.get("macro_decision_id", ""))
        if actual == enabled:
            return {
                "status": "unchanged",
                "macro_buy_enabled": actual,
                "macro_decision_id": current_decision,
            }
        if enabled and not current_decision.startswith("roc-buy-guard:"):
            return {
                "status": "preserved_external_gate",
                "macro_buy_enabled": actual,
                "macro_decision_id": current_decision,
            }
        profile["macro_buy_enabled"] = enabled
        # Never alter the SELL gate: it may be controlled by a different lease.
        profile["macro_decision_id"] = decision_id
        response = self.api.update_controller(
            bot_name, controller_name, profile
        )
        return {
            "status": "applied",
            "macro_buy_enabled": enabled,
            "macro_sell_enabled": bool(profile.get("macro_sell_enabled", True)),
            "macro_decision_id": decision_id,
            "response": response,
        }

    def _apply_roc_buy_guard(
        self,
        snapshots: Dict[str, Dict[str, Any]],
        *,
        risk_actions_enabled: bool,
    ) -> None:
        if not self.roc_buy_guard_enabled:
            return
        guard_state = self.state.setdefault(
            "roc_buy_guard",
            {"active": False, "controlled_bots": []},
        )
        try:
            signal = self._roc_buy_signal()
        except Exception as exc:
            guard_state["last_error"] = repr(exc)
            guard_state["last_error_at"] = time.time()
            return
        guard_state.pop("last_error", None)
        guard_state["latest"] = signal
        was_active = bool(guard_state.get("active", False))
        active = was_active
        transition = ""
        if not active and signal["trigger"]:
            active = True
            transition = "risk_off"
            guard_state["triggered_at"] = time.time()
        elif active and signal["recover"]:
            active = False
            transition = "recovered"
            guard_state["recovered_at"] = time.time()
        guard_state["active"] = active
        if transition:
            self._audit(
                f"roc_buy_guard_{transition}",
                signal=signal,
                thresholds={
                    "roc_48h_pct": self.roc_trigger_pct,
                    "sqzmom_pct": self.sqz_trigger_pct,
                },
            )
            if risk_actions_enabled:
                self._notify(
                    "DCA ROC BUY GUARD "
                    + ("RISK OFF: BUY stopped" if active else "RECOVERED: BUY enabled")
                    + f"\nROC48={signal['roc_48h_pct']:.2f}% "
                    + f"SQZMOM={signal['sqzmom_pct']:.2f}%"
                )
        if not risk_actions_enabled:
            return
        controlled = set(guard_state.get("controlled_bots", []))
        desired_enabled = not active
        decision_id = (
            f"roc-buy-guard:{'resume' if desired_enabled else 'risk-off'}:"
            f"{signal['bar_close_time']}"
        )
        for bot_name, snapshot in snapshots.items():
            bot_state = self.state.get("bots", {}).get(bot_name, {})
            if bot_state.get("tripped"):
                continue
            if desired_enabled and bot_name not in controlled:
                continue
            try:
                result = self._set_roc_buy_gate(
                    bot_name,
                    snapshot,
                    enabled=desired_enabled,
                    decision_id=decision_id,
                )
                if not desired_enabled and (
                    result["status"] == "applied"
                    or str(result.get("macro_decision_id", "")).startswith(
                        "roc-buy-guard:"
                    )
                ):
                    controlled.add(bot_name)
                if desired_enabled and result["status"] in {
                    "applied", "unchanged", "preserved_external_gate"
                }:
                    controlled.discard(bot_name)
                if result["status"] != "unchanged":
                    self._audit(
                        "roc_buy_guard_gate_update",
                        bot=bot_name,
                        desired_buy_enabled=desired_enabled,
                        result=result,
                    )
            except Exception as exc:
                self._audit(
                    "roc_buy_guard_gate_update_failed",
                    bot=bot_name,
                    desired_buy_enabled=desired_enabled,
                    error=repr(exc),
                )
        guard_state["controlled_bots"] = sorted(controlled)

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

    def _trip(self, bot_name: str, reason: str, snapshot: Optional[Dict[str, Any]]) -> None:
        bot_state = self.state["bots"].setdefault(bot_name, {})
        if bot_state.get("action_complete"):
            return
        bot_state.update({"tripped": True, "trip_reason": reason, "tripped_at": time.time()})
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
        for pair, spec in LIVE_PAIRS.items():
            if spec.bot_name not in status_text:
                continue
            bot_state = self.state["bots"].setdefault(spec.bot_name, {
                "pair": pair,
                "started_at": now,
                "tripped": False,
                "action_complete": False,
            })
            snapshot = self._snapshot(spec.bot_name, pair)
            if snapshot is None:
                continue
            snapshots[spec.bot_name] = snapshot
            bot_state["latest"] = snapshot
            if risk_actions_enabled and bot_state.get("tripped"):
                if not bot_state.get("action_complete"):
                    self._trip(spec.bot_name, bot_state.get("trip_reason", "retry"), snapshot)
                continue
            pnl = Decimal(snapshot["pnl_quote"])
            if risk_actions_enabled and pnl <= -SINGLE_BOT_LOSS_LIMIT:
                self._trip(spec.bot_name, f"single bot PnL {pnl} <= -{SINGLE_BOT_LOSS_LIMIT} USDT", snapshot)

        combined_pnl = sum((Decimal(item["pnl_quote"]) for item in snapshots.values()), Decimal("0"))
        self.state["combined_pnl_quote"] = str(combined_pnl)
        if (
            risk_actions_enabled
            and len(snapshots) == len(LIVE_PAIRS)
            and combined_pnl <= -COMBINED_LOSS_LIMIT
        ):
            for bot_name, snapshot in snapshots.items():
                self._trip(bot_name, f"combined PnL {combined_pnl} <= -{COMBINED_LOSS_LIMIT} USDT", snapshot)
        self._apply_roc_buy_guard(
            snapshots, risk_actions_enabled=risk_actions_enabled
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
