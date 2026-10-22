"""Shared emergency exchange and Docker clients, independent of Guard orchestration."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import socket
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlencode

import requests

try:
    from runtime_endpoints import OFFICIAL_BINANCE_API, guarded_endpoint, scenario_mode
except ModuleNotFoundError:
    from live_guard.runtime_endpoints import OFFICIAL_BINANCE_API, guarded_endpoint, scenario_mode
try:
    from get_only_read_client import read_retry_session
except ModuleNotFoundError:
    from scripts.get_only_read_client import read_retry_session


BINANCE_API = OFFICIAL_BINANCE_API


def _read_retry_session() -> requests.Session:
    return read_retry_session(retry_total=2)


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
        if scenario_mode():
            self.session.trust_env = False
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
                "detected spotBNBBurn=true; Binance Spot BNB fee deduction must "
                "be disabled (spotBNBBurn=false)"
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

    def order_by_id(self, pair: str, order_id: str) -> dict:
        value = self._signed("GET", "/api/v3/order", {
            "symbol": self.symbol(pair), "orderId": order_id,
        })
        if Decimal(str(value.get("executedQty", "0"))) > 0:
            value["fills"] = self.order_trades(pair, str(value["orderId"]))
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
            **({"tradeId": str(row.get("id", row.get("tradeId")))}
               if row.get("id", row.get("tradeId")) is not None else {}),
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
