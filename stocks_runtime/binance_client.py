from __future__ import annotations

import hashlib
import hmac
import time
from decimal import Decimal
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import aiohttp


class BinanceStocksReadClient:
    """Small read-only client for runtime reconciliation.

    Deliberately contains no place/cancel method. Economic actions must pass
    through ExecutorService and the managed connector.
    """

    def __init__(self, rest_url: str, api_key: str = "", api_secret: str = ""):
        self.rest_url = rest_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self._session: Optional[aiohttp.ClientSession] = None
        self._time_offset_ms = 0

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        try:
            response = await self._request("GET", "/api/v3/time")
            server_ms = int(response.get("serverTime", int(time.time() * 1000)))
            self._time_offset_ms = server_ms - int(time.time() * 1000)
        except Exception:
            self._time_offset_ms = 0

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False
    ) -> Any:
        if self._session is None:
            raise RuntimeError("Binance client is not started")
        query = dict(params or {})
        headers: Dict[str, str] = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise PermissionError("read-only Binance Stocks credentials are unavailable")
            query.update({"timestamp": int(time.time() * 1000) + self._time_offset_ms, "recvWindow": 5000})
            encoded = urlencode(query)
            query["signature"] = hmac.new(self.api_secret, encoded.encode(), hashlib.sha256).hexdigest()
            headers["X-MBX-APIKEY"] = self.api_key
        async with self._session.request(method, f"{self.rest_url}{path}", params=query, headers=headers) as response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                raise IOError(f"Binance Stocks {path} failed HTTP {response.status}: {payload}")
            if isinstance(payload, dict) and str(payload.get("status", "S")).upper() == "F":
                raise IOError(f"Binance Stocks {path} rejected request: {payload}")
            return payload

    @staticmethod
    def payload(response: Any) -> Any:
        if isinstance(response, dict):
            for key in ("data", "result"):
                if key in response and response[key] is not None:
                    return response[key]
        return response

    async def exchange_info(self) -> Any:
        return self.payload(await self._request("GET", "/sapi/v1/equity/market/exchangeInfo"))

    async def quote(self, symbol: str) -> Dict[str, Any]:
        return self.payload(await self._request(
            "GET", "/sapi/v1/equity/market/quote", {"symbol": symbol.upper()}
        ))

    async def open_orders(self, symbol: Optional[str] = None) -> Any:
        params = {"symbol": symbol.upper()} if symbol else {}
        return self.payload(await self._request(
            "GET", "/sapi/v1/equity/order/open-orders", params, signed=True
        ))

    async def order_history(self, symbol: Optional[str] = None, limit: int = 100) -> Any:
        params: Dict[str, Any] = {"limit": min(max(limit, 1), 500)}
        if symbol:
            params["symbol"] = symbol.upper()
        return self.payload(await self._request(
            "GET", "/sapi/v1/equity/order/history", params, signed=True
        ))

    async def trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> Any:
        params: Dict[str, Any] = {"limit": min(max(limit, 1), 500)}
        if symbol:
            params["symbol"] = symbol.upper()
        return self.payload(await self._request(
            "GET", "/sapi/v1/equity/trade/history", params, signed=True
        ))

    async def funding_usdc(self) -> tuple[Decimal, Decimal]:
        payload = self.payload(await self._request(
            "POST",
            "/sapi/v1/asset/get-funding-asset",
            {"asset": "USDC", "needBtcValuation": "false"},
            signed=True,
        ))
        rows = payload if isinstance(payload, list) else []
        row = next((item for item in rows if str(item.get("asset", "")).upper() == "USDC"), None)
        if row is None:
            return Decimal("0"), Decimal("0")
        available = Decimal(str(row.get("free", row.get("available", "0"))))
        total = available + Decimal(str(row.get("locked", row.get("freeze", "0"))))
        return total, available
