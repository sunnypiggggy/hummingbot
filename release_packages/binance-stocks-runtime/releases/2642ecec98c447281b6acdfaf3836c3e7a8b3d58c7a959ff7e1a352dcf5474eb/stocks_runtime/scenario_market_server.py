"""Stateful Binance Stocks market-data simulator for isolated container smoke tests.

It mirrors the official REST/WebSocket payload shapes. Economic endpoints are
present only as tripwires: every call is counted and rejected.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict

from aiohttp import WSMsgType, web


@dataclass
class Market:
    symbol: str = "AAPL"
    bid: Decimal = Decimal("199.90")
    ask: Decimal = Decimal("200.00")
    bid_size: Decimal = Decimal("10")
    ask_size: Decimal = Decimal("10")
    phase: str = "MARKET_OPEN"
    trading_status: str = "TRADING"
    tradability: str = "BOTH"
    trading_date: str = "2026-08-21"


class ScenarioMarketServer:
    def __init__(self):
        if os.getenv("BINANCE_STOCKS_SCENARIO_MODE", "false").lower() != "true":
            raise RuntimeError("scenario market server requires BINANCE_STOCKS_SCENARIO_MODE=true")
        self.api_key = os.getenv("BINANCE_STOCKS_SCENARIO_API_KEY", "paper-market-key")
        self.market = Market()
        self.economic_requests: Dict[str, int] = {"place": 0, "cancel": 0, "other": 0}
        self.app = web.Application()
        self.app.router.add_get("/api/v3/time", self.server_time)
        self.app.router.add_get("/sapi/v1/equity/market/exchangeInfo", self.exchange_info)
        self.app.router.add_get("/sapi/v1/equity/market/quote", self.quote)
        self.app.router.add_post("/sapi/v1/equity/order/place", self.reject_place)
        self.app.router.add_post("/sapi/v1/equity/order/cancel", self.reject_cancel)
        self.app.router.add_route("*", "/sapi/v1/equity/{tail:.*}", self.reject_other_economic)
        self.app.router.add_post("/scenario/quote", self.set_quote)
        self.app.router.add_post("/scenario/market", self.set_market)
        self.app.router.add_get("/scenario/stats", self.stats)
        self.app.router.add_get("/equity/{tail:.*}", self.websocket)

    def _market_key(self, request: web.Request):
        if request.headers.get("X-MBX-APIKEY") != self.api_key:
            raise web.HTTPUnauthorized(text="market-data API key required")

    async def server_time(self, _request):
        return web.json_response({"serverTime": int(time.time() * 1000)})

    async def exchange_info(self, request):
        self._market_key(request)
        m = self.market
        return web.json_response({"data": {"tradingDate": m.trading_date, "symbols": [{
            "symbol": m.symbol, "status": m.trading_status, "fractionalTrading": True,
            "priceIncrement": "0.01", "quantityIncrement": "0.0001",
            "minQuantity": "0.0001", "minOrderValue": "5", "tokenized": False,
        }]}})

    def quote_event(self):
        m = self.market
        return {
            "e": "quote", "s": m.symbol, "bp": str(m.bid), "ap": str(m.ask),
            "bs": str(m.bid_size), "as": str(m.ask_size), "T": int(time.time() * 1000),
        }

    async def quote(self, request):
        self._market_key(request)
        return web.json_response({"data": self.quote_event()})

    async def set_quote(self, request):
        payload = await request.json()
        m = self.market
        m.symbol = str(payload.get("symbol", m.symbol)).upper()
        m.bid = Decimal(str(payload["bid"]))
        m.ask = Decimal(str(payload["ask"]))
        m.bid_size = Decimal(str(payload.get("bid_size", "10")))
        m.ask_size = Decimal(str(payload.get("ask_size", "10")))
        if m.bid <= 0 or m.ask <= 0 or m.bid > m.ask or m.bid_size <= 0 or m.ask_size <= 0:
            raise web.HTTPBadRequest(text="invalid BBO")
        return web.json_response({"accepted": True, "quote": self.quote_event()})

    async def set_market(self, request):
        payload = await request.json()
        m = self.market
        m.phase = str(payload.get("phase", m.phase)).upper()
        m.trading_status = str(payload.get("trading_status", m.trading_status)).upper()
        m.tradability = str(payload.get("tradability", m.tradability)).upper()
        m.trading_date = str(payload.get("trading_date", m.trading_date))
        return web.json_response({"accepted": True, "phase": m.phase})

    async def _tripwire(self, key: str):
        self.economic_requests[key] += 1
        raise web.HTTPConflict(text="PAPER smoke received a forbidden economic request")

    async def reject_place(self, _request):
        return await self._tripwire("place")

    async def reject_cancel(self, _request):
        return await self._tripwire("cancel")

    async def reject_other_economic(self, _request):
        return await self._tripwire("other")

    async def stats(self, _request):
        return web.json_response({"economic_requests": self.economic_requests, "quote": self.quote_event()})

    async def websocket(self, request):
        websocket = web.WebSocketResponse(heartbeat=15)
        await websocket.prepare(request)
        try:
            while not websocket.closed:
                m = self.market
                events = [
                    ("calendar", {"e": "calendar", "phase": m.phase, "tradingDate": m.trading_date}),
                    (f"{m.symbol}@tradingStatus", {"e": "tradingStatus", "s": m.symbol, "status": m.trading_status}),
                    (f"{m.symbol}@tradability", {"e": "tradability", "s": m.symbol, "tradability": m.tradability}),
                    (f"{m.symbol}@quote", self.quote_event()),
                ]
                for stream, event in events:
                    await websocket.send_json({"stream": stream, "data": event})
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=0.25)
                    if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                        break
                except asyncio.TimeoutError:
                    pass
        finally:
            await websocket.close()
        return websocket


def main():
    server = ScenarioMarketServer()
    web.run_app(server.app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))


if __name__ == "__main__":
    main()
