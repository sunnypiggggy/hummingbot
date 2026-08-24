import hashlib
import hmac
import time
from typing import Any, Dict
from urllib.parse import urlencode

from aiohttp import web


class BinanceStocksStatefulSimulator:
    """Small, stateful Stocks API used by connector integration tests.

    It validates the real RestAssistant request signature and persists economic
    orders by clientOrderId so retries can be tested without a real account.
    """

    def __init__(self, api_key: str = "key", secret: str = "secret"):
        self.api_key = api_key
        self.secret = secret
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.place_requests = 0
        self.runner = None
        self.site = None
        self.base_url = ""
        self.ws_url = ""
        self.app = web.Application()
        self.app.router.add_get("/api/v3/time", self.server_time)
        self.app.router.add_get("/sapi/v1/equity/market/exchangeInfo", self.exchange_info)
        self.app.router.add_get("/sapi/v1/equity/market/quote", self.quote)
        self.app.router.add_post("/sapi/v1/equity/order/place", self.place_order)
        self.app.router.add_post("/sapi/v1/equity/order/cancel", self.cancel_order)
        self.app.router.add_get("/sapi/v1/equity/order/detail", self.order_detail)
        self.app.router.add_post("/sapi/v1/equity/listenKey", self.listen_key)
        self.app.router.add_get("/equity/{tail:.*}", self.websocket)

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.ws_url = f"ws://127.0.0.1:{port}/equity"

    async def stop(self):
        if self.runner is not None:
            await self.runner.cleanup()

    async def server_time(self, request):
        return web.json_response({"serverTime": int(time.time() * 1000)})

    async def exchange_info(self, request):
        if request.headers.get("X-MBX-APIKEY") != self.api_key:
            raise web.HTTPUnauthorized(text="market data api key required")
        return web.json_response(
            {
                "data": {
                    "symbols": [
                        {
                            "symbol": "AAPL",
                            "status": "TRADING",
                            "fractionalTrading": True,
                            "priceIncrement": "0.01",
                            "quantityIncrement": "0.0001",
                            "minQuantity": "0.0001",
                            "minOrderValue": "5",
                        }
                    ]
                }
            }
        )

    async def quote(self, request):
        if request.headers.get("X-MBX-APIKEY") != self.api_key:
            raise web.HTTPUnauthorized(text="market data api key required")
        return web.json_response(
            {
                "data": {
                    "e": "quote",
                    "s": request.query.get("symbol", "AAPL"),
                    "bp": "199.90",
                    "bs": "10",
                    "ap": "200.00",
                    "as": "10",
                    "T": int(time.time() * 1000),
                }
            }
        )

    async def _signed_params(self, request) -> Dict[str, Any]:
        if request.headers.get("X-MBX-APIKEY") != self.api_key:
            raise web.HTTPUnauthorized(text="bad api key")
        if request.method == "GET":
            params = dict(request.query)
        else:
            try:
                params = await request.json()
            except Exception:
                params = dict(await request.post())
        provided = params.pop("signature", None)
        expected = hmac.new(self.secret.encode(), urlencode(params).encode(), hashlib.sha256).hexdigest()
        if provided != expected:
            raise web.HTTPUnauthorized(text=f"bad signature expected={expected} got={provided}")
        return params

    async def place_order(self, request):
        params = await self._signed_params(request)
        self.place_requests += 1
        client_order_id = params["clientOrderId"]
        if client_order_id not in self.orders:
            self.orders[client_order_id] = {
                **params,
                "orderId": str(len(self.orders) + 1),
                "orderStatus": "NEW",
                "filledQuantity": "0",
                "filledAmount": "0",
                "transactTime": int(time.time() * 1000),
            }
        return web.json_response({"status": "S", "data": self.orders[client_order_id]})

    async def cancel_order(self, request):
        params = await self._signed_params(request)
        order = next(value for value in self.orders.values() if value["orderId"] == str(params["orderId"]))
        order["orderStatus"] = "CANCELED"
        return web.json_response({"status": "S", "data": {"orderId": order["orderId"]}})

    async def order_detail(self, request):
        params = await self._signed_params(request)
        if "clientOrderId" in params:
            order = self.orders.get(params["clientOrderId"])
        else:
            order = next(
                (value for value in self.orders.values() if value["orderId"] == str(params.get("orderId"))), None
            )
        if order is None:
            raise web.HTTPNotFound(text="order not found")
        return web.json_response({"status": "S", "data": order})

    async def listen_key(self, request):
        await self._signed_params(request)
        return web.json_response({"status": "S", "data": {"listenKey": "test-listen-key"}})

    async def websocket(self, request):
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        if "orderReport" in request.path:
            await websocket.send_json({"e": "orderReport", "clientOrderId": "unknown", "orderStatus": "NEW"})
        else:
            await websocket.send_json({"stream": "calendar", "data": {"e": "calendar", "phase": "MARKET_OPEN"}})
        async for _ in websocket:
            pass
        return websocket
