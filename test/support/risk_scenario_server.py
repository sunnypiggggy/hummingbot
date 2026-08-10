"""Stateful HTTP simulation of Binance, Hummingbot control, and Telegram.

This server intentionally implements the real endpoints used by the Guard
clients.  It validates Binance signatures and mutates balances/orders so tests
cross the HTTP boundary instead of monkeypatching production methods.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import threading
import time
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse


def _d(value):
    return Decimal(str(value if value is not None else "0"))


class ScenarioState:
    def __init__(self, document):
        self.lock = threading.RLock()
        self.api_key = document.get("api_key", "scenario-key")
        self.api_secret = document.get("api_secret", "scenario-secret")
        self.balances = {
            asset: {"free": _d(row.get("free")), "locked": _d(row.get("locked"))}
            for asset, row in document.get("balances", {}).items()
        }
        self.prices = {key: _d(value) for key, value in document.get("prices", {}).items()}
        self.filters = document.get("filters", {})
        self.orders = {}
        self.order_visibility_misses = {}
        self.open_orders = list(document.get("open_orders", []))
        self.pending_cancel_queries = {}
        self.trades = []
        self.telegram = []
        self.control = document.get("control", {"containers": [], "bots": []})
        self.control.setdefault("controller_updates", [])
        self.faults = {key: list(value) for key, value in document.get("faults", {}).items()}
        self.next_order_id = int(document.get("next_order_id", 900000))
        self.next_message_id = 1

    def fault(self, key):
        with self.lock:
            queue = self.faults.get(key, [])
            return queue.pop(0) if queue else {}

    def public(self):
        with self.lock:
            return {
                "balances": {
                    asset: {key: str(value) for key, value in row.items()}
                    for asset, row in self.balances.items()
                },
                "orders": list(self.orders.values()),
                "open_orders": list(self.open_orders),
                "trades": list(self.trades),
                "telegram": list(self.telegram),
                "controller_updates": list(self.control.get("controller_updates", [])),
                "faults_remaining": {key: len(value) for key, value in self.faults.items()},
            }


class ScenarioHandler(BaseHTTPRequestHandler):
    server_version = "RiskScenario/1"

    @property
    def state(self):
        return self.server.scenario_state

    def log_message(self, _format, *_args):
        return

    def _json(self, status, value):
        body = json.dumps(value, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _params(self):
        return dict(parse_qsl(urlparse(self.path).query, keep_blank_values=True))

    def _signed(self):
        if self.headers.get("X-MBX-APIKEY") != self.state.api_key:
            self._json(401, {"code": -2015, "msg": "Invalid API-key"})
            return False
        pairs = parse_qsl(urlparse(self.path).query, keep_blank_values=True)
        provided = dict(pairs).get("signature", "")
        unsigned = urlencode([(key, value) for key, value in pairs if key != "signature"])
        expected = hmac.new(
            self.state.api_secret.encode(), unsigned.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(provided, expected):
            self._json(400, {"code": -1022, "msg": "Signature invalid"})
            return False
        timestamp = int(dict(pairs).get("timestamp", "0"))
        if abs(int(time.time() * 1000) - timestamp) > int(dict(pairs).get("recvWindow", "5000")):
            self._json(400, {"code": -1021, "msg": "Timestamp outside recvWindow"})
            return False
        return True

    def do_GET(self):
        parsed = urlparse(self.path)
        path, params = parsed.path, self._params()
        if path == "/scenario/state":
            return self._json(200, self.state.public())
        if path == "/api/v3/time":
            return self._json(200, {"serverTime": int(time.time() * 1000)})
        if path == "/api/v3/ticker/price":
            symbol = params["symbol"]
            return self._json(200, {"symbol": symbol, "price": str(self.state.prices[symbol])})
        if path == "/api/v3/ticker/bookTicker":
            symbol, price = params["symbol"], self.state.prices[params["symbol"]]
            return self._json(200, {
                "symbol": symbol, "bidPrice": str(price), "askPrice": str(price),
            })
        if path == "/api/v3/exchangeInfo":
            symbol = params["symbol"]
            fault = self.state.fault("GET /api/v3/exchangeInfo")
            if fault.get("mode") == "http_error":
                return self._json(int(fault.get("status", 500)), {
                    "code": -1000, "msg": "injected exchangeInfo failure",
                })
            spec = {**self.state.filters[symbol], **fault.get("filters", {})}
            filters = [
                {"filterType": "LOT_SIZE", "stepSize": spec["lot_step"]},
                {"filterType": "MARKET_LOT_SIZE", "stepSize": spec["market_step"]},
                {"filterType": "MIN_NOTIONAL", "minNotional": spec["min_notional"]},
            ]
            return self._json(200, {"symbols": [{"symbol": symbol, "filters": filters}]})
        if path == "/api/v3/klines":
            now = int(time.time() // 300 * 300000)
            price = self.state.prices[params["symbol"]]
            return self._json(200, [[now, str(price), str(price), str(price), str(price), "1", now+299999]])
        if path == "/api/v3/account":
            if not self._signed():
                return
            fault = self.state.fault("GET /api/v3/account")
            if fault.get("mode") == "http_error" or fault.get("code") is not None:
                status = int(fault.get("status", 400))
                code = int(fault.get("code", -1021))
                return self._json(status, {"code": code, "msg": fault.get("message", "injected")})
            if fault.get("delay_seconds"):
                time.sleep(float(fault["delay_seconds"]))
            with self.state.lock:
                balances = fault.get("balances") or self.state.balances
                rows = [{
                    "asset": asset, "free": str(value["free"]),
                    "locked": str(value["locked"]),
                } for asset, value in balances.items()]
            return self._json(200, {"canTrade": True, "balances": rows})
        if path == "/api/v3/openOrders":
            if not self._signed():
                return
            symbol = params.get("symbol")
            with self.state.lock:
                remaining = int(self.state.pending_cancel_queries.get(symbol, 0))
                if remaining > 0:
                    self.state.pending_cancel_queries[symbol] = remaining - 1
                elif symbol in self.state.pending_cancel_queries:
                    self.state.open_orders = [
                        row for row in self.state.open_orders
                        if row.get("symbol") != symbol
                    ]
                    del self.state.pending_cancel_queries[symbol]
            return self._json(200, [
                row for row in self.state.open_orders if not symbol or row.get("symbol") == symbol
            ])
        if path == "/api/v3/order":
            if not self._signed():
                return
            key = (params.get("symbol"), params.get("origClientOrderId"))
            misses = int(self.state.order_visibility_misses.get(key, 0))
            if misses > 0:
                self.state.order_visibility_misses[key] = misses - 1
                return self._json(400, {"code": -2013, "msg": "Order not visible yet"})
            order = self.state.orders.get(key)
            if order is None:
                return self._json(400, {"code": -2013, "msg": "Order does not exist"})
            return self._json(200, order)
        if path == "/api/v3/myTrades":
            if not self._signed():
                return
            order_id = str(params.get("orderId", ""))
            symbol = params.get("symbol")
            return self._json(200, [
                row for row in self.state.trades
                if str(row.get("orderId")) == order_id and row.get("symbol") == symbol
            ])
        if path == "/bot-orchestration/status":
            return self._json(200, self.state.control.get("bots", []))
        if path == "/docker/active-containers":
            name = params.get("name_filter", "")
            return self._json(200, [
                {"name": value} for value in self.state.control.get("containers", [])
                if not name or name in value
            ])
        return self._json(404, {"error": path})

    def do_DELETE(self):
        path, params = urlparse(self.path).path, self._params()
        if path == "/api/v3/openOrders":
            if not self._signed():
                return
            symbol = params.get("symbol")
            fault = self.state.fault("DELETE /api/v3/openOrders")
            if fault.get("mode") == "http_error":
                return self._json(int(fault.get("status", 500)), {
                    "code": -1000, "msg": "injected cancel failure",
                })
            with self.state.lock:
                canceled = [row for row in self.state.open_orders if row.get("symbol") == symbol]
                delay_queries = int(fault.get("delay_queries", 0))
                if delay_queries > 0:
                    self.state.pending_cancel_queries[symbol] = delay_queries
                else:
                    self.state.open_orders = [
                        row for row in self.state.open_orders if row.get("symbol") != symbol
                    ]
            return self._json(200, canceled)
        return self._json(404, {"error": path})

    def do_POST(self):
        parsed = urlparse(self.path)
        path, params = parsed.path, self._params()
        if path == "/api/v3/order":
            if not self._signed():
                return
            fault = self.state.fault("POST /api/v3/order")
            if fault.get("mode") == "http_error":
                return self._json(int(fault.get("status", 500)), {"code": -1000, "msg": "injected"})
            symbol, client_id = params["symbol"], params.get("newClientOrderId", "")
            key = (symbol, client_id)
            with self.state.lock:
                if key in self.state.orders:
                    return self._json(400, {"code": -2010, "msg": "Duplicate order sent"})
                requested = _d(params["quantity"])
                price = self.state.prices[symbol]
                spec = self.state.filters[symbol]
                market_step = _d(spec.get("market_step"))
                step = market_step if market_step > 0 else _d(spec["lot_step"])
                minimum_notional = _d(spec["min_notional"])
                if requested <= 0 or step <= 0 or requested % step != 0:
                    return self._json(400, {
                        "code": -1013,
                        "msg": f"Filter failure: MARKET_LOT_SIZE step={step}",
                    })
                if requested * price < minimum_notional:
                    return self._json(400, {
                        "code": -1013,
                        "msg": f"Filter failure: MIN_NOTIONAL {minimum_notional}",
                    })
                fraction = _d(fault.get("fill_fraction", "1"))
                executed = requested * fraction
                quote = executed * price
                base = "BTC" if symbol.startswith("BTC") else "ETH"
                quote_asset = symbol[len(base):]
                if params["side"] == "SELL":
                    if self.state.balances[base]["free"] < executed:
                        return self._json(400, {"code": -2010, "msg": "insufficient balance"})
                    self.state.balances[base]["free"] -= executed
                    self.state.balances.setdefault(quote_asset, {"free": _d(0), "locked": _d(0)})
                    self.state.balances[quote_asset]["free"] += quote
                self.state.next_order_id += 1
                order_id = self.state.next_order_id
                commission = _d(fault.get("commission", "0"))
                commission_asset = fault.get("commission_asset", "BNB")
                if commission:
                    self.state.balances.setdefault(commission_asset, {"free": _d(0), "locked": _d(0)})
                    self.state.balances[commission_asset]["free"] -= commission
                trade = {
                    "symbol": symbol, "orderId": order_id, "price": str(price),
                    "qty": str(executed), "commission": str(commission),
                    "commissionAsset": commission_asset,
                    "time": int(time.time() * 1000),
                }
                self.state.trades.append(trade)
                order = {
                    "symbol": symbol, "orderId": order_id,
                    "clientOrderId": client_id,
                    "status": fault.get("status", "FILLED"),
                    "origQty": str(requested), "executedQty": str(executed),
                    "cummulativeQuoteQty": str(quote),
                    "transactTime": int(time.time() * 1000),
                    "fills": [{
                        "price": str(price), "qty": str(executed),
                        "commission": str(commission),
                        "commissionAsset": commission_asset,
                    }],
                }
                self.state.orders[key] = order
                self.state.order_visibility_misses[key] = int(
                    fault.get("visibility_misses", 0)
                )
            if fault.get("drop_response"):
                self.close_connection = True
                return
            return self._json(200, order)
        if path.startswith("/bot") and path.rsplit("/", 1)[-1] in {
            "sendMessage", "sendPhoto", "sendDocument",
        }:
            fault = self.state.fault("POST telegram")
            if fault:
                return self._json(int(fault.get("status", 429)), {"ok": False})
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            with self.state.lock:
                message_id = self.state.next_message_id
                self.state.next_message_id += 1
                self.state.telegram.append({
                    "method": path.rsplit("/", 1)[-1], "message_id": message_id,
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                    "received_at": time.time(),
                })
            return self._json(200, {"ok": True, "result": {"message_id": message_id}})
        if path == "/bot-orchestration/stop-bot":
            return self._json(200, {"status": "stopped"})
        if path == "/trading/orders":
            return self._json(200, {"status": "accepted-by-control-sim"})
        if path.startswith("/controllers/bots/") and path.endswith("/config"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body or b"{}")
            with self.state.lock:
                self.state.control["controller_updates"].append({
                    "path": path, "payload": payload, "received_at": time.time(),
                })
            return self._json(200, {"status": "updated"})
        return self._json(404, {"error": path})


class RiskScenarioServer:
    def __init__(self, document, host="127.0.0.1", port=0):
        self.state = ScenarioState(document)
        self.server = ThreadingHTTPServer((host, port), ScenarioHandler)
        self.server.scenario_state = self.state
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    document = json.loads(args.scenario.read_text(encoding="utf-8"))
    server = RiskScenarioServer(document, args.host, args.port)
    print(json.dumps({"status": "ready", "base_url": server.base_url}), flush=True)
    server.server.serve_forever()


if __name__ == "__main__":
    main()
