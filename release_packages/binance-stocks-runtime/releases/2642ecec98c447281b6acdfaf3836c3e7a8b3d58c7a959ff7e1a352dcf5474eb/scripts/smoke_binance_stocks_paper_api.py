from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any


class Client:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}
        self.calls: list[tuple[str, str, int]] = []

    def request(self, method: str, path: str, payload: Any = None, expected=(200,)):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                status = response.status
                result = json.loads(response.read().decode() or "null")
        except urllib.error.HTTPError as exc:
            status = exc.code
            result = json.loads(exc.read().decode() or "null")
        self.calls.append((method, path, status))
        if status not in expected:
            raise AssertionError(f"{method} {path}: HTTP {status}, expected {expected}: {result}")
        return result


def market_request(base_url: str, path: str, payload: Any = None):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def wait_for(client: Client, path: str, predicate, timeout: float = 20):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.request("GET", path)
        if predicate(last):
            return last
        time.sleep(0.5)
    raise AssertionError(f"timeout waiting for {path}: {last}")


def wait_executor_terminal(client: Client, executor_id: str, timeout: float = 20):
    active = {"RESERVED", "RUNNING", "EXIT_PENDING", "CLOSING"}
    return wait_for(
        client, f"/stocks/executors/{executor_id}",
        lambda value: str((value.get("ledger") or {}).get("status", "")) not in active,
        timeout=timeout,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--market-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()
    client = Client(args.url, args.username, args.password)
    suffix = str(int(time.time()))[-8:]

    health = client.request("GET", "/stocks/health")
    if health.get("runtime_mode") != "PAPER" or not health.get("scenario_mode"):
        raise AssertionError("smoke requires an isolated PAPER scenario runtime")
    market_request(args.market_url, "/scenario/market", {
        "phase": "MARKET_OPEN", "trading_status": "TRADING", "tradability": "BOTH",
        "trading_date": "2026-08-21",
    })
    client.request("POST", "/stocks/scenario/market-state", {
        "symbol": "AAPL", "phase": "MARKET_OPEN", "trading_status": "TRADING",
        "tradability": "BOTH", "trading_date": "2026-08-21",
    })
    market_request(args.market_url, "/scenario/quote", {
        "symbol": "AAPL", "bid": "199.90", "ask": "200.00", "bid_size": "10", "ask_size": "10",
    })
    time.sleep(2)

    client.request("GET", "/stocks/markets")
    client.request("GET", "/stocks/whitelist")
    client.request("PUT", "/stocks/whitelist/AAPL", {
        "symbol": "AAPL", "enabled": True, "max_position_notional": "200",
    })
    limits = client.request("GET", "/stocks/limits")
    client.request("PUT", "/stocks/limits", {
        "max_order_notional": limits["active"]["max_order_notional"],
        "max_symbol_exposure": limits["active"]["max_symbol_exposure"],
        "max_managed_exposure": limits["active"]["max_managed_exposure"],
    })
    client.request("GET", "/stocks/quotes/AAPL")
    client.request("GET", "/stocks/market-status/AAPL")
    client.request("GET", "/stocks/account-summary")
    client.request("GET", "/stocks/managed-positions")
    client.request("GET", "/stocks/open-orders")
    client.request("GET", "/stocks/order-history")
    client.request("GET", "/stocks/trade-history")
    client.request("GET", "/stocks/executors")

    preview_config = {
        "id": f"preview-{suffix}", "type": "order_executor", "connector_name": "binance_stocks",
        "trading_pair": "AAPL-USDC", "side": "BUY", "amount": "0.5", "price": "200",
        "execution_strategy": "LIMIT",
    }
    client.request("POST", "/stocks/executors/preview", {"executor_config": preview_config})
    client.request("POST", "/stocks/order-executors/preview", {
        "id": f"typed-preview-{suffix}", "symbol": "AAPL", "side": "BUY", "amount": "0.5",
        "order_type": "LIMIT", "price": "200",
    })
    client.request("POST", "/stocks/position-executors/preview", {
        "id": f"position-preview-{suffix}", "symbol": "AAPL", "amount": "0.5",
        "entry_order_type": "LIMIT", "entry_price": "200", "stop_loss": "0.02",
        "take_profit": "0.03", "time_limit": 3600,
    })

    buy_id = f"limitbuy-{suffix}"
    client.request("POST", "/stocks/order-executors", {
        "id": buy_id, "symbol": "AAPL", "side": "BUY", "amount": "0.5",
        "order_type": "LIMIT", "price": "200",
    })
    wait_for(client, "/stocks/paper/trades?symbol=AAPL", lambda x: len(x["items"]) >= 1)
    client.request("GET", f"/stocks/executors/{buy_id}")

    sell_id = f"limitsell-{suffix}"
    client.request("POST", "/stocks/order-executors", {
        "id": sell_id, "symbol": "AAPL", "side": "SELL", "amount": "0.5",
        "order_type": "LIMIT", "price": "205", "source_owner": "unassigned",
    })
    market_request(args.market_url, "/scenario/quote", {
        "symbol": "AAPL", "bid": "205.00", "ask": "205.10", "bid_size": "10", "ask_size": "10",
    })
    wait_for(client, "/stocks/paper/trades?symbol=AAPL", lambda x: len(x["items"]) >= 2)
    wait_executor_terminal(client, buy_id)
    wait_executor_terminal(client, sell_id)
    performance = client.request("GET", "/stocks/paper/performance?window=all")
    if float(performance["net_pnl"]) <= 0:
        raise AssertionError(f"completed profitable round trip did not produce positive PnL: {performance}")

    position_id = f"position-{suffix}"
    market_request(args.market_url, "/scenario/quote", {
        "symbol": "AAPL", "bid": "199.90", "ask": "200.00", "bid_size": "10", "ask_size": "10",
    })
    client.request("POST", "/stocks/position-executors", {
        "id": position_id, "symbol": "AAPL", "amount": "0.5", "entry_order_type": "LIMIT",
        "entry_price": "200", "stop_loss": "0.02", "take_profit": "0.02", "time_limit": 3600,
        "trailing_activation": "0.01", "trailing_delta": "0.005",
    })
    wait_for(client, "/stocks/paper/trades?symbol=AAPL", lambda x: len(x["items"]) >= 3)
    market_request(args.market_url, "/scenario/quote", {
        "symbol": "AAPL", "bid": "204.10", "ask": "204.20", "bid_size": "10", "ask_size": "10",
    })
    wait_for(client, "/stocks/paper/trades?symbol=AAPL", lambda x: len(x["items"]) >= 4, timeout=30)
    wait_executor_terminal(client, position_id, timeout=30)
    client.request("GET", f"/stocks/executors/{position_id}")

    # Reduction against a fully closed batch must fail safely rather than use
    # another Executor's inventory.
    client.request("POST", f"/stocks/executors/{position_id}/reduce", {
        "amount": "0.1", "request_id": f"reduce-{suffix}",
    }, expected=(409,))

    pending_id = f"pending-{suffix}"
    client.request("POST", "/stocks/executors", {"executor_config": {
        "id": pending_id, "type": "order_executor", "connector_name": "binance_stocks",
        "trading_pair": "AAPL-USDC", "side": "BUY", "amount": "0.1", "price": "100",
        "execution_strategy": "LIMIT",
    }, "controller_id": "stocks-paper-smoke"})
    client.request("POST", f"/stocks/executors/{pending_id}/cancel")
    wait_executor_terminal(client, pending_id)

    close_id = f"closepos-{suffix}"
    client.request("POST", "/stocks/position-executors", {
        "id": close_id, "symbol": "AAPL", "amount": "0.1", "entry_order_type": "LIMIT",
        "entry_price": "100", "stop_loss": "0.02", "time_limit": 3600,
    })
    client.request("POST", f"/stocks/executors/{close_id}/close")
    wait_executor_terminal(client, close_id)

    client.request("GET", "/stocks/paper/account")
    client.request("GET", "/stocks/paper/positions")
    client.request("GET", "/stocks/paper/orders")
    client.request("GET", "/stocks/paper/trades")
    client.request("GET", "/stocks/paper/performance?window=4h")
    client.request("GET", "/stocks/paper/performance?window=24h")
    client.request("GET", "/stocks/paper/performance?window=7d")
    client.request("GET", "/stocks/paper/equity")
    all_trades = client.request("GET", "/stocks/paper/trades")["items"]
    completed_performance = client.request("GET", "/stocks/paper/performance?window=all")
    if float(completed_performance["net_pnl"]) <= 0:
        raise AssertionError(f"full Order/Position lifecycle is not profitable: {completed_performance}")
    summary = client.request("GET", "/stocks/paper/summary")
    if summary.get("schema") != "binance-stocks-paper-summary-v1":
        raise AssertionError(f"unexpected Paper summary contract: {summary}")
    if not summary.get("valuation_complete") or not summary.get("reconciliation", {}).get("ok"):
        raise AssertionError(f"Paper summary does not reconcile: {summary}")
    if not any(row.get("symbol") == "AAPL" for row in summary.get("positions", [])):
        raise AssertionError(f"Paper summary has no AAPL attribution: {summary}")
    if summary.get("windows", {}).get("4h", {}).get("window_complete"):
        raise AssertionError("fresh scenario run must not claim a complete 4h window")
    client.request("POST", "/stocks/paper/reset", {"confirmation": "WRONG"}, expected=(422,))
    client.request("DELETE", "/stocks/whitelist/AAPL")
    client.request("PUT", "/stocks/whitelist/AAPL", {
        "symbol": "AAPL", "enabled": True, "max_position_notional": "200",
    })
    reset = client.request("POST", "/stocks/paper/reset", {
        "confirmation": "RESET PAPER ACCOUNT TO 2000 USDC",
    })
    if reset["paper_run_id"] == reset["previous_run_id"]:
        raise AssertionError("paper reset did not create an immutable new run")

    final_health = client.request("GET", "/stocks/health")
    stats = market_request(args.market_url, "/scenario/stats")
    economic = stats["economic_requests"]
    if any(int(value) for value in economic.values()) or final_health["economic_http_request_count"] != 0:
        raise AssertionError(f"forbidden economic request detected: market={economic}, runtime={final_health}")
    print(json.dumps({
        "status": "PASS", "http_calls": len(client.calls), "fills": len(all_trades),
        "net_pnl": completed_performance["net_pnl"], "economic_requests": economic,
        "paper_run_id": final_health["paper_run_id"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
