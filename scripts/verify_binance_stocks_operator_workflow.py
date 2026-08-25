from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable

import requests


ACTIVE_EXECUTOR_STATES = {
    "RESERVED", "RUNNING", "EXIT_PENDING", "CLOSING", "QUEUED",
    "WAITING_SESSION", "WAITING_PREFLIGHT", "ACTIVATING",
}


class Client:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.auth = (os.environ["STOCKS_API_USERNAME"], os.environ["STOCKS_API_PASSWORD"])
        self.calls: list[dict[str, Any]] = []

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None,
        expected: tuple[int, ...] = (200,),
    ) -> Any:
        response = requests.request(
            method, self.base_url + path, auth=self.auth, json=payload, timeout=30,
        )
        try:
            body = response.json()
        except Exception:
            body = response.text
        self.calls.append({"method": method, "path": path, "status": response.status_code})
        if response.status_code not in expected:
            raise AssertionError(
                f"{method} {path}: HTTP {response.status_code}, expected {expected}: {body}"
            )
        return body


def wait_for(fetch: Callable[[], Any], predicate: Callable[[Any], bool], timeout: float = 60) -> Any:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fetch()
        if predicate(last):
            return last
        time.sleep(1)
    raise AssertionError(f"condition did not converge within {timeout}s: {last}")


def assert_paper_boundary(client: Client) -> dict[str, Any]:
    health = client.request("GET", "/stocks/health")
    expected = {
        "runtime_mode": "PAPER", "live_authorized": False,
        "economic_requests_enabled": False, "economic_http_request_count": 0,
    }
    for key, value in expected.items():
        if health.get(key) != value:
            raise AssertionError(f"unsafe PAPER boundary {key}: {health.get(key)!r} != {value!r}")
    if not health.get("connector_ready") or health.get("market_state_conflict"):
        raise AssertionError(f"connector is not ready for acceptance: {health}")
    return health


def active_executors(client: Client) -> list[dict[str, Any]]:
    return list(client.request("GET", "/stocks/executors?active_only=true").get("items", []))


def cleanup_and_reset(client: Client) -> dict[str, Any]:
    health = assert_paper_boundary(client)
    before = client.request("GET", "/stocks/paper/account")
    had_risk = bool(active_executors(client) or before.get("positions"))
    if had_risk:
        client.request(
            "POST", "/stocks/paper/reset",
            {"confirmation": "RESET PAPER ACCOUNT TO 2000 USDC"}, expected=(409,),
        )
    for executor in active_executors(client):
        executor_id = str(executor["executor_id"])
        action = "close" if str(executor.get("executor_type")) == "position_executor" else "cancel"
        client.request("POST", f"/stocks/executors/{executor_id}/{action}", expected=(200, 409))
    wait_for(lambda: active_executors(client), lambda rows: not rows, timeout=90)
    wait_for(
        lambda: client.request("GET", "/stocks/paper/orders?open_only=true").get("items", []),
        lambda rows: not rows, timeout=90,
    )
    managed = client.request("GET", "/stocks/managed-positions")
    for index, lot in enumerate(managed.get("lots", managed.get("items", []))):
        amount = Decimal(str(lot.get("available_base", lot.get("total_base", 0))))
        if amount <= 0:
            continue
        owner = str(lot["owner_id"])
        cleanup_id = f"accept-cleanup-{index}-{int(time.time())}"
        client.request("POST", "/stocks/order-executors", {
            "id": cleanup_id, "symbol": str(lot["symbol"]), "side": "SELL",
            "amount": str(amount), "order_type": "MARKET", "source_owner": owner,
        })
        wait_executor_terminal(client, cleanup_id)
    positions = wait_for(
        lambda: client.request("GET", "/stocks/paper/positions").get("items", []),
        lambda rows: not any(Decimal(str(row.get("total", 0))) > 0 for row in rows), timeout=90,
    )
    reset = client.request(
        "POST", "/stocks/paper/reset",
        {"confirmation": "RESET PAPER ACCOUNT TO 2000 USDC"},
    )
    after = client.request("GET", "/stocks/paper/account")
    if Decimal(after["equity"]) != Decimal("2000") or after.get("positions") or positions:
        raise AssertionError(f"reset did not produce a clean 2000 USDC run: {after}")
    return {"before": before, "reset": reset, "after": after, "health": health}


def configuration_check(client: Client, temporary_symbol: str) -> dict[str, Any]:
    assert_paper_boundary(client)
    original = client.request("GET", "/stocks/whitelist").get("items", [])
    required = {"AAPL", "TSLA", "SPY", "QQQ"}
    enabled = {str(row["symbol"]) for row in original if row.get("enabled")}
    if not required.issubset(enabled):
        raise AssertionError(f"required whitelist entries are missing: {required - enabled}")

    client.request("PUT", f"/stocks/whitelist/{temporary_symbol}", {
        "symbol": temporary_symbol, "enabled": True, "max_position_notional": "900",
    })
    client.request("PUT", f"/stocks/whitelist/{temporary_symbol}", {
        "symbol": temporary_symbol, "enabled": True, "max_position_notional": "1000",
    })
    wait_for(
        lambda: client.request("GET", f"/stocks/quotes/{temporary_symbol}", expected=(200, 503)),
        lambda value: isinstance(value, dict) and Decimal(str(value.get("bidPrice", 0))) > 0,
        timeout=90,
    )
    deleted = client.request("DELETE", f"/stocks/whitelist/{temporary_symbol}")
    blocked = client.request("POST", "/stocks/order-executors/preview", {
        "id": "acceptance-deleted-symbol-buy", "symbol": temporary_symbol, "side": "BUY",
        "amount": "0.1", "order_type": "LIMIT", "price": "100",
    })
    if blocked.get("allowed") is not False:
        raise AssertionError(f"deleted symbol still allows BUY: {blocked}")

    changed = {"max_order_notional": "499", "max_symbol_exposure": "999",
               "max_managed_exposure": "1999", "daily_loss_limit": "199"}
    client.request("PUT", "/stocks/limits", changed)
    persisted = client.request("GET", "/stocks/limits")
    if persisted.get("active") != changed:
        raise AssertionError(f"operator limits did not persist exactly: {persisted}")
    restored = {"max_order_notional": "500", "max_symbol_exposure": "1000",
                "max_managed_exposure": "2000", "daily_loss_limit": "200"}
    client.request("PUT", "/stocks/limits", restored)

    blocked_limit = client.request("POST", "/stocks/order-executors/preview", {
        "id": "acceptance-limit-blocked", "symbol": "AAPL", "side": "BUY",
        "amount": "100", "order_type": "LIMIT", "price": "300",
    })
    if blocked_limit.get("allowed") is not False:
        raise AssertionError(f"business limit did not return allowed=false: {blocked_limit}")
    client.request("POST", "/stocks/order-executors/preview", {
        "id": "acceptance-invalid-limit", "symbol": "AAPL", "side": "BUY",
        "amount": "0.1", "order_type": "LIMIT",
    }, expected=(422,))

    symbols = {}
    for symbol in sorted(required):
        symbols[symbol] = {
            "quote": client.request("GET", f"/stocks/quotes/{symbol}"),
            "status": client.request("GET", f"/stocks/market-status/{symbol}"),
        }
        if not symbols[symbol]["status"].get("quote_fresh"):
            raise AssertionError(f"{symbol} quote is not fresh")
    return {
        "original_whitelist": original, "temporary_symbol": temporary_symbol,
        "temporary_deleted": deleted, "limit_round_trip": {"changed": changed, "restored": restored},
        "symbols": symbols, "http_semantics": {"business": 200, "malformed": 422},
    }


def amount_for_budget(price: Decimal, budget: Decimal = Decimal("20")) -> Decimal:
    return (budget / price).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)


def wait_executor_terminal(client: Client, executor_id: str, timeout: float = 90) -> dict[str, Any]:
    return wait_for(
        lambda: client.request("GET", f"/stocks/executors/{executor_id}"),
        lambda row: str((row.get("ledger") or {}).get("status", "")) not in ACTIVE_EXECUTOR_STATES,
        timeout=timeout,
    )


def trade_and_prepare_restart(client: Client, state_path: Path) -> dict[str, Any]:
    health = assert_paper_boundary(client)
    if health.get("market_phase") != "MARKET_OPEN":
        raise AssertionError(f"real-quote acceptance requires MARKET_OPEN: {health}")
    quote = client.request("GET", "/stocks/quotes/AAPL")
    bid, ask = Decimal(str(quote["bidPrice"])), Decimal(str(quote["askPrice"]))
    suffix = str(int(time.time()))

    pending_id = f"accept-limit-cancel-{suffix}"
    pending_price = (bid * Decimal("0.8")).quantize(Decimal("0.01"))
    pending_amount = amount_for_budget(pending_price)
    client.request("POST", "/stocks/order-executors", {
        "id": pending_id, "symbol": "AAPL", "side": "BUY", "amount": str(pending_amount),
        "order_type": "LIMIT", "price": str(pending_price),
    })
    client.request(
        "POST", "/stocks/paper/reset", {"confirmation": "RESET PAPER ACCOUNT TO 2000 USDC"},
        expected=(409,),
    )
    client.request("POST", f"/stocks/executors/{pending_id}/cancel")
    wait_executor_terminal(client, pending_id)

    buy_id = f"accept-limit-fill-{suffix}"
    fill_price = (ask * Decimal("1.01")).quantize(Decimal("0.01"))
    fill_amount = amount_for_budget(ask)
    trades_before = len(client.request("GET", "/stocks/paper/trades").get("items", []))
    client.request("POST", "/stocks/order-executors", {
        "id": buy_id, "symbol": "AAPL", "side": "BUY", "amount": str(fill_amount),
        "order_type": "LIMIT", "price": str(fill_price),
    })
    wait_for(
        lambda: client.request("GET", "/stocks/paper/trades").get("items", []),
        lambda rows: len(rows) > trades_before, timeout=90,
    )
    buy = wait_executor_terminal(client, buy_id)
    buy_order = next(row for row in (buy.get("ledger") or {}).get("orders", []) if row.get("side") == "BUY")
    filled = Decimal(str(buy_order["cumulative_base"]))

    sell_id = f"accept-market-sell-{suffix}"
    client.request("POST", "/stocks/order-executors", {
        "id": sell_id, "symbol": "AAPL", "side": "SELL", "amount": str(filled),
        "order_type": "MARKET", "source_owner": "unassigned",
    })
    wait_executor_terminal(client, sell_id)

    position_id = f"accept-position-{suffix}"
    client.request("POST", "/stocks/position-executors", {
        "id": position_id, "symbol": "AAPL", "amount": str(fill_amount),
        "entry_order_type": "MARKET", "quote_budget": "20", "stop_loss": "0.02",
        "take_profit": "0.03", "time_limit": 3600,
        "trailing_activation": "0.01", "trailing_delta": "0.005",
    })
    wait_for(
        lambda: client.request("GET", "/stocks/paper/positions").get("items", []),
        lambda rows: any(Decimal(str(row.get("total", 0))) > 0 for row in rows), timeout=90,
    )
    record = client.request("GET", f"/stocks/executors/{position_id}")
    owned = Decimal(str((record.get("ledger") or {}).get("requested_base", fill_amount)))
    reduce_amount = (owned / 2).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN)
    client.request("POST", f"/stocks/executors/{position_id}/reduce", {
        "amount": str(reduce_amount), "request_id": f"reduce-{suffix}",
    })
    time.sleep(3)

    restart_id = f"accept-restart-{suffix}"
    client.request("POST", "/stocks/order-executors", {
        "id": restart_id, "symbol": "AAPL", "side": "BUY", "amount": str(pending_amount),
        "order_type": "LIMIT", "price": str(pending_price),
    })
    state = {"position_id": position_id, "restart_id": restart_id, "suffix": suffix}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return {"pending_id": pending_id, "round_trip": {"buy": buy_id, "sell": sell_id}, **state}


def finish_after_restart(client: Client, state_path: Path) -> dict[str, Any]:
    assert_paper_boundary(client)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    restart_record = client.request("GET", f"/stocks/executors/{state['restart_id']}")
    if not restart_record.get("ledger"):
        raise AssertionError("restart did not restore the pending OrderExecutor")
    client.request("POST", f"/stocks/executors/{state['restart_id']}/cancel", expected=(200, 409))
    client.request("POST", f"/stocks/executors/{state['position_id']}/close", expected=(200, 409))
    wait_for(lambda: active_executors(client), lambda rows: not rows, timeout=90)
    wait_for(
        lambda: client.request("GET", "/stocks/paper/positions").get("items", []),
        lambda rows: not any(Decimal(str(row.get("total", 0))) > 0 for row in rows), timeout=90,
    )
    summary = client.request("GET", "/stocks/paper/summary")
    if not summary.get("valuation_complete") or not summary.get("reconciliation", {}).get("ok"):
        raise AssertionError(f"PAPER summary did not reconcile: {summary}")
    reset = client.request(
        "POST", "/stocks/paper/reset", {"confirmation": "RESET PAPER ACCOUNT TO 2000 USDC"},
    )
    final = client.request("GET", "/stocks/paper/summary")
    account = final.get("account", {})
    if Decimal(str(account.get("equity", 0))) != Decimal("2000"):
        raise AssertionError(f"final PAPER equity is not 2000: {final}")
    if account.get("positions") or final.get("totals", {}).get("open_order_count"):
        raise AssertionError(f"final PAPER account is not empty: {final}")
    return {"restored_executor": state["restart_id"], "pre_reset_summary": summary,
            "reset": reset, "final_summary": final}


def write_result(output: Path, stage: str, result: dict[str, Any], calls: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "binance-stocks-operator-acceptance-v1", "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(), "result": result, "http_calls": calls,
    }
    (output / f"{stage}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8",
    )
    print(json.dumps({"stage": stage, "status": "PASS", "http_calls": len(calls)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cleanup-reset", "configuration", "trade-prepare", "trade-finish"))
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("/tmp/stocks-operator-acceptance"))
    parser.add_argument("--state", type=Path, default=Path("/tmp/stocks-operator-acceptance-state.json"))
    parser.add_argument("--temporary-symbol", default="MSFT")
    args = parser.parse_args()
    client = Client(args.url)
    if args.stage == "cleanup-reset":
        result = cleanup_and_reset(client)
    elif args.stage == "configuration":
        result = configuration_check(client, args.temporary_symbol.upper())
    elif args.stage == "trade-prepare":
        result = trade_and_prepare_restart(client, args.state)
    else:
        result = finish_after_restart(client, args.state)
    write_result(args.output, args.stage, result, client.calls)


if __name__ == "__main__":
    main()
