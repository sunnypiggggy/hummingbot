#!/usr/bin/env python3
"""Read current OCI DCA TradeFill rows without scanning executor history."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


SCALE = Decimal("1000000")
SPECS = {
    "BTC-USDT": (
        "dca-btcusdt-20260711-064046",
        "dca-btcusdt-20260711-064046.sqlite",
    ),
    "ETH-USDT": (
        "dca-ethusdt-20260711-064046",
        "dca-ethusdt-20260711-064046.sqlite",
    ),
}


def mark_price(pair: str) -> Decimal:
    query = urlencode({"symbol": pair.replace("-", "")})
    with urlopen(f"https://api.binance.com/api/v3/ticker/price?{query}", timeout=15) as response:
        return Decimal(str(json.load(response)["price"]))


def decimal_value(value) -> Decimal:
    return Decimal(int(value or 0)) / SCALE


def collect(database: Path, pair: str) -> dict:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        rows = connection.execute(
            """
            SELECT timestamp, trade_type, price, amount, trade_fee_in_quote,
                   order_id, exchange_trade_id
            FROM TradeFill
            WHERE symbol=?
            ORDER BY timestamp
            """,
            (pair,),
        ).fetchall()
    finally:
        connection.close()

    mark = mark_price(pair)
    buy_quote = Decimal("0")
    sell_quote = Decimal("0")
    buy_base = Decimal("0")
    sell_base = Decimal("0")
    fees = Decimal("0")
    fills = []
    for timestamp, side, raw_price, raw_amount, raw_fee, order_id, trade_id in rows:
        price = decimal_value(raw_price)
        amount = decimal_value(raw_amount)
        fee = decimal_value(raw_fee)
        quote = price * amount
        if str(side).upper() == "BUY":
            buy_quote += quote
            buy_base += amount
        elif str(side).upper() == "SELL":
            sell_quote += quote
            sell_base += amount
        fees += fee
        fills.append({
            "timestamp": int(timestamp),
            "side": str(side).upper(),
            "price": str(price),
            "amount": str(amount),
            "quote": str(quote),
            "fee_quote": str(fee),
            "order_id": order_id,
            "trade_id": trade_id,
        })
    net_base = buy_base - sell_base
    cash_flow = sell_quote - buy_quote
    mtm_pnl = cash_flow + net_base * mark - fees
    return {
        "pair": pair,
        "database": str(database),
        "database_bytes": database.stat().st_size,
        "mark_price": str(mark),
        "trades": len(rows),
        "buys": sum(row["side"] == "BUY" for row in fills),
        "sells": sum(row["side"] == "SELL" for row in fills),
        "buy_quote": str(buy_quote),
        "sell_quote": str(sell_quote),
        "buy_base": str(buy_base),
        "sell_base": str(sell_base),
        "net_base": str(net_base),
        "fees_quote": str(fees),
        "cash_flow_quote": str(cash_flow),
        "inventory_mtm_quote": str(net_base * mark),
        "flow_mtm_pnl_quote": str(mtm_pnl),
        "first_fill": fills[0]["timestamp"] if fills else None,
        "last_fill": fills[-1]["timestamp"] if fills else None,
        "fills": fills,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=Path("/home/ubuntu/extra_drive/hummingbot/api-files/bots/instances"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bots = []
    for pair, (bot, filename) in SPECS.items():
        database = args.instances_root / bot / "data" / filename
        bots.append({"bot": bot, **collect(database, pair)})
    total = sum(Decimal(bot["flow_mtm_pnl_quote"]) for bot in bots)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_status": "live_tradefill_snapshot",
        "valuation": "sell_quote - buy_quote + net_base * current_mark - recorded_fees",
        "combined_flow_mtm_pnl_quote": str(total),
        "bots": bots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "generated_at": payload["generated_at"],
        "combined_flow_mtm_pnl_quote": payload["combined_flow_mtm_pnl_quote"],
        "bots": [
            {key: bot[key] for key in (
                "pair", "trades", "buys", "sells", "fees_quote",
                "net_base", "mark_price", "flow_mtm_pnl_quote",
            )}
            for bot in bots
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
