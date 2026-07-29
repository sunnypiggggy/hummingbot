#!/usr/bin/env python3
"""Print a secret-free, read-only Grid balance and open-order summary."""

import json
import os
from pathlib import Path

from dca_live_guard import BinanceEmergencyClient


def main() -> int:
    secret = Path(os.getenv(
        "GRID_BINANCE_EMERGENCY_CREDENTIALS_FILE",
        "/run/secrets/grid_binance_emergency_credentials",
    ))
    client = BinanceEmergencyClient.from_secret_file(secret)
    account = client._signed("GET", "/api/v3/account")
    balances = {
        row["asset"]: {"free": row["free"], "locked": row["locked"]}
        for row in account["balances"]
        if row["asset"] in {"FDUSD", "BTC", "ETH"}
    }
    orders = {}
    for pair in ("BTC-FDUSD", "ETH-FDUSD"):
        rows = client.open_orders(pair)
        orders[pair] = {
            "count": len(rows),
            "BUY": sum(row["side"] == "BUY" for row in rows),
            "SELL": sum(row["side"] == "SELL" for row in rows),
        }
    print(json.dumps({"balances": balances, "orders": orders}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
