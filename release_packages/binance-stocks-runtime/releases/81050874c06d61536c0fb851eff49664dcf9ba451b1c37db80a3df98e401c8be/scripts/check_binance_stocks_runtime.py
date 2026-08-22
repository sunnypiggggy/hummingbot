from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def request(base: str, auth: str, path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=data,
        headers={"Authorization": auth, "Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--create-paper-check", action="store_true")
    args = parser.parse_args()
    env = read_env(args.env)
    auth = "Basic " + base64.b64encode(
        f"{env['STOCKS_API_USERNAME']}:{env['STOCKS_API_PASSWORD']}".encode()
    ).decode()
    result = {"health": request(args.base, auth, "/stocks/health")[1]}
    if args.create_paper_check:
        result["create"] = request(
            args.base,
            auth,
            "/executors/",
            {
                "account_name": "stocks_managed",
                "controller_id": "stocks-runtime",
                "executor_config": {
                    "id": "paper-aapl-check-0001",
                    "type": "order_executor",
                    "connector_name": "binance_stocks",
                    "trading_pair": "AAPL-USDC",
                    "side": "BUY",
                    "amount": "0.1",
                    "price": "100",
                    "execution_strategy": "LIMIT",
                },
            },
        )[1]
    result["summary"] = request(args.base, auth, "/stocks/account-summary")[1]
    try:
        request(args.base, auth, "/trading/orders")
        result["disabled_raw_trading"] = "unexpectedly_allowed"
    except urllib.error.HTTPError as exc:
        result["disabled_raw_trading"] = exc.code
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
