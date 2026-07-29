#!/usr/bin/env python3
"""Cancel dedicated DCA account orders without printing decrypted credentials."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import signal
import time
from urllib.parse import urlencode

import requests

from hummingbot.client.config.config_crypt import ETHKeyFileSecretManger
from hummingbot.client.config.security import Security


BASE_URL = "https://api.binance.com"


def signed_request(session: requests.Session, method: str, path: str,
                   api_key: str, api_secret: str, **parameters):
    parameters.update({"timestamp": int(time.time() * 1000), "recvWindow": 10000})
    query = urlencode(parameters)
    signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    response = session.request(
        method,
        f"{BASE_URL}{path}?{query}&signature={signature}",
        headers={"X-MBX-APIKEY": api_key},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--freeze-pid-one", action="store_true")
    args = parser.parse_args()
    if not Security.login(ETHKeyFileSecretManger(os.environ["CONFIG_PASSWORD"])):
        raise RuntimeError("connector credential password validation failed")
    keys = Security.api_keys("binance")
    api_key = str(keys["binance_api_key"])
    api_secret = str(keys["binance_api_secret"])
    if args.freeze_pid_one:
        os.kill(1, signal.SIGSTOP)
    session = requests.Session()
    results = []
    for symbol in args.symbols:
        cancelled = signed_request(
            session, "DELETE", "/api/v3/openOrders", api_key, api_secret, symbol=symbol
        )
        remaining = signed_request(
            session, "GET", "/api/v3/openOrders", api_key, api_secret, symbol=symbol
        )
        results.append((symbol, len(cancelled), len(remaining)))
    for symbol, cancelled, remaining in results:
        print(f"{symbol}: cancelled={cancelled} remaining={remaining}", flush=True)
    if args.freeze_pid_one:
        os.kill(1, signal.SIGKILL)
    return 0 if all(remaining == 0 for _, _, remaining in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
