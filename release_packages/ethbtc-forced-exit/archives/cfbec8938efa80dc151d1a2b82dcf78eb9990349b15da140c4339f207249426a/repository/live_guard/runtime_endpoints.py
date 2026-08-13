"""Guarded endpoint overrides used by isolated scenario tests.

Production Guard processes may not redirect exchange or notification traffic.
Endpoint overrides are accepted only when the explicit scenario interlock is
enabled and the target is a loopback or Docker-internal host.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


OFFICIAL_BINANCE_API = "https://api.binance.com"
OFFICIAL_TELEGRAM_API = "https://api.telegram.org"
_SCENARIO_HOSTS = {
    "127.0.0.1", "localhost", "::1", "binance-sim", "control-sim",
    "telegram-sim", "host.docker.internal",
}


def scenario_mode() -> bool:
    return os.getenv("GUARD_SCENARIO_MODE", "false").lower() == "true"


def guarded_endpoint(value: str, *, official: str, purpose: str) -> str:
    endpoint = str(value or official).rstrip("/")
    if endpoint == official.rstrip("/"):
        if scenario_mode():
            raise RuntimeError(
                f"{purpose} scenario mode requires an explicit isolated endpoint"
            )
        return endpoint
    if not scenario_mode():
        raise RuntimeError(
            f"{purpose} endpoint override requires GUARD_SCENARIO_MODE=true"
        )
    scenario_id = os.getenv("GUARD_SCENARIO_ID", "").strip()
    if not scenario_id:
        raise RuntimeError("GUARD_SCENARIO_ID is required in scenario mode")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _SCENARIO_HOSTS:
        raise RuntimeError(
            f"{purpose} scenario endpoint must use an isolated host: {endpoint}"
        )
    return endpoint


def binance_api_base() -> str:
    return guarded_endpoint(
        os.getenv("BINANCE_API_BASE_URL", OFFICIAL_BINANCE_API),
        official=OFFICIAL_BINANCE_API,
        purpose="Binance",
    )


def telegram_api_base() -> str:
    return guarded_endpoint(
        os.getenv("TELEGRAM_API_BASE_URL", OFFICIAL_TELEGRAM_API),
        official=OFFICIAL_TELEGRAM_API,
        purpose="Telegram",
    )
