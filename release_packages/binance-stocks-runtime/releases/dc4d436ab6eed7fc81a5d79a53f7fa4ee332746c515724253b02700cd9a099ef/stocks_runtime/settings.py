from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit


def dedicated_database_url(source: str, database_name: str = "hummingbot_stocks") -> str:
    parsed = urlsplit(source)
    if not database_name.replace("_", "").isalnum():
        raise ValueError("unsafe Stocks database name")
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, parsed.fragment))


@dataclass(frozen=True)
class StocksRuntimeSettings:
    mode: str
    account_name: str
    credentials_file: Path
    disclaimer_confirmed: bool
    live_authorized: bool
    order_prefix: str
    max_order_notional: Decimal
    max_symbol_exposure: Decimal
    max_managed_exposure: Decimal
    daily_loss_limit: Decimal
    database_url: str
    rest_url: str
    ws_url: str
    scenario_mode: bool
    paper_initial_usdc: Decimal
    paper_fill_latency_ms: int
    paper_market_timeout_seconds: float
    paper_quote_max_age_seconds: float
    paper_equity_snapshot_seconds: float

    @classmethod
    def from_env(cls) -> "StocksRuntimeSettings":
        mode = os.getenv("BINANCE_STOCKS_RUNTIME_MODE", "PAPER").strip().upper()
        if mode not in {"PAPER", "SHADOW", "LIVE"}:
            raise ValueError("BINANCE_STOCKS_RUNTIME_MODE must be PAPER, SHADOW, or LIVE")
        if mode != "PAPER" and (not os.getenv("STOCKS_API_USERNAME") or not os.getenv("STOCKS_API_PASSWORD")):
            raise ValueError("SHADOW/LIVE requires independent STOCKS_API_USERNAME and STOCKS_API_PASSWORD")
        live_authorized = os.getenv("BINANCE_STOCKS_LIVE_AUTHORIZED", "false").lower() == "true"
        if mode != "LIVE" and live_authorized:
            raise ValueError("live authorization cannot be enabled outside LIVE mode")
        scenario_mode = os.getenv("BINANCE_STOCKS_SCENARIO_MODE", "false").lower() == "true"
        rest_url = os.getenv("BINANCE_STOCKS_REST_URL", "https://api.binance.com").rstrip("/")
        ws_url = os.getenv("BINANCE_STOCKS_WS_URL", "wss://nbstream.binance.com/equity").rstrip("/")
        if not scenario_mode and (
            rest_url != "https://api.binance.com" or ws_url != "wss://nbstream.binance.com/equity"
        ):
            raise ValueError("custom Stocks REST/WS endpoints require BINANCE_STOCKS_SCENARIO_MODE=true")
        return cls(
            mode=mode,
            account_name=os.getenv("BINANCE_STOCKS_ACCOUNT", "stocks_managed"),
            credentials_file=Path(os.getenv(
                "BINANCE_STOCKS_CREDENTIALS_FILE", "/run/secrets/binance_stocks_credentials"
            )),
            disclaimer_confirmed=os.getenv("BINANCE_STOCKS_DISCLAIMER_CONFIRMED", "false").lower() == "true",
            live_authorized=live_authorized,
            order_prefix=os.getenv("BINANCE_STOCKS_ORDER_PREFIX", "x-HBSTK"),
            max_order_notional=Decimal(os.getenv("BINANCE_STOCKS_MAX_ORDER_USDC", "500")),
            max_symbol_exposure=Decimal(os.getenv("BINANCE_STOCKS_MAX_SYMBOL_USDC", "1000")),
            max_managed_exposure=Decimal(os.getenv("BINANCE_STOCKS_MAX_EXPOSURE_USDC", "2000")),
            daily_loss_limit=Decimal(os.getenv("BINANCE_STOCKS_DAILY_LOSS_USDC", "200")),
            database_url=dedicated_database_url(
                os.environ["DATABASE_URL"], os.getenv("BINANCE_STOCKS_DATABASE_NAME", "hummingbot_stocks")
            ),
            rest_url=rest_url,
            ws_url=ws_url,
            scenario_mode=scenario_mode,
            paper_initial_usdc=Decimal(os.getenv("BINANCE_STOCKS_PAPER_INITIAL_USDC", "2000")),
            paper_fill_latency_ms=int(os.getenv("BINANCE_STOCKS_PAPER_FILL_LATENCY_MS", "1000")),
            paper_market_timeout_seconds=float(
                os.getenv("BINANCE_STOCKS_PAPER_MARKET_TIMEOUT_SECONDS", "5")
            ),
            paper_quote_max_age_seconds=float(
                os.getenv("BINANCE_STOCKS_PAPER_QUOTE_MAX_AGE_SECONDS", "10")
            ),
            paper_equity_snapshot_seconds=float(
                os.getenv("BINANCE_STOCKS_PAPER_EQUITY_SNAPSHOT_SECONDS", "60")
            ),
        )

    def read_credentials(self) -> Dict[str, Any]:
        if not self.credentials_file.exists():
            if self.mode == "PAPER":
                return {}
            raise FileNotFoundError(f"Stocks credentials secret is missing: {self.credentials_file}")
        payload = json.loads(self.credentials_file.read_text(encoding="utf-8"))
        api_key = str(payload.get("api_key", payload.get("binance_stocks_api_key", ""))).strip()
        api_secret = str(payload.get("api_secret", payload.get("binance_stocks_api_secret", ""))).strip()
        if not api_key:
            if self.mode == "PAPER":
                return {}
            raise ValueError("Stocks credentials secret must contain api_key and api_secret")
        if not api_secret and self.mode != "PAPER":
            raise ValueError("SHADOW/LIVE Stocks credentials must contain api_key and api_secret")
        return {
            "binance_stocks_api_key": api_key,
            "binance_stocks_api_secret": api_secret,
            "quote_asset": "USDC",
            "wallet_type": "CARD",
            "trading_session": "EXTENDED",
            "time_in_force": "DAY",
            "disclaimer_confirmed": self.disclaimer_confirmed,
        }
