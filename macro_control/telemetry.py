from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Callable


def build_sanitized_snapshot(
    bot_statuses: list[dict],
    market: dict,
    *,
    observed_at: datetime | None = None,
) -> dict:
    """Build a sanitized snapshot without credentials or account-wide balances."""
    observed_at = observed_at or datetime.now(timezone.utc)
    bots: list[dict] = []
    total_equity = 0.0
    total_peak = 0.0
    all_healthy = True
    observed_age = max(
        0.0, (datetime.now(timezone.utc) - observed_at.astimezone(timezone.utc)).total_seconds()
    )
    for raw in bot_statuses:
        equity = float(raw.get("strategy_equity", 0.0))
        peak = max(float(raw.get("peak_strategy_equity", equity)), equity)
        total_equity += equity
        total_peak += peak
        fresh = float(raw.get("data_age_seconds", 10**9)) <= 60
        healthy = bool(raw.get("healthy", False)) and fresh
        all_healthy = all_healthy and healthy
        bots.append(
            {
                "bot_name": raw["bot_name"],
                "controller_name": raw["controller_name"],
                "trading_pair": raw["trading_pair"],
                "strategy_equity": equity,
                "peak_strategy_equity": peak,
                "drawdown": 0.0 if peak <= 0 else max(0.0, 1 - equity / peak),
                "strategy_owned_long_base": float(
                    raw.get("strategy_owned_long_base", 0.0)
                ),
                "active_executors": int(raw.get("active_executors", 0)),
                "active_buy_executors": int(raw.get("active_buy_executors", 0)),
                "trading_buy_executors": int(raw.get("trading_buy_executors", 0)),
                "active_sell_executors": int(raw.get("active_sell_executors", 0)),
                "trading_sell_executors": int(
                    raw.get("trading_sell_executors", 0)
                ),
                "open_orders": int(raw.get("open_orders", 0)),
                "fills": int(raw.get("fills", 0)),
                "macro_buy_enabled": bool(raw.get("macro_buy_enabled", True)),
                "macro_sell_enabled": bool(
                    raw.get("macro_sell_enabled", True)
                ),
                "macro_decision_id": str(raw.get("macro_decision_id", "")),
                "data_age_seconds": float(raw.get("data_age_seconds", 10**9)),
                "observation_age_seconds": float(
                    raw.get("observation_age_seconds", raw.get("data_age_seconds", 10**9))
                ),
                "database_event_age_seconds": float(
                    raw.get("database_event_age_seconds", 10**9)
                ),
                "database_event_at": raw.get("database_event_at"),
                "healthy": healthy,
                "hard_circuit_breaker": bool(raw.get("hard_circuit_breaker", False)),
                "buy_circuit_breaker": bool(raw.get("buy_circuit_breaker", False)),
            }
        )
    drawdown = 0.0 if total_peak <= 0 else max(0.0, 1 - total_equity / total_peak)
    safe_market = {
        pair: {
            "mid_price": float(value["mid_price"]),
            "spread_bps": float(value["spread_bps"]),
            "volatility_ratio_30m": float(value["volatility_ratio_30m"]),
            "data_age_seconds": float(value["data_age_seconds"]),
        }
        for pair, value in market.items()
    }
    all_healthy = all_healthy and all(
        value["data_age_seconds"] <= 60 for value in safe_market.values()
    )
    all_healthy = all_healthy and observed_age <= 60
    canonical = json.dumps(
        {"at": observed_at.isoformat(), "bots": bots, "market": safe_market},
        sort_keys=True,
        separators=(",", ":"),
    )
    snapshot_id = hashlib.sha256(canonical.encode()).hexdigest()[:24]
    return {
        "snapshot_id": snapshot_id,
        "observed_at": observed_at.isoformat(),
        "observation_age_seconds": observed_age,
        "portfolio": {
            "strategy_equity": total_equity,
            "peak_strategy_equity": total_peak,
            "drawdown": drawdown,
        },
        "bots": bots,
        "market": safe_market,
        "volatility_ratio_30m": max(
            (value["volatility_ratio_30m"] for value in safe_market.values()),
            default=0.0,
        ),
        "telemetry_healthy": all_healthy,
        "macro_policy_version": "dca-macro-v3",
    }


class TelemetryProvider:
    def __init__(
        self,
        bot_reader: Callable[[], list[dict]],
        market_reader: Callable[[], dict],
    ) -> None:
        self.bot_reader = bot_reader
        self.market_reader = market_reader

    def snapshot(self) -> dict:
        return build_sanitized_snapshot(self.bot_reader(), self.market_reader())
