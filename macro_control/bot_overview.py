from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock

from .hummingbot_api import HummingbotAPI


class BotOverviewUnavailable(RuntimeError):
    pass


class BotOverviewProvider:
    """Build a secret-free overview of live strategy bots and Grid risk PnL."""

    PERFORMANCE_FIELDS = (
        "realized_pnl_quote",
        "unrealized_pnl_quote",
        "global_pnl_quote",
        "global_pnl_pct",
        "volume_traded",
    )

    def __init__(self, api: HummingbotAPI, grid_guard_path: Path) -> None:
        self.api = api
        self.grid_guard_path = grid_guard_path
        self._lock = Lock()

    @classmethod
    def _controllers(cls, raw: object) -> dict:
        if not isinstance(raw, dict):
            return {}
        result = {}
        for controller_id, report in raw.items():
            if not isinstance(report, dict):
                continue
            performance = report.get("performance", {})
            result[str(controller_id)] = {
                "status": str(report.get("status", "unknown")),
                "performance": {
                    field: performance[field]
                    for field in cls.PERFORMANCE_FIELDS
                    if isinstance(performance, dict) and field in performance
                },
            }
        return result

    @staticmethod
    def _grid_report(state: dict, *, now: float) -> dict:
        bots = state.get("bots", {})
        bot_name = "grid-live-fdusd-400"
        bot = bots.get(bot_name, {}) if isinstance(bots, dict) else {}
        latest = bot.get("latest", {}) if isinstance(bot, dict) else {}
        observed_at = float(latest.get("observed_at", 0) or 0)
        age = max(0.0, now - observed_at) if observed_at else None
        return {
            "bot_name": bot_name,
            "currency": "FDUSD",
            "pnl_method": "strategy_owned_mark_to_market",
            "mtm_pnl_quote": latest.get("pnl"),
            "equity_quote": latest.get("equity"),
            "peak_equity_quote": latest.get("peak_equity"),
            "drawdown_pct": latest.get("drawdown_pct"),
            "pairs": latest.get("pairs", {}),
            "armed": bool(state.get("armed")),
            "shadow": bool(state.get("shadow")),
            "emergency_ready": bool(state.get("emergency_ready")),
            "technical_buy_gate": state.get("technical_buy_gate", {}),
            "observed_at_epoch": observed_at or None,
            "data_age_seconds": age,
            "fresh": age is not None and age <= 90,
        }

    def snapshot(self) -> dict:
        try:
            api_payload = self.api.all_bot_statuses()
            with self._lock:
                grid_state = json.loads(
                    self.grid_guard_path.read_text(encoding="utf-8")
                )
        except Exception as exc:
            raise BotOverviewUnavailable(
                f"bot overview source is unavailable: {type(exc).__name__}"
            ) from exc
        raw_bots = api_payload.get("data", {})
        if not isinstance(raw_bots, dict):
            raise BotOverviewUnavailable("orchestration status data is invalid")
        bots = []
        for bot_name, raw in sorted(raw_bots.items()):
            if not isinstance(raw, dict):
                continue
            bots.append({
                "bot_name": str(bot_name),
                "status": str(raw.get("status", "unknown")),
                "source": str(raw.get("source", "unknown")),
                "recently_active": bool(raw.get("recently_active")),
                "error_count": len(raw.get("error_logs", []))
                if isinstance(raw.get("error_logs", []), list) else 0,
                "controllers": self._controllers(raw.get("performance", {})),
            })
        now = time.time()
        return {
            "schema_version": "hermes-bot-overview-v1",
            "generated_at_epoch": now,
            "running_count": sum(bot["status"] == "running" for bot in bots),
            "bots": bots,
            "grid": self._grid_report(grid_state, now=now),
            "read_only": True,
        }
